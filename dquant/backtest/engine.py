"""
回测引擎
"""

from typing import Dict, List, Optional

import pandas as pd

from dquant.backtest.metrics import Metrics
from dquant.backtest.portfolio import Portfolio
from dquant.backtest.result import BacktestResult
from dquant.constants import DEFAULT_COMMISSION, DEFAULT_INITIAL_CASH, DEFAULT_SLIPPAGE
from dquant.logger import get_logger
from dquant.strategy.base import BaseStrategy, Signal

logger = get_logger(__name__)

# A 股涨跌停限制
DEFAULT_PRICE_LIMIT = 0.10  # 主板 ±10%
ST_PRICE_LIMIT = 0.05  # ST 板块 ±5%
BJ_PRICE_LIMIT = 0.30  # 北交所 ±30%


def _get_price_limit(symbol: str) -> float:
    """根据股票代码判断涨跌停幅度"""
    code = symbol.split(".")[0] if "." in symbol else symbol
    # 北交所: 4/8 开头
    if code and code[0] in ("4", "8"):
        return BJ_PRICE_LIMIT
    return DEFAULT_PRICE_LIMIT


class BacktestEngine:
    """
    向量化回测引擎

    支持多股票、多策略的回测。
    信号在 T 日产生，T+1 日执行，消除前视偏差。
    """

    def __init__(
        self,
        data: pd.DataFrame,
        strategy: BaseStrategy,
        initial_cash: float = DEFAULT_INITIAL_CASH,
        commission: float = DEFAULT_COMMISSION,
        slippage: float = DEFAULT_SLIPPAGE,
        benchmark: Optional[str] = None,
        enforce_price_limit: bool = True,
    ):
        """
        初始化回测引擎

        Args:
            data: 市场数据 (index=date, columns=[symbol, open, high, low, close, volume, ...])
            strategy: 交易策略
            initial_cash: 初始资金
            commission: 手续费率
            slippage: 滑点
            benchmark: 基准代码
            enforce_price_limit: 是否强制涨跌停限制（A股 ±10%，北交所 ±30%）
        """
        self.data = data
        self.strategy = strategy
        self.initial_cash = initial_cash
        self.commission = commission
        self.slippage = slippage
        self.benchmark = benchmark
        self.enforce_price_limit = enforce_price_limit

        self.portfolio = Portfolio(initial_cash=initial_cash)
        self.trades: List[dict] = []

        # 预计算前日收盘价（用于涨跌停判断）
        self._prev_close: Dict[str, float] = {}

    def run(self) -> BacktestResult:
        """
        运行回测

        信号生成与执行时序:
        1. 预先一次性生成所有信号 (O(N))
        2. 按信号 timestamp 归组
        3. 信号在 timestamp 的下一个交易日执行 (T+1)
        """
        if self.data is None or self.data.empty:
            logger.warning("[BACKTEST] 数据为空，返回空结果")
            return self._create_result()

        dates = sorted(self.data.index.unique())
        data_by_date = dict(iter(self.data.groupby(self.data.index)))

        # ---- 预计算信号，构建 T+1 执行映射 ----
        exec_map = self._build_exec_map(dates)

        # ---- 逐日执行 ----
        for date in dates:
            daily_data = data_by_date.get(date)
            if daily_data is None:
                continue

            prices = dict(zip(daily_data["symbol"], daily_data["close"]))
            opens = dict(zip(daily_data["symbol"], daily_data["open"])) \
                if "open" in daily_data.columns else prices

            buy_trade_prices = {
                symbol: price * (1 + self.slippage) for symbol, price in prices.items()
            }
            sell_trade_prices = {
                symbol: price * (1 - self.slippage) for symbol, price in prices.items()
            }

            # 涨跌停检查
            limit_up = set()
            limit_down = set()
            if self.enforce_price_limit:
                limit_up, limit_down = self._check_price_limits(prices, opens)

            # 更新持仓价格 (在执行交易前记录 NAV)
            self.portfolio.update_prices(prices, date)

            # 取当日应执行的信号
            date_key = pd.Timestamp(date).normalize()
            day_signals = exec_map.get(date_key, [])

            # 过滤涨跌停信号：涨停日不可买入，跌停日不可卖出
            buy_sigs = [s for s in day_signals if s.is_buy and s.symbol not in limit_up]
            sell_sigs = [s for s in day_signals if s.is_sell and s.symbol not in limit_down]

            # 执行买入
            if buy_sigs:
                target_weights = {s.symbol: s.strength for s in buy_sigs}
                self.portfolio.rebalance(
                    target_weights,
                    buy_trade_prices,
                    self.commission,
                )

                for s in buy_sigs:
                    self.trades.append(
                        {
                            "date": date,
                            "symbol": s.symbol,
                            "action": "BUY",
                            "price": prices.get(s.symbol, 0),
                            "score": s.metadata.get("score", 0) if s.metadata else 0,
                        }
                    )

            # 执行卖出
            for s in sell_sigs:
                if s.symbol in self.portfolio.positions:
                    pos = self.portfolio.positions[s.symbol]
                    self.portfolio.sell(
                        s.symbol,
                        pos.shares,
                        sell_trade_prices.get(s.symbol, pos.current_price),
                        self.commission,
                    )
                    self.trades.append(
                        {
                            "date": date,
                            "symbol": s.symbol,
                            "action": "SELL",
                            "price": prices.get(s.symbol, 0),
                            "score": s.metadata.get("score", 0) if s.metadata else 0,
                        }
                    )

            # 保存当日收盘价供下日涨跌停判断
            self._prev_close = dict(prices)

        return self._create_result()

    def _check_price_limits(
        self, prices: Dict[str, float], opens: Dict[str, float]
    ) -> tuple:
        """
        检查涨跌停股票

        Returns:
            (涨停股票集合, 跌停股票集合)
        """
        limit_up = set()
        limit_down = set()

        for symbol, close_price in prices.items():
            prev = self._prev_close.get(symbol)
            if prev is None or prev <= 0:
                continue

            limit_pct = _get_price_limit(symbol)
            pct_change = (close_price - prev) / prev

            # 涨停：涨幅达到限制（精确到小数点后 4 位以避免浮点误差）
            if round(pct_change, 4) >= round(limit_pct, 4):
                limit_up.add(symbol)
            # 跌停：跌幅达到限制
            elif round(pct_change, 4) <= round(-limit_pct, 4):
                limit_down.add(symbol)

        return limit_up, limit_down

    def _build_exec_map(self, dates: list) -> Dict[pd.Timestamp, List[Signal]]:
        """
        预计算信号并构建 T+1 执行映射

        Returns:
            {execution_date: [signals]} — 信号在 timestamp 的下一个交易日执行
        """
        signals = self.strategy.generate_signals(self.data)
        if not signals:
            return {}

        # 构建 sorted dates 索引，用于快速查找下一个交易日
        sorted_dates = sorted(pd.Timestamp(d).normalize() for d in dates)
        date_index_map = {d: i for i, d in enumerate(sorted_dates)}

        exec_map: Dict[pd.Timestamp, List[Signal]] = {}

        for sig in signals:
            # 信号无 timestamp → 分配数据最后一天（保守处理：不执行）
            if sig.timestamp is None:
                logger.debug(f"[BACKTEST] 信号无 timestamp，跳过: {sig.symbol} {sig.signal_type}")
                continue

            sig_date = pd.Timestamp(sig.timestamp).normalize()

            # 查找下一个交易日 (T+1)
            idx = date_index_map.get(sig_date)
            if idx is None:
                # 信号日期不在交易日列表中，找最近的后一个交易日
                continue

            if idx + 1 < len(sorted_dates):
                exec_date = sorted_dates[idx + 1]
            else:
                # 信号在最后一个交易日产生，无后续执行日
                continue

            exec_map.setdefault(exec_date, []).append(sig)

        return exec_map

    def _create_result(self) -> BacktestResult:
        """创建回测结果"""
        nav_df = self.portfolio.to_dataframe()

        # 计算绩效
        metrics = Metrics.from_nav(nav_df["nav"])
        metrics.total_trades = len(self.trades)

        # 计算基准净值
        benchmark_nav = self._compute_benchmark_nav()

        # 创建交易记录 DataFrame
        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()

        return BacktestResult(
            portfolio=self.portfolio,
            trades=trades_df,
            metrics=metrics,
            benchmark_nav=benchmark_nav,
        )

    def _compute_benchmark_nav(self) -> Optional[pd.Series]:
        """计算基准净值曲线"""
        if not self.benchmark:
            return None

        # 从数据中筛选 benchmark 股票
        if "symbol" not in self.data.columns:
            return None

        bench_data = self.data[self.data["symbol"] == self.benchmark]
        if bench_data.empty:
            return None

        if "close" not in bench_data.columns:
            return None

        close = bench_data["close"]
        bench_nav = close / close.iloc[0]
        bench_nav.index = bench_data.index
        return bench_nav
