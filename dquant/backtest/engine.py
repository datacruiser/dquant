"""
回测引擎
"""

from typing import Optional, Dict, List
from datetime import datetime
import pandas as pd

from dquant.strategy.base import BaseStrategy, Signal, SignalType
from dquant.backtest.portfolio import Portfolio
from dquant.backtest.metrics import Metrics
from dquant.backtest.result import BacktestResult
from dquant.constants import DEFAULT_COMMISSION, DEFAULT_SLIPPAGE, DEFAULT_INITIAL_CASH
from dquant.logger import get_logger

logger = get_logger(__name__)


class BacktestEngine:
    """
    向量化回测引擎

    支持多股票、多策略的回测。
    """

    def __init__(
        self,
        data: pd.DataFrame,
        strategy: BaseStrategy,
        initial_cash: float = DEFAULT_INITIAL_CASH,
        commission: float = DEFAULT_COMMISSION,
        slippage: float = DEFAULT_SLIPPAGE,
        benchmark: Optional[str] = None,
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
        """
        self.data = data
        self.strategy = strategy
        self.initial_cash = initial_cash
        self.commission = commission
        self.slippage = slippage
        self.benchmark = benchmark

        self.portfolio = Portfolio(initial_cash=initial_cash)
        self.trades: List[dict] = []

    def run(self) -> BacktestResult:
        """运行回测"""
        # 空数据保护
        if self.data is None or self.data.empty:
            logger.warning("[BACKTEST] 数据为空，返回空结果")
            return self._create_result()

        # 生成信号
        signals = self.strategy.generate_signals(self.data)

        # 按日期分组信号
        signal_df = pd.DataFrame([s.to_dict() for s in signals])
        if len(signal_df) == 0:
            return self._create_result()

        signal_df['timestamp'] = pd.to_datetime(signal_df['timestamp'])
        daily_signals = signal_df.groupby('timestamp')

        # 预分区数据避免 O(n²) 扫描
        dates = sorted(self.data.index.unique())
        data_by_date = dict(iter(self.data.groupby(self.data.index)))

        for date in dates:
            # O(1) 获取当日数据
            daily_data = data_by_date.get(date)
            if daily_data is None:
                continue
            prices = dict(zip(daily_data['symbol'], daily_data['close']))

            # 构建滑点调整后的交易价格
            # 买入加滑点（实际成交价更高），卖出减滑点（实际成交价更低）
            buy_trade_prices = {
                s: p * (1 + self.slippage) for s, p in prices.items()
            }
            sell_trade_prices = {
                s: p * (1 - self.slippage) for s, p in prices.items()
            }

            # 更新持仓价格
            self.portfolio.update_prices(prices, date)

            # 检查是否有调仓信号
            if date not in daily_signals.groups:
                continue

            day_signals = daily_signals.get_group(date)

            # 处理买入信号
            buy_signals = day_signals[day_signals['signal_type'] == SignalType.BUY.value]
            if len(buy_signals) > 0:
                target_weights = dict(zip(
                    buy_signals['symbol'],
                    buy_signals['strength']
                ))

                self.portfolio.rebalance(
                    target_weights,
                    buy_trade_prices,
                    self.commission
                )

                for _, row in buy_signals.iterrows():
                    self.trades.append({
                        'date': date,
                        'symbol': row['symbol'],
                        'action': 'BUY',
                        'price': prices.get(row['symbol'], 0),
                        'score': row['metadata'].get('score', 0) if row['metadata'] else 0,
                    })

            # 处理卖出信号
            sell_signals = day_signals[day_signals['signal_type'] == SignalType.SELL.value]
            for _, row in sell_signals.iterrows():
                symbol = row['symbol']
                if symbol in self.portfolio.positions:
                    pos = self.portfolio.positions[symbol]
                    self.portfolio.sell(
                        symbol, pos.shares,
                        sell_trade_prices.get(symbol, pos.current_price),
                        self.commission,
                    )
                    self.trades.append({
                        'date': date,
                        'symbol': symbol,
                        'action': 'SELL',
                        'price': prices.get(symbol, 0),
                        'score': row['metadata'].get('score', 0) if row['metadata'] else 0,
                    })

        return self._create_result()

    def _create_result(self) -> BacktestResult:
        """创建回测结果"""
        nav_df = self.portfolio.to_dataframe()

        # 计算绩效
        metrics = Metrics.from_nav(nav_df['nav'])
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
        if 'symbol' not in self.data.columns:
            return None

        bench_data = self.data[self.data['symbol'] == self.benchmark]
        if bench_data.empty:
            return None

        if 'close' not in bench_data.columns:
            return None

        close = bench_data['close']
        bench_nav = close / close.iloc[0]
        bench_nav.index = bench_data.index
        return bench_nav
