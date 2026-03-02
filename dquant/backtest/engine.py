"""
回测引擎
"""

from typing import Optional, Dict, List
from datetime import datetime
import pandas as pd
import numpy as np

from dquant.strategy.base import BaseStrategy, Signal, SignalType
from dquant.backtest.portfolio import Portfolio
from dquant.backtest.metrics import Metrics
from dquant.backtest.result import BacktestResult
from dquant.constants import DEFAULT_COMMISSION, DEFAULT_SLIPPAGE, DEFAULT_STAMP_DUTY, DEFAULT_INITIAL_CASH, MIN_SHARES, DEFAULT_WINDOW


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
        # 生成信号
        signals = self.strategy.generate_signals(self.data)

        # 按日期分组信号
        signal_df = pd.DataFrame([s.to_dict() for s in signals])
        if len(signal_df) == 0:
            print("[WARN] No signals generated")
            return self._create_result()

        signal_df['timestamp'] = pd.to_datetime(signal_df['timestamp'])
        daily_signals = signal_df.groupby('timestamp')

        # 遍历每个交易日
        dates = sorted(self.data.index.unique())

        for i, date in enumerate(dates):
            # 获取当日价格
            daily_data = self.data[self.data.index == date]
            prices = dict(zip(daily_data['symbol'], daily_data['close']))

            # 更新持仓价格
            self.portfolio.update_prices(prices, date)

            # 检查是否有调仓信号
            if date in daily_signals.groups:
                # 获取当日信号
                day_signals = daily_signals.get_group(date)

                # 计算目标权重
                buy_signals = day_signals[day_signals['signal_type'] == 1]
                if len(buy_signals) > 0:
                    target_weights = dict(zip(
                        buy_signals['symbol'],
                        buy_signals['strength']
                    ))

                    # 再平衡
                    self.portfolio.rebalance(
                        target_weights,
                        prices,
                        self.commission
                    )

                    # 记录交易
                    for _, row in buy_signals.iterrows():
                        self.trades.append({
                            'date': date,
                            'symbol': row['symbol'],
                            'action': 'BUY',
                            'price': prices.get(row['symbol'], 0),
                            'score': row['metadata'].get('score', 0) if row['metadata'] else 0,
                        })

        return self._create_result()

    def _create_result(self) -> BacktestResult:
        """创建回测结果"""
        nav_df = self.portfolio.to_dataframe()

        # 计算绩效
        metrics = Metrics.from_nav(nav_df['nav'])
        metrics.total_trades = len(self.trades)

        # 创建交易记录 DataFrame
        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()

        return BacktestResult(
            portfolio=self.portfolio,
            trades=trades_df,
            metrics=metrics,
        )
