"""
止损止盈策略装饰器

包装任意策略，自动检测持仓的止损/止盈条件并生成 SELL 信号。
"""

from typing import List, Dict, Optional
import pandas as pd

from dquant.strategy.base import BaseStrategy, Signal, SignalType
from dquant.constants import DEFAULT_STOP_LOSS, DEFAULT_TAKE_PROFIT
from dquant.logger import get_logger

logger = get_logger(__name__)


class StopLossTakeProfitStrategy(BaseStrategy):
    """
    止损止盈策略装饰器

    包装任意策略，在 generate_signals() 中:
    1. 先调用被包装策略获取原始信号
    2. 检查所有持仓是否触发止损或止盈
    3. 将触发的 SELL 信号追加到结果中

    Usage:
        base = MoneyFlowStrategy(top_k=10)
        strategy = StopLossTakeProfitStrategy(
            base_strategy=base,
            stop_loss=0.05,     # 5% 止损
            take_profit=0.10,   # 10% 止盈
        )
        signals = strategy.generate_signals(data)
    """

    def __init__(
        self,
        base_strategy: BaseStrategy,
        stop_loss: float = DEFAULT_STOP_LOSS,
        take_profit: float = DEFAULT_TAKE_PROFIT,
        name: str = "StopLossTakeProfit",
    ):
        super().__init__(name=name)
        self.base_strategy = base_strategy
        self.stop_loss = stop_loss
        self.take_profit = take_profit

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        生成信号：原始信号 + 止损止盈 SELL 信号
        """
        # 获取原始信号
        signals = self.base_strategy.generate_signals(data)

        # 检查止损止盈
        stop_signals = self._check_stop_conditions(data)
        signals.extend(stop_signals)

        return signals

    def _check_stop_conditions(self, data: pd.DataFrame) -> List[Signal]:
        """
        检查止损止盈条件

        需要数据中包含 position_avg_cost 列（持仓成本）和 close 列。
        如果数据不含 position_avg_cost，则跳过检查。
        """
        sell_signals = []

        if 'close' not in data.columns:
            return sell_signals

        # 只检查最新日期
        if len(data) == 0:
            return sell_signals

        latest_date = data.index.max()
        latest = data[data.index == latest_date]

        for _, row in latest.iterrows():
            symbol = row.get('symbol', '')
            close = row.get('close', 0)
            avg_cost = row.get('position_avg_cost', None)

            if avg_cost is None or avg_cost <= 0 or close <= 0:
                continue

            pct_change = (close - avg_cost) / avg_cost

            if pct_change <= -self.stop_loss:
                # 止损
                sell_signals.append(Signal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    strength=1.0,
                    price=close,
                    timestamp=latest_date,
                    metadata={
                        'reason': 'stop_loss',
                        'pct_change': pct_change,
                        'avg_cost': avg_cost,
                    }
                ))
                logger.debug(
                    f"[SL] 止损触发: {symbol} "
                    f"亏损 {pct_change:.1%} <= -{self.stop_loss:.1%}"
                )

            elif pct_change >= self.take_profit:
                # 止盈
                sell_signals.append(Signal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    strength=1.0,
                    price=close,
                    timestamp=latest_date,
                    metadata={
                        'reason': 'take_profit',
                        'pct_change': pct_change,
                        'avg_cost': avg_cost,
                    }
                ))
                logger.debug(
                    f"[TP] 止盈触发: {symbol} "
                    f"盈利 {pct_change:.1%} >= {self.take_profit:.1%}"
                )

        return sell_signals

    def on_bar(self, bar: pd.Series) -> Optional[Signal]:
        """逐条模式：委托给基础策略"""
        return self.base_strategy.on_bar(bar)

    def on_tick(self, tick: dict) -> Optional[Signal]:
        """Tick 模式：委托给基础策略"""
        return self.base_strategy.on_tick(tick)
