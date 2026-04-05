"""
Phase 2 Step 6: Strategy on_bar 集成测试
"""

from datetime import datetime
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from dquant.strategy.base import BaseStrategy, Signal, SignalType


class TestOnBarDetection:
    """检测策略是否 override 了 on_bar"""

    def test_default_strategy_not_override(self):
        """BaseStrategy 的 on_bar 返回 None"""

        class DefaultStrategy(BaseStrategy):
            def generate_signals(self, data):
                return []

        strategy = DefaultStrategy()
        assert type(strategy).on_bar is BaseStrategy.on_bar

    def test_override_detected(self):
        """Override on_bar 的策略被检测到"""

        class BarStrategy(BaseStrategy):
            def generate_signals(self, data):
                return []

            def on_bar(self, bar):
                if bar.get("close", 0) > 10:
                    return Signal(
                        symbol=bar.get("symbol", ""), signal_type=SignalType.BUY
                    )
                return None

        strategy = BarStrategy()
        assert type(strategy).on_bar is not BaseStrategy.on_bar

    def test_on_bar_returns_signal(self):
        """on_bar 返回 Signal"""

        class BarStrategy(BaseStrategy):
            def generate_signals(self, data):
                return []

            def on_bar(self, bar):
                if bar.get("close", 0) > 10:
                    return Signal(
                        symbol=bar.get("symbol", "TEST"),
                        signal_type=SignalType.BUY,
                        strength=0.5,
                    )
                return None

        strategy = BarStrategy()

        # 行情低于阈值 → None
        low_bar = pd.Series({"symbol": "TEST", "close": 5.0})
        assert strategy.on_bar(low_bar) is None

        # 行情高于阈值 → Signal
        high_bar = pd.Series({"symbol": "TEST", "close": 15.0})
        sig = strategy.on_bar(high_bar)
        assert sig is not None
        assert sig.symbol == "TEST"
        assert sig.is_buy

    def test_on_bar_with_dataframe_iteration(self):
        """模拟 live loop 的逐行调用"""

        class BarStrategy(BaseStrategy):
            def generate_signals(self, data):
                return []

            def on_bar(self, bar):
                if bar.get("close", 0) > 10:
                    return Signal(
                        symbol=row.get("symbol", ""),
                        signal_type=SignalType.BUY,
                    )
                return None

        # 实际用例应该直接传递 bar
        class BarStrategy2(BaseStrategy):
            def generate_signals(self, data):
                return []

            def on_bar(self, bar):
                close = (
                    bar.get("close", 0)
                    if isinstance(bar, pd.Series)
                    else bar.get("close", 0)
                )
                symbol = (
                    bar.get("symbol", "")
                    if isinstance(bar, pd.Series)
                    else bar.get("symbol", "")
                )
                if close > 10:
                    return Signal(symbol=symbol, signal_type=SignalType.BUY)
                return None

        strategy = BarStrategy2()
        df = pd.DataFrame(
            {
                "symbol": ["A", "B", "C"],
                "close": [5.0, 15.0, 8.0],
            }
        )

        signals = []
        for _, row in df.iterrows():
            sig = strategy.on_bar(row)
            if sig is not None:
                signals.append(sig)

        assert len(signals) == 1
        assert signals[0].symbol == "B"

    def test_on_tick_default_returns_none(self):
        """on_tick 默认返回 None"""
        strategy = DefaultStrategy()
        assert strategy.on_tick({}) is None


class DefaultStrategy(BaseStrategy):
    def generate_signals(self, data):
        return []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
