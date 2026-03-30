"""
Phase 3 Step 2: 止损止盈策略装饰器测试
"""

import pytest
import pandas as pd
import numpy as np

from dquant.strategy.base import BaseStrategy, Signal, SignalType
from dquant.strategy.stop_loss_take_profit import StopLossTakeProfitStrategy


class SimpleBuyStrategy(BaseStrategy):
    """测试用：固定买入策略"""

    def generate_signals(self, data):
        return [Signal(symbol='TEST', signal_type=SignalType.BUY, strength=1.0)]


class TestStopLossTakeProfit:

    def _make_data(self, close=10.0, avg_cost=10.0, symbol='TEST'):
        """构造带 position_avg_cost 的数据"""
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        df = pd.DataFrame({
            'symbol': [symbol] * 5,
            'close': [close] * 5,
            'position_avg_cost': [avg_cost] * 5,
        }, index=dates)
        return df

    def test_stop_loss_triggered(self):
        """亏损超过阈值触发止损"""
        # 成本 10，现价 9 → 亏 10%
        df = self._make_data(close=9.0, avg_cost=10.0)
        strategy = StopLossTakeProfitStrategy(
            base_strategy=SimpleBuyStrategy(),
            stop_loss=0.05,
            take_profit=0.10,
        )
        signals = strategy.generate_signals(df)

        sell_signals = [s for s in signals if s.is_sell]
        assert len(sell_signals) == 1
        assert sell_signals[0].metadata['reason'] == 'stop_loss'

    def test_take_profit_triggered(self):
        """盈利超过阈值触发止盈"""
        # 成本 10，现价 12 → 盈 20%
        df = self._make_data(close=12.0, avg_cost=10.0)
        strategy = StopLossTakeProfitStrategy(
            base_strategy=SimpleBuyStrategy(),
            stop_loss=0.05,
            take_profit=0.10,
        )
        signals = strategy.generate_signals(df)

        sell_signals = [s for s in signals if s.is_sell]
        assert len(sell_signals) == 1
        assert sell_signals[0].metadata['reason'] == 'take_profit'

    def test_no_trigger_within_range(self):
        """在止损止盈范围内不触发"""
        # 成本 10，现价 10.3 → 盈 3%，未到止盈
        df = self._make_data(close=10.3, avg_cost=10.0)
        strategy = StopLossTakeProfitStrategy(
            base_strategy=SimpleBuyStrategy(),
            stop_loss=0.05,
            take_profit=0.10,
        )
        signals = strategy.generate_signals(df)

        sell_signals = [s for s in signals if s.is_sell]
        assert len(sell_signals) == 0

    def test_base_signals_preserved(self):
        """原始策略信号被保留"""
        df = self._make_data(close=10.0, avg_cost=10.0)
        strategy = StopLossTakeProfitStrategy(
            base_strategy=SimpleBuyStrategy(),
            stop_loss=0.05,
            take_profit=0.10,
        )
        signals = strategy.generate_signals(df)

        buy_signals = [s for s in signals if s.is_buy]
        assert len(buy_signals) == 1
        assert buy_signals[0].symbol == 'TEST'

    def test_no_avg_cost_no_sell(self):
        """数据中没有 position_avg_cost 时不触发"""
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        df = pd.DataFrame({
            'symbol': ['TEST'] * 5,
            'close': [9.0] * 5,  # 亏损
        }, index=dates)

        strategy = StopLossTakeProfitStrategy(
            base_strategy=SimpleBuyStrategy(),
            stop_loss=0.05,
            take_profit=0.10,
        )
        signals = strategy.generate_signals(df)

        sell_signals = [s for s in signals if s.is_sell]
        assert len(sell_signals) == 0

    def test_stop_loss_exact_threshold(self):
        """恰好等于止损阈值时触发"""
        # 成本 10，现价 9.5 → 亏 5% = 止损阈值
        df = self._make_data(close=9.5, avg_cost=10.0)
        strategy = StopLossTakeProfitStrategy(
            base_strategy=SimpleBuyStrategy(),
            stop_loss=0.05,
            take_profit=0.10,
        )
        signals = strategy.generate_signals(df)

        sell_signals = [s for s in signals if s.is_sell]
        assert len(sell_signals) == 1
        assert sell_signals[0].metadata['reason'] == 'stop_loss'

    def test_multiple_stops(self):
        """多只股票同时触发止损"""
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        rows = []
        for date in dates:
            for symbol, close in [('AAA', 8.0), ('BBB', 9.0)]:
                rows.append({
                    'symbol': symbol,
                    'close': close,
                    'position_avg_cost': 10.0,
                })
        df = pd.DataFrame(rows, index=[d for d in dates for _ in range(2)])

        # MultiBuyStrategy returns BUY for both
        class MultiBuy(BaseStrategy):
            def generate_signals(self, data):
                return [
                    Signal(symbol='AAA', signal_type=SignalType.BUY, strength=0.5),
                    Signal(symbol='BBB', signal_type=SignalType.BUY, strength=0.5),
                ]

        strategy = StopLossTakeProfitStrategy(
            base_strategy=MultiBuy(),
            stop_loss=0.05,
            take_profit=0.10,
        )
        signals = strategy.generate_signals(df)

        sell_signals = [s for s in signals if s.is_sell]
        sell_symbols = {s.symbol for s in sell_signals}
        assert 'AAA' in sell_symbols
        assert 'BBB' in sell_symbols


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
