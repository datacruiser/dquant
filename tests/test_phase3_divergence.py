"""
Phase 3 Step 1: FlowDivergenceStrategy 顶背离卖出信号测试
"""

import numpy as np
import pandas as pd
import pytest

from dquant.strategy.base import SignalType
from dquant.strategy.flow_strategy import FlowDivergenceStrategy


def _make_divergence_data():
    """构造带有顶背离和底背离的测试数据"""
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    rows = []
    for date in dates:
        # Stock A: 先跌后涨（底背离 → 顶背离）
        # Stock B: 稳定上涨，主力持续流入
        for symbol, main_inflow in [("A", 100), ("B", 50)]:
            rows.append(
                {
                    "symbol": symbol,
                    "close": 10.0,  # 会被覆盖
                    "main_net_inflow": main_inflow,
                }
            )

    df = pd.DataFrame(rows, index=dates.repeat(2))
    # 设置价格变化路径
    # A: 模拟前10天下跌，后10天上涨
    for i in range(20):
        offset = i * 2  # 每天两行 (A, B)
        if i < 10:
            df.iloc[offset]["close"] = 12.0  # A
            df.iloc[offset + 1]["close"] = 11.0  # B
        else:
            df.iloc[offset]["close"] = 8.0  # A
            df.iloc[offset + 1]["close"] = 12.0  # B
    return df


class TestFlowDivergenceSellSignal:

    def test_strategy_emits_sell_on_top_divergence(self):
        """顶背离（价格上涨 + 主力流出）应生成 SELL 信号"""
        # 构造数据：window=5 前价格 8，当前 10 → 涨 25%
        # 主力净流出 → 触发顶背离
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        rows = []
        for i, date in enumerate(dates):
            # 前 5 天低价
            # 后 5 天高价 + 主力流出
            if i < 5:
                close = 8.0
                main_flow = 100  # 流入
            else:
                close = 10.0
                main_flow = -200  # 流出

            rows.append(
                {
                    "symbol": "TEST",
                    "close": close,
                    "main_net_inflow": main_flow,
                }
            )

        df = pd.DataFrame(rows, index=dates)
        strategy = FlowDivergenceStrategy(window=5, top_k=5)
        signals = strategy.generate_signals(df)

        sell_signals = [s for s in signals if s.signal_type == SignalType.SELL]
        # 最后几天应该出现顶背离卖出信号
        assert len(sell_signals) > 0
        for s in sell_signals:
            assert s.symbol == "TEST"
            assert s.is_sell
            assert s.metadata["divergence_type"] == "top"

    def test_strategy_emits_buy_on_bottom_divergence(self):
        """底背离（价格下跌 + 主力流入）应生成 BUY 信号"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        rows = []
        for i, date in enumerate(dates):
            if i < 5:
                close = 10.0
                main_flow = -50
            else:
                close = 8.0
                main_flow = 200  # 流入

            rows.append(
                {
                    "symbol": "TEST",
                    "close": close,
                    "main_net_inflow": main_flow,
                }
            )

        df = pd.DataFrame(rows, index=dates)
        strategy = FlowDivergenceStrategy(window=5, top_k=5)
        signals = strategy.generate_signals(df)

        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
        assert len(buy_signals) > 0
        for s in buy_signals:
            assert s.is_buy
            assert s.metadata["divergence_type"] == "bottom"

    def test_no_signal_when_no_divergence(self):
        """无背离时不产生信号"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        rows = []
        for i, date in enumerate(dates):
            rows.append(
                {
                    "symbol": "TEST",
                    "close": 10.0,  # 价格不变
                    "main_net_inflow": 0,  # 无资金流
                }
            )

        df = pd.DataFrame(rows, index=dates)
        strategy = FlowDivergenceStrategy(window=5, top_k=5)
        signals = strategy.generate_signals(df)
        assert len(signals) == 0

    def test_top_divergence_metadata(self):
        """顶背离信号包含正确的 metadata"""
        dates = pd.date_range("2024-01-01", periods=8, freq="D")
        rows = []
        for i, date in enumerate(dates):
            close = 12.0 if i < 4 else 8.0
            # 让 window=4 后正好出现顶背离
            # Day 4-7: close=8.0, window=4 前的 close=12.0 → price_change = 8/12-1 = -33%
            # 这其实是底背离，我们反过来构造
            pass

        # 简化：直接测试 metadata 结构
        dates = pd.date_range("2024-01-01", periods=8, freq="D")
        rows = []
        for i, date in enumerate(dates):
            # 前 4 天: close=8, 后 4 天 close=10.4 → price_change > 5% (30%)
            close = 8.0 if i < 4 else 10.4
            main_flow = 100 if i < 4 else -300  # 后期主力流出
            rows.append(
                {
                    "symbol": "XYZ",
                    "close": close,
                    "main_net_inflow": main_flow,
                }
            )

        df = pd.DataFrame(rows, index=dates)
        strategy = FlowDivergenceStrategy(window=4, top_k=5)
        signals = strategy.generate_signals(df)

        sell_signals = [s for s in signals if s.is_sell]
        assert len(sell_signals) > 0
        for s in sell_signals:
            assert "divergence_type" in s.metadata
            assert "price_change" in s.metadata
            assert s.metadata["divergence_type"] == "top"

    def test_multiple_stocks_sell(self):
        """多只股票同时触发顶背离"""
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        rows = []
        for i, date in enumerate(dates):
            for symbol in ["AAA", "BBB"]:
                close = 8.0 if i < 5 else 10.0  # 涨 25%
                main_flow = 100 if i < 5 else -200  # 流出
                rows.append(
                    {
                        "symbol": symbol,
                        "close": close,
                        "main_net_inflow": main_flow,
                    }
                )

        df = pd.DataFrame(rows, index=dates.repeat(2))
        strategy = FlowDivergenceStrategy(window=5, top_k=10)
        signals = strategy.generate_signals(df)

        sell_signals = [s for s in signals if s.is_sell]
        sell_symbols = {s.symbol for s in sell_signals}
        assert "AAA" in sell_symbols
        assert "BBB" in sell_symbols


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
