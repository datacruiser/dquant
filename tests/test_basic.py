"""
基础测试
"""

import numpy as np
import pandas as pd


def test_imports():
    """测试导入"""
    from dquant import BaseStrategy, Engine, Signal

    print("✓ 导入成功")


def test_signal():
    """测试信号"""
    from dquant.strategy.base import Signal, SignalType

    signal = Signal(
        symbol="000001.SZ",
        signal_type=SignalType.BUY,
        strength=0.5,
    )

    assert signal.is_buy
    assert not signal.is_sell
    assert signal.strength == 0.5
    print("✓ Signal 测试通过")


def test_portfolio():
    """测试组合"""
    from dquant.backtest.portfolio import Portfolio

    portfolio = Portfolio(initial_cash=1000000)
    assert portfolio.cash == 1000000
    assert portfolio.total_value == 1000000

    portfolio.update_prices({}, pd.Timestamp("2023-01-01"))

    # 买入
    portfolio.buy("000001.SZ", 1000, 10.0)
    assert portfolio.cash == 990000
    assert "000001.SZ" in portfolio.positions

    portfolio.update_prices({"000001.SZ": 10.0}, pd.Timestamp("2023-01-02"))

    # 卖出
    portfolio.sell("000001.SZ", 500, 11.0)
    assert portfolio.positions["000001.SZ"].shares == 500
    print("✓ Portfolio 测试通过")


def test_metrics():
    """测试绩效"""
    from dquant.backtest.metrics import Metrics

    # 模拟净值
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    nav = pd.Series(1 + np.cumsum(np.random.randn(100) * 0.01), index=dates)

    metrics = Metrics.from_nav(nav)

    assert metrics.total_return != 0
    assert metrics.volatility > 0
    print("✓ Metrics 测试通过")


if __name__ == "__main__":
    test_imports()
    test_signal()
    test_portfolio()
    test_metrics()
    print("\n✓ 所有测试通过!")
