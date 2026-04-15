"""修复版回测测试。"""

import pandas as pd

from dquant import BacktestEngine
from dquant.strategy.base import BaseStrategy, Signal, SignalType


def create_test_data(days=100):
    """创建测试数据"""
    dates = pd.date_range("2023-01-01", periods=days, freq="D")
    symbols = ["000001.SZ", "600000.SH"]
    data_list = []

    for symbol in symbols:
        for i, date in enumerate(dates):
            base_price = 10 + i * 0.05
            data_list.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": base_price,
                    "high": base_price * 1.02,
                    "low": base_price * 0.98,
                    "close": base_price * 1.01,
                    "volume": 1000000,
                }
            )

    return pd.DataFrame(data_list).set_index("date")


class BuyHoldStrategy(BaseStrategy):
    def generate_signals(self, data):
        first_date = data.index.min()
        return [
            Signal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                strength=1.0,
                timestamp=first_date,
            )
            for symbol in data["symbol"].unique()
        ]


def test_backtest():
    """回测可以在 T+1 执行后生成稳定结果。"""
    data = create_test_data()
    result = BacktestEngine(data, BuyHoldStrategy(), initial_cash=1_000_000).run()

    assert result.metrics.total_trades == 2
    assert result.metrics.total_return > 0
    assert result.metrics.annual_return > 0
    assert result.metrics.max_drawdown >= 0
    assert result.metrics.sharpe != 0
