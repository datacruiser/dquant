"""
共享测试 fixtures 和工具函数

统一管理所有测试的数据工厂、mock 工具，消除跨文件重复代码。
"""

from datetime import datetime
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from dquant.broker.base import Order, OrderResult


# ============================================================
# 数据工厂 fixtures
# ============================================================


@pytest.fixture
def test_data():
    """创建测试数据（确定性线性趋势，适用于回测测试）

    Args 可通过 fixture 参数控制:
        days: 天数 (默认 100)
        symbols: 股票代码列表 (默认 ['000001.SZ', '600000.SH'])
    """

    def _create(days=100, symbols=None, noisy=False):
        if symbols is None:
            symbols = ["000001.SZ", "600000.SH"]

        dates = pd.date_range("2023-01-01", periods=days, freq="D")
        data_list = []

        for symbol in symbols:
            for i, date in enumerate(dates):
                base_price = 10 + i * 0.05
                if noisy:
                    base_price += np.random.randn() * 0.5

                data_list.append(
                    {
                        "date": date,
                        "symbol": symbol,
                        "open": base_price,
                        "high": base_price * 1.02,
                        "low": base_price * 0.98,
                        "close": base_price * (1 + (np.random.randn() * 0.01 if noisy else 0.01)),
                        "volume": int(1000000 + (np.random.randn() * 100000 if noisy else 0)),
                    }
                )

        data = pd.DataFrame(data_list)
        data = data.set_index("date")
        return data

    return _create


@pytest.fixture
def random_walk_data():
    """创建随机游走测试数据（适用于因子测试）

    使用业务日频率，多股票随机游走价格模型。
    """

    def _create(n_stocks=10, n_days=100, seed=42):
        np.random.seed(seed)

        dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
        symbols = [f"{i:06d}.SH" for i in range(1, n_stocks + 1)]

        data = []
        for symbol in symbols:
            price = 10 + np.random.randn() * 5
            for date in dates:
                ret = np.random.randn() * 0.02
                price *= 1 + ret
                data.append(
                    {
                        "date": date,
                        "symbol": symbol,
                        "open": price * (1 + np.random.randn() * 0.01),
                        "high": price * 1.02,
                        "low": price * 0.98,
                        "close": price,
                        "volume": int(np.random.exponential(1000000)),
                    }
                )

        df = pd.DataFrame(data)
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")

    return _create


@pytest.fixture
def flat_price_data():
    """创建固定价格测试数据（适用于精确断言的测试）

    接受外部 dates 序列，生成平坦价格数据。
    """

    def _create(dates, symbol="TEST.SZ", price=10.0):
        rows = []
        for d in dates:
            rows.append(
                {
                    "symbol": symbol,
                    "open": price,
                    "high": price * 1.01,
                    "low": price * 0.99,
                    "close": price,
                    "volume": 1000000,
                }
            )
        return pd.DataFrame(rows, index=pd.DatetimeIndex(dates))

    return _create


@pytest.fixture
def position_data():
    """创建带 position_avg_cost 的测试数据（适用于止损/止盈测试）"""

    def _create(close=10.0, avg_cost=10.0, symbol="TEST", days=5):
        dates = pd.date_range("2024-01-01", periods=days, freq="D")
        return pd.DataFrame(
            {
                "symbol": [symbol] * days,
                "close": [close] * days,
                "position_avg_cost": [avg_cost] * days,
            },
            index=dates,
        )

    return _create


# ============================================================
# Broker 工厂 fixtures
# ============================================================


@pytest.fixture
def make_order():
    """创建 Order 对象的工厂函数"""

    def _make(**overrides):
        defaults = dict(
            symbol="000001.SZ",
            side="BUY",
            quantity=100,
            price=10.0,
            order_type="LIMIT",
        )
        defaults.update(overrides)
        return Order(**defaults)

    return _make


@pytest.fixture
def make_result():
    """创建 OrderResult 对象的工厂函数"""

    def _make(**overrides):
        defaults = dict(
            order_id="test-001",
            symbol="000001.SZ",
            side="BUY",
            filled_quantity=100,
            filled_price=10.0,
            commission=3.0,
            timestamp=datetime.now(),
            status="FILLED",
        )
        defaults.update(overrides)
        return OrderResult(**defaults)

    return _make


@pytest.fixture
def mock_strategy():
    """创建 MockStrategy"""

    def _make(signals=None):
        strategy = MagicMock()
        strategy.generate_signals.return_value = signals or []
        return strategy

    return _make


@pytest.fixture
def mock_data():
    """创建 MockDataSource"""

    def _make(df=None):
        source = MagicMock()
        source.load.return_value = df if df is not None else pd.DataFrame()
        return source

    return _make
