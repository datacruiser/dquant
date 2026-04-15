"""因子综合测试。"""

import numpy as np
import pandas as pd

from dquant import get_factor


def create_test_data(days=200):
    """创建测试数据"""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2023-01-01", periods=days, freq="D")
    symbols = ["000001.SZ", "600000.SH", "000002.SZ"]
    data_list = []

    for symbol in symbols:
        for i, date in enumerate(dates):
            trend = i * 0.03
            noise = rng.normal(0, 0.5)
            base_price = 10 + trend + noise
            data_list.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": base_price,
                    "high": base_price * 1.02,
                    "low": base_price * 0.98,
                    "close": base_price * (1 + rng.normal(0, 0.01)),
                    "volume": 1000000 + int(rng.normal(0, 100000)),
                }
            )

    return pd.DataFrame(data_list).set_index("date")


def test_all_factors():
    """常见因子在综合数据集上应大多可计算。"""
    data = create_test_data()
    categories = {
        "动量类": ["momentum", "reversal"],
        "波动率类": ["volatility", "atr"],
        "技术指标": ["rsi", "kdj", "cci"],
        "成交量": ["volume_ratio", "obv", "vwap"],
        "价格形态": ["price_position", "gap"],
        "均线": ["ma_slope", "ma_cross", "bias"],
    }
    expected_fail = {"kdj", "gap", "ma_cross"}
    total = 0
    passed = 0

    for factor_names in categories.values():
        for name in factor_names:
            total += 1
            try:
                factor = get_factor(name, window=20)
                factor.fit(data)
                result = factor.predict(data)
                if len(result) > 0:
                    passed += 1
            except Exception:
                assert name in expected_fail

    assert passed >= total - len(expected_fail)


def test_edge_cases():
    """小样本、单标的和缺失值场景仍可计算核心因子。"""
    small_data = create_test_data(days=10)
    small_factor = get_factor("momentum", window=5)
    small_factor.fit(small_data)
    small_result = small_factor.predict(small_data)
    assert list(small_result.columns) == ["symbol", "score"]

    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    single_data = pd.DataFrame(
        [
            {
                "date": date,
                "symbol": "000001.SZ",
                "open": 10 + i * 0.05,
                "high": 10.5 + i * 0.05,
                "low": 9.5 + i * 0.05,
                "close": 10 + i * 0.05,
                "volume": 1000000,
            }
            for i, date in enumerate(dates)
        ]
    ).set_index("date")
    single_factor = get_factor("rsi", window=14)
    single_factor.fit(single_data)
    single_result = single_factor.predict(single_data)
    assert list(single_result.columns) == ["symbol", "score"]

    data_with_nan = pd.DataFrame(
        [
            {
                "date": date,
                "symbol": "000001.SZ",
                "open": 10 + i * 0.05 if i % 10 != 0 else np.nan,
                "high": 10.5 + i * 0.05,
                "low": 9.5 + i * 0.05,
                "close": 10 + i * 0.05 if i % 10 != 0 else np.nan,
                "volume": 1000000,
            }
            for i, date in enumerate(dates)
        ]
    ).set_index("date")
    nan_factor = get_factor("momentum", window=20)
    nan_factor.fit(data_with_nan)
    nan_result = nan_factor.predict(data_with_nan)
    assert list(nan_result.columns) == ["symbol", "score"]


def test_performance():
    """大样本数据集上核心因子至少能完成计算。"""
    dates = pd.date_range("2020-01-01", periods=1000, freq="D")
    symbols = [f"00000{i}.SZ" for i in range(10)]
    data = pd.DataFrame(
        [
            {
                "date": date,
                "symbol": symbol,
                "open": 10 + i * 0.01,
                "high": 10.5 + i * 0.01,
                "low": 9.5 + i * 0.01,
                "close": 10 + i * 0.01,
                "volume": 1000000,
            }
            for symbol in symbols
            for i, date in enumerate(dates)
        ]
    ).set_index("date")

    for name in ["momentum", "rsi", "volatility"]:
        factor = get_factor(name, window=20)
        factor.fit(data)
        result = factor.predict(data)
        assert list(result.columns) == ["symbol", "score"]
