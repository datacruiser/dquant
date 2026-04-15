"""扩展因子测试。"""

import numpy as np
import pandas as pd

from dquant import get_factor


def create_test_data(days=200):
    """创建测试数据"""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2023-01-01", periods=days, freq="D")
    symbols = ["000001.SZ", "600000.SH"]
    data_list = []

    for symbol in symbols:
        for i, date in enumerate(dates):
            base_price = 10 + i * 0.03 + rng.normal(0, 0.5)
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


def test_extended_factors():
    """扩展因子至少大部分可以在测试数据上成功计算。"""
    data = create_test_data()
    categories = {
        "技术指标": ["adx", "aroon", "stochastic", "roc", "cmo", "mfi"],
        "量价关系": ["ad_line", "chaikin_osc", "eom", "force_index", "vpt"],
        "统计因子": ["hurst", "autocorr", "variance_ratio", "beta", "alpha"],
    }

    total = 0
    passed = 0

    for factor_names in categories.values():
        for name in factor_names:
            total += 1
            factor = get_factor(name)
            factor.fit(data)
            result = factor.predict(data)
            if list(result.columns) == ["symbol", "score"] and len(result) > 0:
                passed += 1

    assert passed >= total * 0.6
