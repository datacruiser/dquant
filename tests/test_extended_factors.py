"""
扩展因子测试
"""

import numpy as np
import pandas as pd


def create_test_data(days=200):
    """创建测试数据"""
    dates = pd.date_range("2023-01-01", periods=days, freq="D")
    symbols = ["000001.SZ", "600000.SH"]
    data_list = []

    for symbol in symbols:
        for i, date in enumerate(dates):
            base_price = 10 + i * 0.03 + np.random.randn() * 0.5
            data_list.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": base_price,
                    "high": base_price * 1.02,
                    "low": base_price * 0.98,
                    "close": base_price * (1 + np.random.randn() * 0.01),
                    "volume": 1000000 + int(np.random.randn() * 100000),
                }
            )

    data = pd.DataFrame(data_list)
    data = data.set_index("date")
    return data


def test_extended_factors():
    """测试扩展因子"""
    print("【扩展因子测试】")
    print("=" * 60)

    from dquant import list_factors

    factors = list_factors()
    print(f"注册因子总数: {len(factors)}\n")

    data = create_test_data()

    # 分类测试
    categories = {
        "技术指标": ["adx", "aroon", "stochastic", "roc", "cmo", "mfi"],
        "量价关系": ["ad_line", "chaikin_osc", "eom", "force_index", "vpt"],
        "统计因子": ["hurst", "autocorr", "variance_ratio", "beta", "alpha"],
    }

    total = 0
    passed = 0

    for category, factor_names in categories.items():
        print(f"\n{category}:")
        for name in factor_names:
            total += 1
            try:
                from dquant import get_factor

                factor = get_factor(name, window=20)
                factor.fit(data)
                result = factor.predict(data)

                if len(result) > 0:
                    print(f"  ✓ {name:20s} {len(result):4d} 行")
                    passed += 1
                else:
                    print(f"  ⚠ {name:20s}    0 行")
            except Exception as e:
                print(f"  ✗ {name:20s} 错误: {str(e)[:40]}")

    print("\n" + "=" * 60)
    print(f"测试结果: {passed}/{total} 通过 ({passed/total*100:.1f}%)")

    return passed >= total * 0.6


if __name__ == "__main__":
    import sys

    success = test_extended_factors()
    sys.exit(0 if success else 1)
