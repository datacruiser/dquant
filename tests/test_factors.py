"""
因子测试
"""

import numpy as np
import pandas as pd


def generate_test_data(n_stocks=10, n_days=100):
    """生成测试数据"""
    np.random.seed(42)

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


def test_factor_registry():
    """测试因子注册表"""
    from dquant import get_factor, list_factors

    factors = list_factors()
    assert len(factors) >= 30, f"预期至少 30 个因子，实际 {len(factors)}"

    # 测试创建因子
    momentum = get_factor("momentum", window=20)
    assert momentum.name == "Momentum_20"

    rsi = get_factor("rsi", window=14)
    assert "RSI" in rsi.name

    print(f"✓ 因子注册表测试通过 ({len(factors)} 个因子)")


def test_momentum_factor():
    """测试动量因子"""
    from dquant import get_factor

    data = generate_test_data()
    momentum = get_factor("momentum", window=20)

    result = momentum.predict(data)

    assert len(result) > 0, "动量因子结果为空"
    assert "score" in result.columns
    assert result["score"].notna().sum() > 0

    print("✓ 动量因子测试通过")


def test_rsi_factor():
    """测试 RSI 因子"""
    from dquant import get_factor

    data = generate_test_data()
    rsi = get_factor("rsi", window=14)

    result = rsi.predict(data)

    assert len(result) > 0
    assert "score" in result.columns

    print("✓ RSI 因子测试通过")


def test_macd_factor():
    """测试 MACD 因子"""
    from dquant import get_factor

    data = generate_test_data()
    macd = get_factor("macd")

    result = macd.predict(data)

    assert len(result) > 0
    assert "score" in result.columns

    print("✓ MACD 因子测试通过")


def test_factor_combiner():
    """测试因子组合器"""
    from dquant import FactorCombiner, get_factor

    data = generate_test_data(20, 200)

    combiner = FactorCombiner()
    combiner.add_factor("momentum", get_factor("momentum", window=20))
    combiner.add_factor("volatility", get_factor("volatility", window=20))

    combiner.fit(data)

    # 等权组合
    combined = combiner.combine(method="equal")
    assert len(combined) > 0

    # 相关性矩阵
    corr = combiner.get_factor_correlation()
    assert corr.shape == (2, 2)

    print("✓ 因子组合器测试通过")


def test_combined_factor():
    """测试组合因子类"""
    from dquant import CombinedFactor, get_factor

    data = generate_test_data(20, 200)

    combined = CombinedFactor(
        factors={
            "momentum": get_factor("momentum", window=20),
            "volatility": get_factor("volatility", window=20),
        },
        weights={"momentum": 0.6, "volatility": 0.4},
    )

    combined.fit(data)
    result = combined.predict(data)

    assert len(result) > 0
    assert "score" in result.columns

    print("✓ 组合因子类测试通过")


def test_all_factors():
    """测试所有因子"""
    from dquant import get_factor, list_factors

    data = generate_test_data(10, 100)
    factors = list_factors()

    failed = []
    for name in factors:
        try:
            factor = get_factor(name)
            result = factor.predict(data)
            if len(result) == 0:
                failed.append(name)
        except Exception as e:
            failed.append(f"{name}: {str(e)}")

    # 因子中有些需要真实财务数据列（如 pe, pb, roe 等），测试数据中不可用
    KNOWN_MISSING_DATA_FACTORS = {
        "pe",
        "pb",
        "roe",
        "revenue_growth",
        "profit_growth",
        "market_cap",
        "hurst",
    }

    unexpected_failures = [f for f in failed if f not in KNOWN_MISSING_DATA_FACTORS]

    if unexpected_failures:
        print(f"⚠️  {len(unexpected_failures)} 个因子测试异常失败:")
        for f in unexpected_failures:
            print(f"  - {f}")
    else:
        print(
            f"✓ 所有因子测试通过 ({len(factors)} 个, {len(KNOWN_MISSING_DATA_FACTORS)} 个需真实数据跳过)"
        )

    assert (
        len(unexpected_failures) == 0
    ), f"Unexpected failed factors: {unexpected_failures}"


if __name__ == "__main__":
    print("=" * 60)
    print("因子测试")
    print("=" * 60)
    print()

    test_factor_registry()
    test_momentum_factor()
    test_rsi_factor()
    test_macd_factor()
    test_factor_combiner()
    test_combined_factor()
    test_all_factors()

    print()
    print("=" * 60)
    print("✓ 所有因子测试通过!")
    print("=" * 60)
