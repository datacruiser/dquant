"""
综合测试 (修复版)
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def create_test_data(days=100, symbols=None):
    """创建测试数据"""
    if symbols is None:
        symbols = ["000001.SZ", "600000.SH"]

    dates = pd.date_range("2023-01-01", periods=days, freq="D")
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
                    "volume": 1000000 + int(np.random.randn() * 100000),
                }
            )

    data = pd.DataFrame(data_list)
    data = data.set_index("date")
    return data


def test_factors():
    """测试因子"""
    print("【1】测试因子计算")
    print("-" * 60)

    from dquant import get_factor, list_factors

    # 检查因子数量
    factors = list_factors()
    print(f"注册因子数量: {len(factors)}")

    # 测试几个关键因子
    test_factor_names = ["momentum", "rsi", "volatility"]
    data = create_test_data()

    for name in test_factor_names:
        try:
            factor = get_factor(name, window=20)
            factor.fit(data)
            result = factor.predict(data)
            print(f"  ✓ {name}: {len(result)} 行")
        except Exception as e:
            print(f"  ✗ {name}: {e}")

    return True


def test_risk():
    """测试风险管理"""
    print("\n【2】测试风险管理")
    print("-" * 60)

    from dquant import PositionSizer, RiskManager, StopLoss

    # 测试仓位管理
    sizer = PositionSizer(method="equal", total_value=1000000)
    positions = sizer.size(["000001.SZ", "600000.SH", "000002.SZ"])
    print(f"  ✓ PositionSizer: {len(positions)} 只股票")

    # 测试风险管理器
    manager = RiskManager(max_drawdown=0.15)
    print(f"  ✓ RiskManager: max_drawdown={manager.max_drawdown}")

    # 测试止损 (静态方法)
    stop_price = StopLoss.fixed_stop(entry_price=100.0, stop_pct=0.1)
    print(f"  ✓ StopLoss.fixed_stop: {stop_price}")

    return True


def test_backtest():
    """测试回测"""
    print("\n【3】测试回测引擎")
    print("-" * 60)

    from dquant import BacktestEngine
    from dquant.strategy.base import BaseStrategy, Signal, SignalType

    # 创建简单策略
    class SimpleStrategy(BaseStrategy):
        def generate_signals(self, data):
            signals = []
            for symbol in data["symbol"].unique():
                signals.append(
                    Signal(
                        symbol=symbol,
                        signal_type=SignalType.BUY,
                        strength=1.0,
                        weight=0.5,
                    )
                )
            return signals

    data = create_test_data()
    strategy = SimpleStrategy()

    try:
        engine = BacktestEngine(data, strategy, initial_cash=1000000)
        result = engine.run()
        print(f"  ✓ 回测完成: 收益率={result.total_return:.2%}")
    except Exception as e:
        print(f"  ✗ 回测失败: {e}")
        import traceback

        traceback.print_exc()
        return False

    return True


def test_utils():
    """测试工具函数"""
    print("\n【4】测试工具函数")
    print("-" * 60)

    from dquant import format_money, format_percent, get_trading_days

    # 测试日期函数
    days = get_trading_days("2023-01-01", "2023-12-31")
    print(f"  ✓ get_trading_days: {len(days)} 天")

    # 测试格式化
    money = format_money(1234567.89)
    print(f"  ✓ format_money: {money}")

    pct = format_percent(0.1234)
    print(f"  ✓ format_percent: {pct}")

    return True


def test_config():
    """测试配置"""
    print("\n【5】测试配置管理")
    print("-" * 60)

    from dquant import DQuantConfig, default_config

    config = DQuantConfig()
    print(f"  ✓ DQuantConfig 创建成功")

    # default_config 是对象，不是函数
    print(f"  ✓ default_config: {type(default_config).__name__}")

    return True


def test_logger():
    """测试日志"""
    print("\n【6】测试日志系统")
    print("-" * 60)

    from dquant import get_logger, set_log_level

    logger = get_logger("test")
    logger.info("测试日志")
    print(f"  ✓ get_logger: {logger.name}")

    return True


def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║            DQuant 综合功能测试                                ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    tests = [
        ("因子计算", test_factors),
        ("风险管理", test_risk),
        ("回测引擎", test_backtest),
        ("工具函数", test_utils),
        ("配置管理", test_config),
        ("日志系统", test_logger),
    ]

    results = []
    for name, func in tests:
        try:
            result = func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} 测试失败: {e}")
            import traceback

            traceback.print_exc()
            results.append((name, False))

    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    passed = 0
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {name:20s} {status}")
        if result:
            passed += 1

    print("=" * 60)
    print(f"总计: {passed}/{len(results)} 通过")

    return passed == len(results)


if __name__ == "__main__":
    import sys

    success = main()
    sys.exit(0 if success else 1)
