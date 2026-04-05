"""
修复版回测测试
"""

import numpy as np
import pandas as pd


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

    data = pd.DataFrame(data_list)
    data = data.set_index("date")
    return data


def test_backtest():
    """回测测试"""
    print("【回测测试】")
    print("-" * 60)

    from datetime import datetime

    from dquant import BacktestEngine
    from dquant.strategy.base import BaseStrategy, Signal, SignalType

    # 创建策略 - 在第一天买入
    class BuyHoldStrategy(BaseStrategy):
        def generate_signals(self, data):
            # 只在第一天生成买入信号
            first_date = data.index.min()

            signals = []
            for symbol in data["symbol"].unique():
                signals.append(
                    Signal(
                        symbol=symbol,
                        signal_type=SignalType.BUY,
                        strength=1.0,
                        timestamp=first_date,
                    )
                )
            return signals

    data = create_test_data()
    strategy = BuyHoldStrategy()

    try:
        engine = BacktestEngine(data, strategy, initial_cash=1000000)
        result = engine.run()

        print(f"  ✓ 回测完成:")
        print(f"    总收益率: {result.metrics.total_return:.2%}")
        print(f"    年化收益: {result.metrics.annual_return:.2%}")
        print(f"    最大回撤: {result.metrics.max_drawdown:.2%}")
        print(f"    夏普比率: {result.metrics.sharpe:.2f}")
        print(f"    交易次数: {result.metrics.total_trades}")

        return True
    except Exception as e:
        print(f"  ✗ 回测失败: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys

    success = test_backtest()
    sys.exit(0 if success else 1)
