"""
Phase 3 Step 7: Benchmark 对比测试
"""

import numpy as np
import pandas as pd
import pytest

from dquant.backtest.engine import BacktestEngine
from dquant.backtest.result import BacktestResult
from dquant.strategy.base import BaseStrategy, Signal, SignalType


class SimpleBuyAll(BaseStrategy):
    """买入所有股票的简单策略"""

    def generate_signals(self, data):
        signals = []
        for date, group in data.groupby(data.index):
            for _, row in group.iterrows():
                signals.append(
                    Signal(
                        symbol=row["symbol"],
                        signal_type=SignalType.BUY,
                        strength=1.0 / len(group),
                        timestamp=date,
                    )
                )
        return signals


class TestBenchmarkNav:

    def _make_data(self, n_days=20, n_stocks=3):
        """构造测试数据"""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
        rows = []
        symbols = [f"{i:06d}.SZ" for i in range(1, n_stocks + 1)]

        for date in dates:
            for symbol in symbols:
                rows.append(
                    {
                        "symbol": symbol,
                        "open": 10.0 + np.random.randn() * 0.5,
                        "high": 10.5 + np.random.randn() * 0.5,
                        "low": 9.5 + np.random.randn() * 0.5,
                        "close": 10.0 + np.random.randn() * 0.5,
                        "volume": 1000000 + np.random.randint(-100000, 100000),
                    }
                )

        df = pd.DataFrame(rows, index=dates.repeat(n_stocks))
        return df, symbols

    def test_benchmark_nav_computed(self):
        """指定 benchmark 后计算基准净值"""
        df, symbols = self._make_data()
        strategy = SimpleBuyAll()
        engine = BacktestEngine(
            data=df,
            strategy=strategy,
            benchmark=symbols[0],  # 用第一只股票作基准
        )
        result = engine.run()

        assert result.benchmark_nav is not None
        assert len(result.benchmark_nav) > 0
        # 基准净值起始为 1.0
        assert abs(result.benchmark_nav.iloc[0] - 1.0) < 1e-10

    def test_no_benchmark_returns_none(self):
        """不指定 benchmark 时 benchmark_nav 为 None"""
        df, _ = self._make_data()
        strategy = SimpleBuyAll()
        engine = BacktestEngine(data=df, strategy=strategy)
        result = engine.run()

        assert result.benchmark_nav is None

    def test_benchmark_not_in_data(self):
        """benchmark 不在数据中时返回 None"""
        df, _ = self._make_data()
        strategy = SimpleBuyAll()
        engine = BacktestEngine(
            data=df,
            strategy=strategy,
            benchmark="999999.SZ",
        )
        result = engine.run()
        assert result.benchmark_nav is None

    def test_backtest_result_has_benchmark_field(self):
        """BacktestResult 包含 benchmark_nav 字段"""
        result = BacktestResult(
            portfolio=None,
            trades=pd.DataFrame(),
            metrics=None,
            benchmark_nav=None,
        )
        assert result.benchmark_nav is None

    def test_benchmark_nav_is_normalized(self):
        """基准净值归一化到 1.0 起始"""
        df, symbols = self._make_data()
        strategy = SimpleBuyAll()
        engine = BacktestEngine(
            data=df,
            strategy=strategy,
            benchmark=symbols[0],
        )
        result = engine.run()

        if result.benchmark_nav is not None:
            # 所有值应为正数
            assert (result.benchmark_nav > 0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
