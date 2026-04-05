"""
性能测试
"""

import time

import numpy as np
import pandas as pd
import pytest


def generate_large_data(n_stocks=100, n_days=1000):
    """生成大数据集"""
    np.random.seed(42)

    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
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


class TestPerformance:
    """性能测试"""

    @pytest.mark.slow
    def test_factor_calculation_speed(self):
        """测试因子计算速度"""
        from dquant import get_factor

        data = generate_large_data(n_stocks=50, n_days=500)

        # 测试多个因子
        factors_to_test = ["momentum", "rsi", "macd", "volatility"]

        for factor_name in factors_to_test:
            start = time.time()

            factor = get_factor(factor_name)
            result = factor.predict(data)

            elapsed = time.time() - start

            print(f"\n{factor_name}: {elapsed:.4f}s ({len(result)} rows)")

            # 断言：单个因子计算应该 < 5s
            assert elapsed < 5.0, f"{factor_name} 太慢: {elapsed:.2f}s"

    @pytest.mark.slow
    def test_factor_combination_speed(self):
        """测试因子组合速度"""
        from dquant import FactorCombiner, get_factor

        data = generate_large_data(n_stocks=50, n_days=500)

        combiner = FactorCombiner()

        # 添加多个因子
        for name in ["momentum", "volatility", "rsi", "macd"]:
            combiner.add_factor(name, get_factor(name))

        start = time.time()
        combiner.fit(data)
        combined = combiner.combine(method="equal")
        elapsed = time.time() - start

        print(f"\n因子组合: {elapsed:.4f}s")

        # 断言：组合计算应该 < 10s
        assert elapsed < 10.0

    @pytest.mark.slow
    def test_data_validation_speed(self):
        """测试数据验证速度"""
        from dquant import DataValidator

        data = generate_large_data(n_stocks=100, n_days=1000)
        validator = DataValidator()

        start = time.time()
        result = validator.validate(data)
        elapsed = time.time() - start

        print(f"\n数据验证: {elapsed:.4f}s")

        # 断言：验证应该 < 2s
        assert elapsed < 2.0

    @pytest.mark.skip(reason="Local functions cannot be pickled for multiprocessing")
    def test_parallel_processing(self):
        """测试并行处理"""
        try:
            from dquant.performance import ParallelProcessor

            def process_chunk(df):
                return df["close"].mean()

            data = generate_large_data(n_stocks=50, n_days=200)

            processor = ParallelProcessor(n_workers=2)

            # 按股票分组并行处理
            start = time.time()
            result = processor.apply_to_dataframe(
                data.reset_index(), process_chunk, groupby="symbol"
            )
            elapsed = time.time() - start

            print(f"\n并行处理: {elapsed:.4f}s")

        except ImportError:
            pytest.skip("并行处理模块不可用")

    def test_cache_effectiveness(self):
        """测试缓存效果"""
        try:
            from dquant.performance import CacheManager

            cache = CacheManager(max_size=10)

            # 模拟耗时计算
            @cache.memoize
            def expensive_computation(n):
                time.sleep(0.1)
                return n * 2

            # 第一次调用
            start1 = time.time()
            result1 = expensive_computation(10)
            time1 = time.time() - start1

            # 第二次调用（应该命中缓存）
            start2 = time.time()
            result2 = expensive_computation(10)
            time2 = time.time() - start2

            print(f"\n第一次: {time1:.4f}s")
            print(f"第二次: {time2:.4f}s (缓存)")

            # 缓存应该显著加快
            assert time2 < time1 / 10

        except ImportError:
            pytest.skip("缓存模块不可用")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
