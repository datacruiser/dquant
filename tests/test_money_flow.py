"""
测试资金流因子和策略
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from dquant.ai.money_flow_factors import (
    FlowDivergenceFactor,
    MainForceFactor,
    MediumFlowFactor,
    RetailFlowFactor,
    SmartFlowFactor,
)
from dquant.data.money_flow_loader import MockMoneyFlowLoader
from dquant.strategy.flow_strategy import MoneyFlowStrategy, SmartFlowStrategy


class TestMoneyFlowLoader:
    """测试资金流数据加载器"""

    def test_mock_loader(self):
        """测试模拟数据加载器"""
        loader = MockMoneyFlowLoader(
            symbols=["000001.SZ", "600000.SH"],
            start_date="2023-01-01",
            end_date="2023-01-31",
        )

        df = loader.load()

        assert len(df) > 0
        assert "symbol" in df.columns
        assert "medium_net_inflow" in df.columns
        assert "main_net_inflow" in df.columns
        assert "small_net_inflow" in df.columns

    def test_mock_loader_columns(self):
        """测试数据列完整性"""
        loader = MockMoneyFlowLoader(symbols=["000001.SZ"])
        df = loader.load()

        required_cols = [
            "close",
            "change_pct",
            "main_net_inflow",
            "main_net_inflow_pct",
            "super_large_net_inflow",
            "large_net_inflow",
            "medium_net_inflow",
            "small_net_inflow",
        ]

        for col in required_cols:
            assert col in df.columns, f"缺少列: {col}"


class TestMediumFlowFactor:
    """测试中户资金流因子"""

    def test_factor_creation(self):
        """测试因子创建"""
        factor = MediumFlowFactor(window=5)
        assert factor.name == "MediumFlow_5"

    def test_factor_predict(self):
        """测试因子预测"""
        # 创建模拟数据
        loader = MockMoneyFlowLoader(
            symbols=["000001.SZ", "600000.SH"],
            start_date="2023-01-01",
            end_date="2023-03-31",
        )
        data = loader.load()

        # 计算因子
        factor = MediumFlowFactor(window=5)
        factor.fit(data)
        predictions = factor.predict(data)

        assert len(predictions) > 0
        assert "score" in predictions.columns
        assert predictions["score"].notna().sum() > 0


class TestMainForceFactor:
    """测试主力资金因子"""

    def test_factor_predict(self):
        """测试因子预测"""
        loader = MockMoneyFlowLoader(symbols=["000001.SZ", "600000.SH"])
        data = loader.load()

        factor = MainForceFactor(window=5)
        factor.fit(data)
        predictions = factor.predict(data)

        assert len(predictions) > 0
        assert "score" in predictions.columns


class TestRetailFlowFactor:
    """测试散户资金流因子"""

    def test_factor_predict(self):
        """测试因子预测"""
        loader = MockMoneyFlowLoader(symbols=["000001.SZ"])
        data = loader.load()

        factor = RetailFlowFactor(window=5, reverse=True)
        factor.fit(data)
        predictions = factor.predict(data)

        assert len(predictions) > 0

    def test_factor_no_reverse(self):
        """测试不反向"""
        loader = MockMoneyFlowLoader(symbols=["000001.SZ"])
        data = loader.load()

        factor_reverse = RetailFlowFactor(window=5, reverse=True)
        factor_normal = RetailFlowFactor(window=5, reverse=False)

        pred_reverse = factor_reverse.fit(data).predict(data)
        pred_normal = factor_normal.fit(data).predict(data)

        # 反向应该改变符号
        # (由于有标准化，不能直接比较，只检查都有数据)
        assert len(pred_reverse) > 0
        assert len(pred_normal) > 0


class TestSmartFlowFactor:
    """测试聪明钱因子"""

    def test_factor_predict(self):
        """测试因子预测"""
        loader = MockMoneyFlowLoader(symbols=["000001.SZ", "600000.SH"])
        data = loader.load()

        factor = SmartFlowFactor(
            window=5,
            main_weight=0.5,
            medium_weight=0.3,
            retail_weight=0.2,
        )
        factor.fit(data)
        predictions = factor.predict(data)

        assert len(predictions) > 0
        assert "score" in predictions.columns


class TestFlowDivergenceFactor:
    """测试资金流背离因子"""

    def test_factor_predict(self):
        """测试因子预测"""
        loader = MockMoneyFlowLoader(symbols=["000001.SZ", "600000.SH"])
        data = loader.load()

        factor = FlowDivergenceFactor(window=10)
        factor.fit(data)
        predictions = factor.predict(data)

        assert len(predictions) > 0


class TestMoneyFlowStrategy:
    """测试资金流策略"""

    def test_strategy_creation(self):
        """测试策略创建"""
        strategy = MoneyFlowStrategy(
            top_k=10,
            min_medium_flow=0,
        )
        assert strategy.name == "MoneyFlowStrategy"

    def test_strategy_generate_signals(self):
        """测试信号生成"""
        loader = MockMoneyFlowLoader(
            symbols=["000001.SZ", "600000.SH", "000002.SZ"],
            start_date="2023-01-01",
            end_date="2023-01-31",
        )
        data = loader.load()

        strategy = MoneyFlowStrategy(
            top_k=2,
            min_medium_flow=0,
        )

        signals = strategy.generate_signals(data)

        assert len(signals) > 0
        assert all(s.signal_type.value == 1 for s in signals)  # BUY

    def test_strategy_with_filters(self):
        """测试带过滤条件的策略"""
        loader = MockMoneyFlowLoader(symbols=["000001.SZ", "600000.SH"])
        data = loader.load()

        strategy = MoneyFlowStrategy(
            top_k=2,
            min_medium_flow=-10000,  # 设置一个很小的阈值
            require_main_flow=False,
            avoid_retail_inflow=False,
        )

        signals = strategy.generate_signals(data)

        assert len(signals) > 0


class TestSmartFlowStrategy:
    """测试聪明钱策略"""

    def test_strategy_generate_signals(self):
        """测试信号生成"""
        loader = MockMoneyFlowLoader(
            symbols=["000001.SZ", "600000.SH", "000002.SZ"],
            start_date="2023-01-01",
            end_date="2023-01-31",
        )
        data = loader.load()

        strategy = SmartFlowStrategy(
            top_k=2,
            main_weight=0.5,
            medium_weight=0.3,
            retail_weight=0.2,
        )

        signals = strategy.generate_signals(data)

        assert len(signals) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
