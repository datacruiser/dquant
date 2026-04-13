"""
Walk-Forward 验证测试
"""

import numpy as np
import pandas as pd
import pytest

from dquant.ai.walk_forward import (
    WalkForwardResult,
    WalkForwardSplit,
    WalkForwardValidator,
)


class TestWalkForwardSplit:
    def test_split_generation(self):
        wf = WalkForwardValidator(n_splits=3, train_ratio=0.7, purge_gap=2)
        splits = wf.split(200)
        assert len(splits) == 3
        for s in splits:
            assert s.train_end <= s.test_start
            assert s.train_start < s.train_end
            assert s.test_start < s.test_end

    def test_purge_gap(self):
        wf = WalkForwardValidator(n_splits=2, purge_gap=10)
        splits = wf.split(100)
        for s in splits:
            assert s.test_start - s.train_end >= 10

    def test_too_few_samples(self):
        wf = WalkForwardValidator(n_splits=3)
        with pytest.raises(ValueError, match="at least 10"):
            wf.split(5)

    def test_expanding_window(self):
        wf = WalkForwardValidator(n_splits=3, expanding=True)
        splits = wf.split(100)
        # 扩展窗口：所有 fold 的 train_start 都应该是 0
        for s in splits:
            assert s.train_start == 0

    def test_rolling_window(self):
        wf = WalkForwardValidator(n_splits=3, expanding=False, train_ratio=0.7)
        splits = wf.split(100)
        # 滚动窗口：后面的 fold 的 train_start 应该更大
        if len(splits) >= 2:
            assert splits[0].train_start <= splits[1].train_start


class TestWalkForwardValidate:
    def test_basic_validation(self):
        """使用简单线性模型验证 Walk-Forward"""
        from sklearn.linear_model import LinearRegression

        # 创建简单数据
        np.random.seed(42)
        n = 500
        data = pd.DataFrame(
            {
                "feature1": np.random.randn(n),
                "feature2": np.random.randn(n),
                "return_5d": np.random.randn(n) * 0.01,
            }
        )

        # 创建简单因子适配器
        class SimpleFactor:
            def __init__(self):
                self.target = "return_5d"
                self.features = ["feature1", "feature2"]
                self._model = None
                self._is_fitted = False

            def fit(self, data):
                X = data[self.features].values
                y = data[self.target].values
                mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
                self._model = LinearRegression()
                self._model.fit(X[mask], y[mask])
                self._is_fitted = True
                return self

            def predict(self, data):
                X = data[self.features].values
                scores = self._model.predict(X)
                dates = data.index if isinstance(data.index, pd.DatetimeIndex) else range(len(data))
                return pd.DataFrame(
                    {
                        "date": dates,
                        "symbol": "TEST",
                        "score": scores,
                    }
                )

        factor = SimpleFactor()
        wf = WalkForwardValidator(n_splits=3, purge_gap=5)
        result = wf.validate(data, factor)

        assert result.n_folds == 3
        assert isinstance(result.mean_score, float)
        assert isinstance(result.fold_results, list)
        assert len(result.fold_results) == 3

    def test_summary_output(self):
        result = WalkForwardResult(
            n_folds=2,
            mean_score=0.05,
            std_score=0.02,
            sharpe_ratio=2.5,
            fold_results=[
                {
                    "train_start": 0,
                    "train_end": 50,
                    "test_start": 55,
                    "test_end": 70,
                    "score": 0.04,
                },
                {
                    "train_start": 0,
                    "train_end": 60,
                    "test_start": 65,
                    "test_end": 80,
                    "score": 0.06,
                },
            ],
        )
        summary = result.summary()
        assert "2 folds" in summary
        assert "0.0500" in summary

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            WalkForwardValidator(n_splits=0)
        with pytest.raises(ValueError):
            WalkForwardValidator(train_ratio=0)
        with pytest.raises(ValueError):
            WalkForwardValidator(train_ratio=1.5)
