"""
ML 因子
"""

from typing import Optional, List
import pandas as pd
import numpy as np

from dquant.ai.base import BaseFactor
from dquant.constants import DEFAULT_COMMISSION, DEFAULT_SLIPPAGE, DEFAULT_STAMP_DUTY, DEFAULT_INITIAL_CASH, MIN_SHARES, DEFAULT_WINDOW


class XGBoostFactor(BaseFactor):
    """
    XGBoost 因子

    使用 XGBoost 模型预测股票收益。

    Usage:
        factor = XGBoostFactor(
            features=['pe', 'pb', 'momentum_20', 'volatility_20'],
            target='return_5d'
        )

        # 训练
        factor.fit(train_data)

        # 预测
        predictions = factor.predict(test_data)
    """

    def __init__(
        self,
        features: List[str],
        target: str = 'return_5d',
        model_params: Optional[dict] = None,
        name: str = "XGBoostFactor",
    ):
        super().__init__(name=name)
        self.features = features
        self.target = target
        self.model_params = model_params or {
            'n_estimators': MIN_SHARES,
            'max_depth': 5,
            'learning_rate': 0.1,
            'random_state': 42,
        }

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> "XGBoostFactor":
        """训练模型"""
        try:
            import xgboost as xgb
        except ImportError:
            raise ImportError("xgboost not installed. Run: pip install xgboost")

        # 准备特征
        X = data[self.features].values

        # 准备目标
        if target is not None:
            y = target.values
        elif self.target in data.columns:
            y = data[self.target].values
        else:
            raise ValueError(f"Target column '{self.target}' not found in data")

        # 去除 NaN
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X, y = X[mask], y[mask]

        # 训练
        self._model = xgb.XGBRegressor(**self.model_params)
        self._model.fit(X, y)
        self._is_fitted = True

        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """预测"""
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # 准备特征
        X = data[self.features].values

        # 预测
        scores = self._model.predict(X)

        # 构建结果
        results = []
        for i, (idx, row) in enumerate(data.iterrows()):
            if not np.isnan(scores[i]):
                results.append({
                    'date': idx if isinstance(idx, pd.Timestamp) else row.get('date'),
                    'symbol': row.get('symbol', ''),
                    'score': scores[i],
                })

        df = pd.DataFrame(results)
        if len(df) > 0:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        return df

    def get_feature_importance(self) -> Optional[pd.Series]:
        """获取特征重要性"""
        if self._model is None:
            return None
        return pd.Series(
            self._model.feature_importances_,
            index=self.features
        ).sort_values(ascending=False)


class LGBMFactor(BaseFactor):
    """
    LightGBM 因子

    使用 LightGBM 模型预测股票收益。
    """

    def __init__(
        self,
        features: List[str],
        target: str = 'return_5d',
        model_params: Optional[dict] = None,
        name: str = "LGBMFactor",
    ):
        super().__init__(name=name)
        self.features = features
        self.target = target
        self.model_params = model_params or {
            'n_estimators': MIN_SHARES,
            'max_depth': 5,
            'learning_rate': 0.1,
            'random_state': 42,
            'verbose': -1,
        }

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> "LGBMFactor":
        """训练模型"""
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("lightgbm not installed. Run: pip install lightgbm")

        X = data[self.features].values

        if target is not None:
            y = target.values
        elif self.target in data.columns:
            y = data[self.target].values
        else:
            raise ValueError(f"Target column '{self.target}' not found in data")

        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X, y = X[mask], y[mask]

        self._model = lgb.LGBMRegressor(**self.model_params)
        self._model.fit(X, y)
        self._is_fitted = True

        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """预测"""
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        X = data[self.features].values
        scores = self._model.predict(X)

        results = []
        for i, (idx, row) in enumerate(data.iterrows()):
            if not np.isnan(scores[i]):
                results.append({
                    'date': idx if isinstance(idx, pd.Timestamp) else row.get('date'),
                    'symbol': row.get('symbol', ''),
                    'score': scores[i],
                })

        df = pd.DataFrame(results)
        if len(df) > 0:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        return df

    def get_feature_importance(self) -> Optional[pd.Series]:
        """获取特征重要性"""
        if self._model is None:
            return None
        return pd.Series(
            self._model.feature_importances_,
            index=self.features
        ).sort_values(ascending=False)
