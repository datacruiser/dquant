"""
因子组合器

支持多因子加权组合、因子正交化、因子筛选等。
"""

from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod

from dquant.ai.base import BaseFactor


@dataclass
class FactorWeight:
    """因子权重"""
    name: str
    weight: float
    ic: float = 0.0
    ir: float = 0.0


class FactorCombiner:
    """
    因子组合器

    将多个因子按不同方法组合成综合因子。

    Usage:
        from dquant.ai.factor_combiner import FactorCombiner

        combiner = FactorCombiner()
        combiner.add_factor('momentum', MomentumFactor(20))
        combiner.add_factor('volatility', VolatilityFactor(20))

        combiner.fit(data, target)
        combined = combiner.combine(method='ic_weight')
    """

    def __init__(
        self,
        neutralize: bool = True,
        standardize: bool = True,
        winsorize: bool = True,
        winsorize_limit: float = 0.01,
    ):
        self.neutralize = neutralize
        self.standardize = standardize
        self.winsorize = winsorize
        self.winsorize_limit = winsorize_limit

        self.factors: Dict[str, BaseFactor] = {}
        self.factor_values: Dict[str, pd.DataFrame] = {}
        self.weights: Dict[str, FactorWeight] = {}

    def add_factor(self, name: str, factor: BaseFactor, weight: float = 1.0):
        """添加因子"""
        self.factors[name] = factor
        self.weights[name] = FactorWeight(name=name, weight=weight)

    def fit(
        self,
        data: pd.DataFrame,
        target: Optional[pd.Series] = None,
        calculate_ic: bool = True,
    ):
        """计算所有因子值"""
        for name, factor in self.factors.items():
            factor_data = factor.predict(data)
            factor_data = self._preprocess(factor_data)
            self.factor_values[name] = factor_data

            if calculate_ic and target is not None:
                ic = self._calculate_ic(factor_data, target)
                ir = self._calculate_ir(factor_data, target)
                self.weights[name].ic = ic
                self.weights[name].ir = ir

    def _preprocess(self, factor_data: pd.DataFrame) -> pd.DataFrame:
        """因子预处理"""
        df = factor_data.copy()

        if len(df) == 0:
            return df

        if self.winsorize:
            lower = df['score'].quantile(self.winsorize_limit)
            upper = df['score'].quantile(1 - self.winsorize_limit)
            df['score'] = df['score'].clip(lower, upper)

        if self.standardize:
            df['score'] = df.groupby(df.index)['score'].transform(
                lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x
            )

        return df

    def _calculate_ic(self, factor_data: pd.DataFrame, target: pd.Series) -> float:
        """计算 IC"""
        try:
            ics = []

            for date in factor_data.index.unique():
                factor_day = factor_data[factor_data.index == date]

                # 对齐 target
                if isinstance(target.index, pd.DatetimeIndex):
                    target_day = target[target.index == date]
                else:
                    # target 可能是 MultiIndex
                    try:
                        target_day = target.loc[date]
                    except Exception:
                        continue

                if len(factor_day) == 0 or len(target_day) == 0:
                    continue

                # 合并
                merged = factor_day.set_index('symbol')['score'].to_frame('factor')
                if isinstance(target_day, pd.Series):
                    merged['target'] = target_day
                    merged = merged.dropna()

                    if len(merged) > 1:
                        corr = merged['factor'].corr(merged['target'], method='spearman')
                        if pd.notna(corr):
                            ics.append(corr)

            return np.mean(ics) if ics else 0.0
        except Exception:
            return 0.0

    def _calculate_ir(self, factor_data: pd.DataFrame, target: pd.Series) -> float:
        """计算 IR"""
        try:
            ics = []

            for date in factor_data.index.unique():
                factor_day = factor_data[factor_data.index == date]

                if isinstance(target.index, pd.DatetimeIndex):
                    target_day = target[target.index == date]
                else:
                    try:
                        target_day = target.loc[date]
                    except Exception:
                        continue

                if len(factor_day) == 0 or len(target_day) == 0:
                    continue

                merged = factor_day.set_index('symbol')['score'].to_frame('factor')
                if isinstance(target_day, pd.Series):
                    merged['target'] = target_day
                    merged = merged.dropna()

                    if len(merged) > 1:
                        corr = merged['factor'].corr(merged['target'], method='spearman')
                        if pd.notna(corr):
                            ics.append(corr)

            ic_mean = np.mean(ics) if ics else 0.0
            ic_std = np.std(ics) if len(ics) > 1 else 1.0

            return ic_mean / ic_std if ic_std > 0 else 0.0
        except Exception:
            return 0.0

    def combine(
        self,
        method: str = 'equal',
        weights: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """组合因子"""
        if not self.factor_values:
            raise ValueError("No factor values. Call fit() first.")

        if method == 'equal' or weights:
            return self._combine_equal(weights)
        elif method == 'ic_weight':
            return self._combine_ic_weight()
        elif method == 'ir_weight':
            return self._combine_ir_weight()
        elif method == 'pca':
            return self._combine_pca()
        else:
            raise ValueError(f"Unknown method: {method}")

    def _combine_equal(self, weights: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """等权或自定义权重组合"""
        if weights is None:
            weights = {name: 1.0 / len(self.factor_values) for name in self.factor_values}
        else:
            total = sum(weights.values())
            weights = {k: v / total for k, v in weights.items()}

        combined_scores = []

        for name, factor_data in self.factor_values.items():
            weight = weights.get(name, 0)
            if weight > 0:
                temp = factor_data.copy()
                temp['score'] = temp['score'] * weight
                combined_scores.append(temp)

        if not combined_scores:
            return pd.DataFrame()

        result = pd.concat(combined_scores)
        result = result.groupby([result.index, 'symbol'])['score'].sum().reset_index()
        result = result.set_index('date')

        return result

    def _combine_ic_weight(self) -> pd.DataFrame:
        """IC 加权组合"""
        total_ic = sum(abs(w.ic) for w in self.weights.values())

        if total_ic == 0:
            return self._combine_equal()

        weights = {}
        for name, w in self.weights.items():
            weights[name] = w.ic / total_ic if w.ic >= 0 else -w.ic / total_ic

        return self._combine_equal(weights)

    def _combine_ir_weight(self) -> pd.DataFrame:
        """IR 加权组合"""
        total_ir = sum(abs(w.ir) for w in self.weights.values())

        if total_ir == 0:
            return self._combine_equal()

        weights = {}
        for name, w in self.weights.items():
            weights[name] = w.ir / total_ir if w.ir >= 0 else -w.ir / total_ir

        return self._combine_equal(weights)

    def _combine_pca(self) -> pd.DataFrame:
        """PCA 正交化组合"""
        try:
            from sklearn.decomposition import PCA
        except ImportError:
            raise ImportError("sklearn not installed")

        # 构建因子矩阵
        all_dates = set()
        all_symbols = set()
        for fd in self.factor_values.values():
            all_dates.update(fd.index.unique())
            all_symbols.update(fd['symbol'].unique())

        dates = sorted(all_dates)
        symbols = sorted(all_symbols)
        n_factors = len(self.factor_values)

        n_samples = len(dates) * len(symbols)
        X = np.zeros((n_samples, n_factors))

        factor_names = list(self.factor_values.keys())
        for i, name in enumerate(factor_names):
            fd = self.factor_values[name]
            for j, date in enumerate(dates):
                for k, symbol in enumerate(symbols):
                    idx = j * len(symbols) + k
                    row = fd[(fd.index == date) & (fd['symbol'] == symbol)]
                    if len(row) > 0:
                        X[idx, i] = row['score'].values[0]

        # Temporal split: fit PCA on first 80% of data to avoid look-ahead bias
        split_idx = int(n_samples * 0.8)
        pca = PCA(n_components=1)
        pca.fit(X[:split_idx])
        combined = pca.transform(X)

        result = pd.DataFrame({
            'date': np.repeat(dates, len(symbols)),
            'symbol': symbols * len(dates),
            'score': combined.flatten(),
        })
        result['date'] = pd.to_datetime(result['date'])
        result = result.set_index('date')

        return result

    def get_factor_correlation(self) -> pd.DataFrame:
        """获取因子相关性矩阵"""
        if not self.factor_values:
            return pd.DataFrame()

        # 构建因子相关性
        factor_names = list(self.factor_values.keys())
        n_factors = len(factor_names)
        corr_matrix = np.eye(n_factors)

        for i in range(n_factors):
            for j in range(i + 1, n_factors):
                # 计算两个因子的相关性
                f1 = self.factor_values[factor_names[i]]
                f2 = self.factor_values[factor_names[j]]

                # 合并
                merged = f1.set_index('symbol', append=True)['score'].to_frame('f1')
                merged = merged.join(
                    f2.set_index('symbol', append=True)['score'].to_frame('f2'),
                    how='inner'
                )

                if len(merged) > 1:
                    corr = merged['f1'].corr(merged['f2'])
                    corr_matrix[i, j] = corr
                    corr_matrix[j, i] = corr

        return pd.DataFrame(corr_matrix, index=factor_names, columns=factor_names)

    def get_weights_summary(self) -> pd.DataFrame:
        """获取权重摘要"""
        data = []
        for name, w in self.weights.items():
            data.append({
                'factor': name,
                'weight': w.weight,
                'ic': round(w.ic, 4),
                'ir': round(w.ir, 4),
            })
        return pd.DataFrame(data)


class CombinedFactor(BaseFactor):
    """
    组合因子

    将多个因子组合成单个因子。

    Usage:
        from dquant.ai.factor_combiner import CombinedFactor

        combined = CombinedFactor(
            factors={
                'momentum': MomentumFactor(20),
                'volatility': VolatilityFactor(20),
            },
            weights={'momentum': 0.6, 'volatility': 0.4},
        )

        combined.fit(data)
        predictions = combined.predict(data)
    """

    def __init__(
        self,
        factors: Dict[str, BaseFactor],
        weights: Optional[Dict[str, float]] = None,
        combine_method: str = 'equal',
        name: str = "CombinedFactor",
    ):
        super().__init__(name=name)
        self.factors = factors
        self.weights = weights
        self.combine_method = combine_method

        self._combiner = None

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> "CombinedFactor":
        """训练"""
        self._combiner = FactorCombiner()

        for name, factor in self.factors.items():
            self._combiner.add_factor(name, factor)

        self._combiner.fit(data, target)
        self._is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """预测"""
        if not self._is_fitted:
            raise ValueError("Factor not fitted. Call fit() first.")

        # Recompute factor values on the provided data
        self._combiner.fit(data, calculate_ic=False)
        return self._combiner.combine(method=self.combine_method, weights=self.weights)

    def get_weights_summary(self) -> pd.DataFrame:
        """获取权重摘要"""
        if self._combiner is None:
            return pd.DataFrame()
        return self._combiner.get_weights_summary()


# ============================================================
# 因子注册表
# ============================================================

FACTOR_REGISTRY: Dict[str, type] = {}


def register_factor(name: str):
    """注册因子装饰器"""
    def decorator(cls):
        FACTOR_REGISTRY[name] = cls
        return cls
    return decorator


def get_factor(name: str, **kwargs) -> BaseFactor:
    """获取因子实例"""
    if name not in FACTOR_REGISTRY:
        raise ValueError(f"Unknown factor: {name}. Available: {list(FACTOR_REGISTRY.keys())}")
    return FACTOR_REGISTRY[name](**kwargs)


def list_factors() -> List[str]:
    """列出所有已注册因子"""
    return list(FACTOR_REGISTRY.keys())
