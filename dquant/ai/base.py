"""
AI 因子基类
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

import pandas as pd


class BaseFactor(ABC):
    """
    因子基类

    所有因子都需要实现 predict() 方法。
    """

    def __init__(self, name: str = "BaseFactor"):
        self.name = name
        self._model: Any = None
        self._is_fitted = False

    @abstractmethod
    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> "BaseFactor":
        """
        训练因子模型

        Args:
            data: 特征数据
            target: 目标变量 (可选，用于监督学习)

        Returns:
            self
        """
        pass

    @abstractmethod
    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        预测因子值

        Args:
            data: 市场数据

        Returns:
            DataFrame with columns [symbol, score], index=date
        """
        pass

    def get_feature_importance(self) -> Optional[pd.Series]:
        """获取特征重要性"""
        return None


class RuleFactor(BaseFactor):
    """
    规则因子 (基类)

    简单的技术指标因子，不需要训练。
    子类只需实现 _compute_score(group) 方法，predict() 模板会自动处理
    groupby/sort/dropna/DataFrame 构建。
    """

    def __init__(self, name: str = "RuleFactor"):
        super().__init__(name=name)

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> "RuleFactor":
        """规则因子不需要训练"""
        self._is_fitted = True
        return self

    def _compute_score(self, group: pd.DataFrame) -> pd.Series:
        """
        计算单个 symbol 的因子得分序列。

        Args:
            group: 已按时间排序的单个 symbol 的 DataFrame

        Returns:
            与 group 索引对齐的 score Series（可含 NaN）
        """
        raise NotImplementedError("子类必须实现 _compute_score 或 predict")

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算因子值（向量化模板）"""
        parts = []
        for symbol, grp in data.groupby("symbol"):
            grp = grp.sort_index()
            score = self._compute_score(grp)
            if score is not None and len(score) > 0:
                valid = score.dropna()
                if len(valid) > 0:
                    df = pd.DataFrame(
                        {
                            "symbol": symbol,
                            "score": valid.values,
                        },
                        index=valid.index,
                    )
                    parts.append(df)

        if not parts:
            return pd.DataFrame(columns=["symbol", "score"])
        result = pd.concat(parts)
        result.index.name = "date"
        return result
