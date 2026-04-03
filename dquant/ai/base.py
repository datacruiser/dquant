"""
AI 因子基类
"""

from typing import Optional, Any
from abc import ABC, abstractmethod
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
    """

    def __init__(self, name: str = "RuleFactor"):
        super().__init__(name=name)

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> "RuleFactor":
        """规则因子不需要训练"""
        self._is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算因子值"""
        # 子类实现
        raise NotImplementedError


class MomentumFactor(RuleFactor):
    """动量因子"""

    def __init__(self, window: int = 20):
        super().__init__(name=f"Momentum_{window}")
        self.window = window

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算动量"""
        results = []

        for symbol, group in data.groupby('symbol'):
            group = group.sort_index()
            momentum = group['close'].pct_change(self.window)

            valid = momentum.dropna()
            for date, value in valid.items():
                results.append({
                    'date': date,
                    'symbol': symbol,
                    'score': value,
                })

        df = pd.DataFrame(results)
        if len(df) > 0:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        return df


class VolatilityFactor(RuleFactor):
    """波动率因子"""

    def __init__(self, window: int = 20):
        super().__init__(name=f"Volatility_{window}")
        self.window = window

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算波动率 (取负，低波动率得分高)"""
        results = []

        for symbol, group in data.groupby('symbol'):
            group = group.sort_index()
            returns = group['close'].pct_change()
            volatility = returns.rolling(self.window).std()

            for date, value in volatility.items():
                if pd.notna(value):
                    results.append({
                        'date': date,
                        'symbol': symbol,
                        'score': -value,  # 负值，低波动率得分高
                    })

        df = pd.DataFrame(results)
        if len(df) > 0:
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
        return df
