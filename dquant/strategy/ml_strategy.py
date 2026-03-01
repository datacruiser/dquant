"""
ML 因子策略
"""

from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np

from dquant.strategy.base import BaseStrategy, Signal, SignalType
from dquant.ai.base import BaseFactor


class MLFactorStrategy(BaseStrategy):
    """
    ML 因子策略
    
    使用机器学习模型预测股票收益，选取 TopK 做多。
    
    Usage:
        from dquant.ai import XGBoostFactor
        
        # 定义因子
        factor = XGBoostFactor(features=['pe', 'pb', 'momentum', 'volatility'])
        
        # 创建策略
        strategy = MLFactorStrategy(factor=factor, top_k=10)
        
        # 回测
        result = engine.backtest(strategy=strategy)
    """
    
    def __init__(
        self,
        factor: BaseFactor,
        top_k: int = 10,
        rebalance_freq: int = 5,  # 调仓频率 (交易日)
        name: str = "MLFactorStrategy",
    ):
        super().__init__(name=name)
        self.factor = factor
        self.top_k = top_k
        self.rebalance_freq = rebalance_freq
        self._last_rebalance = None
        
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """生成信号"""
        signals = []
        
        # 获取因子预测值
        predictions = self.factor.predict(data)
        
        # 按日期分组
        for date, group in predictions.groupby(predictions.index):
            # 排序选取 TopK
            top_stocks = group.nlargest(self.top_k, 'score')
            
            for _, row in top_stocks.iterrows():
                signal = Signal(
                    symbol=row['symbol'],
                    signal_type=SignalType.BUY,
                    strength=1.0 / self.top_k,  # 等权
                    timestamp=date,
                    metadata={'score': row['score']}
                )
                signals.append(signal)
                
        return signals
    
    def should_rebalance(self, date: pd.Timestamp) -> bool:
        """判断是否应该调仓"""
        if self._last_rebalance is None:
            return True
        
        days_since_last = (date - self._last_rebalance).days
        return days_since_last >= self.rebalance_freq


class TopKStrategy(BaseStrategy):
    """
    TopK 轮动策略 (简化版)
    
    根据单一因子选股，定期调仓。
    """
    
    def __init__(
        self,
        factor_name: str = 'momentum',  # 因子列名
        top_k: int = 10,
        ascending: bool = False,  # False=选大的
        name: str = "TopKStrategy",
    ):
        super().__init__(name=name)
        self.factor_name = factor_name
        self.top_k = top_k
        self.ascending = ascending
        
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """生成信号"""
        signals = []
        
        if self.factor_name not in data.columns:
            raise ValueError(f"Factor '{self.factor_name}' not found in data")
        
        # 按日期分组
        for date, group in data.groupby(data.index):
            # 排序选 TopK
            sorted_group = group.sort_values(self.factor_name, ascending=self.ascending)
            top_stocks = sorted_group.head(self.top_k)
            
            for _, row in top_stocks.iterrows():
                signal = Signal(
                    symbol=row['symbol'],
                    signal_type=SignalType.BUY,
                    strength=1.0 / self.top_k,
                    timestamp=date,
                )
                signals.append(signal)
                
        return signals
