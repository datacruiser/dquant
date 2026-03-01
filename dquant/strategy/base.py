"""
策略基类
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List, Dict, Any
from datetime import datetime
import pandas as pd


class SignalType(Enum):
    """信号类型"""
    BUY = 1
    SELL = -1
    HOLD = 0


@dataclass
class Signal:
    """
    交易信号
    
    统一的信号格式，用于回测和实盘。
    """
    symbol: str
    signal_type: SignalType
    strength: float = 1.0  # 信号强度 0-1
    price: Optional[float] = None  # 目标价格
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = None  # 额外信息
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def is_buy(self) -> bool:
        return self.signal_type == SignalType.BUY
    
    @property
    def is_sell(self) -> bool:
        return self.signal_type == SignalType.SELL
    
    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'signal_type': self.signal_type.value,
            'strength': self.strength,
            'price': self.price,
            'timestamp': self.timestamp,
            'metadata': self.metadata,
        }


class BaseStrategy(ABC):
    """
    策略基类
    
    所有策略都需要实现 generate_signals() 方法。
    """
    
    def __init__(self, name: str = "BaseStrategy"):
        self.name = name
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        生成交易信号
        
        Args:
            data: 市场数据 (包含所有股票的历史数据)
            
        Returns:
            信号列表
        """
        pass
    
    def on_bar(self, bar: pd.Series) -> Optional[Signal]:
        """
        实时 K 线回调 (用于实盘)
        
        Args:
            bar: 当前 K 线数据
            
        Returns:
            单个信号或 None
        """
        return None
    
    def on_tick(self, tick: dict) -> Optional[Signal]:
        """
        实时 Tick 回调 (用于高频交易)
        
        Args:
            tick: 当前 Tick 数据
            
        Returns:
            单个信号或 None
        """
        return None
    
    def get_positions(self, data: pd.DataFrame, signals: List[Signal]) -> Dict[str, float]:
        """
        根据信号计算目标持仓权重
        
        Args:
            data: 市场数据
            signals: 信号列表
            
        Returns:
            {symbol: weight} 目标持仓权重
        """
        positions = {}
        
        for signal in signals:
            if signal.is_buy:
                positions[signal.symbol] = signal.strength
                
        # 归一化权重
        total = sum(positions.values())
        if total > 0:
            positions = {k: v/total for k, v in positions.items()}
            
        return positions
