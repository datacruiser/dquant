"""
券商接口基类
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Order:
    """订单"""
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: float
    price: Optional[float] = None  # None = 市价单
    order_type: str = 'MARKET'  # 'MARKET' or 'LIMIT'
    timestamp: datetime = None
    order_id: str = None
    status: str = 'PENDING'  # PENDING, FILLED, CANCELLED, REJECTED


@dataclass
class OrderResult:
    """订单结果"""
    order_id: str
    symbol: str
    side: str
    filled_quantity: float
    filled_price: float
    commission: float
    timestamp: datetime
    status: str


class BaseBroker(ABC):
    """
    券商接口基类

    所有券商接口都需要实现这些方法。
    """

    def __init__(self, name: str = "BaseBroker"):
        self.name = name

    @abstractmethod
    def connect(self, **kwargs) -> bool:
        """连接券商"""
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        """断开连接"""
        pass

    @abstractmethod
    def get_account(self) -> dict:
        """获取账户信息"""
        pass

    @abstractmethod
    def get_positions(self) -> Dict[str, dict]:
        """获取持仓"""
        pass

    @abstractmethod
    def place_order(self, order: Order) -> OrderResult:
        """下单"""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """撤单"""
        pass

    @abstractmethod
    def get_order_status(self, order_id: str) -> Order:
        """查询订单状态"""
        pass

    @abstractmethod
    def get_market_data(self, symbol: str) -> dict:
        """获取实时行情"""
        pass
