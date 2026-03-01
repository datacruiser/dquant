"""
模拟交易
"""

from typing import Dict, Optional
from datetime import datetime
import uuid

from dquant.broker.base import BaseBroker, Order, OrderResult
from dquant.constants import DEFAULT_COMMISSION, DEFAULT_SLIPPAGE, DEFAULT_STAMP_DUTY, DEFAULT_INITIAL_CASH, MIN_SHARES, DEFAULT_WINDOW


class Simulator(BaseBroker):
    """
    模拟券商
    
    用于回测和模拟交易，不实际下单。
    """
    
    def __init__(self, initial_cash: float = DEFAULT_INITIAL_CASH):
        super().__init__(name="Simulator")
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, dict] = {}
        self.orders: Dict[str, Order] = {}
        
    def connect(self, **kwargs) -> bool:
        """模拟连接"""
        print(f"[{self.name}] Connected (simulated)")
        return True
    
    def disconnect(self) -> bool:
        """模拟断开"""
        print(f"[{self.name}] Disconnected")
        return True
    
    def get_account(self) -> dict:
        """获取账户信息"""
        total_value = self.cash + sum(
            pos['quantity'] * pos['price'] 
            for pos in self.positions.values()
        )
        return {
            'cash': self.cash,
            'total_value': total_value,
            'initial_cash': self.initial_cash,
            'profit': total_value - self.initial_cash,
            'profit_pct': (total_value - self.initial_cash) / self.initial_cash,
        }
    
    def get_positions(self) -> Dict[str, dict]:
        """获取持仓"""
        return self.positions
    
    def place_order(self, order: Order) -> OrderResult:
        """下单"""
        order.order_id = str(uuid.uuid4())[:8]
        order.timestamp = datetime.now()
        
        # 模拟成交
        filled_price = order.price or self._get_simulated_price(order.symbol)
        filled_quantity = order.quantity
        
        if order.side == 'BUY':
            # 买入
            cost = filled_price * filled_quantity
            if cost > self.cash:
                filled_quantity = self.cash / filled_price
                cost = self.cash
            
            self.cash -= cost
            
            if order.symbol in self.positions:
                pos = self.positions[order.symbol]
                total_qty = pos['quantity'] + filled_quantity
                pos['avg_cost'] = (pos['avg_cost'] * pos['quantity'] + filled_price * filled_quantity) / total_qty
                pos['quantity'] = total_qty
            else:
                self.positions[order.symbol] = {
                    'quantity': filled_quantity,
                    'avg_cost': filled_price,
                    'price': filled_price,
                }
                
        elif order.side == 'SELL':
            # 卖出
            if order.symbol not in self.positions:
                filled_quantity = 0
            else:
                pos = self.positions[order.symbol]
                filled_quantity = min(filled_quantity, pos['quantity'])
                revenue = filled_price * filled_quantity
                self.cash += revenue
                pos['quantity'] -= filled_quantity
                
                if pos['quantity'] <= 0:
                    del self.positions[order.symbol]
        
        order.status = 'FILLED'
        self.orders[order.order_id] = order
        
        return OrderResult(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            filled_quantity=filled_quantity,
            filled_price=filled_price,
            commission=filled_price * filled_quantity * DEFAULT_COMMISSION,  # 万三
            timestamp=order.timestamp,
            status='FILLED',
        )
    
    def cancel_order(self, order_id: str) -> bool:
        """撤单"""
        if order_id in self.orders:
            self.orders[order_id].status = 'CANCELLED'
            return True
        return False
    
    def get_order_status(self, order_id: str) -> Order:
        """查询订单状态"""
        return self.orders.get(order_id)
    
    def get_market_data(self, symbol: str) -> dict:
        """获取实时行情"""
        # 返回模拟数据
        return {
            'symbol': symbol,
            'price': self._get_simulated_price(symbol),
            'bid': 0,
            'ask': 0,
            'volume': 0,
            'timestamp': datetime.now(),
        }
    
    def _get_simulated_price(self, symbol: str) -> float:
        """获取模拟价格"""
        if symbol in self.positions:
            return self.positions[symbol].get('price', 10.0)
        return 10.0  # 默认价格
    
    def update_prices(self, prices: Dict[str, float]):
        """更新持仓价格"""
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol]['price'] = price
