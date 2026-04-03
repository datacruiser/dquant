"""
模拟交易
"""

from typing import Dict, Optional
from datetime import datetime
from copy import deepcopy
import uuid

from dquant.broker.base import BaseBroker, Order, OrderResult
from dquant.broker.safety import OrderValidator
from dquant.constants import (
    DEFAULT_COMMISSION, DEFAULT_SLIPPAGE, DEFAULT_STAMP_DUTY,
    DEFAULT_INITIAL_CASH, MIN_SHARES,
)
from dquant.logger import get_logger

logger = get_logger(__name__)


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
        self._connected = True
        logger.info(f"[{self.name}] Connected (simulated)")
        return True

    def disconnect(self) -> bool:
        """模拟断开"""
        self._connected = False
        logger.info(f"[{self.name}] Disconnected")
        return True

    def get_account(self) -> dict:
        """获取账户信息"""
        total_value = self.cash + sum(
            pos['quantity'] * pos['price']
            for pos in self.positions.values()
        )
        profit_pct = (
            (total_value - self.initial_cash) / self.initial_cash
            if self.initial_cash != 0 else 0.0
        )
        return {
            'cash': self.cash,
            'total_value': total_value,
            'initial_cash': self.initial_cash,
            'profit': total_value - self.initial_cash,
            'profit_pct': profit_pct,
        }

    def get_positions(self) -> Dict[str, dict]:
        """获取持仓（防御性拷贝）"""
        return deepcopy(self.positions)

    def place_order(self, order: Order) -> OrderResult:
        """下单"""
        order.order_id = str(uuid.uuid4())
        order.timestamp = datetime.now()

        # 基本订单验证
        valid, msg = OrderValidator.validate_order(order)
        if not valid:
            order.status = 'REJECTED'
            return OrderResult(
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                filled_quantity=0,
                filled_price=0,
                commission=0,
                timestamp=order.timestamp,
                status='REJECTED',
            )

        # 模拟成交
        filled_price = order.price or self._get_simulated_price(order.symbol)

        # 应用滑点
        if order.side == 'BUY':
            filled_price *= (1 + DEFAULT_SLIPPAGE)
        elif order.side == 'SELL':
            filled_price *= (1 - DEFAULT_SLIPPAGE)

        filled_quantity = order.quantity

        if order.side == 'BUY':
            # 买入：成本 = 价格 * 数量 * (1 + 佣金率)
            total_cost = filled_price * filled_quantity * (1 + DEFAULT_COMMISSION)
            if total_cost > self.cash:
                # 按整手调整
                filled_quantity = int(self.cash / (filled_price * (1 + DEFAULT_COMMISSION)) // MIN_SHARES) * MIN_SHARES
                if filled_quantity <= 0:
                    order.status = 'REJECTED'
                    self.orders[order.order_id] = order
                    return OrderResult(
                        order_id=order.order_id,
                        symbol=order.symbol,
                        side=order.side,
                        filled_quantity=0,
                        filled_price=0,
                        commission=0,
                        timestamp=order.timestamp,
                        status='REJECTED',
                    )
                total_cost = filled_price * filled_quantity * (1 + DEFAULT_COMMISSION)

            self.cash -= total_cost

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

            commission = filled_price * filled_quantity * DEFAULT_COMMISSION

        elif order.side == 'SELL':
            if order.symbol not in self.positions or self.positions[order.symbol]['quantity'] <= 0:
                order.status = 'REJECTED'
                self.orders[order.order_id] = order
                return OrderResult(
                    order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side,
                    filled_quantity=0,
                    filled_price=0,
                    commission=0,
                    timestamp=order.timestamp,
                    status='REJECTED',
                )

            pos = self.positions[order.symbol]
            filled_quantity = min(filled_quantity, pos['quantity'])
            revenue = filled_price * filled_quantity
            # A 股卖出：扣佣金 + 印花税
            total_cost = revenue * (DEFAULT_COMMISSION + DEFAULT_STAMP_DUTY)
            self.cash += revenue - total_cost
            pos['quantity'] -= filled_quantity

            # 佣金包含基础佣金 + 印花税，确保 P&L 计算准确
            commission = filled_price * filled_quantity * (DEFAULT_COMMISSION + DEFAULT_STAMP_DUTY)

            if pos['quantity'] <= 0:
                del self.positions[order.symbol]

        order.filled_quantity = filled_quantity
        order.status = 'FILLED'
        self.orders[order.order_id] = order

        return OrderResult(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            filled_quantity=filled_quantity,
            filled_price=filled_price,
            commission=commission,
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
        logger.warning(f"[Simulator] 使用默认价格 10.0 用于未知标的: {symbol}")
        return 10.0  # 默认价格

    def update_prices(self, prices: Dict[str, float]):
        """更新持仓价格"""
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol]['price'] = price
