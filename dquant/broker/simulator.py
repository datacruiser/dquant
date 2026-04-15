"""
模拟交易
"""

import uuid
from copy import deepcopy
from datetime import datetime
from typing import Dict, Optional

from dquant.broker.base import BaseBroker, Order, OrderResult
from dquant.broker.safety import OrderValidator
from dquant.constants import (
    DEFAULT_COMMISSION,
    DEFAULT_INITIAL_CASH,
    DEFAULT_SLIPPAGE,
    DEFAULT_STAMP_DUTY,
    MIN_SHARES,
)
from dquant.logger import get_logger

logger = get_logger(__name__)


class Simulator(BaseBroker):
    """
    模拟券商

    用于回测和模拟交易，不实际下单。

    Note:
        Simulator is NOT thread-safe. Use from a single thread only.

    Args:
        initial_cash: 初始资金
        order_id_prefix: 订单 ID 前缀 (None = 使用 UUID)
        apply_slippage: 是否应用滑点
        validate_orders: 是否通过 OrderValidator 校验订单
        adjust_lots: 买入资金不足时是否向下调整到整手
        strict_sell: 卖出时数量超过持仓是否直接拒绝 (False = min(qty, position))
    """

    def __init__(
        self,
        initial_cash: float = DEFAULT_INITIAL_CASH,
        *,
        order_id_prefix: str | None = None,
        apply_slippage: bool = True,
        validate_orders: bool = True,
        adjust_lots: bool = True,
        strict_sell: bool = False,
    ):
        super().__init__(name="Simulator")
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, dict] = {}
        self.orders: Dict[str, Order] = {}
        self._order_id_prefix = order_id_prefix
        self._apply_slippage = apply_slippage
        self._validate_orders = validate_orders
        self._adjust_lots = adjust_lots
        self._strict_sell = strict_sell

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
            pos["quantity"] * pos["price"] for pos in self.positions.values()
        )
        profit_pct = (
            (total_value - self.initial_cash) / self.initial_cash if self.initial_cash != 0 else 0.0
        )
        return {
            "cash": self.cash,
            "total_value": total_value,
            "initial_cash": self.initial_cash,
            "profit": total_value - self.initial_cash,
            "profit_pct": profit_pct,
        }

    def get_positions(self) -> Dict[str, dict]:
        """获取持仓（防御性拷贝）"""
        return deepcopy(self.positions)

    def _generate_order_id(self, order: Order) -> str:
        """生成订单 ID，子类可覆盖"""
        if self._order_id_prefix:
            return (
                f"{self._order_id_prefix}_{order.symbol}"
                f"_{datetime.now().strftime('%Y%m%d%H%M%S%f')}_{id(order)}"
            )
        return str(uuid.uuid4())

    def _make_rejected_result(self, order: Order) -> OrderResult:
        """构造统一的 REJECTED OrderResult"""
        return OrderResult(
            order_id=order.order_id or "",
            symbol=order.symbol,
            side=order.side,
            filled_quantity=0,
            filled_price=0,
            commission=0,
            timestamp=order.timestamp or datetime.now(),
            status="REJECTED",
        )

    def place_order(self, order: Order) -> OrderResult:
        """下单"""
        order.order_id = self._generate_order_id(order)
        order.timestamp = datetime.now()

        # 基本订单验证（可通过 validate_orders=False 关闭）
        if self._validate_orders:
            valid, msg = OrderValidator.validate_order(order)
            if not valid:
                order.status = "REJECTED"
                self.orders[order.order_id] = order
                return self._make_rejected_result(order)

        # 模拟成交
        filled_price = order.price or self._get_simulated_price(order.symbol)

        # 应用滑点（可通过 apply_slippage=False 关闭）
        if self._apply_slippage:
            if order.side == "BUY":
                filled_price *= 1 + DEFAULT_SLIPPAGE
            elif order.side == "SELL":
                filled_price *= 1 - DEFAULT_SLIPPAGE

        filled_quantity = order.quantity

        if order.side == "BUY":
            # 买入：成本 = 价格 * 数量 * (1 + 佣金率)
            total_cost = filled_price * filled_quantity * (1 + DEFAULT_COMMISSION)
            if total_cost > self.cash:
                if self._adjust_lots:
                    # 按整手调整
                    filled_quantity = (
                        int(self.cash / (filled_price * (1 + DEFAULT_COMMISSION)) // MIN_SHARES)
                        * MIN_SHARES
                    )
                    if filled_quantity <= 0:
                        order.status = "REJECTED"
                        self.orders[order.order_id] = order
                        return self._make_rejected_result(order)
                    total_cost = filled_price * filled_quantity * (1 + DEFAULT_COMMISSION)
                else:
                    order.status = "REJECTED"
                    self.orders[order.order_id] = order
                    return self._make_rejected_result(order)

            self.cash -= total_cost

            if order.symbol in self.positions:
                pos = self.positions[order.symbol]
                total_qty = pos["quantity"] + filled_quantity
                pos["avg_cost"] = (
                    pos["avg_cost"] * pos["quantity"] + filled_price * filled_quantity
                ) / total_qty
                pos["quantity"] = total_qty
                pos["price"] = filled_price
            else:
                self.positions[order.symbol] = {
                    "quantity": filled_quantity,
                    "avg_cost": filled_price,
                    "price": filled_price,
                }

            commission = filled_price * filled_quantity * DEFAULT_COMMISSION

        elif order.side == "SELL":
            if order.symbol not in self.positions or self.positions[order.symbol]["quantity"] <= 0:
                order.status = "REJECTED"
                self.orders[order.order_id] = order
                return self._make_rejected_result(order)

            pos = self.positions[order.symbol]

            if self._strict_sell and order.quantity > pos["quantity"]:
                order.status = "REJECTED"
                self.orders[order.order_id] = order
                return self._make_rejected_result(order)

            filled_quantity = min(filled_quantity, pos["quantity"])
            revenue = filled_price * filled_quantity
            # A 股卖出：扣佣金 + 印花税
            total_cost = revenue * (DEFAULT_COMMISSION + DEFAULT_STAMP_DUTY)
            self.cash += revenue - total_cost
            pos["quantity"] -= filled_quantity
            pos["price"] = filled_price

            # 佣金包含基础佣金 + 印花税，确保 P&L 计算准确
            commission = filled_price * filled_quantity * (DEFAULT_COMMISSION + DEFAULT_STAMP_DUTY)

            if pos["quantity"] <= 0:
                del self.positions[order.symbol]

        order.filled_quantity = filled_quantity
        order.status = "FILLED"
        self.orders[order.order_id] = order

        return OrderResult(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            filled_quantity=filled_quantity,
            filled_price=filled_price,
            commission=commission,
            timestamp=order.timestamp,
            status="FILLED",
        )

    def cancel_order(self, order_id: str) -> bool:
        """撤单"""
        if order_id in self.orders:
            self.orders[order_id].status = "CANCELLED"
            return True
        return False

    def get_order_status(self, order_id: str) -> Optional[Order]:
        """查询订单状态"""
        return self.orders.get(order_id)

    def get_market_data(self, symbol: str) -> dict:
        """获取实时行情"""
        # 返回模拟数据
        return {
            "symbol": symbol,
            "price": self._get_simulated_price(symbol),
            "bid": 0,
            "ask": 0,
            "volume": 0,
            "timestamp": datetime.now(),
        }

    def _get_simulated_price(self, symbol: str) -> float:
        """获取模拟价格"""
        if symbol in self.positions:
            return self.positions[symbol].get("price", 10.0)
        logger.warning(f"[Simulator] 使用默认价格 10.0 用于未知标的: {symbol}")
        return 10.0  # 默认价格

    def update_prices(self, prices: Dict[str, float]):
        """更新持仓价格"""
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol]["price"] = price
