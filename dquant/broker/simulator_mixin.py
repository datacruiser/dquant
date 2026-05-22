"""
Simulator Mixin — shared simulation layer for broker simulators.

XTPSimulator and QMTSimulator share the exact same delegation pattern:
compose a Simulator instance and delegate all methods.  This mixin
eliminates ~100 lines of duplication per broker.
"""

from copy import deepcopy
from datetime import datetime
from typing import Dict, Optional

from dquant.broker.base import Order, OrderResult
from dquant.broker.simulator import Simulator
from dquant.constants import DEFAULT_INITIAL_CASH
from dquant.logger import get_logger

logger = get_logger(__name__)


class SimulatorMixin:
    """Mixin that delegates all broker methods to an internal Simulator.

    Subclass must set:
        self._sim = Simulator(...)
        self._connected = False
        self.name = "YourSimulator"
    """

    _sim: Simulator
    _connected: bool
    name: str

    # ---------- 连接管理 ----------
    def sim_connect(self, **kwargs) -> bool:
        logger.info(f"[{self.name}] Connected (simulated)")
        self._connected = True
        self._sim.connect()
        return True

    def sim_disconnect(self) -> bool:
        self._connected = False
        return True

    # ---------- 账户 & 持仓 ----------
    def sim_get_account(self) -> dict:
        total_value = self._sim.cash + sum(
            p["quantity"] * p.get("price", 0) for p in self._sim.positions.values()
        )
        return {
            "cash": self._sim.cash,
            "total_value": total_value,
            "market_value": total_value - self._sim.cash,
            "available": self._sim.cash,
        }

    def sim_get_positions(self) -> Dict[str, dict]:
        return deepcopy(self._sim.positions)

    # ---------- 交易 ----------
    def sim_place_order(self, order: Order) -> OrderResult:
        if not self._connected:
            return OrderResult(
                order_id="",
                symbol=order.symbol,
                side=order.side,
                filled_quantity=0,
                filled_price=0,
                commission=0,
                timestamp=datetime.now(),
                status="REJECTED",
            )
        return self._sim.place_order(order)

    def sim_cancel_order(self, order_id: str) -> bool:
        return self._sim.cancel_order(order_id)

    def sim_get_order_status(self, order_id: str) -> Optional[Order]:
        return self._sim.get_order_status(order_id)

    # ---------- 行情 ----------
    def sim_get_market_data(self, symbol: str) -> dict:
        return self._sim.get_market_data(symbol)

    # ---------- 属性代理（保持兼容） ----------
    @property
    def initial_cash(self) -> float:
        return self._sim.initial_cash

    @property
    def cash(self) -> float:
        return self._sim.cash

    @cash.setter
    def cash(self, value: float):
        with self._sim._lock:
            self._sim.cash = value

    @property
    def positions(self) -> Dict[str, dict]:
        return self._sim.positions

    @positions.setter
    def positions(self, value: Dict[str, dict]):
        with self._sim._lock:
            self._sim.positions = value

    @property
    def orders(self) -> Dict[str, Order]:
        return self._sim.orders

    @orders.setter
    def orders(self, value: Dict[str, Order]):
        with self._sim._lock:
            self._sim.orders = value

    def update_prices(self, price_map: dict):
        """Delegate price updates to the internal Simulator."""
        self._sim.update_prices(price_map)
