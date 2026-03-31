"""
订单追踪器

跟踪 PENDING/PARTIAL_FILLED 订单，支持超时检测。
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from dquant.broker.base import Order, OrderResult
from dquant.constants import ORDER_TIMEOUT_SECONDS
from dquant.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TrackedOrder:
    """被追踪的订单"""
    order: Order
    result: OrderResult
    remaining_quantity: float
    first_seen: datetime = field(default_factory=datetime.now)
    last_checked: datetime = field(default_factory=datetime.now)
    check_count: int = 0


class OrderTracker:
    """
    订单状态追踪器

    跟踪未完成（PENDING/PARTIAL_FILLED）订单的状态变化。

    Usage:
        tracker = OrderTracker(timeout_seconds=30)

        # 下单后加入追踪
        tracker.add(order, result)

        # 每个周期轮询
        for order_id, tracked in tracker.get_pending().items():
            updated = broker.get_order_status(order_id)
            tracker.update(order_id, updated)

        # 获取超时订单
        for tracked in tracker.get_timed_out():
            broker.cancel_order(tracked.order.order_id)
            tracker.remove(tracked.order.order_id)
    """

    def __init__(self, timeout_seconds: int = ORDER_TIMEOUT_SECONDS):
        self._timeout = timeout_seconds
        self._pending: Dict[str, TrackedOrder] = {}

    def add(self, order: Order, result: OrderResult) -> None:
        """将 PENDING/PARTIAL_FILLED 订单加入追踪"""
        if result.status not in ('PENDING', 'PARTIAL_FILLED'):
            return

        remaining = order.quantity - result.filled_quantity
        if remaining <= 0:
            return

        tracked = TrackedOrder(
            order=order,
            result=result,
            remaining_quantity=remaining,
        )
        self._pending[order.order_id] = tracked
        logger.info(
            f"[TRACKER] 追踪订单: {order.order_id} "
            f"{order.symbol} {order.side} "
            f"剩余: {remaining}"
        )

    def update(self, order_id: str, order: Order) -> Optional[TrackedOrder]:
        """更新订单状态（从 broker.get_order_status 获取）"""
        tracked = self._pending.get(order_id)
        if tracked is None:
            return None

        tracked.last_checked = datetime.now()
        tracked.check_count += 1
        tracked.order = order

        # 如果已完成，移除追踪
        if order.status in ('FILLED', 'CANCELLED', 'REJECTED'):
            self.remove(order_id)
            logger.info(
                f"[TRACKER] 订单完成: {order_id} status={order.status}"
            )
            return tracked

        # 更新剩余数量
        if hasattr(order, 'filled_quantity'):
            tracked.remaining_quantity = order.quantity - order.filled_quantity

        return tracked

    def remove(self, order_id: str) -> None:
        """移除追踪"""
        self._pending.pop(order_id, None)

    def get_pending(self) -> Dict[str, TrackedOrder]:
        """获取所有待完成订单"""
        return dict(self._pending)

    def has_pending(self) -> bool:
        """是否有待完成订单"""
        return len(self._pending) > 0

    def get_timed_out(self) -> List[TrackedOrder]:
        """获取超时的订单"""
        now = datetime.now()
        timed_out = []
        for order_id, tracked in list(self._pending.items()):
            elapsed = (now - tracked.first_seen).total_seconds()
            if elapsed >= self._timeout:
                timed_out.append(tracked)
        return timed_out

    def cancel_all(self, broker) -> List[str]:
        """
        取消所有 pending 订单

        Args:
            broker: 用于调用 cancel_order 的 broker 实例

        Returns:
            成功取消的 order_id 列表
        """
        cancelled = []
        for order_id, tracked in list(self._pending.items()):
            try:
                ok = broker.cancel_order(order_id)
                if ok:
                    cancelled.append(order_id)
                    logger.info(f"[TRACKER] 关机取消订单: {order_id}")
                else:
                    logger.warning(f"[TRACKER] 关机取消订单失败: {order_id}")
            except Exception as e:
                logger.error(f"[TRACKER] 关机取消订单异常: {order_id} — {e}")
        self._pending.clear()
        return cancelled
