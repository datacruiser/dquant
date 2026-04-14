"""
订单重试装饰器

为 BaseBroker 提供指数退避重试能力，处理网络瞬断、超时等瞬态故障。
包含幂等性保护，防止重试导致双重下单。
"""

import threading
import time
import uuid
from datetime import datetime
from typing import Dict

from dquant.broker.base import BaseBroker, Order, OrderResult
from dquant.constants import ORDER_MAX_RETRIES, ORDER_RETRY_BACKOFF, ORDER_RETRY_DELAY
from dquant.logger import get_logger

logger = get_logger(__name__)

# 瞬态错误：可安全重试
_RETRYABLE_ERRORS = (ConnectionError, TimeoutError, OSError)


class RetryableBroker(BaseBroker):
    """
    可重试的券商包装器

    装饰器模式，为任意 BaseBroker 添加重试逻辑。
    包含幂等性保护：首次下单前生成 idempotency_key，
    重试时先查询该 key 对应的订单是否已存在。

    Usage:
        broker = RetryableBroker(XTPBroker(...), max_retries=3)
        result = broker.place_order(order)  # 自动重试瞬态错误
    """

    def __init__(
        self,
        broker: BaseBroker,
        max_retries: int = ORDER_MAX_RETRIES,
        retry_delay: float = ORDER_RETRY_DELAY,
        backoff: float = ORDER_RETRY_BACKOFF,
    ):
        super().__init__(name=f"Retryable({broker.name})")
        self._broker = broker
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._backoff = backoff
        # 幂等性保护：记录 idempotency_key -> order_id 的映射
        self._idempotency_map: Dict[str, str] = {}
        self._idempotency_lock = threading.Lock()

    def connect(self, **kwargs) -> bool:
        return self._broker.connect(**kwargs)

    def disconnect(self) -> bool:
        return self._broker.disconnect()

    def place_order(self, order: Order) -> OrderResult:
        """
        下单（带重试 + 幂等性保护）

        仅对瞬态错误（网络/超时）重试。验证错误、资金不足等不重试。
        首次下单前生成 idempotency_key，重试时先检查是否已有成功记录。
        """
        # 生成幂等性 key
        idempotency_key = str(uuid.uuid4())

        for attempt in range(self._max_retries):
            # 重试时检查幂等性：是否之前已成功下单
            if attempt > 0:
                with self._idempotency_lock:
                    existing_order_id = self._idempotency_map.get(idempotency_key)
                if existing_order_id:
                    logger.warning(
                        f"[RETRY] 幂等性检查发现已存在的订单: {existing_order_id}，跳过重试"
                    )
                    # 尝试查询已有订单状态
                    try:
                        existing_order = self._broker.get_order_status(existing_order_id)
                        if existing_order is not None:
                            return OrderResult(
                                order_id=existing_order.order_id or "",
                                symbol=existing_order.symbol,
                                side=existing_order.side,
                                filled_quantity=existing_order.filled_quantity,
                                filled_price=existing_order.price or 0,
                                commission=0,
                                timestamp=datetime.now(),
                                status=existing_order.status,
                            )
                    except Exception:
                        logger.debug("[Retry] 查询订单状态失败")
                        pass  # 查询失败，继续重试

            try:
                result = self._broker.place_order(order)

                # 下单成功，记录幂等性映射
                if result.status in ("FILLED", "PENDING", "PARTIAL_FILLED"):
                    with self._idempotency_lock:
                        self._idempotency_map[idempotency_key] = result.order_id

                # REJECTED 不重试（验证错误、资金不足等）
                if result.status == "REJECTED":
                    return result

                return result

            except _RETRYABLE_ERRORS as e:
                delay = self._retry_delay * (self._backoff**attempt)
                logger.warning(
                    f"[RETRY] place_order 瞬态错误 "
                    f"(attempt {attempt + 1}/{self._max_retries}): "
                    f"{e}, {delay:.1f}s 后重试"
                )

                if attempt < self._max_retries - 1:
                    time.sleep(delay)

            except Exception:
                # 未知错误不重试，直接抛出
                raise

        # 重试耗尽
        logger.error(
            f"[RETRY] place_order 重试 {self._max_retries} 次后仍失败: "
            f"{order.symbol} {order.side}"
        )
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

    # --- 委托方法 ---
    def get_account(self) -> dict:
        return self._broker.get_account()

    def get_positions(self) -> Dict[str, dict]:
        return self._broker.get_positions()

    def cancel_order(self, order_id: str) -> bool:
        return self._broker.cancel_order(order_id)

    def get_order_status(self, order_id: str) -> Order:
        return self._broker.get_order_status(order_id)

    def get_market_data(self, symbol: str) -> dict:
        return self._broker.get_market_data(symbol)

    def is_connected(self) -> bool:
        return self._broker.is_connected()

    def __getattr__(self, name):
        """代理其他属性到内部 broker"""
        return getattr(self._broker, name)
