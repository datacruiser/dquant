"""
订单重试装饰器

为 BaseBroker 提供指数退避重试能力，处理网络瞬断、超时等瞬态故障。
"""

import time
from datetime import datetime
from typing import Dict

from dquant.broker.base import BaseBroker, Order, OrderResult
from dquant.constants import ORDER_MAX_RETRIES, ORDER_RETRY_DELAY, ORDER_RETRY_BACKOFF
from dquant.logger import get_logger

logger = get_logger(__name__)

# 瞬态错误：可安全重试
_RETRYABLE_ERRORS = (ConnectionError, TimeoutError, OSError)


class RetryableBroker(BaseBroker):
    """
    可重试的券商包装器

    装饰器模式，为任意 BaseBroker 添加重试逻辑。

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

    def connect(self, **kwargs) -> bool:
        return self._broker.connect(**kwargs)

    def disconnect(self) -> bool:
        return self._broker.disconnect()

    def place_order(self, order: Order) -> OrderResult:
        """
        下单（带重试）

        仅对瞬态错误（网络/超时）重试。验证错误、资金不足等不重试。
        """
        for attempt in range(self._max_retries):
            try:
                result = self._broker.place_order(order)

                # REJECTED 不重试（验证错误、资金不足等）
                if result.status == 'REJECTED':
                    return result

                return result

            except _RETRYABLE_ERRORS as e:
                delay = self._retry_delay * (self._backoff ** attempt)
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
            order_id='',
            symbol=order.symbol,
            side=order.side,
            filled_quantity=0,
            filled_price=0,
            commission=0,
            timestamp=datetime.now(),
            status='REJECTED',
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

    def __getattr__(self, name):
        """代理其他属性到内部 broker"""
        return getattr(self._broker, name)
