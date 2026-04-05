"""
Phase 2 Step 2: RetryableBroker 测试
"""

import time
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from dquant.broker.base import BaseBroker, Order, OrderResult
from dquant.broker.retry import RetryableBroker
from dquant.broker.simulator import Simulator


def _make_order(**overrides):
    defaults = dict(
        symbol="000001.SZ",
        side="BUY",
        quantity=100,
        price=10.0,
        order_type="LIMIT",
    )
    defaults.update(overrides)
    return Order(**defaults)


def _make_result(**overrides):
    defaults = dict(
        order_id="test-001",
        symbol="000001.SZ",
        side="BUY",
        filled_quantity=100,
        filled_price=10.0,
        commission=3.0,
        timestamp=datetime.now(),
        status="FILLED",
    )
    defaults.update(overrides)
    return OrderResult(**defaults)


class TestRetryableBroker:

    def test_success_no_retry(self):
        """正常下单不需要重试"""
        inner = Simulator()
        broker = RetryableBroker(inner, max_retries=3)
        order = _make_order()
        result = broker.place_order(order)
        assert result.status == "FILLED"

    def test_delegates_connect(self):
        """connect 委托给内部 broker"""
        inner = MagicMock(spec=BaseBroker)
        inner.connect.return_value = True
        inner.name = "mock"
        broker = RetryableBroker(inner)
        assert broker.connect() is True
        inner.connect.assert_called_once()

    def test_delegates_disconnect(self):
        """disconnect 委托给内部 broker"""
        inner = MagicMock(spec=BaseBroker)
        inner.disconnect.return_value = True
        inner.name = "mock"
        broker = RetryableBroker(inner)
        assert broker.disconnect() is True

    def test_delegates_get_account(self):
        inner = MagicMock(spec=BaseBroker)
        inner.get_account.return_value = {"cash": 100000}
        inner.name = "mock"
        broker = RetryableBroker(inner)
        assert broker.get_account() == {"cash": 100000}

    def test_delegates_get_positions(self):
        inner = MagicMock(spec=BaseBroker)
        inner.get_positions.return_value = {}
        inner.name = "mock"
        broker = RetryableBroker(inner)
        assert broker.get_positions() == {}

    def test_delegates_cancel_order(self):
        inner = MagicMock(spec=BaseBroker)
        inner.cancel_order.return_value = True
        inner.name = "mock"
        broker = RetryableBroker(inner)
        assert broker.cancel_order("123") is True

    def test_delegates_get_order_status(self):
        inner = MagicMock(spec=BaseBroker)
        order = _make_order()
        inner.get_order_status.return_value = order
        inner.name = "mock"
        broker = RetryableBroker(inner)
        assert broker.get_order_status("123") == order

    def test_delegates_get_market_data(self):
        inner = MagicMock(spec=BaseBroker)
        inner.get_market_data.return_value = {"price": 10.0}
        inner.name = "mock"
        broker = RetryableBroker(inner)
        assert broker.get_market_data("000001.SZ") == {"price": 10.0}

    def test_retries_on_connection_error(self):
        """ConnectionError 触发重试"""
        inner = MagicMock(spec=BaseBroker)
        inner.name = "mock"

        # 第一次 ConnectionError，第二次成功
        inner.place_order.side_effect = [
            ConnectionError("网络断开"),
            _make_result(status="FILLED"),
        ]

        broker = RetryableBroker(inner, max_retries=3, retry_delay=0.01, backoff=1.0)
        result = broker.place_order(_make_order())
        assert result.status == "FILLED"
        assert inner.place_order.call_count == 2

    def test_retries_on_timeout_error(self):
        """TimeoutError 触发重试"""
        inner = MagicMock(spec=BaseBroker)
        inner.name = "mock"
        inner.place_order.side_effect = [
            TimeoutError("请求超时"),
            _make_result(status="FILLED"),
        ]

        broker = RetryableBroker(inner, max_retries=3, retry_delay=0.01)
        result = broker.place_order(_make_order())
        assert result.status == "FILLED"

    def test_no_retry_on_value_error(self):
        """ValueError 不重试，直接抛出"""
        inner = MagicMock(spec=BaseBroker)
        inner.name = "mock"
        inner.place_order.side_effect = ValueError("参数错误")

        broker = RetryableBroker(inner, max_retries=3, retry_delay=0.01)
        with pytest.raises(ValueError, match="参数错误"):
            broker.place_order(_make_order())
        assert inner.place_order.call_count == 1

    def test_no_retry_on_rejected(self):
        """REJECTED 状态不重试"""
        inner = MagicMock(spec=BaseBroker)
        inner.name = "mock"
        inner.place_order.return_value = _make_result(status="REJECTED", filled_quantity=0)

        broker = RetryableBroker(inner, max_retries=3, retry_delay=0.01)
        result = broker.place_order(_make_order())
        assert result.status == "REJECTED"
        assert inner.place_order.call_count == 1

    def test_exhausted_retries_returns_rejected(self):
        """重试耗尽返回 REJECTED"""
        inner = MagicMock(spec=BaseBroker)
        inner.name = "mock"
        inner.place_order.side_effect = ConnectionError("持续断连")

        broker = RetryableBroker(inner, max_retries=2, retry_delay=0.01)
        result = broker.place_order(_make_order())
        assert result.status == "REJECTED"
        assert result.filled_quantity == 0
        assert inner.place_order.call_count == 2

    def test_exponential_backoff(self):
        """验证指数退避延迟"""
        inner = MagicMock(spec=BaseBroker)
        inner.name = "mock"
        inner.place_order.side_effect = ConnectionError("fail")

        broker = RetryableBroker(inner, max_retries=3, retry_delay=1.0, backoff=2.0)

        with patch("dquant.broker.retry.time.sleep") as mock_sleep:
            broker.place_order(_make_order())

            # 3 retries: sleep after attempt 0 (delay=1.0) and attempt 1 (delay=2.0)
            # no sleep after attempt 2 (last attempt)
            calls = mock_sleep.call_args_list
            assert len(calls) == 2
            assert calls[0][0][0] == pytest.approx(1.0)  # 1.0 * 2^0
            assert calls[1][0][0] == pytest.approx(2.0)  # 1.0 * 2^1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
