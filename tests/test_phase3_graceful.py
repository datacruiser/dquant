"""
Phase 3 Step 3: 优雅关机 + _poll_pending_orders 测试
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from dquant.broker.base import Order, OrderResult
from dquant.broker.order_tracker import OrderTracker, TrackedOrder


class TestOrderTrackerCancelAll:

    def test_cancel_all_pending(self):
        """cancel_all 取消所有 pending 订单"""
        tracker = OrderTracker(timeout_seconds=30)
        broker = MagicMock()
        broker.cancel_order.return_value = True

        # 添加两个 pending 订单
        order1 = Order(symbol='A', side='BUY', quantity=100, order_id='id1')
        result1 = OrderResult(
            order_id='id1', symbol='A', side='BUY',
            filled_quantity=0, filled_price=0, commission=0,
            timestamp=datetime.now(), status='PENDING',
        )
        order2 = Order(symbol='B', side='BUY', quantity=200, order_id='id2')
        result2 = OrderResult(
            order_id='id2', symbol='B', side='BUY',
            filled_quantity=0, filled_price=0, commission=0,
            timestamp=datetime.now(), status='PENDING',
        )

        tracker.add(order1, result1)
        tracker.add(order2, result2)

        assert tracker.has_pending()
        cancelled = tracker.cancel_all(broker)

        assert len(cancelled) == 2
        assert 'id1' in cancelled
        assert 'id2' in cancelled
        assert not tracker.has_pending()
        assert broker.cancel_order.call_count == 2

    def test_cancel_all_handles_failure(self):
        """cancel_all 处理取消失败"""
        tracker = OrderTracker(timeout_seconds=30)
        broker = MagicMock()
        broker.cancel_order.return_value = False

        order = Order(symbol='A', side='BUY', quantity=100, order_id='id1')
        result = OrderResult(
            order_id='id1', symbol='A', side='BUY',
            filled_quantity=0, filled_price=0, commission=0,
            timestamp=datetime.now(), status='PENDING',
        )
        tracker.add(order, result)

        cancelled = tracker.cancel_all(broker)
        assert len(cancelled) == 0
        assert not tracker.has_pending()  # 仍然清空列表

    def test_cancel_all_handles_exception(self):
        """cancel_all 处理异常"""
        tracker = OrderTracker(timeout_seconds=30)
        broker = MagicMock()
        broker.cancel_order.side_effect = ConnectionError("连接断开")

        order = Order(symbol='A', side='BUY', quantity=100, order_id='id1')
        result = OrderResult(
            order_id='id1', symbol='A', side='BUY',
            filled_quantity=0, filled_price=0, commission=0,
            timestamp=datetime.now(), status='PENDING',
        )
        tracker.add(order, result)

        cancelled = tracker.cancel_all(broker)
        assert len(cancelled) == 0
        assert not tracker.has_pending()

    def test_cancel_all_empty(self):
        """无 pending 订单时 cancel_all 返回空列表"""
        tracker = OrderTracker(timeout_seconds=30)
        broker = MagicMock()
        cancelled = tracker.cancel_all(broker)
        assert cancelled == []
        broker.cancel_order.assert_not_called()


class TestPollPendingOrders:

    def test_poll_pending_orders_timeout(self):
        """_poll_pending_orders 处理超时订单"""
        from dquant.core import Engine
        from dquant.broker.simulator import Simulator

        # 创建 tracker，设一个已经超时的订单
        tracker = OrderTracker(timeout_seconds=0)  # 立即超时
        broker = Simulator()
        journal = MagicMock()

        order = Order(symbol='A', side='BUY', quantity=100, order_id='test_id')
        order.status = 'PENDING'
        result = OrderResult(
            order_id='test_id', symbol='A', side='BUY',
            filled_quantity=0, filled_price=0, commission=0,
            timestamp=datetime.now() - timedelta(seconds=10),
            status='PENDING',
        )
        tracker.add(order, result)

        # 创建一个 Engine 实例用于测试 _poll_pending_orders
        data_mock = MagicMock()
        strategy_mock = MagicMock()
        engine = Engine(data_mock, strategy_mock, broker=broker)

        engine._poll_pending_orders(tracker, journal, "test_strategy")

        # 超时订单应该被取消并移除
        assert not tracker.has_pending()

    def test_poll_pending_orders_no_pending(self):
        """无 pending 订单时不报错"""
        from dquant.core import Engine
        from dquant.broker.simulator import Simulator

        tracker = OrderTracker(timeout_seconds=30)
        broker = Simulator()
        journal = MagicMock()

        data_mock = MagicMock()
        strategy_mock = MagicMock()
        engine = Engine(data_mock, strategy_mock, broker=broker)

        # 应该不抛异常
        engine._poll_pending_orders(tracker, journal, "test_strategy")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
