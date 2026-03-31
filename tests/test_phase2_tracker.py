"""
Phase 2 Step 3: OrderTracker 测试
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from dquant.broker.base import Order, OrderResult
from dquant.broker.order_tracker import OrderTracker, TrackedOrder


def _make_order(**overrides):
    defaults = dict(
        symbol='000001.SZ', side='BUY', quantity=1000, price=10.0,
        order_type='LIMIT', order_id='ORD-001',
    )
    defaults.update(overrides)
    return Order(**defaults)


def _make_result(**overrides):
    defaults = dict(
        order_id='ORD-001', symbol='000001.SZ', side='BUY',
        filled_quantity=0, filled_price=0, commission=0,
        timestamp=datetime.now(), status='PENDING',
    )
    defaults.update(overrides)
    return OrderResult(**defaults)


class TestOrderTracker:

    def test_add_and_has_pending(self):
        tracker = OrderTracker()
        assert not tracker.has_pending()

        order = _make_order()
        result = _make_result(status='PENDING')
        tracker.add(order, result)
        assert tracker.has_pending()

    def test_get_pending(self):
        tracker = OrderTracker()
        order = _make_order()
        result = _make_result(status='PENDING')
        tracker.add(order, result)

        pending = tracker.get_pending()
        assert 'ORD-001' in pending
        assert pending['ORD-001'].order.symbol == '000001.SZ'

    def test_remove(self):
        tracker = OrderTracker()
        order = _make_order()
        result = _make_result(status='PENDING')
        tracker.add(order, result)
        assert tracker.has_pending()

        tracker.remove('ORD-001')
        assert not tracker.has_pending()

    def test_remove_nonexistent(self):
        """移除不存在的 order_id 不报错"""
        tracker = OrderTracker()
        tracker.remove('NONEXISTENT')  # should not raise

    def test_not_track_filled(self):
        """FILLED 订单不追踪"""
        tracker = OrderTracker()
        order = _make_order()
        result = _make_result(status='FILLED', filled_quantity=1000)
        tracker.add(order, result)
        assert not tracker.has_pending()

    def test_not_track_rejected(self):
        """REJECTED 订单不追踪"""
        tracker = OrderTracker()
        order = _make_order()
        result = _make_result(status='REJECTED')
        tracker.add(order, result)
        assert not tracker.has_pending()

    def test_track_partial_filled(self):
        """PARTIAL_FILLED 订单追踪"""
        tracker = OrderTracker()
        order = _make_order(quantity=1000)
        result = _make_result(status='PARTIAL_FILLED', filled_quantity=300)
        tracker.add(order, result)
        assert tracker.has_pending()

        tracked = tracker.get_pending()['ORD-001']
        assert tracked.remaining_quantity == 700  # 1000 - 300

    def test_update(self):
        """更新订单状态"""
        tracker = OrderTracker()
        order = _make_order(quantity=1000)
        result = _make_result(status='PENDING')
        tracker.add(order, result)

        # Simulate fill: update via get_order_status pattern
        updated_order = _make_order(quantity=1000, status='PARTIAL_FILLED')
        updated_order.filled_quantity = 500
        tracked = tracker.update('ORD-001', updated_order)

        assert tracked.remaining_quantity == 500
        assert tracked.check_count == 1

    def test_get_timed_out(self):
        """检测超时订单"""
        tracker = OrderTracker(timeout_seconds=10)
        order = _make_order()
        result = _make_result(status='PENDING')
        tracker.add(order, result)

        # 修改 first_seen 模拟超时
        tracked = tracker.get_pending()['ORD-001']
        tracked.first_seen = datetime.now() - timedelta(seconds=30)

        timed_out = tracker.get_timed_out()
        assert len(timed_out) == 1
        assert timed_out[0].order.symbol == '000001.SZ'

    def test_not_timed_out_recent(self):
        """近期订单不超时"""
        tracker = OrderTracker(timeout_seconds=60)
        order = _make_order()
        result = _make_result(status='PENDING')
        tracker.add(order, result)

        timed_out = tracker.get_timed_out()
        assert len(timed_out) == 0

    def test_update_filled_removes(self):
        """更新为 FILLED 后自动移除"""
        tracker = OrderTracker()
        order = _make_order(quantity=1000)
        result = _make_result(status='PENDING')
        tracker.add(order, result)

        updated_order = _make_order(quantity=1000, status='FILLED')
        updated_order.filled_quantity = 1000
        tracker.update('ORD-001', updated_order)

        assert not tracker.has_pending()

    def test_update_cancelled_removes(self):
        """更新为 CANCELLED 后自动移除"""
        tracker = OrderTracker()
        order = _make_order()
        result = _make_result(status='PENDING')
        tracker.add(order, result)

        updated_order = _make_order(status='CANCELLED')
        tracker.update('ORD-001', updated_order)

        assert not tracker.has_pending()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
