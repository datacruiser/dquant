from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from dquant.broker.base import Order
from dquant.broker.xtp_broker import XTPBroker


class _FakeXTPAPI:
    """Mock XTP C++ SDK API Behavior"""

    def __init__(self):
        self.callbacks = {}
        self.query_orders_result = []
        self.query_orders_error = None
        self.cancelled_order_ids = []

    def register_callback(self, **kwargs):
        self.callbacks = kwargs

    def place_order(self, **kwargs):
        return SimpleNamespace(order_id=123456)

    def query_orders(self):
        if self.query_orders_error is not None:
            raise self.query_orders_error
        return self.query_orders_result

    def cancel_order(self, order_id):
        self.cancelled_order_ids.append(order_id)


@pytest.fixture
def broker():
    """Provides a connected XTPBroker with mocked API."""
    b = XTPBroker(enable_time_check=False)
    b._api = _FakeXTPAPI()
    b._connected = True
    b.get_account = MagicMock(return_value={"cash": 100_000})
    b.get_positions = MagicMock(return_value={})
    b._setup_callbacks()
    return b


class TestXTPBrokerContract:
    """
    Contract Tests for XTPBroker.
    Verifies that XTPBroker correctly translates XTP API callbacks and state
    into standard dquant Order semantics.
    """

    def test_order_lifecycle_callbacks(self, broker):
        """Contract: Order progresses through PENDING -> PARTIAL_FILLED -> FILLED via callbacks."""
        order = Order(
            symbol="000001.SZ", side="BUY", quantity=1_000, price=10.0, order_type="LIMIT"
        )
        result = broker.place_order(order)

        assert result.status == "PENDING"
        assert result.order_id == "123456"

        # 1. API Callback: Order accepted
        broker._api.callbacks["on_order"](SimpleNamespace(order_xt_id=123456, order_status=1), None)
        status = broker.get_order_status(result.order_id)
        assert status.status == "PARTIAL_FILLED"

        # 2. API Callback: Partial fill
        broker._api.callbacks["on_trade"](
            SimpleNamespace(order_xt_id=123456, stock_code="000001.SZ", quantity=400, price=10.05),
            None,
        )
        status = broker.get_order_status(result.order_id)
        assert status.filled_quantity == 400
        assert status.filled_price == 10.05

        # 3. API Callback: Fully filled
        broker._api.callbacks["on_order"](SimpleNamespace(order_xt_id=123456, order_status=2), None)
        status = broker.get_order_status(result.order_id)
        assert status.status == "FILLED"

    def test_api_query_fallback(self, broker):
        """Contract: get_order_status falls back to local cache if API query fails."""
        result = broker.place_order(
            Order(symbol="000001.SZ", side="BUY", quantity=1_000, price=10.0, order_type="LIMIT")
        )

        broker._api.callbacks["on_order"](
            SimpleNamespace(order_xt_id=123456, order_status=2, qty_traded=1_000), None
        )
        broker._api.query_orders_error = RuntimeError("query failed")

        status = broker.get_order_status(result.order_id)
        assert status.status == "FILLED"
        assert status.filled_quantity == 1_000

    def test_api_query_refresh(self, broker):
        """Contract: get_order_status refreshes from API if query succeeds."""
        result = broker.place_order(
            Order(symbol="000001.SZ", side="BUY", quantity=1_000, price=10.0, order_type="LIMIT")
        )

        broker._api.query_orders_result = [
            SimpleNamespace(
                order_id=int(result.order_id),
                ticker="000001.SZ",
                side=1,
                quantity=1_000,
                price=10.0,
                traded_volume=600,
                status=1,
            )
        ]

        status = broker.get_order_status(result.order_id)
        assert status.filled_quantity == 600
        assert status.status == "PARTIAL_FILLED"

    def test_unknown_side_retention(self, broker):
        """Contract: Unknown side from API does not overwrite cached side."""
        result = broker.place_order(
            Order(symbol="000001.SZ", side="BUY", quantity=1_000, price=10.0, order_type="LIMIT")
        )

        broker._api.query_orders_result = [
            SimpleNamespace(
                order_id=int(result.order_id),
                ticker="000001.SZ",
                side=99,  # Unknown side
                quantity=1_000,
                price=10.0,
                traded_volume=100,
                status=1,
            )
        ]

        status = broker.get_order_status(result.order_id)
        assert status.side == "BUY"

    def test_cancel_order_sync(self, broker):
        """Contract: cancel_order immediately sets status to CANCELLED in cache."""
        result = broker.place_order(
            Order(symbol="000001.SZ", side="BUY", quantity=1_000, price=10.0, order_type="LIMIT")
        )
        broker._api.query_orders_error = RuntimeError("query failed")

        assert broker.cancel_order(result.order_id) is True
        assert broker._api.cancelled_order_ids == [int(result.order_id)]

        status = broker.get_order_status(result.order_id)
        assert status.status == "CANCELLED"
        assert status.filled_quantity == 0

    def test_cancel_order_protects_filled(self, broker):
        """Contract: cancel_order does not overwrite FILLED status."""
        result = broker.place_order(
            Order(symbol="000001.SZ", side="BUY", quantity=1_000, price=10.0, order_type="LIMIT")
        )
        broker._orders[result.order_id]["status"] = "FILLED"
        broker._orders[result.order_id]["filled"] = 1_000
        broker._api.query_orders_error = RuntimeError("query failed")

        assert broker.cancel_order(result.order_id) is True

        status = broker.get_order_status(result.order_id)
        assert status.status == "FILLED"
        assert status.filled_quantity == 1_000

    def test_vwap_calculation(self, broker):
        """Contract: Multiple trade events correctly calculate VWAP (Volume-Weighted Average Price)."""
        result = broker.place_order(
            Order(symbol="000001.SZ", side="BUY", quantity=1_000, price=10.0, order_type="LIMIT")
        )

        broker._api.callbacks["on_trade"](
            SimpleNamespace(
                order_xt_id=int(result.order_id), stock_code="000001.SZ", quantity=400, price=10.0
            ),
            None,
        )
        status = broker.get_order_status(result.order_id)
        assert status.filled_price == 10.0

        broker._api.callbacks["on_trade"](
            SimpleNamespace(
                order_xt_id=int(result.order_id), stock_code="000001.SZ", quantity=600, price=10.5
            ),
            None,
        )
        status = broker.get_order_status(result.order_id)
        assert status.filled_price == 10.3
        assert status.filled_quantity == 1000
        assert status.status == "FILLED"

    def test_mixed_events_sync(self, broker):
        """Contract: Mixed order and trade events safely synchronize quantity without double counting."""
        result = broker.place_order(
            Order(symbol="000001.SZ", side="BUY", quantity=1_000, price=10.0, order_type="LIMIT")
        )
        broker._api.query_orders_error = RuntimeError("query failed")

        # Order event provides cumulative volume
        broker._api.callbacks["on_order"](
            SimpleNamespace(order_xt_id=int(result.order_id), order_status=1, qty_traded=400),
            None,
        )
        status = broker.get_order_status(result.order_id)
        assert status.filled_quantity == 400

        # Trade event provides incremental volume
        broker._api.callbacks["on_trade"](
            SimpleNamespace(
                order_xt_id=int(result.order_id), stock_code="000001.SZ", quantity=200, price=10.5
            ),
            None,
        )
        status = broker.get_order_status(result.order_id)
        assert status.filled_quantity == 600


def test_qmt_whitelist():
    """Verify QMT _call_qmt whitelist contains critical functions."""
    import inspect

    from dquant.broker.qmt_broker import QMTBroker

    # Extract the embedded script from _call_qmt
    source = inspect.getsource(QMTBroker._call_qmt)

    # Verify critical functions are whitelisted
    for func in [
        "stock_order",
        "stock_cancel",
        "query_account",
        "query_position",
        "query_asset",
        "get_market_data",
    ]:
        assert func in source, f"Missing {func} in QMT whitelist"
