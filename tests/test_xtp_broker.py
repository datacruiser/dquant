from types import SimpleNamespace
from unittest.mock import MagicMock

from dquant.broker.base import Order
from dquant.broker.xtp_broker import XTPBroker


class _FakeXTPAPI:
    def __init__(self):
        self.callbacks = {}

    def register_callback(self, **kwargs):
        self.callbacks = kwargs

    def place_order(self, **kwargs):
        return SimpleNamespace(order_id=123456)


def test_xtp_broker_order_callbacks_update_cached_status_flow():
    broker = XTPBroker(enable_time_check=False)
    broker._api = _FakeXTPAPI()
    broker._connected = True
    broker.get_account = MagicMock(return_value={"cash": 100_000})
    broker.get_positions = MagicMock(return_value={})

    broker._setup_callbacks()

    order = Order(
        symbol="000001.SZ",
        side="BUY",
        quantity=1_000,
        price=10.0,
        order_type="LIMIT",
    )
    result = broker.place_order(order)

    assert result.status == "PENDING"
    assert result.order_id == "123456"
    assert broker._orders["123456"]["status"] == "PENDING"
    assert broker._orders["123456"]["filled"] == 0

    broker._api.callbacks["on_order"](
        SimpleNamespace(order_xt_id=123456, order_status=1),
        None,
    )
    assert broker._orders["123456"]["status"] == "PARTIAL_FILLED"

    broker._api.callbacks["on_trade"](
        SimpleNamespace(
            order_xt_id=123456,
            stock_code="000001.SZ",
            traded_quantity=400,
            traded_price=10.05,
        ),
        None,
    )
    assert broker._orders["123456"]["filled"] == 400
    assert broker._orders["123456"]["filled_price"] == 10.05

    broker._api.callbacks["on_order"](
        SimpleNamespace(order_xt_id=123456, order_status=2),
        None,
    )
    assert broker._orders["123456"]["status"] == "FILLED"
