"""
实盘极端边界情况测试
"""

import threading
import time
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from dquant.broker.base import Order, OrderResult
from dquant.broker.simulator import Simulator
from dquant.core import Engine
from dquant.strategy.base import BaseStrategy


class DummyStrategy(BaseStrategy):
    def generate_signals(self, data):
        return []


def test_live_network_disconnect():
    """测试实盘中断网重连"""
    broker = Simulator(initial_cash=100000)
    from dquant.data.data_manager import DataManager
    data = DataManager()
    engine = Engine(strategy=DummyStrategy(), data=data, broker=broker)

    # 模拟断网
    broker.is_connected = MagicMock(side_effect=[False, False, True, True])
    broker.connect = MagicMock(return_value=True)

    # 用一个极短的循环测试
    engine._running = True

    def stop_engine():
        time.sleep(0.1)
        engine._running = False

    t = threading.Thread(target=stop_engine)
    t.start()

    with patch("dquant.calendar.is_trading_day", return_value=True), patch(
        "dquant.broker.safety.TradingTimeChecker.is_trading_time", return_value=(True, "交易中")
    ):
        # 即使断网，引擎也应该尝试重连而不是直接崩溃
        engine.live(interval=0.01, dry_run=True, max_consecutive_errors=2)

    t.join()
    # 验证是否调用了重连
    assert broker.connect.call_count >= 1


def test_live_order_rejected():
    """测试实盘订单被拒的情况"""
    broker = Simulator(initial_cash=100000)
    # 修改 broker.place_order 让它直接拒绝
    def mock_place_order(order):
        return OrderResult(
            order_id="REJECTED_1",
            symbol=order.symbol,
            side=order.side,
            filled_quantity=0,
            filled_price=0.0,
            commission=0.0,
            status="REJECTED",
            timestamp=datetime.now()
        )
    broker.place_order = MagicMock(side_effect=mock_place_order)
    
    from dquant.data.data_manager import DataManager
    data = DataManager()
    engine = Engine(strategy=DummyStrategy(), data=data, broker=broker)
    
    from dquant.strategy.base import Signal, SignalType
    sig = Signal(symbol="000001.SZ", signal_type=SignalType.BUY, strength=1.0)
    
    # 手动触发 _execute_buys
    from dquant.broker.order_tracker import OrderTracker
    from dquant.broker.trade_journal import TradeJournal
    
    tracker = OrderTracker()
    journal = TradeJournal("test")
    
    engine._execute_buys(
        buy_signals=[sig],
        available_cash=100000,
        dry_run=False,
        journal=journal,
        strategy_name="test",
        tracker=tracker
    )
    
    # 被拒订单不应进入 pending tracker
    assert not tracker.has_pending()
