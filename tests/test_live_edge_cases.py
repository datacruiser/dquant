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

    # 模拟断网，最后一次抛出 Exception 让循环自己退出来而不是用 sleep 杀它
    broker.is_connected = MagicMock(side_effect=[False, False, True, Exception("stop")])
    broker.connect = MagicMock(return_value=True)

    with patch("dquant.calendar.is_trading_day", return_value=True), \
         patch("dquant.broker.safety.TradingTimeChecker.is_trading_time", return_value=(True, "交易中")), \
         patch("dquant.core.Engine._fetch_realtime_data", return_value=MagicMock()):
        
        # 即使断网，引擎也应该尝试重连而不是直接崩溃
        engine.live(interval=0.01, dry_run=True, max_consecutive_errors=2)

    # 验证是否调用了重连
    assert broker.connect.call_count >= 1


def test_live_order_rejected():
    """测试实盘订单被拒的情况（覆盖买单和卖单）"""
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
    buy_sig = Signal(symbol="000001.SZ", signal_type=SignalType.BUY, strength=1.0)
    sell_sig = Signal(symbol="000002.SZ", signal_type=SignalType.SELL, strength=1.0)
    
    # 手动触发 _execute_buys 和 _execute_sell
    from dquant.broker.order_tracker import OrderTracker
    from dquant.broker.trade_journal import TradeJournal
    
    tracker = OrderTracker()
    journal = TradeJournal("test")
    
    # 模拟买单被拒
    engine._execute_buys(
        buy_signals=[buy_sig],
        available_cash=100000,
        dry_run=False,
        journal=journal,
        strategy_name="test",
        tracker=tracker
    )
    assert not tracker.has_pending()

    # 模拟卖单被拒
    # 为引擎伪造持仓
    fake_positions = {"000002.SZ": {"quantity": 1000, "available": 1000}}
    res = engine._execute_sell(
        sig=sell_sig,
        positions=fake_positions,
        dry_run=False,
        journal=journal,
        strategy_name="test"
    )
    if res:
        order, result = res
        if result.status in ("PENDING", "PARTIAL_FILLED"):
            tracker.add(order, result)
            
    # 被拒订单不应进入 pending tracker
    assert not tracker.has_pending()
