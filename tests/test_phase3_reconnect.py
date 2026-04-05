"""
Phase 3 Step 4: Broker 自动重连测试
"""

from unittest.mock import MagicMock, patch

import pytest

from dquant.broker.base import BaseBroker
from dquant.broker.simulator import Simulator


class TestBaseBrokerConnected:

    def test_default_not_connected(self):
        """新创建的 broker 默认 _connected=False"""
        sim = Simulator()
        # Simulator.connect() 才设置 _connected=True
        broker = Simulator.__new__(Simulator)
        broker._connected = False
        assert not broker.is_connected()

    def test_connected_after_connect(self):
        """connect() 后 is_connected() 返回 True"""
        sim = Simulator()
        sim.connect()
        assert sim.is_connected()

    def test_disconnected_after_disconnect(self):
        """disconnect() 后 is_connected() 返回 False"""
        sim = Simulator()
        sim.connect()
        assert sim.is_connected()
        sim.disconnect()
        assert not sim.is_connected()


class TestTryReconnect:

    def test_reconnect_success_on_first_try(self):
        """第一次就重连成功"""
        from dquant.core import Engine

        data_mock = MagicMock()
        strategy_mock = MagicMock()
        broker = Simulator()
        broker.disconnect()  # 断开连接

        engine = Engine(data_mock, strategy_mock, broker=broker)
        assert not broker.is_connected()

        result = engine._try_reconnect()
        assert result is True
        assert broker.is_connected()

    @patch("dquant.core.time.sleep")
    def test_reconnect_after_failures(self, mock_sleep):
        """前几次失败，最后一次成功"""
        from dquant.core import Engine

        data_mock = MagicMock()
        strategy_mock = MagicMock()

        # 模拟 broker：前两次连接失败，第三次成功
        broker = MagicMock()
        broker.is_connected.return_value = False
        broker.connect.side_effect = [False, False, True]

        engine = Engine(data_mock, strategy_mock, broker=broker)
        result = engine._try_reconnect()

        assert result is True
        assert broker.connect.call_count == 3

    @patch("dquant.core.time.sleep")
    def test_reconnect_all_fail(self, mock_sleep):
        """所有重连尝试都失败"""
        from dquant.core import Engine

        data_mock = MagicMock()
        strategy_mock = MagicMock()

        broker = MagicMock()
        broker.is_connected.return_value = False
        broker.connect.return_value = False

        engine = Engine(data_mock, strategy_mock, broker=broker)
        result = engine._try_reconnect()

        assert result is False
        assert broker.connect.call_count == 5  # BROKER_MAX_RECONNECT

    @patch("dquant.core.time.sleep")
    def test_reconnect_exception_handled(self, mock_sleep):
        """重连过程中异常被捕获"""
        from dquant.core import Engine

        data_mock = MagicMock()
        strategy_mock = MagicMock()

        broker = MagicMock()
        broker.is_connected.return_value = False
        broker.connect.side_effect = ConnectionError("网络错误")

        engine = Engine(data_mock, strategy_mock, broker=broker)
        result = engine._try_reconnect()

        assert result is False
        assert broker.connect.call_count == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
