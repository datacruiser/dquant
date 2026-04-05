"""
Phase 3 Step 5: 通用持仓价格更新测试
"""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from dquant.broker.base import Order
from dquant.broker.simulator import Simulator
from dquant.core import Engine


class TestUpdatePositionPricesGeneric:

    def test_simulator_still_works(self):
        """Simulator broker 仍使用 update_prices"""
        sim = Simulator()
        sim.connect()
        order = Order(
            symbol="000001.SZ",
            side="BUY",
            quantity=100,
            price=10.0,
            order_type="MARKET",
        )
        sim.place_order(order)

        data_mock = MagicMock()
        strategy_mock = MagicMock()
        engine = Engine(data_mock, strategy_mock, broker=sim)

        df = pd.DataFrame(
            {
                "symbol": ["000001.SZ"],
                "price": [12.0],
            }
        )

        engine._update_position_prices(df)
        assert sim.positions["000001.SZ"]["price"] == 12.0

    def test_non_simulator_calls_get_market_data(self):
        """非 Simulator broker 调用 get_market_data 估值"""
        broker = MagicMock()
        broker.get_positions.return_value = {
            "AAA": {"quantity": 100, "avg_cost": 10.0},
            "BBB": {"quantity": 200, "avg_cost": 20.0},
        }
        broker.get_market_data.side_effect = [
            {"symbol": "AAA", "price": 11.0},
            {"symbol": "BBB", "price": 22.0},
        ]

        data_mock = MagicMock()
        strategy_mock = MagicMock()
        engine = Engine(data_mock, strategy_mock, broker=broker)

        df = pd.DataFrame({"symbol": ["AAA", "BBB"], "price": [11.0, 22.0]})
        engine._update_position_prices(df)

        # 验证 get_market_data 被调用了
        assert broker.get_market_data.call_count == 2

    def test_non_simulator_no_positions(self):
        """无持仓时不调用 get_market_data"""
        broker = MagicMock()
        broker.get_positions.return_value = {}

        data_mock = MagicMock()
        strategy_mock = MagicMock()
        engine = Engine(data_mock, strategy_mock, broker=broker)

        df = pd.DataFrame({"symbol": ["AAA"], "price": [11.0]})
        engine._update_position_prices(df)

        broker.get_market_data.assert_not_called()

    def test_non_simulator_market_data_failure(self):
        """get_market_data 异常不中断循环"""
        broker = MagicMock()
        broker.get_positions.return_value = {
            "AAA": {"quantity": 100},
            "BBB": {"quantity": 200},
        }
        # AAA 抛异常，BBB 正常
        broker.get_market_data.side_effect = [
            Exception("timeout"),
            {"symbol": "BBB", "price": 22.0},
        ]

        data_mock = MagicMock()
        strategy_mock = MagicMock()
        engine = Engine(data_mock, strategy_mock, broker=broker)

        df = pd.DataFrame({"symbol": ["AAA", "BBB"], "price": [11.0, 22.0]})
        # 不应抛异常
        engine._update_position_prices(df)

        # BBB 仍然被估值
        assert broker.get_market_data.call_count == 2

    def test_simulator_missing_columns(self):
        """Simulator 数据缺少必要列时不报错"""
        sim = Simulator()
        sim.connect()

        data_mock = MagicMock()
        strategy_mock = MagicMock()
        engine = Engine(data_mock, strategy_mock, broker=sim)

        # 缺少 price 列
        df = pd.DataFrame({"symbol": ["TEST"]})
        engine._update_position_prices(df)  # 不应报错


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
