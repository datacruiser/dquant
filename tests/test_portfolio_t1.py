import pytest
from datetime import datetime
import pandas as pd
from dquant.backtest.portfolio import Portfolio, Position


class TestTPlus1Execution:
    """测试回测引擎的 T+1 和 T+0(对于新买入的不可用部分) 相关限制"""

    def test_same_day_buy_sell(self):
        """测试同日买入不可卖出"""
        pf = Portfolio(initial_cash=100000)

        # Day 1: 买入 1000 股
        pf.buy("TEST.SZ", 1000, 10.0, commission=0)
        assert pf.positions["TEST.SZ"].shares == 1000
        assert pf.positions["TEST.SZ"].available_shares == 0  # T+1 规则：当天不可用

        # Day 1 尝试卖出：不应成交
        pf.sell("TEST.SZ", 500, 10.0, commission=0)
        assert pf.positions["TEST.SZ"].shares == 1000

        # Day 2: 价格更新触发冻结释放
        pf.update_prices({"TEST.SZ": 10.0}, datetime(2023, 1, 2))
        assert pf.positions["TEST.SZ"].available_shares == 1000

        # Day 2 尝试卖出：应成交
        pf.sell("TEST.SZ", 500, 10.0, commission=0)
        assert pf.positions["TEST.SZ"].shares == 500

    def test_multi_day_accumulated_locked_shares(self):
        """测试多日累积 locked_shares 和解冻过程"""
        pf = Portfolio(initial_cash=100000)

        # Day 1: 买入 500 股
        pf.buy("TEST.SZ", 500, 10.0)
        assert pf.positions["TEST.SZ"].shares == 500
        assert pf.positions["TEST.SZ"].locked_shares == 500
        assert pf.positions["TEST.SZ"].available_shares == 0

        # Day 2: 更新时间解冻，然后再买入 300 股
        pf.update_prices({"TEST.SZ": 10.0}, datetime(2023, 1, 2))
        assert pf.positions["TEST.SZ"].locked_shares == 0
        assert pf.positions["TEST.SZ"].available_shares == 500

        pf.buy("TEST.SZ", 300, 10.0)
        assert pf.positions["TEST.SZ"].shares == 800
        assert pf.positions["TEST.SZ"].locked_shares == 300
        assert pf.positions["TEST.SZ"].available_shares == 500

        # Day 2: 尝试卖出 600 股，但由于只有 500 股可用，只能卖 500 股
        pf.sell("TEST.SZ", 600, 10.0)
        assert pf.positions["TEST.SZ"].shares == 300
        assert pf.positions["TEST.SZ"].locked_shares == 300
        assert pf.positions["TEST.SZ"].available_shares == 0

        # Day 3: 更新时间全部解冻
        pf.update_prices({"TEST.SZ": 10.0}, datetime(2023, 1, 3))
        assert pf.positions["TEST.SZ"].shares == 300
        assert pf.positions["TEST.SZ"].locked_shares == 0
        assert pf.positions["TEST.SZ"].available_shares == 300
