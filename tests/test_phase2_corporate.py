"""
Phase 2 Step 7: Corporate Action 测试
"""

import pytest
from datetime import datetime

from dquant.backtest.portfolio import Portfolio, CorporateAction
from dquant.constants import DEFAULT_INITIAL_CASH


class TestCorporateAction:

    def test_create_dividend(self):
        action = CorporateAction(
            symbol="000001.SZ",
            action_type="dividend",
            ex_date="2024-06-15",
            amount=0.5,
        )
        assert action.symbol == "000001.SZ"
        assert action.action_type == "dividend"
        assert action.amount == 0.5
        assert action.metadata == {}

    def test_create_split(self):
        action = CorporateAction(
            symbol="000001.SZ",
            action_type="split",
            ex_date="2024-06-15",
            amount=2.0,  # 1拆2
        )
        assert action.action_type == "split"
        assert action.amount == 2.0

    def test_create_bonus_shares(self):
        action = CorporateAction(
            symbol="000001.SZ",
            action_type="bonus_shares",
            ex_date="2024-06-15",
            ratio=0.1,  # 每10股送1股
        )
        assert action.ratio == 0.1

    def test_apply_corporate_action_stub(self):
        """apply_corporate_action 存根不改变 portfolio"""
        pf = Portfolio(initial_cash=DEFAULT_INITIAL_CASH)
        pf.buy("000001.SZ", 1000, 10.0, 0.0003)

        initial_cash = pf.cash
        initial_shares = pf.positions["000001.SZ"].shares

        action = CorporateAction(
            symbol="000001.SZ",
            action_type="dividend",
            ex_date="2024-06-15",
            amount=0.5,
        )
        pf.apply_corporate_action(action)

        # 存根模式：不做任何实际修改
        assert pf.positions["000001.SZ"].shares == initial_shares
        assert pf.cash == initial_cash

    def test_metadata_default_empty(self):
        action = CorporateAction(
            symbol="TEST",
            action_type="dividend",
            ex_date="2024-01-01",
            amount=1.0,
        )
        assert action.metadata == {}

    def test_metadata_custom(self):
        action = CorporateAction(
            symbol="TEST",
            action_type="dividend",
            ex_date="2024-01-01",
            amount=1.0,
            metadata={"note": "test"},
        )
        assert action.metadata["note"] == "test"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
