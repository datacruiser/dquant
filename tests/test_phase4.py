"""
Phase 4 综合测试: Alpha101 + 组合优化 + 期货
"""

import numpy as np
import pandas as pd
import pytest


def _make_market_data(n_days=200, n_stocks=5, seed=42):
    """创建标准市场数据"""
    np.random.seed(seed)
    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
    symbols = [f"{i:06d}.SH" for i in range(1, n_stocks + 1)]
    rows = []
    for sym in symbols:
        price = 10.0
        for d in dates:
            ret = np.random.randn() * 0.02
            price *= 1 + ret
            rows.append(
                {
                    "date": d,
                    "symbol": sym,
                    "open": price * (1 + np.random.randn() * 0.005),
                    "high": price * (1 + abs(np.random.randn()) * 0.02),
                    "low": price * (1 - abs(np.random.randn()) * 0.02),
                    "close": price,
                    "volume": int(np.random.exponential(1e6)),
                }
            )
    return pd.DataFrame(rows).set_index("date")


# ============================================================
# Alpha101 测试
# ============================================================


class TestAlpha101:
    @pytest.fixture(scope="class")
    def data(self):
        return _make_market_data()

    def test_list_alphas(self):
        from dquant.ai.alpha101 import list_alphas

        alphas = list_alphas()
        assert len(alphas) == 25
        assert "alpha001" in alphas
        assert "alpha101" in alphas

    def test_get_alpha(self):
        from dquant.ai.alpha101 import get_alpha

        f = get_alpha("alpha001")
        assert f.name == "Alpha001"

    def test_unknown_alpha_raises(self):
        from dquant.ai.alpha101 import get_alpha

        with pytest.raises(ValueError, match="Unknown Alpha101"):
            get_alpha("alpha999")

    @pytest.mark.parametrize(
        "alpha_name",
        [
            "alpha001",
            "alpha002",
            "alpha003",
            "alpha005",
            "alpha006",
            "alpha012",
            "alpha018",
            "alpha023",
            "alpha041",
            "alpha043",
            "alpha053",
            "alpha101",
        ],
    )
    def test_alpha_produces_output(self, data, alpha_name):
        from dquant.ai.alpha101 import get_alpha

        f = get_alpha(alpha_name)
        result = f.predict(data)
        assert len(result) > 0
        assert "symbol" in result.columns
        assert "score" in result.columns
        # score 不应全为 NaN
        assert result["score"].notna().sum() > 0

    def test_alpha_fit_required(self):
        from dquant.ai.alpha101 import get_alpha

        f = get_alpha("alpha001")
        assert f._is_fitted is False
        f.fit(pd.DataFrame())
        assert f._is_fitted is True


# ============================================================
# 多策略组合测试
# ============================================================


class TestMultiStrategyPortfolio:
    def test_add_and_remove(self):
        from unittest.mock import MagicMock

        from dquant.portfolio_optimizer import MultiStrategyPortfolio

        msp = MultiStrategyPortfolio()
        s = MagicMock(spec=["generate_signals"])
        msp.add_strategy("test", s, weight=0.5)
        assert "test" in msp.strategy_names
        msp.remove_strategy("test")
        assert "test" not in msp.strategy_names

    def test_signal_merging(self):
        from dquant.portfolio_optimizer import MultiStrategyPortfolio
        from dquant.strategy.base import BaseStrategy, Signal, SignalType

        class BuyA(BaseStrategy):
            def generate_signals(self, data):
                return [Signal(symbol="A", signal_type=SignalType.BUY, strength=0.8)]

        class BuyAStrong(BaseStrategy):
            def generate_signals(self, data):
                return [Signal(symbol="A", signal_type=SignalType.BUY, strength=0.5)]

        msp = MultiStrategyPortfolio()
        msp.add_strategy("s1", BuyA(), weight=1.0)
        msp.add_strategy("s2", BuyAStrong(), weight=1.0)

        signals = msp.generate_signals(pd.DataFrame())
        assert len(signals) == 1
        # 合并强度应为 0.8 + 0.5 = 1.3
        assert abs(signals[0].strength - 1.3) < 0.01
        assert signals[0].metadata["n_sources"] == 2

    def test_signal_filter(self):
        from dquant.portfolio_optimizer import MultiStrategyPortfolio
        from dquant.strategy.base import BaseStrategy, Signal, SignalType

        class BothSignals(BaseStrategy):
            def generate_signals(self, data):
                return [
                    Signal(symbol="A", signal_type=SignalType.BUY, strength=0.5),
                    Signal(symbol="B", signal_type=SignalType.SELL, strength=0.5),
                ]

        msp = MultiStrategyPortfolio()
        msp.add_strategy("test", BothSignals(), signal_filter="buy")

        signals = msp.generate_signals(pd.DataFrame())
        assert len(signals) == 1
        assert signals[0].symbol == "A"

    def test_negative_weight_raises(self):
        from dquant.portfolio_optimizer import MultiStrategyPortfolio

        msp = MultiStrategyPortfolio()
        with pytest.raises(ValueError):
            msp.add_strategy("bad", None, weight=-0.5)


# ============================================================
# 组合优化器测试
# ============================================================


class TestPortfolioOptimizer:
    @pytest.fixture
    def returns(self):
        np.random.seed(42)
        dates = pd.date_range("2023-01-01", periods=252, freq="B")
        return pd.DataFrame(
            np.random.randn(252, 4) * 0.01,
            index=dates,
            columns=["A", "B", "C", "D"],
        )

    def test_equal_weight(self, returns):
        from dquant.portfolio_optimizer import PortfolioOptimizer

        opt = PortfolioOptimizer(returns)
        result = opt.optimize("equal_weight")
        assert abs(sum(result.weights.values()) - 1.0) < 0.01
        assert result.method == "equal_weight"

    def test_risk_parity(self, returns):
        from dquant.portfolio_optimizer import PortfolioOptimizer

        opt = PortfolioOptimizer(returns)
        result = opt.optimize("risk_parity")
        assert abs(sum(result.weights.values()) - 1.0) < 0.01
        assert result.expected_volatility > 0

    def test_min_variance(self, returns):
        from dquant.portfolio_optimizer import PortfolioOptimizer

        opt = PortfolioOptimizer(returns)
        result = opt.optimize("min_variance")
        assert abs(sum(result.weights.values()) - 1.0) < 0.01
        assert result.expected_volatility > 0

    def test_mean_variance(self, returns):
        from dquant.portfolio_optimizer import PortfolioOptimizer

        opt = PortfolioOptimizer(returns)
        result = opt.optimize("mean_variance")
        assert abs(sum(result.weights.values()) - 1.0) < 0.01

    def test_black_litterman(self, returns):
        from dquant.portfolio_optimizer import PortfolioOptimizer

        opt = PortfolioOptimizer(returns)
        result = opt.optimize("black_litterman", views={"A": 0.15, "B": 0.10})
        assert abs(sum(result.weights.values()) - 1.0) < 0.01

    def test_unknown_method_raises(self, returns):
        from dquant.portfolio_optimizer import PortfolioOptimizer

        opt = PortfolioOptimizer(returns)
        with pytest.raises(ValueError, match="Unknown optimization"):
            opt.optimize("invalid_method")

    def test_sharpe_reasonable(self, returns):
        from dquant.portfolio_optimizer import PortfolioOptimizer

        opt = PortfolioOptimizer(returns)
        for method in ["equal_weight", "risk_parity", "min_variance"]:
            result = opt.optimize(method)
            # Sharpe 应在合理范围
            assert abs(result.sharpe_ratio) < 10


# ============================================================
# 期货支持测试
# ============================================================


class TestFutures:
    def test_contract_creation(self):
        from dquant.futures import FuturesContract, FuturesType

        c = FuturesContract(
            symbol="IF2401",
            underlying="IF",
            futures_type=FuturesType.COMMODITY,
            multiplier=300,
            margin_rate=0.12,
        )
        assert c.is_index_futures is False
        c2 = FuturesContract(
            symbol="IF2401", underlying="IF", futures_type=FuturesType.INDEX, multiplier=300
        )
        assert c2.is_index_futures is True

    def test_margin_calculation(self):
        from dquant.futures import INDEX_FUTURES

        if_contract = INDEX_FUTURES["IF"]
        # IF 价格 4000, 乘数 300, 保证金比例 12%
        margin = if_contract.margin_required(4000.0)
        assert abs(margin - 4000 * 300 * 0.12) < 0.01  # 144000

    def test_notional_value(self):
        from dquant.futures import INDEX_FUTURES

        if_contract = INDEX_FUTURES["IF"]
        assert if_contract.notional_value(4000.0) == 1_200_000

    def test_tick_value(self):
        from dquant.futures import INDEX_FUTURES

        if_contract = INDEX_FUTURES["IF"]
        assert if_contract.tick_value() == 0.2 * 300  # 60 元

    def test_futures_account_open_close(self):
        from dquant.futures import FuturesAccount

        account = FuturesAccount(initial_capital=1_000_000)

        # 开多 IF 1手 @ 4000
        success = account.open_position("IF", "long", 1, 4000.0)
        assert success is True
        assert len(account.positions) == 1

        # 平仓 @ 4050
        pnl = account.close_position("IF", "long", 1, 4050.0)
        assert pnl is not None
        assert pnl > 0  # 赚了
        assert len(account.positions) == 0

    def test_short_position(self):
        from dquant.futures import FuturesAccount

        account = FuturesAccount(initial_capital=1_000_000)

        # 开空
        account.open_position("IF", "short", 1, 4000.0)
        assert len(account.positions) == 1

        # 价格下跌到 3900，空头盈利
        pnl = account.close_position("IF", "short", 1, 3900.0)
        assert pnl is not None
        assert pnl > 0  # 空头赚 100 * 300 = 30000

    def test_insufficient_margin(self):
        from dquant.futures import FuturesAccount

        account = FuturesAccount(initial_capital=100_000)

        # IF 保证金约 144,000 > 100,000
        success = account.open_position("IF", "long", 1, 4000.0)
        assert success is False

    def test_mark_to_market(self):
        from dquant.futures import FuturesAccount

        account = FuturesAccount(initial_capital=1_000_000)
        account.open_position("IF", "long", 1, 4000.0)

        account.mark_to_market({"IF": 4100.0})
        pos = list(account.positions.values())[0]
        assert pos.unrealized_pnl > 0  # 盈利 100*300=30000

    def test_margin_usage_ratio(self):
        from dquant.futures import FuturesAccount

        account = FuturesAccount(initial_capital=1_000_000)
        account.open_position("IF", "long", 1, 4000.0)
        assert account.margin_usage_ratio > 0
        assert account.total_equity > 0

    def test_index_futures_templates(self):
        from dquant.futures import INDEX_FUTURES

        assert "IF" in INDEX_FUTURES
        assert "IC" in INDEX_FUTURES
        assert "IM" in INDEX_FUTURES
        assert "IH" in INDEX_FUTURES

    def test_commodity_futures_templates(self):
        from dquant.futures import COMMODITY_FUTURES

        assert "RB" in COMMODITY_FUTURES
        assert "CU" in COMMODITY_FUTURES
        assert "AU" in COMMODITY_FUTURES
