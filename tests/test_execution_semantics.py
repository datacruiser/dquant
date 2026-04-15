from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd

from dquant.backtest.engine import BacktestEngine
from dquant.broker.base import OrderResult
from dquant.broker.order_tracker import OrderTracker
from dquant.broker.simulator import Simulator
from dquant.broker.trade_journal import TradeJournal
from dquant.core import Engine
from dquant.data.data_manager import DataManager
from dquant.strategy.base import BaseStrategy, Signal, SignalType


def _make_multisymbol_data(dates, symbols, price_map):
    rows = []
    index = []
    for dt in dates:
        for symbol in symbols:
            price = price_map[symbol]
            rows.append(
                {
                    "symbol": symbol,
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": 1_000_000,
                }
            )
            index.append(dt)
    return pd.DataFrame(rows, index=pd.DatetimeIndex(index))


class _SingleBuyStrategy(BaseStrategy):
    def __init__(self, symbol, signal_date):
        super().__init__("single-buy")
        self.symbol = symbol
        self.signal_date = pd.Timestamp(signal_date)

    def generate_signals(self, data):
        return [
            Signal(
                symbol=self.symbol,
                signal_type=SignalType.BUY,
                strength=1.0,
                timestamp=self.signal_date,
            )
        ]


class _WeightedBuyStrategy(BaseStrategy):
    def __init__(self, signal_date):
        super().__init__("weighted-buy")
        self.signal_date = pd.Timestamp(signal_date)

    def generate_signals(self, data):
        return [
            Signal(
                symbol="AAA.SZ",
                signal_type=SignalType.BUY,
                strength=1.0,
                timestamp=self.signal_date,
            ),
            Signal(
                symbol="BBB.SZ",
                signal_type=SignalType.BUY,
                strength=3.0,
                timestamp=self.signal_date,
            ),
        ]


class _DummyStrategy(BaseStrategy):
    def generate_signals(self, data):
        return []


def test_backtest_buy_signal_does_not_clear_existing_position():
    dates = pd.date_range("2023-01-02", periods=3, freq="B")
    data = _make_multisymbol_data(dates, ["AAA.SZ", "BBB.SZ"], {"AAA.SZ": 10.0, "BBB.SZ": 10.0})

    strategy = _SingleBuyStrategy("BBB.SZ", dates[0])
    engine = BacktestEngine(
        data=data,
        strategy=strategy,
        initial_cash=100_000,
        commission=0,
        slippage=0,
        enforce_price_limit=False,
    )

    engine.portfolio.buy("AAA.SZ", 100, 10.0, commission=0)
    engine.portfolio.positions["AAA.SZ"].locked_shares = 0

    result = engine.run()

    assert "AAA.SZ" in result.portfolio.positions
    assert "BBB.SZ" in result.portfolio.positions


def test_backtest_buy_strength_is_normalized_before_allocation():
    dates = pd.date_range("2023-01-02", periods=3, freq="B")
    data = _make_multisymbol_data(dates, ["AAA.SZ", "BBB.SZ"], {"AAA.SZ": 10.0, "BBB.SZ": 10.0})

    engine = BacktestEngine(
        data=data,
        strategy=_WeightedBuyStrategy(dates[0]),
        initial_cash=100_000,
        commission=0,
        slippage=0,
        enforce_price_limit=False,
    )

    result = engine.run()
    positions = result.portfolio.positions
    assert positions["BBB.SZ"].shares > positions["AAA.SZ"].shares
    assert positions["BBB.SZ"].shares == positions["AAA.SZ"].shares * 3


def test_backtest_nav_is_recorded_after_trade_execution():
    dates = pd.date_range("2023-01-02", periods=3, freq="B")
    data = _make_multisymbol_data(dates, ["AAA.SZ"], {"AAA.SZ": 10.0})

    engine = BacktestEngine(
        data=data,
        strategy=_SingleBuyStrategy("AAA.SZ", dates[0]),
        initial_cash=100_000,
        commission=0.001,
        slippage=0.001,
        enforce_price_limit=False,
    )

    result = engine.run()
    nav_df = result.portfolio.to_dataframe()

    execution_day = dates[1]
    assert execution_day in nav_df.index
    assert nav_df.loc[execution_day, "nav"] < 1.0
    assert result.metrics.total_trades == 1


def test_live_execute_buys_uses_latest_market_price():
    broker = Simulator(initial_cash=100_000)
    broker.place_order = MagicMock(
        return_value=OrderResult(
            order_id="ORDER-1",
            symbol="000001.SZ",
            side="BUY",
            filled_quantity=2_000,
            filled_price=50.0,
            commission=0.0,
            timestamp=datetime.now(),
            status="FILLED",
        )
    )
    broker.get_market_data = MagicMock(return_value={"price": 50.0})

    engine = Engine(strategy=_DummyStrategy(), data=DataManager(), broker=broker)
    tracker = OrderTracker()
    journal = TradeJournal("test")

    engine._execute_buys(
        buy_signals=[Signal(symbol="000001.SZ", signal_type=SignalType.BUY, strength=1.0)],
        available_cash=100_000,
        dry_run=False,
        journal=journal,
        strategy_name="test",
        tracker=tracker,
        latest_prices={"000001.SZ": 50.0},
    )

    placed_order = broker.place_order.call_args[0][0]
    assert placed_order.quantity == 2_000


def test_optimize_minimizes_max_drawdown():
    strategy = _DummyStrategy()
    strategy.window = 1
    engine = Engine(strategy=strategy, data=DataManager(), broker=Simulator())

    fake_results = {
        1: SimpleNamespace(metrics=SimpleNamespace(max_drawdown=0.30)),
        2: SimpleNamespace(metrics=SimpleNamespace(max_drawdown=0.10)),
    }

    def _fake_backtest(**kwargs):
        return fake_results[strategy.window]

    engine.backtest = _fake_backtest
    result = engine.optimize({"window": [1, 2]}, metric="max_drawdown")

    assert result["best_params"] == {"window": 2}
    assert result["best_score"] == 0.10
