"""
涨跌停限制测试
"""

import pandas as pd
import pytest

from dquant.backtest.engine import (
    BJ_PRICE_LIMIT,
    DEFAULT_PRICE_LIMIT,
    BacktestEngine,
    _get_price_limit,
)
from dquant.strategy.base import BaseStrategy, Signal, SignalType


class _BuyAll(BaseStrategy):
    """每天都买入所有股票"""

    def generate_signals(self, data):
        signals = []
        for date, group in data.groupby(data.index):
            for _, row in group.iterrows():
                signals.append(
                    Signal(
                        symbol=row["symbol"],
                        signal_type=SignalType.BUY,
                        strength=0.5,
                        timestamp=date,
                    )
                )
        return signals


class _SellAll(BaseStrategy):
    """每天都卖出所有持仓"""

    def generate_signals(self, data):
        signals = []
        for date, group in data.groupby(data.index):
            for _, row in group.iterrows():
                signals.append(
                    Signal(
                        symbol=row["symbol"],
                        signal_type=SignalType.SELL,
                        strength=0.5,
                        timestamp=date,
                    )
                )
        return signals


def _make_limit_data():
    """创建模拟涨跌停数据"""
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    rows = []
    # 股票 A: 正常涨跌
    for i, d in enumerate(dates):
        price = 10 + i * 0.1
        rows.append(
            {
                "date": d,
                "symbol": "000001.SZ",
                "open": price,
                "high": price * 1.02,
                "low": price * 0.98,
                "close": price,
                "volume": 1000000,
            }
        )
    # 股票 B: 第3天涨停 (+10%)
    base = 20.0
    for i, d in enumerate(dates):
        if i == 2:
            close = base * 1.10  # 涨停
        else:
            close = base + i * 0.05
        rows.append(
            {
                "date": d,
                "symbol": "600000.SH",
                "open": base + i * 0.05,
                "high": close * 1.01,
                "low": (base + i * 0.05) * 0.99,
                "close": close,
                "volume": 1000000,
            }
        )
        base = close  # 下一天的前收盘 = 今天收盘

    df = pd.DataFrame(rows)
    df = df.set_index("date")
    return df


class TestGetPriceLimit:
    def test_main_board(self):
        assert _get_price_limit("000001.SZ") == DEFAULT_PRICE_LIMIT
        assert _get_price_limit("600000.SH") == DEFAULT_PRICE_LIMIT

    def test_bj_board(self):
        assert _get_price_limit("430001.BJ") == BJ_PRICE_LIMIT
        assert _get_price_limit("830001.BJ") == BJ_PRICE_LIMIT


class TestPriceLimitEnforcement:
    def test_limit_up_blocks_buy(self):
        """涨停日不应执行买入"""
        data = _make_limit_data()
        engine = BacktestEngine(
            data=data,
            strategy=_BuyAll(),
            initial_cash=1_000_000,
            enforce_price_limit=True,
        )
        result = engine.run()

        # 确认有交易但涨停日应被过滤
        if not result.trades.empty:
            limit_up_trades = result.trades[
                (result.trades["symbol"] == "600000.SH")
                & (result.trades["action"] == "BUY")
            ]
            # 涨停日的买入信号应被过滤
            for _, trade in limit_up_trades.iterrows():
                assert trade["date"] != pd.Timestamp("2024-01-03")

    def test_no_limit_check_when_disabled(self):
        """关闭涨跌停检查时应有更多交易"""
        data = _make_limit_data()

        engine_with = BacktestEngine(
            data=data, strategy=_BuyAll(), enforce_price_limit=True
        )
        engine_without = BacktestEngine(
            data=data, strategy=_BuyAll(), enforce_price_limit=False
        )

        result_with = engine_with.run()
        result_without = engine_without.run()

        # 不检查涨跌停的交易数 >= 检查的
        assert len(result_without.trades) >= len(result_with.trades)

    def test_backward_compatible(self):
        """新参数默认值不应破坏旧测试"""
        dates = pd.date_range("2023-01-01", periods=30, freq="D")
        rows = []
        for d in dates:
            for symbol in ["000001.SZ", "600000.SH"]:
                price = 10.0
                rows.append(
                    {
                        "date": d,
                        "symbol": symbol,
                        "open": price,
                        "high": price * 1.02,
                        "low": price * 0.98,
                        "close": price * 1.01,
                        "volume": 1000000,
                    }
                )
        df = pd.DataFrame(rows).set_index("date")

        engine = BacktestEngine(data=df, strategy=_BuyAll())
        result = engine.run()
        assert result.metrics is not None
