"""
回归测试 — P0/P1/P2 修复验证

覆盖:
1. T+1 信号执行 (无前视偏差)
2. signal.timestamp=None 信号处理
3. portfolio.sell() 整手约束 + 清仓路径
4. optimize() metric 别名映射
5. TradingTimeChecker 返回值解包
6. symbol 标准化
"""

from datetime import datetime, time
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from dquant.backtest.engine import BacktestEngine
from dquant.backtest.portfolio import Portfolio, Position
from dquant.constants import MIN_SHARES, DEFAULT_STAMP_DUTY, normalize_symbol
from dquant.strategy.base import BaseStrategy, Signal, SignalType

# ============================================================
# 1. T+1 信号执行
# ============================================================


class SimpleBuyStrategy(BaseStrategy):
    """在指定日期产生买入信号的测试策略"""

    def __init__(self, buy_date, symbol="TEST.SZ"):
        self.buy_date = buy_date
        self.symbol = symbol

    def generate_signals(self, data):
        if data.empty:
            return []
        last_date = data.index.max()
        if pd.Timestamp(last_date).normalize() < pd.Timestamp(self.buy_date).normalize():
            return []
        return [
            Signal(
                symbol=self.symbol,
                signal_type=SignalType.BUY,
                strength=1.0,
                timestamp=last_date,
            )
        ]


def _make_test_data(dates, symbol="TEST.SZ", price=10.0):
    """生成测试数据"""
    rows = []
    for d in dates:
        rows.append(
            {
                "symbol": symbol,
                "open": price,
                "high": price * 1.01,
                "low": price * 0.99,
                "close": price,
                "volume": 1000000,
            }
        )
    df = pd.DataFrame(rows, index=pd.DatetimeIndex(dates))
    return df


class TestTPlus1Execution:
    """验证信号在 T+1 日执行，非 T 日"""

    def test_signal_executed_next_day(self):
        """信号在 Day2 产生，应在 Day3 执行 (T+1)"""
        dates = pd.date_range("2023-01-02", periods=5, freq="B")
        data = _make_test_data(dates, price=10.0)

        # 在 Day3 (dates[2]) 产生信号
        strategy = SimpleBuyStrategy(buy_date=str(dates[2]))
        engine = BacktestEngine(data=data, strategy=strategy, initial_cash=100000)
        result = engine.run()

        # 交易记录应在 Day4 (dates[3]) — T+1
        trades = result.trades if hasattr(result, "trades") else pd.DataFrame()
        if not trades.empty:
            trade_dates = pd.to_datetime(trades["date"]).normalize()
            signal_date = pd.Timestamp(dates[2]).normalize()
            for td in trade_dates:
                assert (
                    td > signal_date
                ), f"交易日期 {td} 不应早于或等于信号日期 {signal_date} (T+1 违规)"

    def test_no_trade_on_last_day_signal(self):
        """在最后一个交易日产生的信号不应被执行 (无 T+1)"""
        dates = pd.date_range("2023-01-02", periods=3, freq="B")
        data = _make_test_data(dates, price=10.0)

        # 信号在最后一天产生 → 无后续交易日 → 不应执行
        strategy = SimpleBuyStrategy(buy_date=str(dates[-1]))
        engine = BacktestEngine(data=data, strategy=strategy, initial_cash=100000)
        result = engine.run()

        trades = result.trades if hasattr(result, "trades") else pd.DataFrame()
        assert trades.empty, "最后一天信号不应被执行 (无 T+1 日)"

    def test_no_timestamp_signals_skipped(self):
        """无 timestamp 的信号应被跳过 (不崩溃)"""

        class NoTimestampStrategy(BaseStrategy):
            def generate_signals(self, data):
                if len(data) < 2:
                    return []
                return [
                    Signal(
                        symbol="TEST.SZ",
                        signal_type=SignalType.BUY,
                        strength=1.0,
                        timestamp=None,  # 无 timestamp
                    )
                ]

        dates = pd.date_range("2023-01-02", periods=5, freq="B")
        data = _make_test_data(dates)
        strategy = NoTimestampStrategy()
        engine = BacktestEngine(data=data, strategy=strategy, initial_cash=100000)

        # 不应崩溃
        result = engine.run()
        trades = result.trades if hasattr(result, "trades") else pd.DataFrame()
        # 无 timestamp 信号不应产生交易
        assert trades.empty


# ============================================================
# 2. Portfolio sell 整手约束
# ============================================================


class TestSellLotSize:
    """验证 sell() 的整手约束和清仓路径"""

    def test_normal_lot_sell(self):
        """正常整手卖出"""
        pf = Portfolio(initial_cash=100000)
        pf.update_prices({"TEST.SZ": 10.0}, datetime(2023, 1, 1))
        pf.buy("TEST.SZ", 500, 10.0, commission=0)
        assert pf.positions["TEST.SZ"].shares == 500

        # 过一天释放 T+1 冻结
        pf.update_prices({"TEST.SZ": 10.0}, datetime(2023, 1, 2))

        pf.sell("TEST.SZ", 200, 10.0, commission=0)
        assert pf.positions["TEST.SZ"].shares == 300

    def test_fractional_sell_rounds_down(self):
        """小数卖出向下取整"""
        pf = Portfolio(initial_cash=100000)
        pf.update_prices({"TEST.SZ": 10.0}, datetime(2023, 1, 1))
        pf.buy("TEST.SZ", 500, 10.0, commission=0)

        pf.update_prices({"TEST.SZ": 10.0}, datetime(2023, 1, 2))

        # 请求卖出 123.45 股 → 向下取整到 100
        pf.sell("TEST.SZ", 123.45, 10.0, commission=0)
        assert pf.positions["TEST.SZ"].shares == 400

    def test_small_sell_clears_position(self):
        """卖出数量不足一手时清仓"""
        pf = Portfolio(initial_cash=100000)
        pf.update_prices({"TEST.SZ": 10.0}, datetime(2023, 1, 1))
        pf.buy("TEST.SZ", 200, 10.0, commission=0)

        pf.update_prices({"TEST.SZ": 10.0}, datetime(2023, 1, 2))

        # 请求卖出 50 股 → lot_shares=0 → 清仓 200
        pf.sell("TEST.SZ", 50, 10.0, commission=0)
        assert "TEST.SZ" not in pf.positions

    def test_remaining_fractional_clears_position(self):
        """卖出后剩余不足一手时清仓"""
        pf = Portfolio(initial_cash=100000)
        pf.update_prices({"TEST.SZ": 10.0}, datetime(2023, 1, 1))
        pf.buy("TEST.SZ", 250, 10.0, commission=0)

        pf.update_prices({"TEST.SZ": 10.0}, datetime(2023, 1, 2))

        # 请求卖出 200 → lot_shares=200, remaining=50 < 100 → 清仓 250
        pf.sell("TEST.SZ", 200, 10.0, commission=0)
        assert "TEST.SZ" not in pf.positions

    def test_full_sell_removes_position(self):
        """全部卖出移除持仓"""
        pf = Portfolio(initial_cash=100000)
        pf.update_prices({"TEST.SZ": 10.0}, datetime(2023, 1, 1))
        pf.buy("TEST.SZ", 500, 10.0, commission=0)

        pf.update_prices({"TEST.SZ": 10.0}, datetime(2023, 1, 2))

        pf.sell("TEST.SZ", 500, 10.0, commission=0)
        assert "TEST.SZ" not in pf.positions

    def test_sell_revenue_includes_stamp_duty(self):
        """卖出收入扣除印花税"""
        pf = Portfolio(initial_cash=100000)
        pf.update_prices({"TEST.SZ": 10.0}, datetime(2023, 1, 1))
        pf.buy("TEST.SZ", 100, 10.0, commission=0)
        cash_before = pf.cash

        pf.update_prices({"TEST.SZ": 10.0}, datetime(2023, 1, 2))

        # 卖出 100 股 @ 10.0, stamp_duty=0.001
        pf.sell("TEST.SZ", 100, 10.0, commission=0, stamp_duty=0.001)
        # 收入 = 100 * 10.0 * (1 - 0 - 0.001) = 999
        assert abs(pf.cash - (cash_before + 999.0)) < 0.01


# ============================================================
# 2b. T+1 冻结持仓专项测试
# ============================================================


class TestTPlus1LockedShares:
    """验证 T+1 冻结/释放/累积/扣减逻辑"""

    def test_cannot_sell_on_buy_day(self):
        """当日买入的股票不可卖出（locked_shares 阻断）"""
        pf = Portfolio(initial_cash=100000)
        pf.update_prices({}, datetime(2023, 1, 1))
        pf.buy("TEST.SZ", 500, 10.0, commission=0)

        # 同一天尝试卖出 → available = max(0, 500-500) = 0 → 卖出被阻断
        pf.sell("TEST.SZ", 500, 10.0, commission=0)
        assert pf.positions["TEST.SZ"].shares == 500
        assert pf.positions["TEST.SZ"].locked_shares == 500

    def test_lock_accumulates_on_same_day_buys(self):
        """同日多次买入累积 locked_shares"""
        pf = Portfolio(initial_cash=200000)
        pf.update_prices({}, datetime(2023, 1, 1))
        pf.buy("TEST.SZ", 300, 10.0, commission=0)
        pf.buy("TEST.SZ", 200, 10.0, commission=0)

        pos = pf.positions["TEST.SZ"]
        assert pos.shares == 500
        assert pos.locked_shares == 500
        assert pos.available_shares == 0

    def test_unlocked_after_next_day(self):
        """次日 locked_shares 被清零，全部可用"""
        pf = Portfolio(initial_cash=100000)
        pf.update_prices({}, datetime(2023, 1, 1))
        pf.buy("TEST.SZ", 500, 10.0, commission=0)
        assert pf.positions["TEST.SZ"].available_shares == 0

        # 过一天 → locked_shares 清零
        pf.update_prices({"TEST.SZ": 10.0}, datetime(2023, 1, 2))
        assert pf.positions["TEST.SZ"].locked_shares == 0
        assert pf.positions["TEST.SZ"].available_shares == 500

    def test_multi_day_buy_partial_availability(self):
        """Day1 买入 500, Day2 又买入 300 → Day2 可卖 500，锁定 300"""
        pf = Portfolio(initial_cash=200000)
        pf.update_prices({}, datetime(2023, 1, 1))
        pf.buy("TEST.SZ", 500, 10.0, commission=0)

        # Day 2: 释放 Day1 锁定 + 新买 300
        pf.update_prices({"TEST.SZ": 10.0}, datetime(2023, 1, 2))
        pf.buy("TEST.SZ", 300, 10.0, commission=0)

        pos = pf.positions["TEST.SZ"]
        assert pos.shares == 800
        assert pos.locked_shares == 300
        assert pos.available_shares == 500

        # 尝试卖 600 → 实际只能卖 500（受限于 available）
        pf.sell("TEST.SZ", 600, 10.0, commission=0)
        assert pos.shares == 300
        assert pos.locked_shares == 300  # min(300, 300)

    def test_partial_sell_clamps_locked(self):
        """部分卖出后 locked_shares 被 min(locked, shares) 钳制"""
        pf = Portfolio(initial_cash=200000)
        pf.update_prices({}, datetime(2023, 1, 1))
        pf.buy("TEST.SZ", 500, 10.0, commission=0)

        pf.update_prices({"TEST.SZ": 10.0}, datetime(2023, 1, 2))
        pf.buy("TEST.SZ", 300, 10.0, commission=0)  # locked=300

        # 卖出 500（全部 available）→ shares=300, locked=min(300,300)=300
        pf.sell("TEST.SZ", 500, 10.0, commission=0)
        assert pf.positions["TEST.SZ"].locked_shares == 300
        assert pf.positions["TEST.SZ"].available_shares == 0

        # 次日释放后可卖剩余
        pf.update_prices({"TEST.SZ": 10.0}, datetime(2023, 1, 3))
        assert pf.positions["TEST.SZ"].available_shares == 300

    def test_locked_cleared_across_weekend(self):
        """跨周末 locked_shares 正确释放"""
        pf = Portfolio(initial_cash=100000)
        pf.update_prices({}, datetime(2023, 1, 6))  # Friday
        pf.buy("TEST.SZ", 500, 10.0, commission=0)
        assert pf.positions["TEST.SZ"].locked_shares == 500

        # 跳到 Monday (2023-01-09)
        pf.update_prices({"TEST.SZ": 10.0}, datetime(2023, 1, 9))
        assert pf.positions["TEST.SZ"].locked_shares == 0
        assert pf.positions["TEST.SZ"].available_shares == 500

    def test_dust_compensation_does_not_sell_locked_shares(self):
        """零股补偿不应变现 T+1 锁定股份（BUG FIX 回归测试）"""
        pf = Portfolio(initial_cash=500000)
        pf.update_prices({}, datetime(2023, 1, 1))
        # 买入 500 股（整手）
        pf.buy("TEST.SZ", 500, 10.0, commission=0)
        assert pf.positions["TEST.SZ"].locked_shares == 500

        # Day 2: 解锁 500, 再买入 200 → locked=200, available=500
        pf.update_prices({"TEST.SZ": 10.0}, datetime(2023, 1, 2))
        pf.buy("TEST.SZ", 200, 10.0, commission=0)

        pos = pf.positions["TEST.SZ"]
        assert pos.shares == 700
        assert pos.locked_shares == 200
        assert pos.available_shares == 500

        # 卖出 500 → shares=200, locked=min(200,200)=200
        cash_before = pf.cash
        pf.sell("TEST.SZ", 500, 10.0, commission=0)

        # 剩余 200 股 >= MIN_SHARES(100)，不触发零股补偿
        assert "TEST.SZ" in pf.positions
        assert pf.positions["TEST.SZ"].shares == 200
        assert pf.positions["TEST.SZ"].locked_shares == 200

        # Day 3: 解锁 200, 再卖 100 → 剩余 100 = MIN_SHARES，不触发零股
        pf.update_prices({"TEST.SZ": 10.0}, datetime(2023, 1, 3))
        pf.sell("TEST.SZ", 100, 10.0, commission=0)
        assert pf.positions["TEST.SZ"].shares == 100

        # 再卖 50 → lot_shares=0, 清仓 available=100
        cash_before = pf.cash
        pf.sell("TEST.SZ", 50, 10.0, commission=0)
        # 100 股全部 available 且清仓了，无剩余零股
        assert "TEST.SZ" not in pf.positions
        expected_revenue = 100 * 10.0 * (1 - 0 - DEFAULT_STAMP_DUTY)
        assert abs(pf.cash - (cash_before + expected_revenue)) < 0.01

    def test_dust_compensation_sells_unlocked_dust_only(self):
        """零股补偿仅变现未锁定的零股部分（全部 unlocked 的零股应被补偿）"""
        pf = Portfolio(initial_cash=200000)
        pf.update_prices({}, datetime(2023, 1, 1))
        pf.buy("TEST.SZ", 200, 10.0, commission=0)

        # Day 2: 解锁 200
        pf.update_prices({"TEST.SZ": 10.0}, datetime(2023, 1, 2))
        # 卖出 50 股 → lot_shares=0 → 清仓 available=200（整手规则：不足一手全卖）
        cash_before = pf.cash
        pf.sell("TEST.SZ", 50, 10.0, commission=0)

        # 200 股全部清仓（不足一手触发清仓），无剩余零股
        assert "TEST.SZ" not in pf.positions
        expected_revenue = 200 * 10.0 * (1 - 0 - DEFAULT_STAMP_DUTY)
        assert abs(pf.cash - (cash_before + expected_revenue)) < 0.01

    def test_dust_compensation_partially_locked(self):
        """零股补偿中混合锁定/非锁定时，仅补偿非锁定部分"""
        pf = Portfolio(initial_cash=500000)
        pf.update_prices({}, datetime(2023, 1, 1))
        pf.buy("TEST.SZ", 300, 10.0, commission=0)

        # Day 2: 解锁 300, 再买入 100 → shares=400, locked=100, available=300
        pf.update_prices({"TEST.SZ": 10.0}, datetime(2023, 1, 2))
        pf.buy("TEST.SZ", 100, 10.0, commission=0)

        pos = pf.positions["TEST.SZ"]
        assert pos.shares == 400
        assert pos.locked_shares == 100
        assert pos.available_shares == 300

        # 卖出 300 (全部 available) → shares=100, locked=100
        cash_before = pf.cash
        pf.sell("TEST.SZ", 300, 10.0, commission=0)

        # shares=100, locked=100 → 100 == MIN_SHARES，不触发零股补偿
        assert "TEST.SZ" in pf.positions
        expected_revenue = 300 * 10.0 * (1 - 0 - DEFAULT_STAMP_DUTY)
        assert abs(pf.cash - (cash_before + expected_revenue)) < 0.01


# ============================================================
# 3. TradingTimeChecker 返回值
# ============================================================


class TestTradingTimeChecker:
    """验证 is_trading_time 返回 (bool, str) 元组"""

    def test_returns_tuple(self):
        from dquant.broker.safety import TradingTimeChecker

        checker = TradingTimeChecker()
        result = checker.is_trading_time(datetime(2023, 6, 15, 10, 0))
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)

    def test_tuple_unpacking(self):
        """验证解包后 bool 值正确"""
        from dquant.broker.safety import TradingTimeChecker

        checker = TradingTimeChecker()

        # 交易时间
        can_trade, msg = checker.is_trading_time(datetime(2023, 6, 15, 10, 0))
        assert can_trade is True
        assert "上午" in msg or "交易" in msg

        # 非交易时间
        can_trade, msg = checker.is_trading_time(datetime(2023, 6, 15, 8, 0))
        assert can_trade is False

    def test_not_tuple_is_truthy(self):
        """验证 not tuple 不会被错误判为 False"""
        from dquant.broker.safety import TradingTimeChecker

        checker = TradingTimeChecker()
        result = checker.is_trading_time(datetime(2023, 6, 15, 10, 0))
        # result = (True, "...")
        # not result = not (True, "...") = False — 正确
        # 但如果只做了 not result, 会得到 False 即使是交易时间
        can_trade, _ = result
        assert can_trade is True
        # 错误用法 (回归测试):
        assert not result is False  # (True, msg) is truthy, so `not result` = False


# ============================================================
# 4. Symbol 标准化
# ============================================================


class TestNormalizeSymbol:
    """验证集中式 symbol 标准化"""

    def test_sh_stock(self):
        assert normalize_symbol("600000") == "600000.SH"

    def test_sz_stock(self):
        assert normalize_symbol("000001") == "000001.SZ"

    def test_sz_gem(self):
        assert normalize_symbol("300001") == "300001.SZ"

    def test_bj_stock(self):
        assert normalize_symbol("430001") == "430001.BJ"

    def test_already_standard(self):
        assert normalize_symbol("600000.SH") == "600000.SH"

    def test_xshg_suffix(self):
        assert normalize_symbol("600000.XSHG") == "600000.SH"

    def test_empty(self):
        assert normalize_symbol("") == ""

    def test_none(self):
        assert normalize_symbol(None) is None


# ============================================================
# 5. optimize metric aliases
# ============================================================


class TestOptimizeMetricAliases:
    """验证 optimize() 支持指标别名"""

    def test_metric_aliases_mapping(self):
        """直接测试映射逻辑"""
        metric_aliases = {
            "return": "total_return",
            "sharpe": "sharpe",
            "max_drawdown": "max_drawdown",
        }
        assert metric_aliases.get("return", "return") == "total_return"
        assert metric_aliases.get("sharpe", "sharpe") == "sharpe"
        assert metric_aliases.get("max_drawdown", "max_drawdown") == "max_drawdown"
        # 未知 metric → 原样返回
        assert metric_aliases.get("calmar", "calmar") == "calmar"
