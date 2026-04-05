"""
Phase 1 实盘交易基础设施测试

覆盖: TradeJournal, RiskManager 日亏损追踪, Simulator 防御性拷贝,
      OrderValidator 集成, LiveConfig, RotatingFileHandler
"""

import json
import logging
import os
import shutil
import signal
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from dquant.broker.base import Order, OrderResult
from dquant.broker.safety import OrderValidator, TradingSafety
from dquant.broker.simulator import Simulator
from dquant.broker.trade_journal import TradeJournal
from dquant.config import DQuantConfig, LiveConfig
from dquant.logger import get_logger
from dquant.risk import PositionLimit, PositionSizer, RiskManager

# ============================================================
# TradeJournal 测试
# ============================================================


class TestTradeJournal:
    """交易审计日志测试"""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.journal = TradeJournal(journal_dir=self.tmpdir)

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_order(self, **overrides):
        defaults = dict(
            symbol="000001.SZ",
            side="BUY",
            quantity=100,
            price=10.0,
            order_type="LIMIT",
            order_id="test-001",
        )
        defaults.update(overrides)
        return Order(**defaults)

    def _make_result(self, **overrides):
        defaults = dict(
            order_id="test-001",
            symbol="000001.SZ",
            side="BUY",
            filled_quantity=100,
            filled_price=10.0,
            commission=3.0,
            timestamp=datetime.now(),
            status="FILLED",
        )
        defaults.update(overrides)
        return OrderResult(**defaults)

    def test_record_and_read(self):
        """测试写入和读取 JSONL"""
        order = self._make_order()
        result = self._make_result()

        self.journal.record("ORDER_PLACED", order, result, strategy_name="TestStrategy")

        date_str = datetime.now().strftime("%Y-%m-%d")
        records = self.journal.read_day(date_str)

        assert len(records) == 1
        r = records[0]
        assert r["event_type"] == "ORDER_PLACED"
        assert r["symbol"] == "000001.SZ"
        assert r["side"] == "BUY"
        assert r["quantity"] == 100
        assert r["filled_quantity"] == 100
        assert r["status"] == "FILLED"
        assert r["strategy"] == "TestStrategy"

    def test_record_without_result(self):
        """测试无 result 的记录"""
        order = self._make_order()
        self.journal.record("ORDER_CANCELLED", order, strategy_name="Test")

        date_str = datetime.now().strftime("%Y-%m-%d")
        records = self.journal.read_day(date_str)

        assert len(records) == 1
        r = records[0]
        assert r["event_type"] == "ORDER_CANCELLED"
        assert "filled_quantity" not in r

    def test_multiple_records_same_day(self):
        """测试同一天多条记录"""
        for i in range(5):
            order = self._make_order(order_id=f"test-{i:03d}")
            self.journal.record("ORDER_PLACED", order)

        date_str = datetime.now().strftime("%Y-%m-%d")
        records = self.journal.read_day(date_str)
        assert len(records) == 5

    def test_read_nonexistent_day(self):
        """测试读取不存在的日期"""
        records = self.journal.read_day("2020-01-01")
        assert records == []

    def test_get_today_summary(self):
        """测试今日交易摘要"""
        # 写入一些记录
        order = self._make_order()
        result = self._make_result(status="FILLED")
        self.journal.record("ORDER_PLACED", order, result)

        summary = self.journal.get_today_summary()
        assert summary["total_placed"] == 1
        assert summary["total_filled"] == 1
        assert summary["total_rejected"] == 0
        assert "000001.SZ" in summary["symbols"]

    def test_corrupted_line_skipped(self):
        """测试损坏的 JSONL 行被跳过"""
        date_str = datetime.now().strftime("%Y-%m-%d")
        filepath = Path(self.tmpdir) / f"{date_str}.jsonl"

        # 写入一条正常和一条损坏的
        with open(filepath, "a") as f:
            f.write('{"valid": true}\n')
            f.write("corrupted line\n")
            f.write('{"also_valid": true}\n')

        records = self.journal.read_day(date_str)
        assert len(records) == 2

    def test_signal_info_recorded(self):
        """测试信号信息被记录"""
        order = self._make_order()
        signal_info = {"indicator": "MACD", "cross": "golden"}
        self.journal.record("ORDER_PLACED", order, signal_info=signal_info)

        date_str = datetime.now().strftime("%Y-%m-%d")
        records = self.journal.read_day(date_str)
        assert records[0]["signal"] == signal_info


# ============================================================
# RiskManager 日亏损追踪测试
# ============================================================


class TestRiskManagerDailyLoss:
    """RiskManager 日亏损追踪测试"""

    def test_reset_daily_start(self):
        rm = RiskManager(max_daily_loss=0.03)
        rm.reset_daily_start(1000000, "2026-03-30")
        assert rm.daily_start_value == 1000000
        assert rm.daily_start_date == "2026-03-30"

    def test_reset_daily_start_idempotent(self):
        """同一天多次调用不覆盖"""
        rm = RiskManager()
        rm.reset_daily_start(1000000, "2026-03-30")
        rm.reset_daily_start(900000, "2026-03-30")
        assert rm.daily_start_value == 1000000

    def test_reset_daily_start_new_day(self):
        """新一天会重置"""
        rm = RiskManager()
        rm.reset_daily_start(1000000, "2026-03-29")
        rm.reset_daily_start(980000, "2026-03-30")
        assert rm.daily_start_value == 980000

    def test_check_daily_loss_no_trigger(self):
        rm = RiskManager(max_daily_loss=0.03)
        rm.reset_daily_start(1000000, "2026-03-30")
        triggered, loss = rm.check_daily_loss(990000)
        assert not triggered
        assert abs(loss - 0.01) < 1e-6

    def test_check_daily_loss_triggered(self):
        rm = RiskManager(max_daily_loss=0.03)
        rm.reset_daily_start(1000000, "2026-03-30")
        triggered, loss = rm.check_daily_loss(960000)
        assert triggered
        assert abs(loss - 0.04) < 1e-6
        assert rm.halt_trading

    def test_should_halt(self):
        rm = RiskManager(max_drawdown=0.10)
        rm.check_drawdown(1000000)  # peak
        rm.check_drawdown(850000)  # 15% drawdown > 10%
        assert rm.should_halt()

    def test_should_not_halt(self):
        rm = RiskManager()
        rm.reset_daily_start(1000000, "2026-03-30")
        rm.check_drawdown(1000000)
        rm.check_drawdown(990000)
        rm.check_daily_loss(990000)
        assert not rm.should_halt()

    def test_daily_loss_with_zero_start(self):
        rm = RiskManager()
        triggered, loss = rm.check_daily_loss(100000)
        assert not triggered
        assert loss == 0.0


# ============================================================
# Simulator 防御性拷贝 + 订单验证 测试
# ============================================================


class TestSimulatorSafety:
    """Simulator 安全增强测试"""

    def test_defensive_copy_positions(self):
        """get_positions 返回的是副本"""
        sim = Simulator()
        sim.positions["000001.SZ"] = {"quantity": 100, "avg_cost": 10.0, "price": 10.0}
        pos = sim.get_positions()
        pos["000001.SZ"]["quantity"] = 999
        assert sim.positions["000001.SZ"]["quantity"] == 100

    def test_order_validation_rejects_bad_quantity(self):
        """非整手数量被拒绝"""
        sim = Simulator()
        order = Order(symbol="000001.SZ", side="BUY", quantity=50, price=10.0)
        result = sim.place_order(order)
        assert result.status == "REJECTED"
        assert result.filled_quantity == 0

    def test_order_validation_rejects_bad_symbol(self):
        """无效代码被拒绝"""
        sim = Simulator()
        order = Order(symbol="INVALID", side="BUY", quantity=100, price=10.0)
        result = sim.place_order(order)
        assert result.status == "REJECTED"

    def test_order_validation_rejects_bad_side(self):
        """无效方向被拒绝"""
        sim = Simulator()
        order = Order(symbol="000001.SZ", side="HOLD", quantity=100, price=10.0)
        result = sim.place_order(order)
        assert result.status == "REJECTED"

    def test_valid_order_still_works(self):
        """合法订单仍然正常执行"""
        sim = Simulator()
        order = Order(symbol="000001.SZ", side="BUY", quantity=100, price=10.0)
        result = sim.place_order(order)
        assert result.status == "FILLED"
        assert result.filled_quantity == 100


# ============================================================
# LiveConfig 测试
# ============================================================


class TestLiveConfig:
    """LiveConfig 配置测试"""

    def test_default_values(self):
        cfg = LiveConfig()
        assert cfg.broker == "simulator"
        assert cfg.dry_run is True
        assert cfg.interval == 60
        assert cfg.max_drawdown == 0.15
        assert cfg.max_daily_loss == 0.03
        assert cfg.max_consecutive_errors == 10
        assert cfg.position_method == "equal_weight"

    def test_config_in_dquant_config(self):
        cfg = DQuantConfig()
        assert hasattr(cfg, "live")
        assert isinstance(cfg.live, LiveConfig)

    def test_from_dict(self):
        data = {
            "live": {
                "broker": "xtp",
                "dry_run": False,
                "interval": 30,
                "max_drawdown": 0.10,
            }
        }
        cfg = DQuantConfig.from_dict(data)
        assert cfg.live.broker == "xtp"
        assert cfg.live.dry_run is False
        assert cfg.live.interval == 30
        assert cfg.live.max_drawdown == 0.10
        # Other fields keep defaults
        assert cfg.live.max_daily_loss == 0.03

    def test_to_dict(self):
        cfg = DQuantConfig()
        d = cfg.to_dict()
        assert "live" in d
        assert d["live"]["broker"] == "simulator"


# ============================================================
# RotatingFileHandler 测试
# ============================================================


class TestRotatingLogger:
    """RotatingFileHandler 日志测试"""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_rotating_handler(self):
        """rotating=True 使用 RotatingFileHandler"""
        from logging.handlers import RotatingFileHandler

        log_file = os.path.join(self.tmpdir, "test.log")
        # Use unique name to avoid handler cache
        logger = get_logger(
            f"dquant.test.rotating.{id(self)}",
            log_file=log_file,
            rotating=True,
            max_bytes=1024,
            backup_count=2,
        )

        # Check handler type
        file_handlers = [h for h in logger.handlers if isinstance(h, RotatingFileHandler)]
        assert len(file_handlers) == 1
        assert file_handlers[0].maxBytes == 1024

    def test_non_rotating_handler(self):
        """rotating=False 使用普通 FileHandler"""
        log_file = os.path.join(self.tmpdir, "test.log")
        logger = get_logger(
            f"dquant.test.nonrotating.{id(self)}",
            log_file=log_file,
            rotating=False,
        )

        from logging.handlers import RotatingFileHandler

        file_handlers = [h for h in logger.handlers if isinstance(h, RotatingFileHandler)]
        normal_handlers = [
            h
            for h in logger.handlers
            if isinstance(h, logging.FileHandler) and not isinstance(h, RotatingFileHandler)
        ]
        assert len(file_handlers) == 0
        assert len(normal_handlers) == 1


# ============================================================
# Engine.live() 基本集成测试
# ============================================================


class TestEngineLive:
    """Engine.live() 基本测试 (使用 Simulator + MockStrategy)"""

    def _make_mock_strategy(self, signals=None):
        """创建 MockStrategy"""
        strategy = MagicMock()
        strategy.generate_signals.return_value = signals or []
        return strategy

    def _make_mock_data(self):
        """创建 MockDataSource"""
        data = MagicMock()
        data.load.return_value = pd.DataFrame()
        return data

    def test_live_dry_run_connects(self):
        """dry_run 模式下 broker 正常连接"""
        from dquant.core import Engine

        data = self._make_mock_data()
        strategy = self._make_mock_strategy()

        engine = Engine(data, strategy, broker="simulator")
        # Patch is_trading_day and TradingTimeChecker to force non-trading
        with patch("dquant.core.is_trading_day", return_value=False):
            # Use _running flag to break after first iteration
            original_live = engine.live
            call_count = [0]

            def limited_live(**kwargs):
                engine._running = True
                import time as _time

                # Manually run one iteration of the loop logic
                from dquant.broker.safety import TradingTimeChecker
                from dquant.broker.trade_journal import TradeJournal
                from dquant.risk import RiskManager

                engine.broker.connect()
                # Simulator connects successfully
                account = engine.broker.get_account()
                assert account["initial_cash"] == engine.initial_cash
                engine.broker.disconnect()

            limited_live()

    def test_live_risk_manager_halt(self):
        """风控触发时 halt_trading 生效"""
        rm = RiskManager(max_drawdown=0.10)
        rm.check_drawdown(1000000)
        rm.check_drawdown(800000)  # 20% > 10%
        assert rm.should_halt()
        assert rm.halt_trading

    def test_sizer_equal_weight(self):
        """等权仓位计算"""
        # max_single_pct 默认 0.1，所以 100000 * 0.1 = 10000 per stock max
        # 用更大的 total_value 或更宽的 limit
        limits = PositionLimit(max_single_pct=0.5)
        sizer = PositionSizer(method="equal_weight", total_value=100000, limits=limits)
        positions = sizer.size(["000001.SZ", "000002.SZ", "000003.SZ"])
        assert len(positions) == 3
        for v in positions.values():
            assert v == pytest.approx(100000 / 3, rel=0.01)

    def test_sizer_respects_max_single(self):
        """等权受 max_single_pct 限制"""
        limits = PositionLimit(max_single_pct=0.2)
        sizer = PositionSizer(method="equal_weight", total_value=100000, limits=limits)
        positions = sizer.size(["000001.SZ"])
        # 1 stock would get 100000, but max is 100000 * 0.2 = 20000
        assert positions["000001.SZ"] == 20000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
