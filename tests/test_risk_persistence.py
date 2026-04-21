"""
RiskManager 状态持久化测试
"""

import json
import os
import tempfile

import pytest

from dquant.risk import PositionLimit, RiskManager


class TestRiskManagerPersistence:
    def test_save_and_restore(self):
        """测试保存和恢复状态"""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            state_path = f.name

        try:
            rm = RiskManager(max_drawdown=0.15, max_daily_loss=0.03)
            rm.enable_persistence(state_path)

            # 模拟运行状态
            rm.peak_value = 1_100_000
            rm.current_drawdown = 0.05
            rm.daily_start_value = 1_050_000
            rm.daily_start_date = "2026-04-10"
            rm.save_state()

            # 新建 RiskManager 并恢复
            rm2 = RiskManager(max_drawdown=0.15, max_daily_loss=0.03)
            rm2.enable_persistence(state_path)

            assert rm2.peak_value == 1_100_000
            assert rm2.current_drawdown == 0.05
            assert rm2.daily_start_value == 1_050_000
            assert rm2.daily_start_date == "2026-04-10"
            assert rm2.halt_trading is False

        finally:
            os.unlink(state_path)

    def test_halt_trading_persists(self):
        """测试 halt 状态正确持久化"""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            state_path = f.name

        try:
            rm = RiskManager(max_drawdown=0.05)
            rm.peak_value = 1_000_000
            rm.enable_persistence(state_path)

            # 触发回撤止损
            triggered, dd = rm.check_drawdown(900_000)
            assert triggered is True
            assert rm.halt_trading is True

            # 恢复
            rm2 = RiskManager(max_drawdown=0.05)
            rm2.enable_persistence(state_path)
            assert rm2.halt_trading is True

        finally:
            os.unlink(state_path)

    def test_daily_loss_persists(self):
        """测试日亏损状态持久化"""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            state_path = f.name

        try:
            rm = RiskManager(max_daily_loss=0.02)
            rm.enable_persistence(state_path)

            rm.reset_daily_start(1_000_000, "2026-04-10")
            triggered, loss = rm.check_daily_loss(970_000)
            assert triggered is True

            rm2 = RiskManager(max_daily_loss=0.02)
            rm2.enable_persistence(state_path)
            assert rm2.halt_trading is True
            assert rm2.daily_start_value == 1_000_000
            assert rm2.daily_start_date == "2026-04-10"

        finally:
            os.unlink(state_path)

    def test_no_persistence_by_default(self):
        """默认不启用持久化，save_state 不报错"""
        rm = RiskManager()
        rm.peak_value = 500_000
        rm.save_state()  # 不应报错

    def test_restore_nonexistent_file(self):
        """恢复不存在的文件返回 False"""
        rm = RiskManager()
        assert rm.restore_state() is False

    def test_corrupted_state_file(self):
        """损坏的状态文件不会导致崩溃"""
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            f.write("{invalid json")
            state_path = f.name

        try:
            rm = RiskManager()
            rm._state_path = __import__("pathlib").Path(state_path)
            result = rm.restore_state()
            assert result is False
            # 默认值不变
            assert rm.peak_value == 0.0
        finally:
            os.unlink(state_path)


class TestPositionLimits:
    """仓位限制检查测试"""

    def test_single_position_pass(self):
        rm = RiskManager()
        ok, msg = rm.check_position_limit("000001.SZ", 50_000, 1_000_000)
        assert ok is True
        assert msg == "OK"

    def test_single_position_exceeds(self):
        limits = PositionLimit(max_single_pct=0.05)
        rm = RiskManager(limits=limits)
        ok, msg = rm.check_position_limit("000001.SZ", 60_000, 1_000_000)
        assert ok is False
        assert "单只股票仓位" in msg

    def test_sector_limit_exceeds(self):
        limits = PositionLimit(max_sector_pct=0.2)
        rm = RiskManager(limits=limits)
        ok, msg = rm.check_position_limit(
            "000001.SZ", 80_000, 1_000_000,
            sector="银行", sector_value=150_000,
        )
        assert ok is False
        assert "行业" in msg

    def test_sector_limit_passes(self):
        limits = PositionLimit(max_sector_pct=0.3)
        rm = RiskManager(limits=limits)
        ok, msg = rm.check_position_limit(
            "000001.SZ", 80_000, 1_000_000,
            sector="银行", sector_value=150_000,
        )
        assert ok is True

    def test_total_position_exceeds(self):
        limits = PositionLimit(max_total_pct=0.9)
        rm = RiskManager(limits=limits)
        ok, msg = rm.check_position_limit(
            "000001.SZ", 100_000, 1_000_000,
            total_position_value=850_000,
        )
        assert ok is False
        assert "总仓位" in msg

    def test_total_position_passes(self):
        limits = PositionLimit(max_total_pct=0.95)
        rm = RiskManager(limits=limits)
        ok, msg = rm.check_position_limit(
            "000001.SZ", 100_000, 1_000_000,
            total_position_value=800_000,
        )
        assert ok is True

    def test_zero_total_value(self):
        rm = RiskManager()
        ok, msg = rm.check_position_limit("000001.SZ", 100, 0)
        assert ok is False

    def test_backward_compatible_no_optional_args(self):
        """旧的调用方式仍然有效"""
        rm = RiskManager()
        ok, msg = rm.check_position_limit("000001.SZ", 50_000, 1_000_000)
        assert ok is True
