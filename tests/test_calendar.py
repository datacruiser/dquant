"""
交易日历模块测试
"""

from datetime import datetime

import pytest

from dquant.calendar import (
    EXCHANGE_CALENDARS_AVAILABLE,
    get_next_trading_day,
    get_previous_trading_day,
    get_trading_days,
    is_trading_day,
)


class TestIsTradingDay:
    def test_weekday_is_trading_day(self):
        """普通工作日应为交易日"""
        # 2025-01-06 是周一
        assert is_trading_day("2025-01-06") is True

    def test_weekend_not_trading_day(self):
        """周末不是交易日"""
        # 2025-01-04 是周六
        assert is_trading_day("2025-01-04") is False
        # 2025-01-05 是周日
        assert is_trading_day("2025-01-05") is False

    def test_chinese_new_year_holiday(self):
        """春节假期不是交易日"""
        if not EXCHANGE_CALENDARS_AVAILABLE:
            pytest.skip("exchange_calendars not installed")

        # 2025-01-29 是春节假期 (初一)
        assert is_trading_day("2025-01-29") is False

    def test_national_day_holiday(self):
        """国庆假期不是交易日"""
        if not EXCHANGE_CALENDARS_AVAILABLE:
            pytest.skip("exchange_calendars not installed")

        # 2025-10-01 国庆节
        assert is_trading_day("2025-10-01") is False

    def test_datetime_input(self):
        """datetime 输入"""
        dt = datetime(2025, 1, 6)  # 周一
        assert is_trading_day(dt) is True

    def test_fallback_weekday(self):
        """无 exchange_calendars 时退化为 weekday 判断"""
        # 周一应该为 True
        assert is_trading_day("2025-01-06") is True
        # 周六应该为 False
        assert is_trading_day("2025-01-04") is False


class TestGetTradingDays:
    def test_basic_range(self):
        """基本日期范围"""
        days = get_trading_days("2025-01-06", "2025-01-10")
        # 周一到周五，应该有 5 天
        assert len(days) == 5

    def test_excludes_weekends(self):
        """排除周末"""
        days = get_trading_days("2025-01-04", "2025-01-05")
        # 周六周日
        assert len(days) == 0

    def test_single_day(self):
        """单日范围"""
        days = get_trading_days("2025-01-06", "2025-01-06")
        assert len(days) == 1

    def test_excludes_holiday(self):
        """排除节假日"""
        if not EXCHANGE_CALENDARS_AVAILABLE:
            pytest.skip("exchange_calendars not installed")

        # 2025-01-01 元旦
        days = get_trading_days("2025-01-01", "2025-01-01")
        assert len(days) == 0


class TestPreviousTradingDay:
    def test_basic(self):
        """基本前一个交易日"""
        # 2025-01-06 周一，前一个交易日应为 2025-01-03 周五
        result = get_previous_trading_day("2025-01-06", n=1)
        assert result.weekday() < 5  # 应该是工作日

    def test_skip_weekend(self):
        """跳过周末"""
        # 2025-01-06 周一 -> 前一交易日应该是周五
        result = get_previous_trading_day("2025-01-06", n=1)
        # 不应该是周六或周日
        assert result.weekday() < 5


class TestNextTradingDay:
    def test_basic(self):
        """基本后一个交易日"""
        # 2025-01-03 周五，后一个交易日应为 2025-01-06 周一
        result = get_next_trading_day("2025-01-03", n=1)
        assert result.weekday() < 5

    def test_skip_weekend(self):
        """跳过周末"""
        result = get_next_trading_day("2025-01-03", n=1)
        # 不应该是周六或周日
        assert result.weekday() < 5
