"""
交易日历模块

提供交易日判断功能，优先使用 exchange_calendars 库获取准确的交易日历，
未安装时退化为简单的周一至周五判断。

Usage:
    from dquant.calendar import is_trading_day, get_trading_days

    # 判断是否为交易日
    is_trading_day('2025-01-01')  # 元旦，返回 False（需安装 exchange_calendars）

    # 获取交易日列表
    days = get_trading_days('2025-01-01', '2025-01-31')
"""

from datetime import datetime, timedelta
from typing import List, Union

import pandas as pd

from dquant.logger import get_logger

logger = get_logger(__name__)

# 尝试导入 exchange_calendars
try:
    import exchange_calendars as ec

    EXCHANGE_CALENDARS_AVAILABLE = True
except ImportError:
    EXCHANGE_CALENDARS_AVAILABLE = False

# 市场到交易所代码的映射
_MARKET_EXCHANGE_MAP = {
    "cn": "XSHG",  # 上海证券交易所（覆盖 A 股）
    "xshg": "XSHG",  # 上交所
    "xshe": "XSHE",  # 深交所
}

# 日历实例缓存
_CALENDAR_CACHE: dict = {}


def _get_calendar(market: str = "cn"):
    """
    获取交易日历实例（懒加载、缓存）

    Args:
        market: 市场代码

    Returns:
        exchange_calendars.ExchangeCalendar 或 None
    """
    if not EXCHANGE_CALENDARS_AVAILABLE:
        return None

    exchange = _MARKET_EXCHANGE_MAP.get(market.lower())
    if exchange is None:
        logger.warning(f"Unknown market: {market}, falling back to XSHG")
        exchange = "XSHG"

    if exchange not in _CALENDAR_CACHE:
        try:
            _CALENDAR_CACHE[exchange] = ec.get_calendar(exchange)
        except Exception as e:
            logger.warning(f"Failed to load calendar for {exchange}: {e}")
            return None

    return _CALENDAR_CACHE[exchange]


def _normalize_date(date: Union[str, datetime, pd.Timestamp]) -> pd.Timestamp:
    """将日期统一转换为 pd.Timestamp"""
    if isinstance(date, str):
        return pd.Timestamp(date)
    if isinstance(date, datetime):
        return pd.Timestamp(date)
    return pd.Timestamp(date)


def is_trading_day(
    date: Union[str, datetime, pd.Timestamp],
    market: str = "cn",
) -> bool:
    """
    判断是否为交易日

    Args:
        date: 日期
        market: 市场 ('cn', 'xshg', 'xshe')

    Returns:
        是否为交易日
    """
    ts = _normalize_date(date)

    cal = _get_calendar(market)
    if cal is not None:
        try:
            return cal.is_session(ts)
        except Exception:
            logger.debug(f"exchange_calendars is_session failed for {ts}, falling back to weekday check")

    # 退化为周一至周五
    return ts.weekday() < 5


def get_trading_days(
    start: Union[str, datetime, pd.Timestamp],
    end: Union[str, datetime, pd.Timestamp],
    market: str = "cn",
) -> List[datetime]:
    """
    获取交易日列表

    Args:
        start: 开始日期
        end: 结束日期
        market: 市场

    Returns:
        交易日列表
    """
    start_ts = _normalize_date(start)
    end_ts = _normalize_date(end)

    cal = _get_calendar(market)
    if cal is not None:
        try:
            sessions = cal.sessions_in_range(start_ts, end_ts)
            return sessions.tolist()
        except Exception:
            logger.debug(f"exchange_calendars sessions_in_range failed for {start_ts}-{end_ts}, returning empty list")

    # 退化为工作日
    return pd.date_range(start_ts, end_ts, freq="B").tolist()


def get_previous_trading_day(
    date: Union[str, datetime, pd.Timestamp],
    n: int = 1,
    market: str = "cn",
) -> datetime:
    """
    获取前 n 个交易日

    Args:
        date: 基准日期
        n: 前几个交易日
        market: 市场

    Returns:
        前 n 个交易日
    """
    ts = _normalize_date(date)

    cal = _get_calendar(market)
    if cal is not None:
        try:
            # 从 date 往前找 n 个 session
            sessions = cal.sessions_in_range(
                ts - pd.Timedelta(days=n * 3 + 5),
                ts - pd.Timedelta(days=1),
            )
            if len(sessions) >= n:
                return sessions.iloc[-n].to_pydatetime()
        except Exception:
            logger.debug(f"exchange_calendars get_previous_trading_day failed for {ts}, n={n}")

    # 退化为逐天回退
    result = ts
    count = 0
    while count < n:
        result -= timedelta(days=1)
        if result.weekday() < 5:
            count += 1
    return result.to_pydatetime()


def get_next_trading_day(
    date: Union[str, datetime, pd.Timestamp],
    n: int = 1,
    market: str = "cn",
) -> datetime:
    """
    获取后 n 个交易日

    Args:
        date: 基准日期
        n: 后几个交易日
        market: 市场

    Returns:
        后 n 个交易日
    """
    ts = _normalize_date(date)

    cal = _get_calendar(market)
    if cal is not None:
        try:
            sessions = cal.sessions_in_range(
                ts + pd.Timedelta(days=1),
                ts + pd.Timedelta(days=n * 3 + 5),
            )
            if len(sessions) >= n:
                return sessions.iloc[n - 1].to_pydatetime()
        except Exception:
            logger.debug(f"exchange_calendars get_next_trading_day failed for {ts}, n={n}")

    # 退化为逐天前进
    result = ts
    count = 0
    while count < n:
        result += timedelta(days=1)
        if result.weekday() < 5:
            count += 1
    return result.to_pydatetime()
