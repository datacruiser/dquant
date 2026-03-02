"""
DQuant 工具函数

常用的辅助函数。
"""

from typing import List, Union, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dquant.constants import (
    DEFAULT_COMMISSION,
    DEFAULT_SLIPPAGE,
    DEFAULT_STAMP_DUTY,
    DEFAULT_INITIAL_CASH,
    MIN_SHARES,
    DEFAULT_WINDOW,
)


# ============================================================
# 日期工具
# ============================================================

def get_trading_days(
    start: Union[str, datetime],
    end: Union[str, datetime],
    market: str = 'cn',
) -> List[datetime]:
    """
    获取交易日列表

    Args:
        start: 开始日期
        end: 结束日期
        market: 市场 (cn, us)

    Returns:
        交易日列表
    """
    # 简化版本：周一到周五
    # 实际应用中应该使用交易日历
    if isinstance(start, str):
        start = pd.to_datetime(start)
    if isinstance(end, str):
        end = pd.to_datetime(end)

    days = pd.date_range(start, end, freq='B')  # 工作日
    return days.tolist()


def is_trading_day(date: Union[str, datetime], market: str = 'cn') -> bool:
    """判断是否为交易日"""
    if isinstance(date, str):
        date = pd.to_datetime(date)

    # 简化：周一到周五
    return date.weekday() < 5


def get_previous_trading_day(date: Union[str, datetime], n: int = 1) -> datetime:
    """获取前 n 个交易日"""
    if isinstance(date, str):
        date = pd.to_datetime(date)

    result = date
    count = 0

    while count < n:
        result -= timedelta(days=1)
        if is_trading_day(result):
            count += 1

    return result


def get_next_trading_day(date: Union[str, datetime], n: int = 1) -> datetime:
    """获取后 n 个交易日"""
    if isinstance(date, str):
        date = pd.to_datetime(date)

    result = date
    count = 0

    while count < n:
        result += timedelta(days=1)
        if is_trading_day(result):
            count += 1

    return result


# ============================================================
# 绩效分析工具
# ============================================================

def calculate_returns(
    nav: pd.Series,
    freq: str = 'D',
) -> pd.Series:
    """
    计算收益率序列

    Args:
        nav: 净值序列
        freq: 频率 (D=日, W=周, M=月)

    Returns:
        收益率序列
    """
    if freq == 'D':
        return nav.pct_change()
    elif freq == 'W':
        return nav.resample('W').last().pct_change()
    elif freq == 'M':
        return nav.resample('M').last().pct_change()
    else:
        return nav.pct_change()


def calculate_cumulative_returns(returns: pd.Series) -> pd.Series:
    """计算累积收益率"""
    return (1 + returns).cumprod() - 1


def annualized_return(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """
    计算年化收益率

    Args:
        returns: 收益率序列
        periods_per_year: 一年的交易周期数 (252=日, 52=周, 12=月)

    Returns:
        年化收益率
    """
    if len(returns) == 0:
        return 0.0

    cumulative = (1 + returns).prod()
    n_periods = len(returns)

    return cumulative ** (periods_per_year / n_periods) - 1


def annualized_volatility(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """
    计算年化波动率

    Args:
        returns: 收益率序列
        periods_per_year: 一年的交易周期数

    Returns:
        年化波动率
    """
    if len(returns) == 0:
        return 0.0

    return returns.std() * np.sqrt(periods_per_year)


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.03,
    periods_per_year: int = 252,
) -> float:
    """
    计算 Sharpe 比率

    Args:
        returns: 收益率序列
        risk_free_rate: 无风险利率 (年化)
        periods_per_year: 一年的交易周期数

    Returns:
        Sharpe 比率
    """
    ann_ret = annualized_return(returns, periods_per_year)
    ann_vol = annualized_volatility(returns, periods_per_year)

    if ann_vol == 0:
        return 0.0

    return (ann_ret - risk_free_rate) / ann_vol


def max_drawdown(nav: pd.Series) -> Tuple[float, datetime, datetime]:
    """
    计算最大回撤

    Args:
        nav: 净值序列

    Returns:
        (最大回撤, 开始日期, 结束日期)
    """
    cummax = nav.cummax()
    drawdown = (nav - cummax) / cummax

    max_dd = drawdown.min()
    end_idx = drawdown.idxmin()

    # 找到开始日期
    start_idx = nav[:end_idx].idxmax()

    return max_dd, start_idx, end_idx


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.03,
    periods_per_year: int = 252,
) -> float:
    """
    计算 Sortino 比率

    Args:
        returns: 收益率序列
        risk_free_rate: 无风险利率 (年化)
        periods_per_year: 一年的交易周期数

    Returns:
        Sortino 比率
    """
    ann_ret = annualized_return(returns, periods_per_year)

    # 下行波动率
    negative_returns = returns[returns < 0]
    if len(negative_returns) == 0:
        return float('inf')

    downside_std = negative_returns.std() * np.sqrt(periods_per_year)

    if downside_std == 0:
        return float('inf')

    return (ann_ret - risk_free_rate) / downside_std


def calmar_ratio(
    returns: pd.Series,
    periods_per_year: int = 252,
) -> float:
    """
    计算 Calmar 比率

    Args:
        returns: 收益率序列
        periods_per_year: 一年的交易周期数

    Returns:
        Calmar 比率
    """
    ann_ret = annualized_return(returns, periods_per_year)

    # 最大回撤
    nav = (1 + returns).cumprod()
    max_dd, _, _ = max_drawdown(nav)

    if max_dd == 0:
        return float('inf')

    return ann_ret / abs(max_dd)


# ============================================================
# 数据处理工具
# ============================================================

def winsorize(
    data: pd.Series,
    limits: Tuple[float, float] = (0.01, 0.01),
) -> pd.Series:
    """
    去极值

    Args:
        data: 数据序列
        limits: (下限, 上限) 分位数

    Returns:
        处理后的序列
    """
    lower = data.quantile(limits[0])
    upper = data.quantile(1 - limits[1])

    return data.clip(lower, upper)


def standardize(data: pd.Series) -> pd.Series:
    """标准化"""
    return (data - data.mean()) / data.std()


def normalize(data: pd.Series) -> pd.Series:
    """归一化到 [0, 1]"""
    return (data - data.min()) / (data.max() - data.min())


# ============================================================
# 其他工具
# ============================================================

def format_money(amount: float) -> str:
    """格式化金额"""
    if abs(amount) >= 1e8:
        return f"¥{amount/1e8:.2f}亿"
    elif abs(amount) >= 1e4:
        return f"¥{amount/1e4:.2f}万"
    else:
        return f"¥{amount:.2f}"


def format_percent(value: float) -> str:
    """格式化百分比"""
    return f"{value*MIN_SHARES:.2f}%"


def format_sharpe(sharpe: float) -> str:
    """格式化 Sharpe 比率"""
    if sharpe > 2:
        return f"{sharpe:.2f} (优秀)"
    elif sharpe > 1:
        return f"{sharpe:.2f} (良好)"
    elif sharpe > 0:
        return f"{sharpe:.2f} (一般)"
    else:
        return f"{sharpe:.2f} (较差)"
