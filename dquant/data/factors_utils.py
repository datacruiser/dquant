"""
共享因子计算工具

提供各 DataLoader 通用的技术因子计算逻辑，避免重复代码。
各 Loader 可在此基础上扩展自己特有的因子。
"""

from typing import List, Optional

import numpy as np
import pandas as pd


def calculate_common_factors(
    df: pd.DataFrame,
    symbol_col: str = "symbol",
    momentum_windows: Optional[List[int]] = None,
    volatility_windows: Optional[List[int]] = None,
    ma_windows: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    计算通用技术因子（向量化实现）

    对 DataFrame 按 symbol 分组，在每组内计算动量、波动率、均线等因子。

    Args:
        df: 含 close/volume 列的 DataFrame，需有 symbol 列和时间索引
        symbol_col: symbol 列名
        momentum_windows: 动量因子窗口列表
        volatility_windows: 波动率因子窗口列表
        ma_windows: 均线因子窗口列表

    Returns:
        添加了因子列的 DataFrame
    """
    momentum_windows = momentum_windows or [5, 10, 20]
    volatility_windows = volatility_windows or [5, 10, 20]
    ma_windows = ma_windows or [5, 10, 20]

    if "close" not in df.columns:
        return df

    def _group_factors(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_index()
        close = group["close"]

        # 动量因子
        for w in momentum_windows:
            group[f"momentum_{w}"] = close.pct_change(w)

        # 波动率因子
        returns = close.pct_change()
        for w in volatility_windows:
            group[f"volatility_{w}"] = returns.rolling(w).std()

        # 均线因子
        for w in ma_windows:
            group[f"ma_{w}"] = close.rolling(w).mean()
            group[f"bias_{w}"] = (close - group[f"ma_{w}"]) / group[f"ma_{w}"]

        # 成交量因子
        if "volume" in group.columns:
            group["volume_ma_5"] = group["volume"].rolling(5).mean()
            vol_ma5 = group["volume_ma_5"].replace(0, np.nan)
            group["volume_ratio"] = group["volume"] / vol_ma5

        return group

    return df.groupby(symbol_col, group_keys=False).apply(_group_factors)


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """计算 RSI"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
):
    """计算 MACD，返回 (macd, signal, histogram)"""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist


def calculate_bollinger(
    series: pd.Series,
    window: int = 20,
    num_std: float = 2.0,
):
    """计算布林带，返回 (upper, middle, lower)"""
    middle = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return upper, middle, lower
