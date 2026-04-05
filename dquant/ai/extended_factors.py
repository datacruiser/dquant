"""
扩展因子库

包含更多技术指标、量价关系、统计因子等。
"""

from typing import Optional

import numpy as np
import pandas as pd

from dquant.ai.base import BaseFactor
from dquant.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# 技术指标因子
# ============================================================


class ADXFactor(BaseFactor):
    """
    ADX (Average Directional Index)

    平均趋向指标，衡量趋势强度。
    """

    def __init__(self, window: int = 14, name: Optional[str] = None):
        super().__init__(name=name or f"ADX_{window}")
        self.window = window

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None):
        self._is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算 ADX"""
        results = []

        for symbol, group in data.groupby("symbol"):
            group = group.sort_index()

            high = group["high"]
            low = group["low"]
            close = group["close"]

            # 计算 +DM 和 -DM (Wilder's mutual exclusion rule)
            plus_dm = high.diff()
            minus_dm = -low.diff()
            # Wilder's rule: only keep the larger directional movement
            plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
            minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

            # 计算 TR
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            # 平滑
            atr = tr.rolling(self.window).mean()
            plus_di = 100 * (plus_dm.rolling(self.window).mean() / atr)
            minus_di = 100 * (minus_dm.rolling(self.window).mean() / atr)

            # DX
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)

            # ADX
            adx = dx.rolling(self.window).mean()

            for date, value in adx.items():
                if pd.notna(value):
                    results.append(
                        {
                            "date": date,
                            "symbol": symbol,
                            "score": value,
                        }
                    )

        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")

        return df


class AroonFactor(BaseFactor):
    """
    Aroon Indicator

    阿隆指标，衡量趋势的开始和强度。
    """

    def __init__(self, window: int = 25, name: Optional[str] = None):
        super().__init__(name=name or f"Aroon_{window}")
        self.window = window

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None):
        self._is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算 Aroon"""
        results = []

        for symbol, group in data.groupby("symbol"):
            group = group.sort_index()

            # Aroon Up
            aroon_up = (
                group["high"]
                .rolling(self.window)
                .apply(
                    lambda x: (self.window - (self.window - 1 - np.argmax(x))) / self.window * 100,
                    raw=False,
                )
            )

            # Aroon Down
            aroon_down = (
                group["low"]
                .rolling(self.window)
                .apply(
                    lambda x: (self.window - (self.window - 1 - np.argmin(x))) / self.window * 100,
                    raw=False,
                )
            )

            # Aroon Oscillator
            aroon = aroon_up - aroon_down

            for date, value in aroon.items():
                if pd.notna(value):
                    results.append(
                        {
                            "date": date,
                            "symbol": symbol,
                            "score": value,
                        }
                    )

        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")

        return df


class StochasticFactor(BaseFactor):
    """
    Stochastic Oscillator

    随机指标。
    """

    def __init__(self, k_window: int = 14, d_window: int = 3, name: Optional[str] = None):
        super().__init__(name=name or f"Stochastic_{k_window}")
        self.k_window = k_window
        self.d_window = d_window

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None):
        self._is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算 Stochastic"""
        results = []

        for symbol, group in data.groupby("symbol"):
            group = group.sort_index()

            low_min = group["low"].rolling(self.k_window).min()
            high_max = group["high"].rolling(self.k_window).max()

            # %K
            k = 100 * (group["close"] - low_min) / (high_max - low_min)

            # %D
            d = k.rolling(self.d_window).mean()

            for date, value in d.items():
                if pd.notna(value):
                    results.append(
                        {
                            "date": date,
                            "symbol": symbol,
                            "score": value,
                        }
                    )

        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")

        return df


class ROCFactor(BaseFactor):
    """
    Rate of Change

    变动率指标。
    """

    def __init__(self, window: int = 12, name: Optional[str] = None):
        super().__init__(name=name or f"ROC_{window}")
        self.window = window

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None):
        self._is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算 ROC"""
        results = []

        for symbol, group in data.groupby("symbol"):
            group = group.sort_index()
            roc = (
                (group["close"] - group["close"].shift(self.window))
                / group["close"].shift(self.window)
                * 100
            )

            for date, value in roc.items():
                if pd.notna(value):
                    results.append(
                        {
                            "date": date,
                            "symbol": symbol,
                            "score": value,
                        }
                    )

        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")

        return df


class CMOFactor(BaseFactor):
    """
    Chande Momentum Oscillator

    钱德动量摆动指标。
    """

    def __init__(self, window: int = 14, name: Optional[str] = None):
        super().__init__(name=name or f"CMO_{window}")
        self.window = window

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None):
        self._is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算 CMO"""
        results = []

        for symbol, group in data.groupby("symbol"):
            group = group.sort_index()

            diff = group["close"].diff()

            sum_up = diff.where(diff > 0, 0).rolling(self.window).sum()
            sum_down = abs(diff.where(diff < 0, 0).rolling(self.window).sum())

            cmo = 100 * (sum_up - sum_down) / (sum_up + sum_down)

            for date, value in cmo.items():
                if pd.notna(value):
                    results.append(
                        {
                            "date": date,
                            "symbol": symbol,
                            "score": value,
                        }
                    )

        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")

        return df


class MFIFactor(BaseFactor):
    """
    Money Flow Index

    资金流量指标。
    """

    def __init__(self, window: int = 14, name: Optional[str] = None):
        super().__init__(name=name or f"MFI_{window}")
        self.window = window

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None):
        self._is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算 MFI"""
        results = []

        for symbol, group in data.groupby("symbol"):
            group = group.sort_index()

            # Typical Price
            tp = (group["high"] + group["low"] + group["close"]) / 3

            # Money Flow
            mf = tp * group["volume"]

            # Positive/Negative Money Flow
            diff = tp.diff()

            pmf = mf.where(diff > 0, 0).rolling(self.window).sum()
            nmf = mf.where(diff < 0, 0).rolling(self.window).sum()

            # MFI
            mfi = 100 - 100 / (1 + pmf / nmf)

            for date, value in mfi.items():
                if pd.notna(value):
                    results.append(
                        {
                            "date": date,
                            "symbol": symbol,
                            "score": value,
                        }
                    )

        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")

        return df


# ============================================================
# 量价关系因子
# ============================================================


class ADLineFactor(BaseFactor):
    """
    Accumulation/Distribution Line

    累积/派发线。
    """

    def __init__(self, name: str = "ADLine"):
        super().__init__(name=name)

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None):
        self._is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算 A/D Line"""
        results = []

        for symbol, group in data.groupby("symbol"):
            group = group.sort_index()

            # CLV (Close Location Value)
            clv = ((group["close"] - group["low"]) - (group["high"] - group["close"])) / (
                group["high"] - group["low"]
            )

            clv = clv.fillna(0)

            # A/D Line
            ad = (clv * group["volume"]).cumsum()

            # 使用变化率作为因子
            ad_change = ad.pct_change()

            for date, value in ad_change.items():
                if pd.notna(value):
                    results.append(
                        {
                            "date": date,
                            "symbol": symbol,
                            "score": value,
                        }
                    )

        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")

        return df


class ChaikinOscillatorFactor(BaseFactor):
    """
    Chaikin Oscillator

    佳庆振荡器。
    """

    def __init__(self, fast: int = 3, slow: int = 10, name: Optional[str] = None):
        super().__init__(name=name or f"ChaikinOsc_{fast}_{slow}")
        self.fast = fast
        self.slow = slow

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None):
        self._is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算 Chaikin Oscillator"""
        results = []

        for symbol, group in data.groupby("symbol"):
            group = group.sort_index()

            # CLV
            clv = ((group["close"] - group["low"]) - (group["high"] - group["close"])) / (
                group["high"] - group["low"]
            )
            clv = clv.fillna(0)

            # AD
            ad = (clv * group["volume"]).cumsum()

            # Chaikin Oscillator
            co = ad.ewm(span=self.fast).mean() - ad.ewm(span=self.slow).mean()

            for date, value in co.items():
                if pd.notna(value):
                    results.append(
                        {
                            "date": date,
                            "symbol": symbol,
                            "score": value,
                        }
                    )

        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")

        return df


class EaseOfMovementFactor(BaseFactor):
    """
    Ease of Movement

    简易波动指标。
    """

    def __init__(self, window: int = 14, name: Optional[str] = None):
        super().__init__(name=name or f"EOM_{window}")
        self.window = window

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None):
        self._is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算 EOM"""
        results = []

        for symbol, group in data.groupby("symbol"):
            group = group.sort_index()

            # Distance Moved
            dm = (group["high"] + group["low"]) / 2 - (
                group["high"].shift() + group["low"].shift()
            ) / 2

            # Box Ratio
            br = (group["volume"] / 100000000) / (group["high"] - group["low"])

            # EOM
            eom = dm / br
            eom_ma = eom.rolling(self.window).mean()

            for date, value in eom_ma.items():
                if pd.notna(value):
                    results.append(
                        {
                            "date": date,
                            "symbol": symbol,
                            "score": value,
                        }
                    )

        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")

        return df


class ForceIndexFactor(BaseFactor):
    """
    Force Index

    强力指数。
    """

    def __init__(self, window: int = 13, name: Optional[str] = None):
        super().__init__(name=name or f"ForceIndex_{window}")
        self.window = window

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None):
        self._is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算 Force Index"""
        results = []

        for symbol, group in data.groupby("symbol"):
            group = group.sort_index()

            # Force Index
            fi = group["close"].diff() * group["volume"]

            # Smoothed
            fi_ma = fi.ewm(span=self.window).mean()

            for date, value in fi_ma.items():
                if pd.notna(value):
                    results.append(
                        {
                            "date": date,
                            "symbol": symbol,
                            "score": value,
                        }
                    )

        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")

        return df


class VPTFactor(BaseFactor):
    """
    Volume Price Trend

    量价趋势指标。
    """

    def __init__(self, name: str = "VPT"):
        super().__init__(name=name)

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None):
        self._is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算 VPT"""
        results = []

        for symbol, group in data.groupby("symbol"):
            group = group.sort_index()

            # VPT
            vpt = (group["volume"] * group["close"].pct_change()).cumsum()

            # 使用变化率
            vpt_change = vpt.pct_change()

            for date, value in vpt_change.items():
                if pd.notna(value):
                    results.append(
                        {
                            "date": date,
                            "symbol": symbol,
                            "score": value,
                        }
                    )

        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")

        return df


# ============================================================
# 统计因子
# ============================================================


class HurstExponentFactor(BaseFactor):
    """
    Hurst Exponent

    赫斯特指数，衡量时间序列的长期记忆性。
    """

    def __init__(self, window: int = 100, name: Optional[str] = None):
        super().__init__(name=name or f"Hurst_{window}")
        self.window = window

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None):
        self._is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算 Hurst Exponent"""
        results = []

        for symbol, group in data.groupby("symbol"):
            group = group.sort_index()
            prices = group["close"]

            for i in range(self.window, len(prices)):
                window_data = prices.iloc[i - self.window : i]

                # R/S 分析
                lags = range(2, min(50, self.window))
                tau = [np.std(np.subtract(window_data[lag:], window_data[:-lag])) for lag in lags]

                # 对数回归
                try:
                    # 过滤掉0和负数
                    lags_arr = np.array(lags)
                    tau_arr = np.array(tau)
                    mask = (lags_arr > 0) & (tau_arr > 0)
                    if not mask.any():
                        continue
                    reg = np.polyfit(np.log(lags_arr[mask]), np.log(tau_arr[mask]), 1)
                    hurst = reg[0]

                    results.append(
                        {
                            "date": prices.index[i],
                            "symbol": symbol,
                            "score": hurst,
                        }
                    )
                except Exception as e:
                    logger.warning(f"Operation failed: {e}")

        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")

        return df


class AutocorrelationFactor(BaseFactor):
    """
    Autocorrelation

    自相关系数。
    """

    def __init__(self, window: int = 20, lag: int = 1, name: Optional[str] = None):
        super().__init__(name=name or f"Autocorr_{window}")
        self.window = window
        self.lag = lag

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None):
        self._is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算自相关系数"""
        results = []

        for symbol, group in data.groupby("symbol"):
            group = group.sort_index()
            returns = group["close"].pct_change()

            autocorr = returns.rolling(self.window).apply(
                lambda x: x.autocorr(self.lag) if len(x) > self.lag else np.nan
            )

            for date, value in autocorr.items():
                if pd.notna(value):
                    results.append(
                        {
                            "date": date,
                            "symbol": symbol,
                            "score": value,
                        }
                    )

        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")

        return df


class VarianceRatioFactor(BaseFactor):
    """
    Variance Ratio

    方差比率检验。
    """

    def __init__(self, window: int = 20, name: Optional[str] = None):
        super().__init__(name=name or f"VR_{window}")
        self.window = window

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None):
        self._is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算方差比率"""
        results = []

        for symbol, group in data.groupby("symbol"):
            group = group.sort_index()
            returns = group["close"].pct_change()

            for i in range(self.window * 2, len(returns)):
                window_data = returns.iloc[i - self.window * 2 : i]

                try:
                    # 1期方差
                    var_1 = window_data.var()

                    # 2期方差
                    var_2 = window_data.rolling(2).sum().var()

                    # VR
                    vr = var_2 / (2 * var_1) if var_1 > 0 else 1

                    results.append(
                        {
                            "date": returns.index[i],
                            "symbol": symbol,
                            "score": vr,
                        }
                    )
                except Exception as e:
                    logger.warning(f"Operation failed: {e}")

        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")

        return df


class BetaFactor(BaseFactor):
    """
    Beta

    相对基准的 Beta 系数。
    """

    def __init__(self, window: int = 60, name: Optional[str] = None):
        super().__init__(name=name or f"Beta_{window}")
        self.window = window

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None):
        self._is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算 Beta"""
        results = []

        # 计算市场收益率 (等权平均)
        market_returns = data.groupby(data.index)["close"].apply(lambda x: x.pct_change().mean())

        for symbol, group in data.groupby("symbol"):
            group = group.sort_index()
            returns = group["close"].pct_change()

            # 对齐市场收益率
            aligned_market = market_returns.reindex(returns.index)

            # 计算滚动 Beta
            for i in range(self.window, len(returns)):
                stock_ret = returns.iloc[i - self.window : i]
                market_ret = aligned_market.iloc[i - self.window : i]

                if len(stock_ret) == len(market_ret) and len(stock_ret) > 0:
                    try:
                        covariance = stock_ret.cov(market_ret)
                        variance = market_ret.var()

                        beta = covariance / variance if variance > 0 else 1

                        results.append(
                            {
                                "date": returns.index[i],
                                "symbol": symbol,
                                "score": beta,
                            }
                        )
                    except Exception as e:
                        logger.warning(f"Operation failed: {e}")

        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")

        return df


class AlphaFactor(BaseFactor):
    """
    Alpha

    相对基准的超额收益。
    """

    def __init__(self, window: int = 60, name: Optional[str] = None):
        super().__init__(name=name or f"Alpha_{window}")
        self.window = window

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None):
        self._is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算 Alpha"""
        results = []

        # 计算市场收益率
        market_returns = data.groupby(data.index)["close"].apply(lambda x: x.pct_change().mean())

        for symbol, group in data.groupby("symbol"):
            group = group.sort_index()
            returns = group["close"].pct_change()

            aligned_market = market_returns.reindex(returns.index)

            # 计算滚动 Alpha
            for i in range(self.window, len(returns)):
                stock_ret = returns.iloc[i - self.window : i]
                market_ret = aligned_market.iloc[i - self.window : i]

                if len(stock_ret) == len(market_ret) and len(stock_ret) > 0:
                    try:
                        alpha = stock_ret.mean() - market_ret.mean()

                        results.append(
                            {
                                "date": returns.index[i],
                                "symbol": symbol,
                                "score": alpha,
                            }
                        )
                    except Exception as e:
                        logger.warning(f"Operation failed: {e}")

        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")

        return df


# ============================================================
# 注册扩展因子
# ============================================================

EXTENDED_FACTORS = {
    # 技术指标
    "adx": ADXFactor,
    "aroon": AroonFactor,
    "stochastic": StochasticFactor,
    "roc": ROCFactor,
    "cmo": CMOFactor,
    "mfi": MFIFactor,
    # 量价关系
    "ad_line": ADLineFactor,
    "chaikin_osc": ChaikinOscillatorFactor,
    "eom": EaseOfMovementFactor,
    "force_index": ForceIndexFactor,
    "vpt": VPTFactor,
    # 统计因子
    "hurst": HurstExponentFactor,
    "autocorr": AutocorrelationFactor,
    "variance_ratio": VarianceRatioFactor,
    "beta": BetaFactor,
    "alpha": AlphaFactor,
}
