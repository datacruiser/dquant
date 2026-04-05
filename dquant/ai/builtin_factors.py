"""
DQuant 内置因子库

包含常用的技术因子、基本面因子、情绪因子等。

Author: DQuant Team
Version: 0.1.0
"""

from typing import List, Optional

import numpy as np
import pandas as pd

from dquant.ai.base import BaseFactor

# RSI/ADX/KDJ/Williams 等技术指标公式中的常量 100 直接使用字面值，
# 避免与 A 股最小交易单位 MIN_SHARES 混淆


# ============================================================
# 动量类因子
# ============================================================


class MomentumFactor(BaseFactor):
    """动量因子 - 过去 N 天的收益率"""

    def __init__(self, window: int = 20, name: Optional[str] = None):
        super().__init__(name=name or f"Momentum_{window}")
        self.window = window

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None):
        self._is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        for symbol, group in data.groupby("symbol"):
            group = group.sort_index()
            momentum = group["close"].pct_change(self.window, fill_method=None)
            for date, value in momentum.items():
                if pd.notna(value):
                    results.append({"date": date, "symbol": symbol, "score": value})

        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df


class ReversalFactor(BaseFactor):
    """反转因子 - 短期反转"""

    def __init__(self, window: int = 5, name: Optional[str] = None):
        super().__init__(name=name or f"Reversal_{window}")
        self.window = window

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None):
        self._is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        for symbol, group in data.groupby("symbol"):
            group = group.sort_index()
            ret = group["close"].pct_change(self.window)
            reversal = -ret  # 反转
            for date, value in reversal.items():
                if pd.notna(value):
                    results.append({"date": date, "symbol": symbol, "score": value})

        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df


class AccMomentumFactor(BaseFactor):
    """累积动量因子 - 累积收益率"""

    def __init__(self, window: int = 20, name: Optional[str] = None):
        super().__init__(name=name or f"AccMomentum_{window}")
        self.window = window

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None):
        self._is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        for symbol, group in data.groupby("symbol"):
            group = group.sort_index()
            ret = group["close"].pct_change()
            acc_ret = (1 + ret).rolling(self.window).apply(lambda x: x.prod(), raw=True) - 1
            for date, value in acc_ret.items():
                if pd.notna(value):
                    results.append({"date": date, "symbol": symbol, "score": value})

        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df


# ============================================================
# 波动率类因子
# ============================================================


class VolatilityFactor(BaseFactor):
    """波动率因子 - 收益率标准差"""

    def __init__(self, window: int = 20, prefer_low: bool = True, name: Optional[str] = None):
        super().__init__(name=name or f"Volatility_{window}")
        self.window = window
        self.prefer_low = prefer_low

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None):
        self._is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        for symbol, group in data.groupby("symbol"):
            group = group.sort_index()
            returns = group["close"].pct_change()
            volatility = returns.rolling(self.window).std()
            if self.prefer_low:
                volatility = -volatility
            for date, value in volatility.items():
                if pd.notna(value):
                    results.append({"date": date, "symbol": symbol, "score": value})

        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df


class ATRFactor(BaseFactor):
    """ATR 因子 - Average True Range"""

    def __init__(self, window: int = 14, name: Optional[str] = None):
        super().__init__(name=name or f"ATR_{window}")
        self.window = window

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None):
        self._is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        for symbol, group in data.groupby("symbol"):
            group = group.sort_index()
            high, low, close = group["high"], group["low"], group["close"]
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(self.window).mean()
            atr_ratio = atr / close
            for date, value in atr_ratio.items():
                if pd.notna(value):
                    results.append({"date": date, "symbol": symbol, "score": -value})

        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df


class SkewnessFactor(BaseFactor):
    """偏度因子 - 收益率偏度"""

    def __init__(self, window: int = 20, name: Optional[str] = None):
        super().__init__(name=name or f"Skewness_{window}")
        self.window = window

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None):
        self._is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        for symbol, group in data.groupby("symbol"):
            group = group.sort_index()
            returns = group["close"].pct_change()
            skewness = returns.rolling(self.window).skew()
            for date, value in skewness.items():
                if pd.notna(value):
                    results.append({"date": date, "symbol": symbol, "score": -value})  # 负偏度偏好

        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df


class KurtosisFactor(BaseFactor):
    """峰度因子 - 收益率峰度"""

    def __init__(self, window: int = 20, name: Optional[str] = None):
        super().__init__(name=name or f"Kurtosis_{window}")
        self.window = window

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None):
        self._is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        for symbol, group in data.groupby("symbol"):
            group = group.sort_index()
            returns = group["close"].pct_change()
            kurtosis = returns.rolling(self.window).kurt()
            for date, value in kurtosis.items():
                if pd.notna(value):
                    results.append({"date": date, "symbol": symbol, "score": -value})  # 低峰度偏好

        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df


class MaxDrawdownFactor(BaseFactor):
    """最大回撤因子"""

    def __init__(self, window: int = 20, name: Optional[str] = None):
        super().__init__(name=name or f"MaxDrawdown_{window}")
        self.window = window

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None):
        self._is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        for symbol, group in data.groupby("symbol"):
            group = group.sort_index()
            cummax = group["close"].rolling(self.window).max()
            drawdown = (group["close"] - cummax) / cummax
            max_dd = drawdown.rolling(self.window).min()
            for date, value in max_dd.items():
                if pd.notna(value):
                    results.append({"date": date, "symbol": symbol, "score": -value})  # 小回撤偏好

        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df


# ============================================================
# 技术指标因子
# ============================================================


class RSIFactor(BaseFactor):
    """RSI 因子 - 相对强弱指标"""

    def __init__(self, window: int = 14, name: Optional[str] = None):
        super().__init__(name=name or f"RSI_{window}")
        self.window = window

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None):
        self._is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        for symbol, group in data.groupby("symbol"):
            group = group.sort_index()
            delta = group["close"].diff()
            gain = delta.where(delta > 0, 0).ewm(alpha=1 / self.window, adjust=False).mean()
            loss = (-delta.where(delta < 0, 0)).ewm(alpha=1 / self.window, adjust=False).mean()
            rs = gain / loss.replace(0, float("nan"))
            rsi = 100 - (100 / (1 + rs))
            factor = 50 - rsi  # RSI 越低分数越高
            for date, value in factor.items():
                if pd.notna(value):
                    results.append({"date": date, "symbol": symbol, "score": value})

        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df


class MACDFactor(BaseFactor):
    """MACD 因子"""

    def __init__(
        self,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        name: Optional[str] = None,
    ):
        super().__init__(name=name or f"MACD_{fast}_{slow}")
        self.fast = fast
        self.slow = slow
        self.signal = signal

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None):
        self._is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        for symbol, group in data.groupby("symbol"):
            group = group.sort_index()
            ema_fast = group["close"].ewm(span=self.fast, adjust=False).mean()
            ema_slow = group["close"].ewm(span=self.slow, adjust=False).mean()
            macd = ema_fast - ema_slow
            signal_line = macd.ewm(span=self.signal, adjust=False).mean()
            histogram = macd - signal_line
            for date, value in histogram.items():
                if pd.notna(value):
                    results.append({"date": date, "symbol": symbol, "score": value})

        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df


class BollingerPositionFactor(BaseFactor):
    """布林带位置因子"""

    def __init__(self, window: int = 20, num_std: float = 2.0, name: Optional[str] = None):
        super().__init__(name=name or f"Bollinger_{window}")
        self.window = window
        self.num_std = num_std

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None):
        self._is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        for symbol, group in data.groupby("symbol"):
            group = group.sort_index()
            middle = group["close"].rolling(self.window).mean()
            std = group["close"].rolling(self.window).std()
            upper = middle + self.num_std * std
            lower = middle - self.num_std * std
            band_width = upper - lower
            position = (group["close"] - lower) / band_width.replace(0, float("nan"))
            factor = 0.5 - position  # 越接近下轨分数越高
            for date, value in factor.items():
                if pd.notna(value):
                    results.append({"date": date, "symbol": symbol, "score": value})

        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df


class TrendStrengthFactor(BaseFactor):
    """趋势强度因子 - ADX"""

    def __init__(self, window: int = 14, name: Optional[str] = None):
        super().__init__(name=name or f"TrendStrength_{window}")
        self.window = window

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None):
        self._is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        for symbol, group in data.groupby("symbol"):
            group = group.sort_index()
            high, low, close = group["high"], group["low"], group["close"]

            plus_dm = high.diff()
            minus_dm = -low.diff()
            plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
            minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

            tr = pd.concat(
                [high - low, abs(high - close.shift(1)), abs(low - close.shift(1))],
                axis=1,
            ).max(axis=1)
            atr = tr.rolling(self.window).mean()
            plus_di = 100 * (plus_dm.rolling(self.window).mean() / atr.replace(0, float("nan")))
            minus_di = 100 * (minus_dm.rolling(self.window).mean() / atr.replace(0, float("nan")))
            di_sum = plus_di + minus_di
            dx = 100 * abs(plus_di - minus_di) / di_sum.replace(0, float("nan"))

            for date, value in dx.items():
                if pd.notna(value):
                    results.append({"date": date, "symbol": symbol, "score": value})

        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df


class KDJFactor(BaseFactor):
    """KDJ 因子"""

    def __init__(self, n: int = 9, m1: int = 3, m2: int = 3, name: Optional[str] = None):
        super().__init__(name=name or f"KDJ_{n}")
        self.n = n
        self.m1 = m1
        self.m2 = m2

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None):
        self._is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        for symbol, group in data.groupby("symbol"):
            group = group.sort_index()

            low_n = group["low"].rolling(self.n).min()
            high_n = group["high"].rolling(self.n).max()

            price_range = high_n - low_n
            rsv = (group["close"] - low_n) / price_range.replace(0, float("nan")) * 100
            k = rsv.ewm(alpha=1 / self.m1, adjust=False).mean()
            d = k.ewm(alpha=1 / self.m2, adjust=False).mean()
            j = 3 * k - 2 * d

            # J 值作为因子
            for date, value in j.items():
                if pd.notna(value):
                    results.append(
                        {"date": date, "symbol": symbol, "score": 50 - value}
                    )  # J < 50 偏多

        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df


class CCIFactor(BaseFactor):
    """CCI 因子 - 顺势指标"""

    def __init__(self, window: int = 14, name: Optional[str] = None):
        super().__init__(name=name or f"CCI_{window}")
        self.window = window

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None):
        self._is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        for symbol, group in data.groupby("symbol"):
            group = group.sort_index()

            tp = (group["high"] + group["low"] + group["close"]) / 3
            ma = tp.rolling(self.window).mean()
            md = tp.rolling(self.window).apply(lambda x: np.abs(x - x.mean()).mean())

            cci = (tp - ma) / (0.015 * md.replace(0, float("nan")))

            # CCI 反转
            for date, value in cci.items():
                if pd.notna(value):
                    results.append({"date": date, "symbol": symbol, "score": -value})

        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df


class WilliamsRFactor(BaseFactor):
    """威廉指标因子"""

    def __init__(self, window: int = 14, name: Optional[str] = None):
        super().__init__(name=name or f"WilliamsR_{window}")
        self.window = window

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None):
        self._is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        for symbol, group in data.groupby("symbol"):
            group = group.sort_index()

            high_n = group["high"].rolling(self.window).max()
            low_n = group["low"].rolling(self.window).min()

            price_range = high_n - low_n
            wr = (high_n - group["close"]) / price_range.replace(0, float("nan")) * -100

            # WR 反转
            for date, value in wr.items():
                if pd.notna(value):
                    results.append({"date": date, "symbol": symbol, "score": -value})

        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df


# ============================================================
# 成交量因子
# ============================================================


class VolumeRatioFactor(BaseFactor):
    """量比因子"""

    def __init__(self, window: int = 5, name: Optional[str] = None):
        super().__init__(name=name or f"VolumeRatio_{window}")
        self.window = window

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None):
        self._is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        for symbol, group in data.groupby("symbol"):
            group = group.sort_index()
            vol_ma = group["volume"].rolling(self.window).mean()
            vol_ratio = group["volume"] / vol_ma
            for date, value in vol_ratio.items():
                if pd.notna(value):
                    results.append({"date": date, "symbol": symbol, "score": value})

        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df


class TurnoverRateFactor(BaseFactor):
    """换手率因子"""

    def __init__(self, window: int = 5, name: Optional[str] = None):
        super().__init__(name=name or f"TurnoverRate_{window}")
        self.window = window

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None):
        self._is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        for symbol, group in data.groupby("symbol"):
            group = group.sort_index()
            # 假设有 turnover 字段
            if "turnover" in group.columns:
                turnover = group["turnover"]
            else:
                # 用成交量近似
                turnover = group["volume"] / group["volume"].rolling(self.window).mean()

            avg_turnover = turnover.rolling(self.window).mean()

            for date, value in avg_turnover.items():
                if pd.notna(value):
                    results.append(
                        {"date": date, "symbol": symbol, "score": -value}
                    )  # 低换手率偏好

        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df


class OBVFactor(BaseFactor):
    """OBV 因子 - 能量潮"""

    def __init__(self, window: int = 20, name: Optional[str] = None):
        super().__init__(name=name or f"OBV_{window}")
        self.window = window

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None):
        self._is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        for symbol, group in data.groupby("symbol"):
            group = group.sort_index()

            direction = np.sign(group["close"].diff())
            obv = (direction * group["volume"]).cumsum()
            obv_ma = obv.rolling(self.window).mean()

            # OBV 与价格背离
            obv_trend = obv - obv_ma

            for date, value in obv_trend.items():
                if pd.notna(value):
                    results.append({"date": date, "symbol": symbol, "score": value})

        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df


class VWAPFactor(BaseFactor):
    """VWAP 因子 - 成交量加权均价"""

    def __init__(self, window: int = 20, name: Optional[str] = None):
        super().__init__(name=name or f"VWAP_{window}")
        self.window = window

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None):
        self._is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        for symbol, group in data.groupby("symbol"):
            group = group.sort_index()

            # 简化 VWAP
            typical_price = (group["high"] + group["low"] + group["close"]) / 3
            vol_sum = group["volume"].rolling(self.window).sum()
            vwap = (typical_price * group["volume"]).rolling(self.window).sum() / vol_sum.replace(
                0, float("nan")
            )

            # 价格与 VWAP 的偏离
            deviation = (group["close"] - vwap) / vwap.replace(0, float("nan"))

            for date, value in deviation.items():
                if pd.notna(value):
                    results.append(
                        {"date": date, "symbol": symbol, "score": -value}
                    )  # 低于 VWAP 偏好

        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df


# ============================================================
# 价格形态因子
# ============================================================


class PricePositionFactor(BaseFactor):
    """价格位置因子"""

    def __init__(self, window: int = 20, name: Optional[str] = None):
        super().__init__(name=name or f"PricePosition_{window}")
        self.window = window

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None):
        self._is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        for symbol, group in data.groupby("symbol"):
            group = group.sort_index()
            rolling_high = group["high"].rolling(self.window).max()
            rolling_low = group["low"].rolling(self.window).min()
            position = (group["close"] - rolling_low) / (rolling_high - rolling_low)
            factor = 1 - position
            for date, value in factor.items():
                if pd.notna(value):
                    results.append({"date": date, "symbol": symbol, "score": value})

        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df


class GapFactor(BaseFactor):
    """跳空因子"""

    def __init__(self, name: str = "Gap"):
        super().__init__(name=name)

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None):
        self._is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        for symbol, group in data.groupby("symbol"):
            group = group.sort_index()
            gap = (group["open"] - group["close"].shift(1)) / group["close"].shift(1)
            for date, value in gap.items():
                if pd.notna(value):
                    results.append({"date": date, "symbol": symbol, "score": -abs(value)})

        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df


class IntradayReturnFactor(BaseFactor):
    """日内收益因子"""

    def __init__(self, name: str = "IntradayReturn"):
        super().__init__(name=name)

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None):
        self._is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        for symbol, group in data.groupby("symbol"):
            group = group.sort_index()
            intraday = (group["close"] - group["open"]) / group["open"]
            factor = -intraday
            for date, value in factor.items():
                if pd.notna(value):
                    results.append({"date": date, "symbol": symbol, "score": value})

        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df


class OvernightReturnFactor(BaseFactor):
    """隔夜收益因子"""

    def __init__(self, name: str = "OvernightReturn"):
        super().__init__(name=name)

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None):
        self._is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        for symbol, group in data.groupby("symbol"):
            group = group.sort_index()
            overnight = (group["open"] - group["close"].shift(1)) / group["close"].shift(1)
            for date, value in overnight.items():
                if pd.notna(value):
                    results.append({"date": date, "symbol": symbol, "score": -value})

        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df


# ============================================================
# 均线因子
# ============================================================


class MASlopeFactor(BaseFactor):
    """均线斜率因子"""

    def __init__(self, window: int = 20, slope_window: int = 5, name: Optional[str] = None):
        super().__init__(name=name or f"MASlope_{window}")
        self.window = window
        self.slope_window = slope_window

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None):
        self._is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        for symbol, group in data.groupby("symbol"):
            group = group.sort_index()
            ma = group["close"].rolling(self.window).mean()
            slope = (ma - ma.shift(self.slope_window)) / ma.shift(self.slope_window)
            for date, value in slope.items():
                if pd.notna(value):
                    results.append({"date": date, "symbol": symbol, "score": value})

        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df


class MACrossFactor(BaseFactor):
    """均线交叉因子"""

    def __init__(self, short: int = 5, long: int = 20, name: Optional[str] = None):
        super().__init__(name=name or f"MACross_{short}_{long}")
        self.short = short
        self.long = long

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None):
        self._is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        for symbol, group in data.groupby("symbol"):
            group = group.sort_index()
            ma_short = group["close"].rolling(self.short).mean()
            ma_long = group["close"].rolling(self.long).mean()
            cross = (ma_short - ma_long) / ma_long
            for date, value in cross.items():
                if pd.notna(value):
                    results.append({"date": date, "symbol": symbol, "score": value})

        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df


class BiasFactor(BaseFactor):
    """乖离率因子"""

    def __init__(self, window: int = 20, name: Optional[str] = None):
        super().__init__(name=name or f"Bias_{window}")
        self.window = window

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None):
        self._is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        for symbol, group in data.groupby("symbol"):
            group = group.sort_index()
            ma = group["close"].rolling(self.window).mean()
            bias = (group["close"] - ma) / ma
            factor = -bias
            for date, value in factor.items():
                if pd.notna(value):
                    results.append({"date": date, "symbol": symbol, "score": value})

        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df


# ============================================================
# 基本面因子 (需要额外数据)
# ============================================================


class PERatioFactor(BaseFactor):
    """PE 因子 - 市盈率"""

    def __init__(self, name: str = "PE"):
        super().__init__(name=name)

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None):
        self._is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        if "pe" in data.columns:
            for idx, row in data.iterrows():
                if pd.notna(row["pe"]) and row["pe"] > 0:
                    results.append(
                        {"date": idx, "symbol": row["symbol"], "score": -row["pe"]}
                    )  # 低 PE 偏好

        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df


class PBRatioFactor(BaseFactor):
    """PB 因子 - 市净率"""

    def __init__(self, name: str = "PB"):
        super().__init__(name=name)

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None):
        self._is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        if "pb" in data.columns:
            for idx, row in data.iterrows():
                if pd.notna(row["pb"]) and row["pb"] > 0:
                    results.append(
                        {"date": idx, "symbol": row["symbol"], "score": -row["pb"]}
                    )  # 低 PB 偏好

        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df


class ROEFactor(BaseFactor):
    """ROE 因子 - 净资产收益率"""

    def __init__(self, name: str = "ROE"):
        super().__init__(name=name)

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None):
        self._is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        if "roe" in data.columns:
            for idx, row in data.iterrows():
                if pd.notna(row["roe"]):
                    results.append(
                        {"date": idx, "symbol": row["symbol"], "score": row["roe"]}
                    )  # 高 ROE 偏好

        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df


class RevenueGrowthFactor(BaseFactor):
    """营收增长率因子"""

    def __init__(self, name: str = "RevenueGrowth"):
        super().__init__(name=name)

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None):
        self._is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        if "revenue" in data.columns:
            for symbol, group in data.groupby("symbol"):
                group = group.sort_index()
                growth = group["revenue"].pct_change(4)  # 同比增长
                for date, value in growth.items():
                    if pd.notna(value):
                        results.append({"date": date, "symbol": symbol, "score": value})

        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df


class ProfitGrowthFactor(BaseFactor):
    """利润增长率因子"""

    def __init__(self, name: str = "ProfitGrowth"):
        super().__init__(name=name)

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None):
        self._is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        if "profit" in data.columns:
            for symbol, group in data.groupby("symbol"):
                group = group.sort_index()
                growth = group["profit"].pct_change(4)
                for date, value in growth.items():
                    if pd.notna(value):
                        results.append({"date": date, "symbol": symbol, "score": value})

        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df


class MarketCapFactor(BaseFactor):
    """市值因子"""

    def __init__(self, name: str = "MarketCap"):
        super().__init__(name=name)

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None):
        self._is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        if "market_cap" in data.columns:
            for idx, row in data.iterrows():
                if pd.notna(row["market_cap"]):
                    # 小市值偏好
                    results.append(
                        {
                            "date": idx,
                            "symbol": row["symbol"],
                            "score": -np.log(row["market_cap"]),
                        }
                    )

        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df


# ============================================================
# 情绪因子
# ============================================================


class MoneyFlowFactor(BaseFactor):
    """资金流向因子"""

    def __init__(self, window: int = 5, name: Optional[str] = None):
        super().__init__(name=name or f"MoneyFlow_{window}")
        self.window = window

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None):
        self._is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        results = []

        # 需要有 net_inflow 字段
        if "net_inflow" in data.columns:
            for symbol, group in data.groupby("symbol"):
                group = group.sort_index()
                flow = group["net_inflow"].rolling(self.window).mean()
                for date, value in flow.items():
                    if pd.notna(value):
                        results.append({"date": date, "symbol": symbol, "score": value})
        else:
            # 用成交量变化近似
            for symbol, group in data.groupby("symbol"):
                group = group.sort_index()
                vol_change = group["volume"].pct_change()
                price_change = group["close"].pct_change()
                flow = vol_change * price_change  # 量价配合
                flow_ma = flow.rolling(self.window).mean()
                for date, value in flow_ma.items():
                    if pd.notna(value):
                        results.append({"date": date, "symbol": symbol, "score": value})

        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df


class AmihudIlliquidityFactor(BaseFactor):
    """Amihud 非流动性因子"""

    def __init__(self, window: int = 20, name: Optional[str] = None):
        super().__init__(name=name or f"Amihud_{window}")
        self.window = window

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None):
        self._is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        for symbol, group in data.groupby("symbol"):
            group = group.sort_index()

            ret = abs(group["close"].pct_change())
            volume = group["volume"]

            # Amihud = |ret| / volume
            illiq = ret / (volume + 1)
            illiq_ma = illiq.rolling(self.window).mean()

            for date, value in illiq_ma.items():
                if pd.notna(value):
                    results.append(
                        {"date": date, "symbol": symbol, "score": -value}
                    )  # 高流动性偏好

        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df


# ============================================================
# 因子注册表
# ============================================================

FACTOR_REGISTRY = {
    # 动量类
    "momentum": MomentumFactor,
    "reversal": ReversalFactor,
    "acc_momentum": AccMomentumFactor,
    # 波动率类
    "volatility": VolatilityFactor,
    "atr": ATRFactor,
    "skewness": SkewnessFactor,
    "kurtosis": KurtosisFactor,
    "max_drawdown": MaxDrawdownFactor,
    # 技术指标
    "rsi": RSIFactor,
    "macd": MACDFactor,
    "bollinger": BollingerPositionFactor,
    "trend": TrendStrengthFactor,
    "kdj": KDJFactor,
    "cci": CCIFactor,
    "williams_r": WilliamsRFactor,
    # 成交量
    "volume_ratio": VolumeRatioFactor,
    "turnover_rate": TurnoverRateFactor,
    "obv": OBVFactor,
    "vwap": VWAPFactor,
    # 价格形态
    "price_position": PricePositionFactor,
    "gap": GapFactor,
    "intraday": IntradayReturnFactor,
    "overnight": OvernightReturnFactor,
    # 均线
    "ma_slope": MASlopeFactor,
    "ma_cross": MACrossFactor,
    "bias": BiasFactor,
    # 基本面
    "pe": PERatioFactor,
    "pb": PBRatioFactor,
    "roe": ROEFactor,
    "revenue_growth": RevenueGrowthFactor,
    "profit_growth": ProfitGrowthFactor,
    "market_cap": MarketCapFactor,
    # 情绪
    "money_flow": MoneyFlowFactor,
    "amihud": AmihudIlliquidityFactor,
}


def get_factor(name: str, **kwargs) -> BaseFactor:
    """获取因子实例"""
    if name not in FACTOR_REGISTRY:
        raise ValueError(f"Unknown factor: {name}. Available: {list(FACTOR_REGISTRY.keys())}")
    return FACTOR_REGISTRY[name](**kwargs)


def list_factors() -> List[str]:
    """列出所有内置因子"""
    return list(FACTOR_REGISTRY.keys())


def _register_extended_factors():
    """注册扩展因子 - 在函数内部导入避免循环导入"""
    from dquant.ai.extended_factors import (
        ADLineFactor,
        ADXFactor,
        AlphaFactor,
        AroonFactor,
        AutocorrelationFactor,
        BetaFactor,
        ChaikinOscillatorFactor,
        CMOFactor,
        EaseOfMovementFactor,
        ForceIndexFactor,
        HurstExponentFactor,
        MFIFactor,
        ROCFactor,
        StochasticFactor,
        VarianceRatioFactor,
        VPTFactor,
    )

    # 添加到注册表
    FACTOR_REGISTRY.update(
        {
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
    )


# 注册扩展因子
_register_extended_factors()
