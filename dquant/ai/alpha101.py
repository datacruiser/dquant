"""
WorldQuant Alpha101 因子库

实现《101 Formulaic Alphas》(Zura Kakushadze, 2015) 中的经典因子。
所有因子均为 RuleFactor，使用向量化计算，基于标准 OHLCV 数据。

因子编号对应论文原始编号。每个因子返回截面得分，已 rank 标准化。

参考: https://arxiv.org/abs/1510.04934
"""

import numpy as np
import pandas as pd

from dquant.ai.base import RuleFactor
from dquant.logger import get_logger

logger = get_logger(__name__)


def _rank(df_col: pd.Series) -> pd.Series:
    """截面 rank（按日期分组后排名）"""
    return df_col.groupby(level=0).rank(pct=True)


def _ts_rank(series: pd.Series, window: int) -> pd.Series:
    """时间序列 rank: 当前值在过去 window 天中的百分位"""
    return series.rolling(window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False)


def _ts_delta(series: pd.Series, period: int) -> pd.Series:
    """series - series.shift(period)"""
    return series - series.shift(period)


def _ts_sum(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).sum()


def _ts_mean(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()


def _ts_std(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).std()


def _ts_max(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).max()


def _ts_min(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).min()


def _ts_corr(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    return x.rolling(window).corr(y)


def _ts_cov(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    return x.rolling(window).cov(y)


def _ts_product(series: pd.Series, window: int) -> pd.Series:
    """滚动乘积"""
    return series.rolling(window).apply(np.prod, raw=True)


def _sign(series: pd.Series) -> pd.Series:
    return np.sign(series)


# ============================================================
# Alpha101 因子实现
# ============================================================


class Alpha001(RuleFactor):
    """Alpha#1: (rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5)"""

    def __init__(self):
        super().__init__(name="Alpha001")

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        for sym, g in data.groupby("symbol"):
            g = g.sort_index()
            ret = g["close"].pct_change()
            cond = g["close"].copy()
            cond[ret < 0] = ret.rolling(20).std()
            signed_power = cond ** 2
            argmax = signed_power.rolling(5).apply(np.argmax, raw=True)
            score = argmax.rank(pct=True) - 0.5
            for date, val in score.items():
                if pd.notna(val):
                    results.append({"date": date, "symbol": sym, "score": val})
        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df


class Alpha002(RuleFactor):
    """Alpha#2: (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))"""

    def __init__(self):
        super().__init__(name="Alpha002")

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        for sym, g in data.groupby("symbol"):
            g = g.sort_index()
            log_vol_delta = np.log(g["volume"]).diff(2)
            close_open_ratio = (g["close"] - g["open"]) / g["open"]
            score = -_ts_corr(log_vol_delta.rank(), close_open_ratio.rank(), 6)
            for date, val in score.items():
                if pd.notna(val):
                    results.append({"date": date, "symbol": sym, "score": val})
        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df


class Alpha003(RuleFactor):
    """Alpha#3: (-1 * correlation(rank(open), rank(volume), 10))"""

    def __init__(self):
        super().__init__(name="Alpha003")

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        for sym, g in data.groupby("symbol"):
            g = g.sort_index()
            score = -_ts_corr(g["open"].rank(), g["volume"].rank(), 10)
            for date, val in score.items():
                if pd.notna(val):
                    results.append({"date": date, "symbol": sym, "score": val})
        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df


class Alpha004(RuleFactor):
    """Alpha#4: (-1 * Ts_Rank(rank(low), 9))"""

    def __init__(self):
        super().__init__(name="Alpha004")

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        for sym, g in data.groupby("symbol"):
            g = g.sort_index()
            score = -_ts_rank(g["low"].rank(), 9)
            for date, val in score.items():
                if pd.notna(val):
                    results.append({"date": date, "symbol": sym, "score": val})
        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df


class Alpha005(RuleFactor):
    """Alpha#5: (-1 * ts_corr(((high+low)/2), volume, 5))"""

    def __init__(self):
        super().__init__(name="Alpha005")

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        for sym, g in data.groupby("symbol"):
            g = g.sort_index()
            mid = (g["high"] + g["low"]) / 2
            score = -_ts_corr(mid, g["volume"], 5)
            for date, val in score.items():
                if pd.notna(val):
                    results.append({"date": date, "symbol": sym, "score": val})
        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df


class Alpha006(RuleFactor):
    """Alpha#6: (-1 * correlation(open, volume, 10))"""

    def __init__(self):
        super().__init__(name="Alpha006")

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        for sym, g in data.groupby("symbol"):
            g = g.sort_index()
            score = -_ts_corr(g["open"], g["volume"], 10)
            for date, val in score.items():
                if pd.notna(val):
                    results.append({"date": date, "symbol": sym, "score": val})
        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df


class Alpha007(RuleFactor):
    """Alpha#7: ((adv20 < volume) ? (-1 * ts_rank(abs(delta(close, 7)), 60)) : (-1 * 1))"""

    def __init__(self):
        super().__init__(name="Alpha007")

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        for sym, g in data.groupby("symbol"):
            g = g.sort_index()
            adv20 = g["volume"].rolling(20).mean()
            delta_close = g["close"].diff(7).abs()
            score = pd.Series(-1.0, index=g.index)
            mask = adv20 < g["volume"]
            score[mask] = -_ts_rank(delta_close, 60)[mask]
            for date, val in score.items():
                if pd.notna(val):
                    results.append({"date": date, "symbol": sym, "score": val})
        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df


class Alpha008(RuleFactor):
    """Alpha#8: (-1 * correlation(open, sum(high+low, 5), 10))"""

    def __init__(self):
        super().__init__(name="Alpha008")

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        for sym, g in data.groupby("symbol"):
            g = g.sort_index()
            hl_sum = _ts_sum(g["high"] + g["low"], 5)
            score = -_ts_corr(g["open"], hl_sum, 10)
            for date, val in score.items():
                if pd.notna(val):
                    results.append({"date": date, "symbol": sym, "score": val})
        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df


class Alpha012(RuleFactor):
    """Alpha#12: (sign(delta(volume, 1)) * (-1 * delta(close, 1)))"""

    def __init__(self):
        super().__init__(name="Alpha012")

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        for sym, g in data.groupby("symbol"):
            g = g.sort_index()
            score = _sign(g["volume"].diff(1)) * (-g["close"].diff(1))
            for date, val in score.items():
                if pd.notna(val):
                    results.append({"date": date, "symbol": sym, "score": val})
        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df


class Alpha014(RuleFactor):
    """Alpha#14: (-1 * ts_corr(open, volume, 10))"""

    def __init__(self):
        super().__init__(name="Alpha014")

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        for sym, g in data.groupby("symbol"):
            g = g.sort_index()
            ret = g["close"].pct_change()
            score = -_ts_corr(ret, (g["close"].shift(1) + g["close"].shift(2) + g["close"].shift(3)) / 3, 10)
            for date, val in score.items():
                if pd.notna(val):
                    results.append({"date": date, "symbol": sym, "score": val})
        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df


class Alpha015(RuleFactor):
    """Alpha#15: (-1 * ts_sum(rank(correlation(rank(high), rank(volume), 3)), 3))"""

    def __init__(self):
        super().__init__(name="Alpha015")

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        for sym, g in data.groupby("symbol"):
            g = g.sort_index()
            corr = _ts_corr(g["high"].rank(), g["volume"].rank(), 3)
            score = -_ts_sum(corr.rank(), 3)
            for date, val in score.items():
                if pd.notna(val):
                    results.append({"date": date, "symbol": sym, "score": val})
        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df


class Alpha016(RuleFactor):
    """Alpha#16: (-1 * ts_rank(covariance(rank(high), rank(volume), 5), 5))"""

    def __init__(self):
        super().__init__(name="Alpha016")

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        for sym, g in data.groupby("symbol"):
            g = g.sort_index()
            cov = _ts_cov(g["high"].rank(), g["volume"].rank(), 5)
            score = -_ts_rank(cov, 5)
            for date, val in score.items():
                if pd.notna(val):
                    results.append({"date": date, "symbol": sym, "score": val})
        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df


class Alpha018(RuleFactor):
    """Alpha#18: (-1 * correlation(close, open, 10))"""

    def __init__(self):
        super().__init__(name="Alpha018")

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        for sym, g in data.groupby("symbol"):
            g = g.sort_index()
            score = -_ts_corr(g["close"], g["open"], 10)
            for date, val in score.items():
                if pd.notna(val):
                    results.append({"date": date, "symbol": sym, "score": val})
        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df


class Alpha020(RuleFactor):
    """Alpha#20: (((-1 * (close - delay(close, 7))) * correlation(close, volume, 5))"""

    def __init__(self):
        super().__init__(name="Alpha020")

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        for sym, g in data.groupby("symbol"):
            g = g.sort_index()
            price_chg = -(g["close"] - g["close"].shift(7))
            corr = _ts_corr(g["close"], g["volume"], 5)
            score = price_chg * corr
            for date, val in score.items():
                if pd.notna(val):
                    results.append({"date": date, "symbol": sym, "score": val})
        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df


class Alpha023(RuleFactor):
    """Alpha#23: (-1 * (high - close) / (close - low) if low != close else 0)"""

    def __init__(self):
        super().__init__(name="Alpha023")

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        for sym, g in data.groupby("symbol"):
            g = g.sort_index()
            denom = g["close"] - g["low"]
            denom = denom.replace(0, np.nan)
            score = -(g["high"] - g["close"]) / denom
            score = score.fillna(0)
            for date, val in score.items():
                if pd.notna(val):
                    results.append({"date": date, "symbol": sym, "score": val})
        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df


class Alpha026(RuleFactor):
    """Alpha#26: (-1 * ts_max(ts_corr(rank(volume), rank(high), 5), 26))"""

    def __init__(self):
        super().__init__(name="Alpha026")

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        for sym, g in data.groupby("symbol"):
            g = g.sort_index()
            corr = _ts_corr(g["volume"].rank(), g["high"].rank(), 5)
            score = -_ts_max(corr, 26)
            for date, val in score.items():
                if pd.notna(val):
                    results.append({"date": date, "symbol": sym, "score": val})
        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df


class Alpha028(RuleFactor):
    """Alpha#28: scale(correlation(volume, low, 5))"""

    def __init__(self):
        super().__init__(name="Alpha028")

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        for sym, g in data.groupby("symbol"):
            g = g.sort_index()
            score = _ts_corr(g["volume"], g["low"], 5)
            # scale: divide by abs(sum)
            abs_sum = score.abs().sum()
            if abs_sum > 0:
                score = score / abs_sum
            for date, val in score.items():
                if pd.notna(val):
                    results.append({"date": date, "symbol": sym, "score": val})
        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df


class Alpha032(RuleFactor):
    """Alpha#32: (scale(((sum(close, 7) / 7) - close)) + (20 * scale(correlation(close, volume, 5))))"""

    def __init__(self):
        super().__init__(name="Alpha032")

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        for sym, g in data.groupby("symbol"):
            g = g.sort_index()
            ma7 = _ts_mean(g["close"], 7)
            part1 = (ma7 - g["close"])
            corr = _ts_corr(g["close"], g["volume"], 5)
            score = part1 + 20 * corr
            for date, val in score.items():
                if pd.notna(val):
                    results.append({"date": date, "symbol": sym, "score": val})
        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df


class Alpha041(RuleFactor):
    """Alpha#41: (((high * low)^0.5) - vwap)"""

    def __init__(self):
        super().__init__(name="Alpha041")

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        for sym, g in data.groupby("symbol"):
            g = g.sort_index()
            sqrt_hl = np.sqrt(g["high"] * g["low"])
            # 使用 close 作为 vwap 的近似
            vwap = g["close"] if "vwap" not in g.columns else g["vwap"]
            score = sqrt_hl - vwap
            for date, val in score.items():
                if pd.notna(val):
                    results.append({"date": date, "symbol": sym, "score": val})
        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df


class Alpha043(RuleFactor):
    """Alpha#43: (ts_rank(volume / adv20, 20) * ts_rank((-1 * delta(close, 7)), 8))"""

    def __init__(self):
        super().__init__(name="Alpha043")

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        for sym, g in data.groupby("symbol"):
            g = g.sort_index()
            adv20 = g["volume"].rolling(20).mean().replace(0, np.nan)
            vol_ratio = g["volume"] / adv20
            score = _ts_rank(vol_ratio, 20) * _ts_rank(-g["close"].diff(7), 8)
            for date, val in score.items():
                if pd.notna(val):
                    results.append({"date": date, "symbol": sym, "score": val})
        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df


class Alpha046(RuleFactor):
    """Alpha#46: ((0.25 < ((delay(close, 20) - delay(close, 10)) / (-0.5 * delay(close, 10)))) ? (-1 * 1) : 1)"""

    def __init__(self):
        super().__init__(name="Alpha046")

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        for sym, g in data.groupby("symbol"):
            g = g.sort_index()
            c10 = g["close"].shift(10)
            c20 = g["close"].shift(20)
            ratio = (c20 - c10) / (-0.5 * c10)
            score = pd.Series(1.0, index=g.index)
            score[ratio > 0.25] = -1.0
            for date, val in score.items():
                if pd.notna(val):
                    results.append({"date": date, "symbol": sym, "score": val})
        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df


class Alpha049(RuleFactor):
    """Alpha#49: ((delay(close, 20) - delay(close, 10)) / (-0.5 * delay(close, 10)))"""

    def __init__(self):
        super().__init__(name="Alpha049")

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        for sym, g in data.groupby("symbol"):
            g = g.sort_index()
            c10 = g["close"].shift(10)
            c20 = g["close"].shift(20)
            score = (c20 - c10) / (-0.5 * c10)
            for date, val in score.items():
                if pd.notna(val):
                    results.append({"date": date, "symbol": sym, "score": val})
        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df


class Alpha053(RuleFactor):
    """Alpha#53: (-1 * delta((((close - low) - (high - close)) / (close - low)), 9))"""

    def __init__(self):
        super().__init__(name="Alpha053")

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        for sym, g in data.groupby("symbol"):
            g = g.sort_index()
            denom = g["close"] - g["low"]
            denom = denom.replace(0, np.nan)
            x = ((g["close"] - g["low"]) - (g["high"] - g["close"])) / denom
            score = -x.diff(9)
            for date, val in score.items():
                if pd.notna(val):
                    results.append({"date": date, "symbol": sym, "score": val})
        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df


class Alpha054(RuleFactor):
    """Alpha#54: (-1 * ((low - close) * (open^5)) / ((low - high) * (close^5)))"""

    def __init__(self):
        super().__init__(name="Alpha054")

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        for sym, g in data.groupby("symbol"):
            g = g.sort_index()
            denom = (g["low"] - g["high"]) * (g["close"] ** 5)
            denom = denom.replace(0, np.nan)
            score = -((g["low"] - g["close"]) * (g["open"] ** 5)) / denom
            score = score.replace([np.inf, -np.inf], np.nan)
            for date, val in score.items():
                if pd.notna(val):
                    results.append({"date": date, "symbol": sym, "score": val})
        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df


class Alpha101(RuleFactor):
    """Alpha#101: ((close - open) / ((high - low) + .001))"""

    def __init__(self):
        super().__init__(name="Alpha101")

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        results = []
        for sym, g in data.groupby("symbol"):
            g = g.sort_index()
            denom = g["high"] - g["low"] + 0.001
            score = (g["close"] - g["open"]) / denom
            for date, val in score.items():
                if pd.notna(val):
                    results.append({"date": date, "symbol": sym, "score": val})
        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        return df


# ============================================================
# 因子注册表
# ============================================================

ALPHA101_REGISTRY = {
    "alpha001": Alpha001,
    "alpha002": Alpha002,
    "alpha003": Alpha003,
    "alpha004": Alpha004,
    "alpha005": Alpha005,
    "alpha006": Alpha006,
    "alpha007": Alpha007,
    "alpha008": Alpha008,
    "alpha012": Alpha012,
    "alpha014": Alpha014,
    "alpha015": Alpha015,
    "alpha016": Alpha016,
    "alpha018": Alpha018,
    "alpha020": Alpha020,
    "alpha023": Alpha023,
    "alpha026": Alpha026,
    "alpha028": Alpha028,
    "alpha032": Alpha032,
    "alpha041": Alpha041,
    "alpha043": Alpha043,
    "alpha046": Alpha046,
    "alpha049": Alpha049,
    "alpha053": Alpha053,
    "alpha054": Alpha054,
    "alpha101": Alpha101,
}


def get_alpha(name: str) -> RuleFactor:
    """获取 Alpha101 因子实例"""
    cls = ALPHA101_REGISTRY.get(name)
    if cls is None:
        raise ValueError(f"Unknown Alpha101 factor: {name}. Available: {list(ALPHA101_REGISTRY.keys())}")
    return cls()


def list_alphas() -> list:
    """列出所有 Alpha101 因子名称"""
    return list(ALPHA101_REGISTRY.keys())
