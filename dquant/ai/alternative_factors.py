"""
另类因子

包含情感因子、新闻因子、资金流向因子等。
"""

import pandas as pd

from dquant.ai.base import RuleFactor

# ============================================================
# 情感因子
# ============================================================


class SentimentFactor(RuleFactor):
    """
    市场情感因子

    基于市场整体情感（如 VIX、涨跌比等）。
    """

    def __init__(
        self,
        window: int = 20,
        name: str = None,
    ):
        super().__init__(name=name or f"Sentiment_{window}")
        self.window = window

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算市场情感

        使用涨跌比、成交量变化等指标。
        """
        results = []

        # 按日期分组计算市场情感
        for date, group in data.groupby(data.index):
            # 上涨股票比例
            up_stocks = (group["close"] > group["open"]).sum()
            total_stocks = len(group)
            up_ratio = up_stocks / total_stocks if total_stocks > 0 else 0.5

            # 成交量变化

            # 情感分数
            sentiment = (up_ratio - 0.5) * 2  # 归一化到 [-1, 1]

            for symbol in group["symbol"].unique():
                results.append(
                    {
                        "date": date,
                        "symbol": symbol,
                        "score": sentiment,
                    }
                )

        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")

        return df


class NewsSentimentFactor(RuleFactor):
    """
    新闻情感因子

    基于新闻文本的情感分析。

    注意：需要外部新闻数据。
    """

    def __init__(self, name: str = "NewsSentiment"):
        super().__init__(name=name)

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算新闻情感

        需要数据中包含 'news_sentiment' 列。
        """
        results = []

        if "news_sentiment" in data.columns:
            for idx, row in data.iterrows():
                results.append(
                    {
                        "date": idx,
                        "symbol": row["symbol"],
                        "score": row["news_sentiment"],
                    }
                )

        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")

        return df


class SocialMediaFactor(RuleFactor):
    """
    社交媒体因子

    基于社交媒体讨论热度、情感等。

    注意：需要外部社交媒体数据。
    """

    def __init__(self, name: str = "SocialMedia"):
        super().__init__(name=name)

    def _compute_score(self, group: pd.DataFrame) -> pd.Series:
        """计算社交媒体因子"""
        if "social_mention_count" in group.columns:
            return group["social_mention_count"].pct_change()
        return pd.Series(dtype=float)


# ============================================================
# 资金流向因子
# ============================================================


class NorthboundFlowFactor(RuleFactor):
    """
    北向资金因子

    基于沪股通、深股通资金流向。
    """

    def __init__(self, window: int = 5, name: str = None):
        super().__init__(name=name or f"NorthboundFlow_{window}")
        self.window = window

    def _compute_score(self, group: pd.DataFrame) -> pd.Series:
        """计算北向资金因子"""
        if "northbound_flow" in group.columns:
            return group["northbound_flow"].rolling(self.window).sum()
        return pd.Series(dtype=float)


class MarginTradingFactor(RuleFactor):
    """
    融资融券因子

    基于融资融券余额变化。
    """

    def __init__(self, window: int = 5, name: str = None):
        super().__init__(name=name or f"MarginTrading_{window}")
        self.window = window

    def _compute_score(self, group: pd.DataFrame) -> pd.Series:
        """计算融资融券因子"""
        if "margin_balance" in group.columns:
            return group["margin_balance"].pct_change(self.window)
        return pd.Series(dtype=float)


class InstitutionalFlowFactor(RuleFactor):
    """
    机构资金流向因子

    基于机构买卖数据。
    """

    def __init__(self, window: int = 5, name: str = None):
        super().__init__(name=name or f"InstitutionalFlow_{window}")
        self.window = window

    def _compute_score(self, group: pd.DataFrame) -> pd.Series:
        """计算机构资金流向因子"""
        if "institutional_buy" in group.columns and "institutional_sell" in group.columns:
            net_buy = group["institutional_buy"] - group["institutional_sell"]
            return net_buy.rolling(self.window).mean()
        return pd.Series(dtype=float)


# ============================================================
# 另类数据因子
# ============================================================


class ShortInterestFactor(RuleFactor):
    """
    卖空兴趣因子

    基于卖空比例。
    """

    def __init__(self, name: str = "ShortInterest"):
        super().__init__(name=name)

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算卖空兴趣因子

        需要数据中包含 'short_ratio' 列。
        """
        results = []

        if "short_ratio" in data.columns:
            for idx, row in data.iterrows():
                results.append(
                    {
                        "date": idx,
                        "symbol": row["symbol"],
                        "score": -row["short_ratio"],  # 卖空比例高 = 负面
                    }
                )

        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")

        return df


class AnalystRatingFactor(RuleFactor):
    """
    分析师评级因子

    基于分析师评级变化。
    """

    def __init__(self, window: int = 30, name: str = None):
        super().__init__(name=name or f"AnalystRating_{window}")
        self.window = window

    def _compute_score(self, group: pd.DataFrame) -> pd.Series:
        """计算分析师评级因子"""
        if "analyst_rating" in group.columns:
            return group["analyst_rating"].diff()
        return pd.Series(dtype=float)


class OptionsFlowFactor(RuleFactor):
    """
    期权流向因子

    基于期权市场数据（Put/Call 比率等）。
    """

    def __init__(self, name: str = "OptionsFlow"):
        super().__init__(name=name)

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算期权流向因子

        需要数据中包含 'put_call_ratio' 列。
        """
        results = []

        if "put_call_ratio" in data.columns:
            for idx, row in data.iterrows():
                # Put/Call 比率低 = 看多
                score = 1 - row["put_call_ratio"]
                results.append(
                    {
                        "date": idx,
                        "symbol": row["symbol"],
                        "score": score,
                    }
                )

        df = pd.DataFrame(results)
        if len(df) > 0:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")

        return df


# ============================================================
# 注册另类因子
# ============================================================

ALTERNATIVE_FACTORS = {
    "sentiment": SentimentFactor,
    "news_sentiment": NewsSentimentFactor,
    "social_media": SocialMediaFactor,
    "northbound_flow": NorthboundFlowFactor,
    "margin_trading": MarginTradingFactor,
    "institutional_flow": InstitutionalFlowFactor,
    "short_interest": ShortInterestFactor,
    "analyst_rating": AnalystRatingFactor,
    "options_flow": OptionsFlowFactor,
}
