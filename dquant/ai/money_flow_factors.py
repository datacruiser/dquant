"""
资金流因子

基于资金流数据的交易因子
"""

from typing import Optional

import pandas as pd

from dquant.ai.base import BaseFactor


class MediumFlowFactor(BaseFactor):
    """
    中户资金流因子

    基于中单资金净流入进行选股。

    原理:
    - 中单通常代表中户资金
    - 中单净流入 > 0 表示中户看多
    - 中单净流入 < 0 表示中户看空

    策略思路:
    - 选中单持续净流入的股票
    - 结合主力资金流向判断
    - 避免主力流出但中单流入的陷阱

    Usage:
        factor = MediumFlowFactor(window=5)
        factor.fit(data)
        predictions = factor.predict(data)
    """

    def __init__(self, window: int = 5, name: str = None):
        super().__init__(name=name or f"MediumFlow_{window}")
        self.window = window

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None):
        """拟合因子"""
        self._is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算因子值

        Args:
            data: 包含 'medium_net_inflow' 列的 DataFrame

        Returns:
            DataFrame with columns: date, symbol, score
        """
        results = []

        if "medium_net_inflow" not in data.columns:
            raise ValueError("数据缺少 'medium_net_inflow' 列")

        for symbol, group in data.groupby("symbol"):
            group = group.sort_index()

            # 中单净流入 N 日均值
            flow_ma = group["medium_net_inflow"].rolling(self.window).mean()

            # 标准化
            flow_std = group["medium_net_inflow"].rolling(self.window * 2).std()
            score = flow_ma / (flow_std + 1e-8)

            for date, value in score.items():
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


class MainForceFactor(BaseFactor):
    """
    主力资金因子

    基于主力（超大单 + 大单）资金流向。

    原理:
    - 主力资金通常先于价格变化
    - 主力持续流入是上涨信号
    - 主力持续流出是下跌信号

    Usage:
        factor = MainForceFactor(window=5)
        predictions = factor.predict(data)
    """

    def __init__(self, window: int = 5, name: str = None):
        super().__init__(name=name or f"MainForce_{window}")
        self.window = window

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None):
        self._is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算主力资金因子"""
        results = []

        if "main_net_inflow" not in data.columns:
            raise ValueError("数据缺少 'main_net_inflow' 列")

        for symbol, group in data.groupby("symbol"):
            group = group.sort_index()

            # 主力净流入 N 日均值
            flow_ma = group["main_net_inflow"].rolling(self.window).mean()

            # 标准化
            flow_std = group["main_net_inflow"].rolling(self.window * 2).std()
            score = flow_ma / (flow_std + 1e-8)

            for date, value in score.items():
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


class RetailFlowFactor(BaseFactor):
    """
    散户资金流因子

    基于小单资金流向。

    原理:
    - 散户通常后知后觉
    - 散户大量流入可能是顶部信号
    - 散户大量流出可能是底部信号
    - 常用于反向指标

    Usage:
        factor = RetailFlowFactor(window=5, reverse=True)
        predictions = factor.predict(data)
    """

    def __init__(self, window: int = 5, reverse: bool = True, name: str = None):
        super().__init__(name=name or f"RetailFlow_{window}")
        self.window = window
        self.reverse = reverse  # 是否反向

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None):
        self._is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算散户资金流因子"""
        results = []

        if "small_net_inflow" not in data.columns:
            raise ValueError("数据缺少 'small_net_inflow' 列")

        for symbol, group in data.groupby("symbol"):
            group = group.sort_index()

            # 小单净流入 N 日均值
            flow_ma = group["small_net_inflow"].rolling(self.window).mean()

            # 标准化
            flow_std = group["small_net_inflow"].rolling(self.window * 2).std()
            score = flow_ma / (flow_std + 1e-8)

            # 反向（散户流入记为负分）
            if self.reverse:
                score = -score

            for date, value in score.items():
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


class SmartFlowFactor(BaseFactor):
    """
    聪明钱因子

    综合主力、中户、散户资金流向。

    策略:
    - 主力流入 + 中户流入 + 散户流出 = 强烈买入
    - 主力流出 + 中户流出 + 散户流入 = 强烈卖出

    公式:
    score = 主力权重 × 主力净流入
          + 中户权重 × 中户净流入
          - 散户权重 × 散户净流入

    Usage:
        factor = SmartFlowFactor(
            main_weight=0.5,
            medium_weight=0.3,
            retail_weight=0.2,
        )
        predictions = factor.predict(data)
    """

    def __init__(
        self,
        window: int = 5,
        main_weight: float = 0.5,
        medium_weight: float = 0.3,
        retail_weight: float = 0.2,
        name: str = None,
    ):
        super().__init__(name=name or f"SmartFlow_{window}")
        self.window = window
        self.main_weight = main_weight
        self.medium_weight = medium_weight
        self.retail_weight = retail_weight

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None):
        self._is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算聪明钱因子"""
        results = []

        required_cols = ["main_net_inflow", "medium_net_inflow", "small_net_inflow"]
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"数据缺少 '{col}' 列")

        for symbol, group in data.groupby("symbol"):
            group = group.sort_index()

            # 各类资金 N 日均值
            main_ma = group["main_net_inflow"].rolling(self.window).mean()
            medium_ma = group["medium_net_inflow"].rolling(self.window).mean()
            retail_ma = group["small_net_inflow"].rolling(self.window).mean()

            # 标准化
            main_std = group["main_net_inflow"].rolling(self.window * 2).std()
            medium_std = group["medium_net_inflow"].rolling(self.window * 2).std()
            retail_std = group["small_net_inflow"].rolling(self.window * 2).std()

            main_score = main_ma / (main_std + 1e-8)
            medium_score = medium_ma / (medium_std + 1e-8)
            retail_score = retail_ma / (retail_std + 1e-8)

            # 综合得分
            score = (
                self.main_weight * main_score
                + self.medium_weight * medium_score
                - self.retail_weight * retail_score  # 散户反向
            )

            for date, value in score.items():
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


class FlowDivergenceFactor(BaseFactor):
    """
    资金流背离因子

    检测价格与资金流的背离。

    原理:
    - 价格上涨 + 主力流出 = 顶背离 (卖出信号)
    - 价格下跌 + 主力流入 = 底背离 (买入信号)

    Usage:
        factor = FlowDivergenceFactor(window=10)
        predictions = factor.predict(data)
    """

    def __init__(self, window: int = 10, name: str = None):
        super().__init__(name=name or f"FlowDivergence_{window}")
        self.window = window

    def fit(self, data: pd.DataFrame, target: Optional[pd.Series] = None):
        self._is_fitted = True
        return self

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """计算资金流背离因子"""
        results = []

        required_cols = ["close", "main_net_inflow"]
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"数据缺少 '{col}' 列")

        for symbol, group in data.groupby("symbol"):
            group = group.sort_index()

            # 价格变化率
            price_ret = group["close"].pct_change(self.window)

            # 主力资金累计净流入
            flow_cumsum = group["main_net_inflow"].rolling(self.window).sum()

            # 标准化
            price_std = price_ret.rolling(self.window * 2).std()
            flow_std = flow_cumsum.rolling(self.window * 2).std()

            price_z = price_ret / (price_std + 1e-8)
            flow_z = flow_cumsum / (flow_std + 1e-8)

            # 背离得分
            # 价格上涨但资金流出 -> 负分
            # 价格下跌但资金流入 -> 正分
            score = flow_z - price_z

            for date, value in score.items():
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
