"""
资金流策略

基于资金流向的交易策略
"""

from typing import List

import pandas as pd

from dquant.strategy.base import BaseStrategy, Signal, SignalType


class MoneyFlowStrategy(BaseStrategy):
    """
    资金流策略

    基于中户资金流选股。

    策略逻辑:
    1. 选中单净流入持续为正的股票
    2. 主力资金也流入加分
    3. 散户资金流出加分
    4. 定期调仓

    Usage:
        from dquant import MoneyFlowStrategy

        strategy = MoneyFlowStrategy(
            top_k=10,
            min_medium_flow=100,  # 中单净流入 >= 100万
            require_main_flow=True,  # 要求主力也流入
        )

        result = engine.backtest(strategy=strategy)
    """

    def __init__(
        self,
        top_k: int = 10,
        min_medium_flow: float = 0,  # 最小中单净流入 (万元)
        require_main_flow: bool = False,  # 是否要求主力流入
        avoid_retail_inflow: bool = True,  # 避免散户大量流入
        rebalance_freq: int = 5,  # 调仓频率 (天)
        name: str = "MoneyFlowStrategy",
    ):
        super().__init__(name=name)
        self.top_k = top_k
        self.min_medium_flow = min_medium_flow
        self.require_main_flow = require_main_flow
        self.avoid_retail_inflow = avoid_retail_inflow
        self.rebalance_freq = rebalance_freq
        self._last_rebalance = None

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """生成交易信号"""
        signals = []

        # 检查必要列
        required_cols = ["medium_net_inflow"]
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"数据缺少 '{col}' 列")

        # 按日期分组
        sorted_dates = sorted(data.index.unique())
        for i, date in enumerate(sorted_dates):
            # 调仓频率控制：仅在调仓日生成信号
            if self.rebalance_freq > 1 and i % self.rebalance_freq != 0:
                continue

            grp = (
                data.loc[[date]]
                if isinstance(data.index, pd.DatetimeIndex)
                else data[data.index == date]
            )

            # 筛选条件
            candidates = grp.copy()

            # 1. 中单净流入 >= 阈值
            if self.min_medium_flow > 0:
                candidates = candidates[candidates["medium_net_inflow"] >= self.min_medium_flow]

            # 2. 主力也流入
            if self.require_main_flow and "main_net_inflow" in candidates.columns:
                candidates = candidates[candidates["main_net_inflow"] > 0]

            # 3. 避免散户大量流入（截面过小时跳过 quantile 过滤）
            if (
                self.avoid_retail_inflow
                and "small_net_inflow" in candidates.columns
                and len(candidates) >= 5
            ):
                candidates = candidates[
                    candidates["small_net_inflow"] < candidates["small_net_inflow"].quantile(0.8)
                ]

            # 按中单净流入排序
            if len(candidates) > 0:
                top_stocks = candidates.nlargest(self.top_k, "medium_net_inflow")

                for _, row in top_stocks.iterrows():
                    signal = Signal(
                        symbol=row["symbol"],
                        signal_type=SignalType.BUY,
                        strength=1.0 / self.top_k,  # 等权
                        timestamp=date,
                        metadata={
                            "medium_flow": row["medium_net_inflow"],
                            "main_flow": row.get("main_net_inflow", 0),
                            "small_flow": row.get("small_net_inflow", 0),
                        },
                    )
                    signals.append(signal)

        return signals


class SmartFlowStrategy(BaseStrategy):
    """
    聪明钱策略

    综合主力、中户、散户资金流。

    策略逻辑:
    1. 主力流入 + 中户流入 + 散户流出 = 买入
    2. 按综合得分选 TopK

    Usage:
        strategy = SmartFlowStrategy(
            top_k=10,
            main_weight=0.5,
            medium_weight=0.3,
            retail_weight=0.2,
        )
    """

    def __init__(
        self,
        top_k: int = 10,
        main_weight: float = 0.5,
        medium_weight: float = 0.3,
        retail_weight: float = 0.2,
        window: int = 5,
        name: str = "SmartFlowStrategy",
    ):
        super().__init__(name=name)
        self.top_k = top_k
        self.main_weight = main_weight
        self.medium_weight = medium_weight
        self.retail_weight = retail_weight
        self.window = window

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """生成交易信号"""
        signals = []

        # 检查必要列
        required_cols = ["main_net_inflow", "medium_net_inflow", "small_net_inflow"]
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"数据缺少 '{col}' 列")

        # 按股票分组做时间序列滚动平滑（跨日期，而非截面内）
        if self.window > 1:
            main = data.groupby("symbol")["main_net_inflow"].transform(
                lambda x: x.rolling(self.window, min_periods=1).mean()
            )
            medium = data.groupby("symbol")["medium_net_inflow"].transform(
                lambda x: x.rolling(self.window, min_periods=1).mean()
            )
            small = data.groupby("symbol")["small_net_inflow"].transform(
                lambda x: x.rolling(self.window, min_periods=1).mean()
            )
        else:
            main = data["main_net_inflow"]
            medium = data["medium_net_inflow"]
            small = data["small_net_inflow"]

        # 综合得分
        score = self.main_weight * main + self.medium_weight * medium - self.retail_weight * small
        data_scored = data.assign(_score=score)

        # 按日期分组选股
        for date, grp in data_scored.groupby(data_scored.index):
            top_stocks = grp.nlargest(self.top_k, "_score")

            for _, row in top_stocks.iterrows():
                signal = Signal(
                    symbol=row["symbol"],
                    signal_type=SignalType.BUY,
                    strength=1.0 / self.top_k,
                    timestamp=date,
                    metadata={
                        "composite_score": row["_score"],
                    },
                )
                signals.append(signal)

        return signals


class FlowDivergenceStrategy(BaseStrategy):
    """
    资金流背离策略

    检测价格与资金流的背离，反向操作。

    策略逻辑:
    1. 价格上涨 + 主力流出 = 卖出 (顶背离)
    2. 价格下跌 + 主力流入 = 买入 (底背离)

    Usage:
        strategy = FlowDivergenceStrategy(
            window=10,
            top_k=10,
        )
    """

    def __init__(
        self,
        window: int = 10,
        top_k: int = 10,
        name: str = "FlowDivergenceStrategy",
    ):
        super().__init__(name=name)
        self.window = window
        self.top_k = top_k

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """生成交易信号"""
        signals = []

        # 检查必要列
        required_cols = ["close", "main_net_inflow"]
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"数据缺少 '{col}' 列")

        # 按股票分组计算价格变化（跨日期，避免按日 groupby 后 shift 跨股票污染）
        price_change = data.groupby("symbol")["close"].transform(
            lambda x: x / x.shift(self.window) - 1
        )
        data_with_change = data.assign(price_change=price_change)

        # 按日期分组
        for date, grp in data_with_change.groupby(data_with_change.index):
            # 底背离: 价格下跌但主力流入 → 买入
            bottom_divergence = (grp["price_change"] < -0.05) & (  # 价格下跌 > 5%
                grp["main_net_inflow"] > 0
            )  # 主力流入

            candidates = grp[bottom_divergence]
            if len(candidates) > 0:
                top_stocks = candidates.nlargest(
                    min(self.top_k, len(candidates)), "main_net_inflow"
                )
                for _, row in top_stocks.iterrows():
                    signals.append(
                        Signal(
                            symbol=row["symbol"],
                            signal_type=SignalType.BUY,
                            strength=1.0 / self.top_k,
                            timestamp=date,
                            metadata={
                                "divergence_type": "bottom",
                                "price_change": row["price_change"],
                            },
                        )
                    )

            # 顶背离: 价格上涨但主力流出 → 卖出
            top_divergence = (grp["price_change"] > 0.05) & (  # 价格上涨 > 5%
                grp["main_net_inflow"] < 0
            )  # 主力流出

            sell_candidates = grp[top_divergence]
            if len(sell_candidates) > 0:
                sell_stocks = sell_candidates.nsmallest(
                    min(self.top_k, len(sell_candidates)),
                    "main_net_inflow",  # 流出最多优先
                )
                for _, row in sell_stocks.iterrows():
                    signals.append(
                        Signal(
                            symbol=row["symbol"],
                            signal_type=SignalType.SELL,
                            strength=1.0 / self.top_k,
                            timestamp=date,
                            metadata={
                                "divergence_type": "top",
                                "price_change": row["price_change"],
                            },
                        )
                    )

        return signals
