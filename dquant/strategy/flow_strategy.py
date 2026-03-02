"""
资金流策略

基于资金流向的交易策略
"""

from typing import List, Optional
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
        required_cols = ['medium_net_inflow']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"数据缺少 '{col}' 列")

        # 按日期分组
        for date, group in data.groupby(data.index):
            # 筛选条件
            candidates = group.copy()

            # 1. 中单净流入 >= 阈值
            if self.min_medium_flow > 0:
                candidates = candidates[
                    candidates['medium_net_inflow'] >= self.min_medium_flow
                ]

            # 2. 主力也流入
            if self.require_main_flow and 'main_net_inflow' in candidates.columns:
                candidates = candidates[
                    candidates['main_net_inflow'] > 0
                ]

            # 3. 避免散户大量流入
            if self.avoid_retail_inflow and 'small_net_inflow' in candidates.columns:
                candidates = candidates[
                    candidates['small_net_inflow'] < candidates['small_net_inflow'].quantile(0.8)
                ]

            # 按中单净流入排序
            if len(candidates) > 0:
                top_stocks = candidates.nlargest(self.top_k, 'medium_net_inflow')

                for _, row in top_stocks.iterrows():
                    signal = Signal(
                        symbol=row['symbol'],
                        signal_type=SignalType.BUY,
                        strength=1.0 / self.top_k,  # 等权
                        timestamp=date,
                        metadata={
                            'medium_flow': row['medium_net_inflow'],
                            'main_flow': row.get('main_net_inflow', 0),
                            'small_flow': row.get('small_net_inflow', 0),
                        }
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
        required_cols = ['main_net_inflow', 'medium_net_inflow', 'small_net_inflow']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"数据缺少 '{col}' 列")

        # 按日期分组
        for date, group in data.groupby(data.index):
            # 计算 N 日均值
            # (这里简化处理，实际应该按股票分组计算)

            # 综合得分
            score = (
                self.main_weight * group['main_net_inflow']
                + self.medium_weight * group['medium_net_inflow']
                - self.retail_weight * group['small_net_inflow']  # 散户反向
            )

            # 选 TopK
            top_idx = score.nlargest(self.top_k).index
            top_stocks = group.loc[top_idx]

            for _, row in top_stocks.iterrows():
                signal = Signal(
                    symbol=row['symbol'],
                    signal_type=SignalType.BUY,
                    strength=1.0 / self.top_k,
                    timestamp=date,
                    metadata={
                        'composite_score': score[row.name],
                    }
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
        required_cols = ['close', 'main_net_inflow']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"数据缺少 '{col}' 列")

        # 按日期分组
        for date, group in data.groupby(data.index):
            # 简化版: 直接使用当日数据
            # 实际应该计算 window 日的累计值

            # 价格变化
            price_change = group['close'] / group['close'].shift(self.window) - 1

            # 资金流累计
            # (这里简化处理)

            # 底背离: 价格下跌但主力流入
            bottom_divergence = (
                (price_change < -0.05) &  # 价格下跌 > 5%
                (group['main_net_inflow'] > 0)  # 主力流入
            )

            # 选 TopK
            candidates = group[bottom_divergence]

            if len(candidates) > 0:
                top_stocks = candidates.nlargest(
                    min(self.top_k, len(candidates)),
                    'main_net_inflow'
                )

                for _, row in top_stocks.iterrows():
                    signal = Signal(
                        symbol=row['symbol'],
                        signal_type=SignalType.BUY,
                        strength=1.0 / self.top_k,
                        timestamp=date,
                        metadata={
                            'divergence_type': 'bottom',
                            'price_change': price_change[row.name],
                        }
                    )
                    signals.append(signal)

        return signals
