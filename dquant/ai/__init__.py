"""
DQuant AI Module

提供因子、机器学习、强化学习等 AI 模块。
"""

# 基础因子
from dquant.ai.base import BaseFactor

# 内置因子库
from dquant.ai.builtin_factors import (
    # 动量类
    MomentumFactor,
    ReversalFactor,
    AccMomentumFactor,

    # 波动率类
    VolatilityFactor,
    ATRFactor,
    SkewnessFactor,
    KurtosisFactor,
    MaxDrawdownFactor,

    # 技术指标
    RSIFactor,
    MACDFactor,
    BollingerPositionFactor,
    TrendStrengthFactor,
    KDJFactor,
    CCIFactor,
    WilliamsRFactor,

    # 成交量
    VolumeRatioFactor,
    TurnoverRateFactor,
    OBVFactor,
    VWAPFactor,

    # 价格形态
    PricePositionFactor,
    GapFactor,
    IntradayReturnFactor,
    OvernightReturnFactor,

    # 均线
    MASlopeFactor,
    MACrossFactor,
    BiasFactor,

    # 基本面
    PERatioFactor,
    PBRatioFactor,
    ROEFactor,
    RevenueGrowthFactor,
    ProfitGrowthFactor,
    MarketCapFactor,

    # 情绪
    MoneyFlowFactor,
    AmihudIlliquidityFactor,

    # 注册表
    FACTOR_REGISTRY,
    get_factor,
    list_factors,
)

# 因子组合
from dquant.ai.factor_combiner import (
    FactorCombiner,
    CombinedFactor,
)

# ML 因子
from dquant.ai.ml_factors import XGBoostFactor, LGBMFactor

# Qlib 适配器
from dquant.ai.qlib_adapter import (
    QlibModelAdapter,
    QlibFactorConverter,
    QlibDataHandler,
)

# RL Agent
from dquant.ai.rl_agents import (
    TradingEnvironment,
    BaseRLAgent,
    DQNAgent,
    PPOAgent,
    RLStrategy,
)


__all__ = [
    # 基础
    "BaseFactor",

    # 动量类因子
    "MomentumFactor",
    "ReversalFactor",
    "AccMomentumFactor",

    # 波动率类因子
    "VolatilityFactor",
    "ATRFactor",
    "SkewnessFactor",
    "KurtosisFactor",
    "MaxDrawdownFactor",

    # 技术指标因子
    "RSIFactor",
    "MACDFactor",
    "BollingerPositionFactor",
    "TrendStrengthFactor",
    "KDJFactor",
    "CCIFactor",
    "WilliamsRFactor",

    # 成交量因子
    "VolumeRatioFactor",
    "TurnoverRateFactor",
    "OBVFactor",
    "VWAPFactor",

    # 价格形态因子
    "PricePositionFactor",
    "GapFactor",
    "IntradayReturnFactor",
    "OvernightReturnFactor",

    # 均线因子
    "MASlopeFactor",
    "MACrossFactor",
    "BiasFactor",

    # 基本面因子
    "PERatioFactor",
    "PBRatioFactor",
    "ROEFactor",
    "RevenueGrowthFactor",
    "ProfitGrowthFactor",
    "MarketCapFactor",

    # 情绪因子
    "MoneyFlowFactor",
    "AmihudIlliquidityFactor",

    # 注册表
    "FACTOR_REGISTRY",
    "get_factor",
    "list_factors",

    # 因子组合
    "FactorCombiner",
    "CombinedFactor",

    # ML 因子
    "XGBoostFactor",
    "LGBMFactor",

    # Qlib
    "QlibModelAdapter",
    "QlibFactorConverter",
    "QlibDataHandler",

    # RL
    "TradingEnvironment",
    "BaseRLAgent",
    "DQNAgent",
    "PPOAgent",
    "RLStrategy",
]
