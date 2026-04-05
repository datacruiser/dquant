"""
DQuant - 轻量级AI量化框架
"""

# flake8: noqa: F401
# This file re-exports public API for package users


# AI - Base
from dquant.ai.base import BaseFactor

# AI - Built-in Factors
from dquant.ai.builtin_factors import (  # 动量类; 波动率类; 技术指标; 成交量; 价格形态; 均线; 基本面; 情绪; 注册表
    FACTOR_REGISTRY,
    AccMomentumFactor,
    AmihudIlliquidityFactor,
    ATRFactor,
    BiasFactor,
    BollingerPositionFactor,
    CCIFactor,
    GapFactor,
    IntradayReturnFactor,
    KDJFactor,
    KurtosisFactor,
    MACDFactor,
    MACrossFactor,
    MarketCapFactor,
    MASlopeFactor,
    MaxDrawdownFactor,
    MomentumFactor,
    MoneyFlowFactor,
    OBVFactor,
    OvernightReturnFactor,
    PBRatioFactor,
    PERatioFactor,
    PricePositionFactor,
    ProfitGrowthFactor,
    RevenueGrowthFactor,
    ReversalFactor,
    ROEFactor,
    RSIFactor,
    SkewnessFactor,
    TrendStrengthFactor,
    TurnoverRateFactor,
    VolatilityFactor,
    VolumeRatioFactor,
    VWAPFactor,
    WilliamsRFactor,
    get_factor,
    list_factors,
)

# AI - Factor Combiner
from dquant.ai.factor_combiner import CombinedFactor, FactorCombiner

# AI - ML Factors
from dquant.ai.ml_factors import LGBMFactor, XGBoostFactor

# AI - Qlib
from dquant.ai.qlib_adapter import QlibModelAdapter

# AI - RL
from dquant.ai.rl_agents import DQNAgent, TradingEnvironment

# Backtest
from dquant.backtest.engine import BacktestEngine
from dquant.backtest.metrics import Metrics
from dquant.backtest.portfolio import Portfolio
from dquant.backtest.result import BacktestResult

# Broker
from dquant.broker.base import BaseBroker, Order, OrderResult
from dquant.broker.safety import (
    FundChecker,
    OrderValidator,
    TradingSafety,
    TradingTimeChecker,
)
from dquant.broker.simulator import Simulator
from dquant.core import Engine
from dquant.data.akshare_loader import AKShareLoader, AKShareRealTime

# Data
from dquant.data.base import DataSource
from dquant.data.csv_loader import CSVLoader
from dquant.data.data_manager import DataManager, DataSourceRegistry, load_data
from dquant.data.database_loader import DatabaseLoader, MongoLoader
from dquant.data.jqdata_loader import JQDataFactor, JQDataLoader
from dquant.data.ricequant_loader import RiceQuantLoader
from dquant.data.tdx_loader import TDXBlockLoader, TDXLoader
from dquant.data.tushare_loader import TushareFinancial, TushareLoader
from dquant.data.yahoo_loader import YahooLoader, YahooRealTime

# Strategy
from dquant.strategy.base import BaseStrategy, Signal, SignalType
from dquant.strategy.ml_strategy import MLFactorStrategy, TopKStrategy

# Visualization
from dquant.visualization.plotter import BacktestPlotter, plot_backtest

__version__ = "0.1.0"

__all__ = [
    # Core
    "Engine",
    # Data - Base
    "DataSource",
    "CSVLoader",
    # Data - Online
    "AKShareLoader",
    "AKShareRealTime",
    "TushareLoader",
    "TushareFinancial",
    "YahooLoader",
    "YahooRealTime",
    "JQDataLoader",
    "JQDataFactor",
    "RiceQuantLoader",
    # Data - Local
    "TDXLoader",
    "TDXBlockLoader",
    # Data - Database
    "DatabaseLoader",
    "MongoLoader",
    # Data - Manager
    "DataManager",
    "DataSourceRegistry",
    "load_data",
    # Strategy
    "BaseStrategy",
    "Signal",
    "SignalType",
    "MLFactorStrategy",
    "TopKStrategy",
    # Backtest
    "BacktestEngine",
    "Portfolio",
    "Metrics",
    "BacktestResult",
    # AI - Base
    "BaseFactor",
    # AI - 动量类因子
    "MomentumFactor",
    "ReversalFactor",
    "AccMomentumFactor",
    # AI - 波动率类因子
    "VolatilityFactor",
    "ATRFactor",
    "SkewnessFactor",
    "KurtosisFactor",
    "MaxDrawdownFactor",
    # AI - 技术指标因子
    "RSIFactor",
    "MACDFactor",
    "BollingerPositionFactor",
    "TrendStrengthFactor",
    "KDJFactor",
    "CCIFactor",
    "WilliamsRFactor",
    # AI - 成交量因子
    "VolumeRatioFactor",
    "TurnoverRateFactor",
    "OBVFactor",
    "VWAPFactor",
    # AI - 价格形态因子
    "PricePositionFactor",
    "GapFactor",
    "IntradayReturnFactor",
    "OvernightReturnFactor",
    # AI - 均线因子
    "MASlopeFactor",
    "MACrossFactor",
    "BiasFactor",
    # AI - 基本面因子
    "PERatioFactor",
    "PBRatioFactor",
    "ROEFactor",
    "RevenueGrowthFactor",
    "ProfitGrowthFactor",
    "MarketCapFactor",
    # AI - 情绪因子
    "MoneyFlowFactor",
    "AmihudIlliquidityFactor",
    # AI - Registry
    "FACTOR_REGISTRY",
    "get_factor",
    "list_factors",
    # AI - Combiner
    "FactorCombiner",
    "CombinedFactor",
    # AI - ML
    "XGBoostFactor",
    "LGBMFactor",
    "QlibModelAdapter",
    "DQNAgent",
    "TradingEnvironment",
    # Broker
    "BaseBroker",
    "Order",
    "OrderResult",
    "Simulator",
    "TradingSafety",
    "OrderValidator",
    "FundChecker",
    "TradingTimeChecker",
    # Visualization
    "BacktestPlotter",
    "plot_backtest",
]

# Config
from dquant.config import (
    BacktestConfig,
    DataConfig,
    DQuantConfig,
    FactorConfig,
    MLConfig,
    default_config,
)

# Logger
from dquant.logger import LoggerMixin, debug_mode, get_logger, quiet_mode, set_log_level

# Utils
from dquant.utils import annualized_return  # 日期; 绩效; 数据处理; 格式化
from dquant.utils import (
    annualized_volatility,
    calmar_ratio,
    format_money,
    format_percent,
    get_next_trading_day,
    get_previous_trading_day,
    get_trading_days,
    is_trading_day,
    max_drawdown,
    normalize,
    sharpe_ratio,
    sortino_ratio,
    standardize,
    winsorize,
)

__all__.extend(
    [
        # Config
        "DQuantConfig",
        "BacktestConfig",
        "DataConfig",
        "FactorConfig",
        "MLConfig",
        "default_config",
        # Logger
        "get_logger",
        "set_log_level",
        "quiet_mode",
        "debug_mode",
        "LoggerMixin",
        # Utils
        "get_trading_days",
        "is_trading_day",
        "get_previous_trading_day",
        "get_next_trading_day",
        "annualized_return",
        "annualized_volatility",
        "sharpe_ratio",
        "max_drawdown",
        "sortino_ratio",
        "calmar_ratio",
        "winsorize",
        "standardize",
        "normalize",
        "format_money",
        "format_percent",
    ]
)

# Factor Analysis
from dquant.ai.factor_analysis import FactorAnalysisResult, FactorAnalyzer, FactorReport

# Data Validation
from dquant.data.validators import DataCleaner, DataQualityReport, DataValidator

# Risk Management
from dquant.risk import (
    PositionLimit,
    PositionSizer,
    RiskManager,
    RiskMetrics,
    StopLoss,
    TakeProfit,
)

__all__.extend(
    [
        # Risk Management
        "PositionSizer",
        "PositionLimit",
        "RiskManager",
        "RiskMetrics",
        "StopLoss",
        "TakeProfit",
        # Data Validation
        "DataValidator",
        "DataCleaner",
        "DataQualityReport",
        # Factor Analysis
        "FactorAnalyzer",
        "FactorAnalysisResult",
        "FactorReport",
    ]
)

# Alternative Factors
from dquant.ai.alternative_factors import (
    AnalystRatingFactor,
    InstitutionalFlowFactor,
    MarginTradingFactor,
    NewsSentimentFactor,
    NorthboundFlowFactor,
    OptionsFlowFactor,
    SentimentFactor,
    ShortInterestFactor,
    SocialMediaFactor,
)

# Event-Driven Backtest
from dquant.backtest.event_driven import (
    Event,
    EventDrivenBacktest,
    EventType,
    ExecutionHandler,
    FillEvent,
    MarketEvent,
    OrderEvent,
    SignalEvent,
    SlippageModel,
)

# Performance
from dquant.performance import (
    CacheManager,
    NumbaAccelerator,
    ParallelProcessor,
    PerformanceMonitor,
    VectorizedOperations,
    accelerator,
    cache_manager,
    parallel_processor,
    performance_monitor,
    timing,
)

# Realtime
from dquant.realtime import (
    MockRealtimeSource,
    RealtimeClient,
    RealtimeDataSource,
    RealtimeManager,
    RealtimeQuote,
    RealtimeServer,
    create_mock_realtime_manager,
)

__all__.extend(
    [
        # Performance
        "NumbaAccelerator",
        "ParallelProcessor",
        "VectorizedOperations",
        "CacheManager",
        "PerformanceMonitor",
        "accelerator",
        "parallel_processor",
        "cache_manager",
        "performance_monitor",
        "timing",
        # Realtime
        "RealtimeQuote",
        "RealtimeDataSource",
        "MockRealtimeSource",
        "RealtimeManager",
        "RealtimeServer",
        "RealtimeClient",
        "create_mock_realtime_manager",
        # Event-Driven
        "EventType",
        "Event",
        "MarketEvent",
        "SignalEvent",
        "OrderEvent",
        "FillEvent",
        "SlippageModel",
        "ExecutionHandler",
        "EventDrivenBacktest",
        # Alternative Factors
        "SentimentFactor",
        "NewsSentimentFactor",
        "SocialMediaFactor",
        "NorthboundFlowFactor",
        "MarginTradingFactor",
        "InstitutionalFlowFactor",
        "ShortInterestFactor",
        "AnalystRatingFactor",
        "OptionsFlowFactor",
    ]
)

# Extended Factors
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

__all__.extend(
    [
        # Extended Factors - Technical
        "ADXFactor",
        "AroonFactor",
        "StochasticFactor",
        "ROCFactor",
        "CMOFactor",
        "MFIFactor",
        # Extended Factors - Volume-Price
        "ADLineFactor",
        "ChaikinOscillatorFactor",
        "EaseOfMovementFactor",
        "ForceIndexFactor",
        "VPTFactor",
        # Extended Factors - Statistical
        "HurstExponentFactor",
        "AutocorrelationFactor",
        "VarianceRatioFactor",
        "BetaFactor",
        "AlphaFactor",
    ]
)

# 资金流因子
from dquant.ai.money_flow_factors import (
    FlowDivergenceFactor,
    MainForceFactor,
    MediumFlowFactor,
    RetailFlowFactor,
    SmartFlowFactor,
)

# 资金流策略
from dquant.strategy.flow_strategy import (
    FlowDivergenceStrategy,
    MoneyFlowStrategy,
    SmartFlowStrategy,
)
