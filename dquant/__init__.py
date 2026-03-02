"""
DQuant - 轻量级AI量化框架
"""

from dquant.core import Engine

# Data
from dquant.data.base import DataSource
from dquant.data.csv_loader import CSVLoader
from dquant.data.akshare_loader import AKShareLoader, AKShareRealTime
from dquant.data.tushare_loader import TushareLoader, TushareFinancial
from dquant.data.yahoo_loader import YahooLoader, YahooRealTime
from dquant.data.jqdata_loader import JQDataLoader, JQDataFactor
from dquant.data.ricequant_loader import RiceQuantLoader
from dquant.data.tdx_loader import TDXLoader, TDXBlockLoader
from dquant.data.database_loader import DatabaseLoader, MongoLoader
from dquant.data.data_manager import DataManager, DataSourceRegistry, load_data

# Strategy
from dquant.strategy.base import BaseStrategy, Signal, SignalType
from dquant.strategy.ml_strategy import MLFactorStrategy, TopKStrategy

# Backtest
from dquant.backtest.engine import BacktestEngine
from dquant.backtest.portfolio import Portfolio
from dquant.backtest.metrics import Metrics
from dquant.backtest.result import BacktestResult

# AI - Base
from dquant.ai.base import BaseFactor

# AI - Built-in Factors
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

# AI - Factor Combiner
from dquant.ai.factor_combiner import FactorCombiner, CombinedFactor

# AI - ML Factors
from dquant.ai.ml_factors import XGBoostFactor, LGBMFactor

# AI - Qlib
from dquant.ai.qlib_adapter import QlibModelAdapter

# AI - RL
from dquant.ai.rl_agents import DQNAgent, TradingEnvironment

# Broker
from dquant.broker.base import BaseBroker, Order, OrderResult
from dquant.broker.simulator import Simulator
from dquant.broker.safety import TradingSafety, OrderValidator, FundChecker, TradingTimeChecker

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
    DQuantConfig,
    BacktestConfig,
    DataConfig,
    FactorConfig,
    MLConfig,
    default_config,
)

# Logger
from dquant.logger import (
    get_logger,
    set_log_level,
    quiet_mode,
    debug_mode,
    LoggerMixin,
)

# Utils
from dquant.utils import (
    # 日期
    get_trading_days,
    is_trading_day,
    get_previous_trading_day,
    get_next_trading_day,
    # 绩效
    annualized_return,
    annualized_volatility,
    sharpe_ratio,
    max_drawdown,
    sortino_ratio,
    calmar_ratio,
    # 数据处理
    winsorize,
    standardize,
    normalize,
    # 格式化
    format_money,
    format_percent,
)

__all__.extend([
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
])

# Risk Management
from dquant.risk import (
    PositionSizer,
    PositionLimit,
    RiskManager,
    RiskMetrics,
    StopLoss,
    TakeProfit,
)

# Data Validation
from dquant.data.validators import (
    DataValidator,
    DataCleaner,
    DataQualityReport,
)

# Factor Analysis
from dquant.ai.factor_analysis import (
    FactorAnalyzer,
    FactorAnalysisResult,
    FactorReport,
)

__all__.extend([
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
])

# Performance
from dquant.performance import (
    NumbaAccelerator,
    ParallelProcessor,
    VectorizedOperations,
    CacheManager,
    PerformanceMonitor,
    accelerator,
    parallel_processor,
    cache_manager,
    performance_monitor,
    timing,
)

# Realtime
from dquant.realtime import (
    RealtimeQuote,
    RealtimeDataSource,
    MockRealtimeSource,
    RealtimeManager,
    RealtimeServer,
    RealtimeClient,
    create_mock_realtime_manager,
)

# Event-Driven Backtest
from dquant.backtest.event_driven import (
    EventType,
    Event,
    MarketEvent,
    SignalEvent,
    OrderEvent,
    FillEvent,
    SlippageModel,
    ExecutionHandler,
    EventDrivenBacktest,
)

# Alternative Factors
from dquant.ai.alternative_factors import (
    SentimentFactor,
    NewsSentimentFactor,
    SocialMediaFactor,
    NorthboundFlowFactor,
    MarginTradingFactor,
    InstitutionalFlowFactor,
    ShortInterestFactor,
    AnalystRatingFactor,
    OptionsFlowFactor,
)

__all__.extend([
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
])

# Extended Factors
from dquant.ai.extended_factors import (
    ADXFactor,
    AroonFactor,
    StochasticFactor,
    ROCFactor,
    CMOFactor,
    MFIFactor,
    ADLineFactor,
    ChaikinOscillatorFactor,
    EaseOfMovementFactor,
    ForceIndexFactor,
    VPTFactor,
    HurstExponentFactor,
    AutocorrelationFactor,
    VarianceRatioFactor,
    BetaFactor,
    AlphaFactor,
)

__all__.extend([
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
])

# 资金流因子
from dquant.ai.money_flow_factors import (
    MediumFlowFactor,
    MainForceFactor,
    RetailFlowFactor,
    SmartFlowFactor,
    FlowDivergenceFactor,
)

# 资金流策略
from dquant.strategy.flow_strategy import (
    MoneyFlowStrategy,
    SmartFlowStrategy,
    FlowDivergenceStrategy,
)
