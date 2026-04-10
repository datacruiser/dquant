"""
DQuant - 轻量级AI量化框架

使用延迟导入（lazy import），仅在首次访问时加载对应模块，
避免因缺少可选依赖（xgboost、lightgbm、qlib 等）导致 import 失败。
"""

__version__ = "0.1.0"

# ---------- 延迟导入映射 ----------
# 格式: "公开名称": "module.path:ClassName"
_SUBMODULES = {
    # Core
    "Engine": "dquant.core:Engine",
    # Data - Base
    "DataSource": "dquant.data.base:DataSource",
    "CSVLoader": "dquant.data.csv_loader:CSVLoader",
    # Data - Online
    "AKShareLoader": "dquant.data.akshare_loader:AKShareLoader",
    "AKShareRealTime": "dquant.data.akshare_loader:AKShareRealTime",
    "TushareLoader": "dquant.data.tushare_loader:TushareLoader",
    "TushareFinancial": "dquant.data.tushare_loader:TushareFinancial",
    "YahooLoader": "dquant.data.yahoo_loader:YahooLoader",
    "YahooRealTime": "dquant.data.yahoo_loader:YahooRealTime",
    "JQDataLoader": "dquant.data.jqdata_loader:JQDataLoader",
    "JQDataFactor": "dquant.data.jqdata_loader:JQDataFactor",
    "RiceQuantLoader": "dquant.data.ricequant_loader:RiceQuantLoader",
    # Data - Local
    "TDXLoader": "dquant.data.tdx_loader:TDXLoader",
    "TDXBlockLoader": "dquant.data.tdx_loader:TDXBlockLoader",
    # Data - Database
    "DatabaseLoader": "dquant.data.database_loader:DatabaseLoader",
    "MongoLoader": "dquant.data.database_loader:MongoLoader",
    # Data - Manager
    "DataManager": "dquant.data.data_manager:DataManager",
    "DataSourceRegistry": "dquant.data.data_manager:DataSourceRegistry",
    "load_data": "dquant.data.data_manager:load_data",
    # Data - Validation
    "DataValidator": "dquant.data.validators:DataValidator",
    "DataCleaner": "dquant.data.validators:DataCleaner",
    "DataQualityReport": "dquant.data.validators:DataQualityReport",
    # Strategy
    "BaseStrategy": "dquant.strategy.base:BaseStrategy",
    "Signal": "dquant.strategy.base:Signal",
    "SignalType": "dquant.strategy.base:SignalType",
    "MLFactorStrategy": "dquant.strategy.ml_strategy:MLFactorStrategy",
    "TopKStrategy": "dquant.strategy.ml_strategy:TopKStrategy",
    # Strategy - Flow
    "MoneyFlowStrategy": "dquant.strategy.flow_strategy:MoneyFlowStrategy",
    "SmartFlowStrategy": "dquant.strategy.flow_strategy:SmartFlowStrategy",
    "FlowDivergenceStrategy": "dquant.strategy.flow_strategy:FlowDivergenceStrategy",
    # Backtest
    "BacktestEngine": "dquant.backtest.engine:BacktestEngine",
    "Portfolio": "dquant.backtest.portfolio:Portfolio",
    "Metrics": "dquant.backtest.metrics:Metrics",
    "BacktestResult": "dquant.backtest.result:BacktestResult",
    # Backtest - Event Driven
    "EventType": "dquant.backtest.event_driven:EventType",
    "Event": "dquant.backtest.event_driven:Event",
    "MarketEvent": "dquant.backtest.event_driven:MarketEvent",
    "SignalEvent": "dquant.backtest.event_driven:SignalEvent",
    "OrderEvent": "dquant.backtest.event_driven:OrderEvent",
    "FillEvent": "dquant.backtest.event_driven:FillEvent",
    "SlippageModel": "dquant.backtest.event_driven:SlippageModel",
    "ExecutionHandler": "dquant.backtest.event_driven:ExecutionHandler",
    "EventDrivenBacktest": "dquant.backtest.event_driven:EventDrivenBacktest",
    # AI - Base
    "BaseFactor": "dquant.ai.base:BaseFactor",
    # AI - Registry
    "FACTOR_REGISTRY": "dquant.ai.builtin_factors:FACTOR_REGISTRY",
    "get_factor": "dquant.ai.builtin_factors:get_factor",
    "list_factors": "dquant.ai.builtin_factors:list_factors",
    # AI - 动量类因子
    "MomentumFactor": "dquant.ai.builtin_factors:MomentumFactor",
    "ReversalFactor": "dquant.ai.builtin_factors:ReversalFactor",
    "AccMomentumFactor": "dquant.ai.builtin_factors:AccMomentumFactor",
    # AI - 波动率类因子
    "VolatilityFactor": "dquant.ai.builtin_factors:VolatilityFactor",
    "ATRFactor": "dquant.ai.builtin_factors:ATRFactor",
    "SkewnessFactor": "dquant.ai.builtin_factors:SkewnessFactor",
    "KurtosisFactor": "dquant.ai.builtin_factors:KurtosisFactor",
    "MaxDrawdownFactor": "dquant.ai.builtin_factors:MaxDrawdownFactor",
    # AI - 技术指标因子
    "RSIFactor": "dquant.ai.builtin_factors:RSIFactor",
    "MACDFactor": "dquant.ai.builtin_factors:MACDFactor",
    "BollingerPositionFactor": "dquant.ai.builtin_factors:BollingerPositionFactor",
    "TrendStrengthFactor": "dquant.ai.builtin_factors:TrendStrengthFactor",
    "KDJFactor": "dquant.ai.builtin_factors:KDJFactor",
    "CCIFactor": "dquant.ai.builtin_factors:CCIFactor",
    "WilliamsRFactor": "dquant.ai.builtin_factors:WilliamsRFactor",
    # AI - 成交量因子
    "VolumeRatioFactor": "dquant.ai.builtin_factors:VolumeRatioFactor",
    "TurnoverRateFactor": "dquant.ai.builtin_factors:TurnoverRateFactor",
    "OBVFactor": "dquant.ai.builtin_factors:OBVFactor",
    "VWAPFactor": "dquant.ai.builtin_factors:VWAPFactor",
    # AI - 价格形态因子
    "PricePositionFactor": "dquant.ai.builtin_factors:PricePositionFactor",
    "GapFactor": "dquant.ai.builtin_factors:GapFactor",
    "IntradayReturnFactor": "dquant.ai.builtin_factors:IntradayReturnFactor",
    "OvernightReturnFactor": "dquant.ai.builtin_factors:OvernightReturnFactor",
    # AI - 均线因子
    "MASlopeFactor": "dquant.ai.builtin_factors:MASlopeFactor",
    "MACrossFactor": "dquant.ai.builtin_factors:MACrossFactor",
    "BiasFactor": "dquant.ai.builtin_factors:BiasFactor",
    # AI - 基本面因子
    "PERatioFactor": "dquant.ai.builtin_factors:PERatioFactor",
    "PBRatioFactor": "dquant.ai.builtin_factors:PBRatioFactor",
    "ROEFactor": "dquant.ai.builtin_factors:ROEFactor",
    "RevenueGrowthFactor": "dquant.ai.builtin_factors:RevenueGrowthFactor",
    "ProfitGrowthFactor": "dquant.ai.builtin_factors:ProfitGrowthFactor",
    "MarketCapFactor": "dquant.ai.builtin_factors:MarketCapFactor",
    # AI - 情绪因子
    "MoneyFlowFactor": "dquant.ai.builtin_factors:MoneyFlowFactor",
    "AmihudIlliquidityFactor": "dquant.ai.builtin_factors:AmihudIlliquidityFactor",
    # AI - Combiner
    "FactorCombiner": "dquant.ai.factor_combiner:FactorCombiner",
    "CombinedFactor": "dquant.ai.factor_combiner:CombinedFactor",
    # AI - ML
    "XGBoostFactor": "dquant.ai.ml_factors:XGBoostFactor",
    "LGBMFactor": "dquant.ai.ml_factors:LGBMFactor",
    "QlibModelAdapter": "dquant.ai.qlib_adapter:QlibModelAdapter",
    # AI - RL
    "DQNAgent": "dquant.ai.rl_agents:DQNAgent",
    "TradingEnvironment": "dquant.ai.rl_agents:TradingEnvironment",
    # AI - Factor Analysis
    "FactorAnalyzer": "dquant.ai.factor_analysis:FactorAnalyzer",
    "FactorAnalysisResult": "dquant.ai.factor_analysis:FactorAnalysisResult",
    "FactorReport": "dquant.ai.factor_analysis:FactorReport",
    # AI - Alternative Factors
    "SentimentFactor": "dquant.ai.alternative_factors:SentimentFactor",
    "NewsSentimentFactor": "dquant.ai.alternative_factors:NewsSentimentFactor",
    "SocialMediaFactor": "dquant.ai.alternative_factors:SocialMediaFactor",
    "NorthboundFlowFactor": "dquant.ai.alternative_factors:NorthboundFlowFactor",
    "MarginTradingFactor": "dquant.ai.alternative_factors:MarginTradingFactor",
    "InstitutionalFlowFactor": "dquant.ai.alternative_factors:InstitutionalFlowFactor",
    "ShortInterestFactor": "dquant.ai.alternative_factors:ShortInterestFactor",
    "AnalystRatingFactor": "dquant.ai.alternative_factors:AnalystRatingFactor",
    "OptionsFlowFactor": "dquant.ai.alternative_factors:OptionsFlowFactor",
    # AI - Extended Factors - Technical
    "ADXFactor": "dquant.ai.extended_factors:ADXFactor",
    "AroonFactor": "dquant.ai.extended_factors:AroonFactor",
    "StochasticFactor": "dquant.ai.extended_factors:StochasticFactor",
    "ROCFactor": "dquant.ai.extended_factors:ROCFactor",
    "CMOFactor": "dquant.ai.extended_factors:CMOFactor",
    "MFIFactor": "dquant.ai.extended_factors:MFIFactor",
    # AI - Extended Factors - Volume-Price
    "ADLineFactor": "dquant.ai.extended_factors:ADLineFactor",
    "ChaikinOscillatorFactor": "dquant.ai.extended_factors:ChaikinOscillatorFactor",
    "EaseOfMovementFactor": "dquant.ai.extended_factors:EaseOfMovementFactor",
    "ForceIndexFactor": "dquant.ai.extended_factors:ForceIndexFactor",
    "VPTFactor": "dquant.ai.extended_factors:VPTFactor",
    # AI - Extended Factors - Statistical
    "HurstExponentFactor": "dquant.ai.extended_factors:HurstExponentFactor",
    "AutocorrelationFactor": "dquant.ai.extended_factors:AutocorrelationFactor",
    "VarianceRatioFactor": "dquant.ai.extended_factors:VarianceRatioFactor",
    "BetaFactor": "dquant.ai.extended_factors:BetaFactor",
    "AlphaFactor": "dquant.ai.extended_factors:AlphaFactor",
    # AI - Money Flow Factors
    "MediumFlowFactor": "dquant.ai.money_flow_factors:MediumFlowFactor",
    "MainForceFactor": "dquant.ai.money_flow_factors:MainForceFactor",
    "SmartFlowFactor": "dquant.ai.money_flow_factors:SmartFlowFactor",
    "RetailFlowFactor": "dquant.ai.money_flow_factors:RetailFlowFactor",
    "FlowDivergenceFactor": "dquant.ai.money_flow_factors:FlowDivergenceFactor",
    # Broker
    "BaseBroker": "dquant.broker.base:BaseBroker",
    "Order": "dquant.broker.base:Order",
    "OrderResult": "dquant.broker.base:OrderResult",
    "Simulator": "dquant.broker.simulator:Simulator",
    "TradingSafety": "dquant.broker.safety:TradingSafety",
    "OrderValidator": "dquant.broker.safety:OrderValidator",
    "FundChecker": "dquant.broker.safety:FundChecker",
    "TradingTimeChecker": "dquant.broker.safety:TradingTimeChecker",
    # Risk Management
    "PositionSizer": "dquant.risk:PositionSizer",
    "PositionLimit": "dquant.risk:PositionLimit",
    "RiskManager": "dquant.risk:RiskManager",
    "RiskMetrics": "dquant.risk:RiskMetrics",
    "StopLoss": "dquant.risk:StopLoss",
    "TakeProfit": "dquant.risk:TakeProfit",
    # Config
    "DQuantConfig": "dquant.config:DQuantConfig",
    "BacktestConfig": "dquant.config:BacktestConfig",
    "DataConfig": "dquant.config:DataConfig",
    "FactorConfig": "dquant.config:FactorConfig",
    "MLConfig": "dquant.config:MLConfig",
    "default_config": "dquant.config:default_config",
    # Logger
    "get_logger": "dquant.logger:get_logger",
    "set_log_level": "dquant.logger:set_log_level",
    "quiet_mode": "dquant.logger:quiet_mode",
    "debug_mode": "dquant.logger:debug_mode",
    "LoggerMixin": "dquant.logger:LoggerMixin",
    # Utils
    "get_trading_days": "dquant.utils:get_trading_days",
    "is_trading_day": "dquant.utils:is_trading_day",
    "get_previous_trading_day": "dquant.utils:get_previous_trading_day",
    "get_next_trading_day": "dquant.utils:get_next_trading_day",
    "annualized_return": "dquant.utils:annualized_return",
    "annualized_volatility": "dquant.utils:annualized_volatility",
    "sharpe_ratio": "dquant.utils:sharpe_ratio",
    "max_drawdown": "dquant.utils:max_drawdown",
    "sortino_ratio": "dquant.utils:sortino_ratio",
    "calmar_ratio": "dquant.utils:calmar_ratio",
    "winsorize": "dquant.utils:winsorize",
    "standardize": "dquant.utils:standardize",
    "normalize": "dquant.utils:normalize",
    "format_money": "dquant.utils:format_money",
    "format_percent": "dquant.utils:format_percent",
    # Performance
    "NumbaAccelerator": "dquant.performance:NumbaAccelerator",
    "ParallelProcessor": "dquant.performance:ParallelProcessor",
    "VectorizedOperations": "dquant.performance:VectorizedOperations",
    "CacheManager": "dquant.performance:CacheManager",
    "PerformanceMonitor": "dquant.performance:PerformanceMonitor",
    "accelerator": "dquant.performance:accelerator",
    "parallel_processor": "dquant.performance:parallel_processor",
    "cache_manager": "dquant.performance:cache_manager",
    "performance_monitor": "dquant.performance:performance_monitor",
    "timing": "dquant.performance:timing",
    # Realtime
    "RealtimeQuote": "dquant.realtime:RealtimeQuote",
    "RealtimeDataSource": "dquant.realtime:RealtimeDataSource",
    "MockRealtimeSource": "dquant.realtime:MockRealtimeSource",
    "RealtimeManager": "dquant.realtime:RealtimeManager",
    "RealtimeServer": "dquant.realtime:RealtimeServer",
    "RealtimeClient": "dquant.realtime:RealtimeClient",
    "create_mock_realtime_manager": "dquant.realtime:create_mock_realtime_manager",
    # Visualization
    "BacktestPlotter": "dquant.visualization.plotter:BacktestPlotter",
    "plot_backtest": "dquant.visualization.plotter:plot_backtest",
}

__all__ = list(_SUBMODULES.keys())


def __getattr__(name):
    """延迟导入：仅在首次访问时加载对应模块。"""
    if name in _SUBMODULES:
        module_path, attr_name = _SUBMODULES[name].rsplit(":", 1)
        import importlib

        try:
            mod = importlib.import_module(module_path)
            attr = getattr(mod, attr_name)
        except ImportError as e:
            raise ImportError(
                f"Cannot import '{name}' from {module_path}: {e}. "
                f"You may need to install additional dependencies."
            ) from e
        # 缓存到模块全局，后续访问直接返回
        globals()[name] = attr
        return attr
    raise AttributeError(f"module 'dquant' has no attribute {name!r}")


def __dir__():
    """支持 IDE 自动补全。"""
    return list(_SUBMODULES.keys())
