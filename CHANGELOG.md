# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] - 2026-02-28

### Added

#### Core
- Initial release of DQuant framework
- Core Engine for backtesting and live trading
- Configuration management system (`config.py`)
- Logging system (`logger.py`)
- Utility functions (`utils.py`)

#### Data Sources
- CSV data loader
- AKShare data loader (A股免费数据)
- Tushare data loader (A股数据)
- Yahoo Finance data loader (美股/全球)
- JQData data loader (聚宽)
- RiceQuant data loader (米筐)
- TDX data loader (通达信)
- SQL database loader
- MongoDB loader
- Data Manager with caching support

#### Factors (34 built-in factors)
- **Momentum**: Momentum, Reversal, AccMomentum
- **Volatility**: Volatility, ATR, Skewness, Kurtosis, MaxDrawdown
- **Technical**: RSI, MACD, Bollinger, Trend, KDJ, CCI, WilliamsR
- **Volume**: VolumeRatio, TurnoverRate, OBV, VWAP
- **Price**: PricePosition, Gap, Intraday, Overnight
- **MA**: MASlope, MACross, Bias
- **Fundamental**: PE, PB, ROE, RevenueGrowth, ProfitGrowth, MarketCap
- **Sentiment**: MoneyFlow, Amihud

#### Factor Combination
- FactorCombiner for multi-factor combination
- Support equal weight, IC weight, IR weight, PCA methods
- Factor correlation analysis
- Factor preprocessing (standardize, winsorize)

#### Machine Learning
- XGBoost factor
- LightGBM factor
- Qlib adapter

#### Reinforcement Learning
- Trading environment (OpenAI Gym interface)
- DQN agent
- PPO agent

#### Backtest
- Vectorized backtest engine
- Portfolio management
- Performance metrics (Sharpe, MaxDD, etc.)
- Backtest visualization (NAV, drawdown, heatmap)

#### Strategy
- BaseStrategy class
- Signal generation
- ML-based strategy
- TopK strategy

#### Broker
- Simulator (paper trading)
- XTP broker interface (中泰证券)
- QMT broker interface (迅投)

#### Visualization
- BacktestPlotter
- NAV curve
- Drawdown chart
- Monthly/yearly returns heatmap

#### Documentation
- README with comprehensive examples
- INSTALL guide
- Code examples in `examples/`
- Quick start script (`quickstart.py`)
- Unit tests

#### Development
- requirements.txt (minimal, standard, full)
- pyproject.toml
- Makefile
- .gitignore
- Test suite

### Changed
- N/A (initial release)

### Fixed
- N/A (initial release)

## Roadmap

### [0.2.0] - Planned

#### Data
- [ ] Real-time data streaming
- [ ] More data sources (EastMoney, TongHuaShun)
- [ ] Futures data support
- [ ] Crypto data support

#### Factors
- [ ] More fundamental factors
- [ ] Sentiment factors (news, social media)
- [ ] Alternative data factors

#### ML
- [ ] AutoML for factor selection
- [ ] Deep learning factors (LSTM, Transformer)
- [ ] Factor mining with genetic algorithm

#### Backtest
- [ ] Event-driven backtest engine
- [ ] Transaction cost model
- [ ] Market impact model
- [ ] Multi-asset backtest

#### Portfolio
- [ ] Risk management module
- [ ] Position sizing strategies
- [ ] Portfolio optimization

#### Live Trading
- [ ] More broker integrations
- [ ] Order management system
- [ ] Risk control system
- [ ] Paper trading mode

#### UI
- [ ] Web dashboard
- [ ] Real-time monitoring
- [ ] Strategy configuration UI

#### Other
- [ ] Performance optimization
- [ ] More unit tests
- [ ] Documentation website
- [ ] CI/CD pipeline

## Version Naming

- **Major (X.0.0)**: Breaking changes
- **Minor (0.X.0)**: New features, backward compatible
- **Patch (0.0.X)**: Bug fixes, backward compatible
