<div align="right">
  English | <a href="README.md">中文</a>
</div>

<div align='center'>

# DQuant

### ⚡ Lightweight AI-Powered Quantitative Trading Framework

*A Python framework integrating backtesting, AI/ML factor discovery, and live trading*

[![Stars](https://img.shields.io/github/stars/datacruiser/dquant?style=flat&logo=github&color=yellow)](https://github.com/datacruiser/dquant/stargazers)
[![Forks](https://img.shields.io/github/forks/datacruiser/dquant?style=flat&logo=github&color=blue)](https://github.com/datacruiser/dquant/network/members)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat&logo=python)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-378%20passed-brightgreen?style=flat)](tests/)

</div>

---

## ✨ Features

- **10+ Data Sources** — A-shares, US stocks, HK stocks, and more
- **AI-Friendly** — 43+ built-in factors, ML factors, RL Agent, Qlib adapter
- **Unified Backtest/Live** — Same strategy, seamless switching
- **Production-Ready** — Risk management, order tracking, auto-reconnect, graceful shutdown
- **High Performance** — Vectorized backtest engine, Numba acceleration, parallel computing
- **Extensible** — Modular design for easy addition of new data sources and strategies

## 📦 Installation

```bash
pip install -e .
```

See [Installation Guide](#installation-dependencies) for details.

## 🚀 Quick Start

```python
from dquant import Engine, AKShareLoader, MLFactorStrategy, XGBoostFactor

# 1. Load data
data = AKShareLoader(symbols='hs300', start='2022-01-01').load()

# 2. Create strategy
strategy = MLFactorStrategy(factor=XGBoostFactor(features=['momentum_20']), top_k=10)

# 3. Backtest
engine = Engine(data, strategy)
result = engine.backtest(start='2022-06-01', end='2024-01-01')
print(result.metrics)
```

## 📊 Data Sources

| Source | Type | Description | Install |
|--------|------|-------------|---------|
| CSV | Local | CSV file loader | Built-in |
| AKShare | Online | Free A-share data | `pip install akshare` |
| Tushare | Online | Professional financial data | `pip install tushare` |
| Yahoo | Online | Global market data | `pip install yfinance` |
| JQData | Online | JoinQuant platform | `pip install jqdatasdk` |
| RiceQuant | Online | RiceQuant platform | `pip install rqdatac` |
| TDX | Local | Tongdaxin local files | Built-in |
| SQL | Database | MySQL/SQLite | `pip install sqlalchemy` |
| MongoDB | Database | NoSQL database | `pip install pymongo` |

## 📈 Backtest Engine

- Total return / Annualized return / Sharpe ratio / Max drawdown / Calmar ratio
- Benchmark NAV curve comparison
- Slippage simulation, A-share trading costs (commission + stamp duty)
- Signal-driven sell + implicit rebalance

```python
from dquant import BacktestEngine

engine = BacktestEngine(data, strategy, benchmark='000001.SZ')
result = engine.run()
print(f"Strategy: {result.metrics.total_return:.2%}  Benchmark: {result.benchmark_nav.iloc[-1]:.2f}")
```

## 🔌 Live Trading

### Broker Interfaces

| Interface | Description | Status |
|-----------|-------------|--------|
| Simulator | Simulated trading (A-share rules) | ✅ |
| XTP | Zhongtai Securities | ✅ (requires auth) |
| miniQMT | XunTou QMT | ✅ (requires auth) |

### Live Trading Architecture

```python
from dquant import Engine, Simulator
from dquant.strategy.stop_loss_take_profit import StopLossTakeProfitStrategy

# Stop-loss/take-profit decorator
strategy = StopLossTakeProfitStrategy(
    base_strategy=MoneyFlowStrategy(top_k=10),
    stop_loss=0.05,
    take_profit=0.10,
)

broker = Simulator(initial_cash=1_000_000)
engine = Engine(data, strategy, broker=broker)
engine.live(dry_run=True)
```

### Safety Mechanisms

| Capability | Description |
|------------|-------------|
| Risk Management | Drawdown monitoring + daily loss circuit breaker |
| Order Tracking | Auto-cancel timed-out PENDING/PARTIAL_FILLED orders |
| Order Retry | Exponential backoff for transient network errors |
| Graceful Shutdown | SIGINT/SIGTERM auto-cancels pending orders |
| Auto-Reconnect | Broker reconnection with exponential backoff (max 5 retries) |
| JSONL Audit | Daily trade journal files |
| Notification | DingTalk webhook + log fallback |

## 🧠 AI/ML Module

### Built-in Factors (43+)

| Category | Factors |
|----------|---------|
| Momentum | Momentum, Reversal, AccMomentum |
| Volatility | Volatility, ATR, Skewness, Kurtosis |
| Technical | RSI, MACD, Bollinger, KDJ, CCI |
| Volume | VolumeRatio, TurnoverRate, OBV, VWAP |
| Moving Avg | MASlope, MACross, Bias |
| Money Flow | MediumFlow, MainForce, SmartFlow, FlowDivergence |
| Sentiment | Sentiment, NorthboundFlow, MarginTrading |

### ML Factors

```python
from dquant import XGBoostFactor, LGBMFactor

factor = XGBoostFactor(features=['pe', 'pb', 'momentum_20'])
factor.fit(train_data)
predictions = factor.predict(test_data)
```

### Strategies

| Strategy | Description | Signals |
|----------|-------------|---------|
| MoneyFlowStrategy | Medium investor flow stock selection | BUY |
| SmartFlowStrategy | Composite smart money flow | BUY |
| FlowDivergenceStrategy | Price-flow divergence detection | BUY + SELL |
| MLFactorStrategy | ML factor TopK selection | BUY |
| TopKStrategy | Single-factor TopK rotation | BUY |
| StopLossTakeProfitStrategy | Stop-loss/take-profit decorator | SELL |

## 📁 Project Structure

```
dquant/
├── dquant/
│   ├── core.py              # Core engine (backtest + live)
│   ├── config.py            # Configuration management
│   ├── risk.py              # Risk control (PositionSizer, RiskManager, StopLoss)
│   ├── constants.py         # Centralized constants
│   ├── data/                # Data sources (10+ loaders + DataManager)
│   ├── strategy/            # Strategies (5 strategies + SL/TP decorator)
│   ├── backtest/            # Backtest engine (Portfolio, Metrics, Benchmark)
│   ├── broker/              # Broker interfaces (Simulator, XTP, QMT)
│   │   ├── base.py          # BaseBroker ABC + Order/OrderResult
│   │   ├── simulator.py     # Simulated trading
│   │   ├── retry.py         # RetryableBroker decorator
│   │   ├── order_tracker.py # Order tracking + timeout detection
│   │   ├── trade_journal.py # JSONL audit log
│   │   └── safety.py        # Trading safety checks
│   ├── notify/              # Notifications (DingTalk + log)
│   ├── ai/                  # AI module (factors + ML + RL + Qlib)
│   └── visualization/       # Visualization
├── tests/                   # Tests (197 tests)
└── pyproject.toml
```

## 📝 Roadmap

### Completed

- [x] Phase 1: Live Trading Infrastructure — RiskManager, Simulator safety, Live Loop, JSONL audit
- [x] Phase 2: Reliability — Order retry, Order tracking, Notification, Data validation
- [x] Phase 3: Strategy Enhancement — SELL signals, Stop-loss/Take-profit, Graceful shutdown, Auto-reconnect, Benchmark

### Planned

- [ ] Web dashboard (FastAPI + WebSocket)
- [ ] Config security (env var / keyring)
- [ ] Multi-strategy portfolio
- [ ] More data sources
- [ ] Futures support

## 🔧 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Current status
# 197 passed, 1 skipped, 0 failures
```

## 🤝 Contributing

1. Fork the project
2. Create a feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'feat: add amazing'`)
4. Push to the branch (`git push origin feature/amazing`)
5. Create a Pull Request

## 📄 License

[MIT License](LICENSE)

---

<div align="center">

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=datacruiser/dquant&type=Date)](https://star-history.com/#datacruiser/dquant&Date)

**If this project helps you, please consider giving it a Star ⭐**

</div>
