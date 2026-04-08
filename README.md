<div align="right">
  <a href="README_EN.md">English</a> | 中文
</div>

<div align='center'>

# DQuant

### ⚡ 轻量级 AI 量化交易框架

*整合回测研究、AI/ML 因子挖掘与实盘交易的 Python 量化框架*

[![Stars](https://img.shields.io/github/stars/datacruiser/dquant?style=flat&logo=github&color=yellow)](https://github.com/datacruiser/dquant/stargazers)
[![Forks](https://img.shields.io/github/forks/datacruiser/dquant?style=flat&logo=github&color=blue)](https://github.com/datacruiser/dquant/network/members)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat&logo=python)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-232%20passed-brightgreen?style=flat)](tests/)

</div>

---

## ✨ 特性

- **多数据源支持** — 10+ 数据源，覆盖 A 股、美股、港股等
- **AI 友好** — 内置 43+ 因子、ML 因子、RL Agent、Qlib 适配
- **回测/实盘统一** — 同一策略无缝切换回测与实盘
- **实盘就绪** — 风控管理、订单追踪、自动重连、优雅关机
- **高性能** — 向量化回测引擎，Numba 加速，并行计算
- **T+1 规则** — A 股 T+1 交收制度，回测与实盘统一强制
- **易扩展** — 模块化设计，易于添加新数据源和策略

## 📦 安装

```bash
pip install -e .
```

详见 [安装指南](#安装依赖)。

## 🚀 快速开始

```python
from dquant import Engine, AKShareLoader, MLFactorStrategy, XGBoostFactor

# 1. 加载数据
data = AKShareLoader(symbols='hs300', start='2022-01-01').load()

# 2. 创建策略
strategy = MLFactorStrategy(factor=XGBoostFactor(features=['momentum_20']), top_k=10)

# 3. 回测
engine = Engine(data, strategy)
result = engine.backtest(start='2022-06-01', end='2024-01-01')
print(result.metrics)
```

## 📊 数据源

| 数据源 | 类型 | 说明 | 安装 |
|--------|------|------|------|
| CSV | 本地 | CSV 文件加载 | 内置 |
| AKShare | 在线 | 免费 A 股数据 | `pip install akshare` |
| Tushare | 在线 | 专业金融数据 | `pip install tushare` |
| Yahoo | 在线 | 全球市场数据 | `pip install yfinance` |
| 聚宽 | 在线 | 量化平台数据 | `pip install jqdatasdk` |
| 米筐 | 在线 | 量化平台数据 | `pip install rqdatac` |
| 通达信 | 本地 | 本地行情文件 | 内置 |
| SQL | 数据库 | MySQL/SQLite | `pip install sqlalchemy` |
| MongoDB | 数据库 | NoSQL 数据库 | `pip install pymongo` |

## 📈 回测引擎

- 总收益率 / 年化收益率 / 夏普比率 / 最大回撤 / 卡玛比率
- 支持基准净值曲线对比
- 滑点模拟、A 股交易成本（佣金 + 印花税）
- T+1 冻结/释放：当日买入股份次日方可卖出
- 零股整手约束：不足一手自动清仓并补偿
- 信号驱动卖出 + 策略隐式调仓

```python
from dquant import BacktestEngine

engine = BacktestEngine(data, strategy, benchmark='000001.SZ')
result = engine.run()
print(f"策略: {result.metrics.total_return:.2%}  基准: {result.benchmark_nav.iloc[-1]:.2f}")
```

## 🔌 实盘交易

### 券商接口

| 接口 | 说明 | 状态 |
|------|------|------|
| Simulator | 模拟交易（含 A 股规则） | ✅ |
| XTP | 中泰证券 | ✅ (需权限) |
| miniQMT | 迅投 QMT | ✅ (需权限) |

### 实盘架构

```python
from dquant import Engine, Simulator
from dquant.strategy.stop_loss_take_profit import StopLossTakeProfitStrategy

# 止损止盈装饰器
strategy = StopLossTakeProfitStrategy(
    base_strategy=MoneyFlowStrategy(top_k=10),
    stop_loss=0.05,
    take_profit=0.10,
)

broker = Simulator(initial_cash=1_000_000)
engine = Engine(data, strategy, broker=broker)
engine.live(dry_run=True)
```

### 实盘安全机制

| 能力 | 说明 |
|------|------|
| 风控管理 | 回撤监控 + 日亏损熔断 |
| T+1 锁仓 | 当日买入锁定，次日释放，回测/实盘统一 |
| 并发行情 | ThreadPoolExecutor 并发拉取多标的数据 |
| 订单追踪 | PENDING/PARTIAL_FILLED 超时自动取消 |
| 订单重试 | 指数退避重试瞬态网络错误 |
| 优雅关机 | SIGINT/SIGTERM 自动取消 pending 订单 |
| 自动重连 | Broker 断线指数退避重连 (最多 5 次) |
| JSONL 审计 | 每日独立交易日志文件 |
| 通知系统 | 钉钉 Webhook + 日志回退 |

## 🧠 AI/ML 模块

### 内置因子 (43+)

| 类别 | 因子 |
|------|------|
| 动量 | Momentum, Reversal, AccMomentum |
| 波动率 | Volatility, ATR, Skewness, Kurtosis |
| 技术指标 | RSI, MACD, Bollinger, KDJ, CCI |
| 成交量 | VolumeRatio, TurnoverRate, OBV, VWAP |
| 均线 | MASlope, MACross, Bias |
| 资金流 | MediumFlow, MainForce, SmartFlow, FlowDivergence |
| 情绪 | Sentiment, NorthboundFlow, MarginTrading |

### ML 因子

```python
from dquant import XGBoostFactor, LGBMFactor

factor = XGBoostFactor(features=['pe', 'pb', 'momentum_20'])
factor.fit(train_data)
predictions = factor.predict(test_data)
```

### 策略

| 策略 | 说明 | 信号 |
|------|------|------|
| MoneyFlowStrategy | 中户资金流选股 | BUY |
| SmartFlowStrategy | 综合聪明钱流 | BUY |
| FlowDivergenceStrategy | 价格-资金流背离 | BUY + SELL |
| MLFactorStrategy | ML 因子 TopK | BUY |
| TopKStrategy | 单因子 TopK 轮动 | BUY |
| StopLossTakeProfitStrategy | 止损止盈装饰器 | SELL |

## 📁 项目结构

```
dquant/
├── dquant/
│   ├── core.py              # 核心引擎 (回测 + 实盘 + 并发行情)
│   ├── config.py            # 配置管理
│   ├── risk.py              # 风控 (PositionSizer, RiskManager, StopLoss)
│   ├── constants.py         # 集中常量
│   ├── data/                # 数据源 (10+ loaders + DataManager)
│   ├── strategy/            # 策略层 (5 策略 + 止损止盈装饰器)
│   ├── backtest/            # 回测引擎 (Portfolio, Metrics, Benchmark, T+1)
│   ├── broker/              # 券商接口 (Simulator, XTP, QMT)
│   │   ├── base.py          # BaseBroker ABC + Order/OrderResult
│   │   ├── simulator.py     # 模拟交易
│   │   ├── retry.py         # RetryableBroker 装饰器
│   │   ├── order_tracker.py # 订单追踪 + 超时检测
│   │   ├── trade_journal.py # JSONL 审计日志
│   │   └── safety.py        # 交易安全检查
│   ├── notify/              # 通知 (钉钉 + 日志)
│   ├── ai/                  # AI 模块 (因子 + ML + RL + Qlib)
│   └── visualization/       # 可视化
├── tests/                   # 测试 (232 tests)
└── pyproject.toml
```

## 📝 开发路线

### 已完成

- [x] Phase 1: 实盘基础设施 — RiskManager, Simulator 安全, Live Loop, JSONL 审计
- [x] Phase 2: 可靠性增强 — 订单重试, 订单追踪, 通知系统, 数据验证
- [x] Phase 3: 策略完善 — SELL 信号, 止损止盈, 优雅关机, 自动重连, Benchmark

### 规划中

- [ ] Web 仪表盘 (FastAPI + WebSocket)
- [ ] 配置安全 (env var / keyring)
- [ ] 多策略组合
- [ ] 更多数据源
- [ ] 期货支持

## 🔧 测试

```bash
# 运行全部测试
python -m pytest tests/ -v

# 当前状态
# 232 passed, 1 skipped, 0 failures
```

## 🤝 贡献

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/amazing`)
3. 提交更改 (`git commit -m 'feat: add amazing'`)
4. 推送到分支 (`git push origin feature/amazing`)
5. 创建 Pull Request

## 📄 许可证

[MIT License](LICENSE)

---

<div align="center">

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=datacruiser/dquant&type=Date)](https://star-history.com/#datacruiser/dquant&Date)

**如果这个项目对你有帮助，请给个 Star ⭐**

</div>
