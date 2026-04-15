# DQuant API 文档

本文档详细介绍 DQuant 的主要 API 和使用方法。

---

## 📦 核心模块

### Engine - 交易引擎

主引擎，用于回测和实盘交易。

```python
from dquant import Engine

# 创建引擎
engine = Engine(
    data,              # 数据 (DataFrame)
    strategy,          # 策略对象
    broker=None,       # 券商接口 (可选)
    benchmark=None,    # 基准 (可选)
)

# 回测
result = engine.backtest(
    start='2022-01-01',  # 开始日期
    end='2023-12-31',    # 结束日期
)

# 实盘
engine.live(
    dry_run=True,  # True=模拟, False=实盘
)
```

**参数**:
- `data` (DataFrame): 股票数据，必须包含列: date, symbol, open, high, low, close, volume
- `strategy` (BaseStrategy): 策略对象
- `broker` (BaseBroker): 券商接口，None 表示使用默认模拟器
- `benchmark` (str): 基准指数，如 'hs300'

**返回**:
- `backtest()`: BacktestResult 对象
- `live()`: None (持续运行)

---

## 📊 数据源

### AKShareLoader - A股免费数据

```python
from dquant import AKShareLoader

loader = AKShareLoader(
    symbols='hs300',        # 股票池: 'hs300', 'zz500', 或列表
    start='2022-01-01',     # 开始日期
    end='2023-12-31',       # 结束日期
)

df = loader.load()
```

**参数**:
- `symbols`: 股票池，支持:
  - 'hs300': 沪深300
  - 'zz500': 中证500
  - 'all': 全A股
  - ['600000.SH', '000001.SZ']: 自定义列表
- `start`: 开始日期 (YYYY-MM-DD)
- `end`: 结束日期 (YYYY-MM-DD)

**返回**:
- DataFrame with columns: date, symbol, open, high, low, close, volume

---

### YahooLoader - 美股/全球市场

```python
from dquant import YahooLoader

loader = YahooLoader(
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    start='2022-01-01',
    end='2023-12-31',
)

df = loader.load()
```

---

### TushareLoader - 专业金融数据

```python
from dquant import TushareLoader

loader = TushareLoader(
    symbols='hs300',
    start='2022-01-01',
    end='2023-12-31',
    token='YOUR_TOKEN',  # 从 tushare.pro 获取
)

df = loader.load()
```

---

### DataManager - 数据管理器

统一接口 + 自动缓存。

```python
from dquant import DataManager

dm = DataManager(cache_dir='./cache')

# 加载数据
df = dm.load(
    source='akshare',
    symbols='hs300',
    start='2022-01-01',
)

# 增量更新
df = dm.update(
    source='akshare',
    symbols='hs300',
)
```

---

### load_data - 快捷函数

```python
from dquant import load_data

df = load_data(
    'akshare',
    symbols='hs300',
    start='2022-01-01',
)
```

---

## 🎯 策略

### BaseStrategy - 基础策略

所有策略的基类。

```python
from dquant import BaseStrategy

class MyStrategy(BaseStrategy):
    def __init__(self):
        super().__init__(name='MyStrategy')
    
    def generate_signals(self, data):
        # 实现信号生成逻辑
        signals = []
        for symbol, group in data.groupby('symbol'):
            # 计算信号
            signal = self._calculate_signal(group)
            signals.append({
                'date': group.index[-1],
                'symbol': symbol,
                'signal': signal,
            })
        return pd.DataFrame(signals)
```

---

### TopKStrategy - TopK 选股策略

基于因子排序的选股策略。

```python
from dquant import TopKStrategy, MomentumFactor

strategy = TopKStrategy(
    factor=MomentumFactor(window=20),  # 因子对象
    top_k=10,                          # 选股数量
    rebalance_freq=5,                  # 调仓频率 (天)
)
```

**参数**:
- `factor` (BaseFactor): 因子对象
- `top_k` (int): 选股数量
- `rebalance_freq` (int): 调仓频率，每N天调仓

---

### MLFactorStrategy - 机器学习策略

基于 ML 因子的策略。

```python
from dquant import MLFactorStrategy, XGBoostFactor

strategy = MLFactorStrategy(
    factor=XGBoostFactor(
        features=['momentum_20', 'volatility_20'],
        target='return_5d',
    ),
    top_k=10,
    rebalance_freq=5,
)
```

---

### MoneyFlowStrategy - 资金流策略

基于资金流的策略。

```python
from dquant import MoneyFlowStrategy
from dquant.ai.money_flow_factors import MediumFlowFactor

strategy = MoneyFlowStrategy(
    factor=MediumFlowFactor(window=20),
    top_k=10,
    rebalance_freq=5,
)
```

---

## 📈 因子

### 内置因子

```python
from dquant import (
    MomentumFactor,      # 动量因子
    ReversalFactor,      # 反转因子
    VolatilityFactor,    # 波动率因子
    ATRFactor,          # ATR因子
    RSIFactor,          # RSI因子
    MACDFactor,         # MACD因子
    BollingerPositionFactor,    # 布林带因子
    TrendStrengthFactor,        # 趋势因子
    VolumeRatioFactor,  # 量比因子
    PricePositionFactor, # 价格位置因子
)

# 创建因子
momentum = MomentumFactor(window=20)
rsi = RSIFactor(window=14)

# 计算因子值
result = momentum.predict(data)
```

### 因子方法

```python
# 训练 (ML因子需要)
factor.fit(train_data, target)

# 预测
result = factor.predict(data)

# 获取特征重要性 (ML因子)
importance = factor.get_feature_importance()
```

---

### 因子组合

```python
from dquant import FactorCombiner, MomentumFactor, VolatilityFactor

# 创建组合器
combiner = FactorCombiner()

# 添加因子
combiner.add_factor('momentum', MomentumFactor(20))
combiner.add_factor('volatility', VolatilityFactor(20))

# 计算因子值
combiner.fit(data)

# 组合 (等权)
combined = combiner.combine(method='equal')

# 组合 (IC加权)
combined = combiner.combine(method='ic_weight')

# 查看相关性
corr = combiner.get_factor_correlation()
```

---

## 🤖 机器学习

### XGBoostFactor

```python
from dquant import XGBoostFactor

factor = XGBoostFactor(
    features=['momentum_20', 'volatility_20', 'rsi_14'],
    target='return_5d',
    params={
        'n_estimators': 100,
        'max_depth': 5,
        'learning_rate': 0.1,
    }
)

# 训练
factor.fit(train_data)

# 预测
predictions = factor.predict(test_data)

# 特征重要性
importance = factor.get_feature_importance()
```

---

### LGBMFactor

```python
from dquant import LGBMFactor

factor = LGBMFactor(
    features=['momentum_20', 'volatility_20'],
    target='return_5d',
    params={
        'n_estimators': 100,
        'max_depth': 5,
    }
)
```

---

## 🔌 券商接口

### Simulator - 模拟器

```python
from dquant import Simulator

broker = Simulator(
    initial_cash=1000000,  # 初始资金
    commission=0.0003,     # 佣金率
    slippage=0.0001,       # 滑点
)
```

---

### QMTBroker - 中航证券 QMT

```python
from dquant import QMTBroker

broker = QMTBroker(
    qmt_path='C:/中航证券QMT/userdata_mini',
    account='YOUR_ACCOUNT',
)

# 获取账户信息
account = broker.get_account()

# 获取持仓
positions = broker.get_positions()

# 下单
from dquant.broker.base import Order
order = broker.place_order(Order(
    symbol='600000.SH',
    side='BUY',
    quantity=1000,
    price=10.5,
))

# 撤单
broker.cancel_order(order.order_id)
```

---

### XTPBroker - 中泰证券 XTP

```python
from dquant import XTPBroker

broker = XTPBroker(
    server='120.27.164.138',
    port=6001,
    account='YOUR_ACCOUNT',
    password='YOUR_PASSWORD',
)
```

---

## 🛡️ 安全系统

### TradingSafety - 交易安全控制器

```python
from dquant.broker.safety import TradingSafety

safety = TradingSafety(
    max_position_pct=0.1,   # 单只股票最大仓位 10%
    max_daily_loss=0.05,    # 日最大亏损 5%
    stop_loss_pct=0.03,     # 止损 3%
)

# 检查订单
is_valid, message = safety.check_order(
    symbol='600000.SH',
    side='buy',
    quantity=1000,
    price=10.5,
    account=account,
)

# 检查资金
is_valid, message = safety.check_fund(
    required=10500,
    account=account,
)

# 检查交易时间
is_valid, message = safety.check_time()
```

---

### OrderValidator - 订单验证器

```python
from dquant.broker.safety import OrderValidator

validator = OrderValidator()

# 验证订单
is_valid, message = validator.validate(
    symbol='600000.SH',
    side='buy',
    quantity=1000,
    price=10.5,
)
```

---

## 📊 回测结果

### BacktestResult

```python
result = engine.backtest()

# 绩效指标
metrics = result.metrics
print(f"总收益率: {metrics['total_return']:.2%}")
print(f"年化收益率: {metrics['annual_return']:.2%}")
print(f"夏普比率: {metrics['sharpe_ratio']:.2f}")
print(f"最大回撤: {metrics['max_drawdown']:.2%}")

# 净值曲线
nav = result.nav

# 持仓记录
positions = result.positions

# 交易记录
trades = result.trades
```

---

### BacktestPlotter - 可视化

```python
from dquant import BacktestPlotter

plotter = BacktestPlotter(result)

# 净值曲线
plotter.plot_nav()

# 回撤曲线
plotter.plot_drawdown()

# 月度收益
plotter.plot_monthly_returns()

# 年度收益
plotter.plot_yearly_returns()

# 全部图表
plotter.plot_all()

# 保存
plotter.save('./output/report.html')
```

---

## 📝 配置

### DQuantConfig

```python
from dquant import DQuantConfig

# 从文件加载 (仅支持 JSON 格式)
config = DQuantConfig.from_file('configs/config.json')

# 从环境变量加载
config = DQuantConfig.from_env()

# 访问配置
print(config.data.default_source)
print(config.backtest.initial_cash)
print(config.live.broker)
```

---

## 🔧 工具函数

### list_factors - 列出所有因子

```python
from dquant import list_factors

factors = list_factors()
print(factors)
# ['momentum', 'reversal', 'volatility', 'rsi', ...]
```

---

### get_factor - 获取因子

```python
from dquant import get_factor

# 获取因子
momentum = get_factor('momentum', window=20)
rsi = get_factor('rsi', window=14)
```

---

## 📚 完整示例

### 回测示例

```python
from dquant import (
    Engine,
    AKShareLoader,
    TopKStrategy,
    MomentumFactor,
    BacktestPlotter,
)

# 1. 加载数据
data = AKShareLoader(
    symbols='hs300',
    start='2022-01-01',
    end='2023-12-31'
).load()

# 2. 创建策略
strategy = TopKStrategy(
    factor=MomentumFactor(window=20),
    top_k=10,
    rebalance_freq=5,
)

# 3. 回测
engine = Engine(data, strategy)
result = engine.backtest(start='2022-06-01', end='2023-12-31')

# 4. 查看结果
print(f"总收益率: {result.metrics['total_return']:.2%}")
print(f"夏普比率: {result.metrics['sharpe_ratio']:.2f}")

# 5. 可视化
plotter = BacktestPlotter(result)
plotter.plot_all()
```

---

### 实盘示例

```python
from dquant import (
    Engine,
    QMTBroker,
    TopKStrategy,
    MomentumFactor,
)
from dquant.broker.safety import TradingSafety

# 1. 加载数据
data = load_data('akshare', symbols='hs300')

# 2. 创建策略
strategy = TopKStrategy(
    factor=MomentumFactor(window=20),
    top_k=10,
)

# 3. 创建风控
safety = TradingSafety(
    max_position_pct=0.1,
    max_daily_loss=0.05,
)

# 4. 创建券商接口
broker = QMTBroker(
    qmt_path='C:/中航证券QMT/userdata_mini',
    account='YOUR_ACCOUNT',
    safety=safety,
)

# 5. 运行实盘
engine = Engine(data, strategy, broker=broker)
engine.live(dry_run=False)
```

---

## 📞 更多帮助

- [GitHub](https://github.com/datacruiser/dquant)
- [Issues](https://github.com/datacruiser/dquant/issues)
- [Discord](https://discord.com/invite/clawd)

---

**DQuant API 文档 - v0.1.0**
