# DQuant 实盘交易指南

本指南介绍如何使用 DQuant 进行实盘交易。

## 📋 前提条件

### 1. 硬件环境

- **操作系统**: Windows 10/11 (推荐)
- **内存**: 8GB+ (推荐 16GB)
- **硬盘**: 50GB+ 可用空间
- **网络**: 稳定的互联网连接

### 2. 软件环境

- **Python**: 3.8+
- **券商客户端**: 
  - 中航证券 QMT (推荐)
  - 中泰证券 XTP
- **依赖包**: 
  ```bash
  pip install -r requirements-full.txt
  ```

### 3. 券商账户

- 已开通A股交易权限
- 已开通量化交易接口权限
- 资金账号和密码

---

## 🔧 配置步骤

### 1. 安装券商客户端

#### QMT (中航证券)

1. 联系券商开通 QMT 权限
2. 下载并安装 QMT 客户端
3. 登录客户端确认能正常交易

安装路径示例:
```
C:\中航证券QMT\userdata_mini
```

#### XTP (中泰证券)

1. 联系券商开通 XTP 权限
2. 获取服务器地址和端口
3. 获取资金账号和密码

### 2. 创建配置文件

复制配置模板:
```bash
cd dquant
cp configs/config.local.yaml.example configs/config.local.yaml
```

编辑配置文件:
```yaml
# configs/config.local.yaml

# 券商配置
broker:
  # QMT 配置
  qmt:
    path: "C:/中航证券QMT/userdata_mini"
    account: "YOUR_QMT_ACCOUNT"
  
  # XTP 配置 (可选)
  xtp:
    server: "120.27.164.138"
    port: 6001
    account: "YOUR_XTP_ACCOUNT"
    password: "YOUR_XTP_PASSWORD"

# 实盘交易配置
trading:
  # 初始资金
  initial_cash: 100000  # 10万
  
  # 风控参数
  max_position_pct: 0.1   # 单只股票最大仓位 10%
  max_daily_loss: 0.05    # 日最大亏损 5%
  stop_loss_pct: 0.03     # 止损 3%
  
  # 交易时间
  trading_hours:
    morning_start: "09:30"
    morning_end: "11:30"
    afternoon_start: "13:00"
    afternoon_end: "15:00"
```

### 3. 验证配置

运行验证脚本:
```bash
python scripts/verify_config.py
```

---

## 🚀 开始实盘

### 1. 模拟测试 (推荐先执行)

使用模拟器测试策略:
```python
from dquant import Engine, Simulator, TopKStrategy, MomentumFactor
from dquant.data import AKShareLoader

# 加载数据
data = AKShareLoader(symbols='hs300', start='2023-01-01').load()

# 创建策略
strategy = TopKStrategy(
    factor=MomentumFactor(window=20),
    top_k=10,
)

# 创建模拟器
broker = Simulator(initial_cash=100000)

# 创建引擎
engine = Engine(data, strategy, broker=broker)

# 运行模拟
engine.live(dry_run=True)
```

### 2. 实盘交易

#### QMT 实盘

```python
from dquant import Engine, QMTBroker, TopKStrategy, MomentumFactor
from dquant.data import AKShareLoader
from dquant.broker.safety import TradingSafety

# 加载数据
data = AKShareLoader(symbols='hs300').load()

# 创建策略
strategy = TopKStrategy(
    factor=MomentumFactor(window=20),
    top_k=10,
)

# 创建安全控制器
safety = TradingSafety(
    max_position_pct=0.1,
    max_daily_loss=0.05,
    stop_loss_pct=0.03,
)

# 创建 QMT 接口
broker = QMTBroker(
    qmt_path='C:/中航证券QMT/userdata_mini',
    account='YOUR_ACCOUNT',
    safety=safety,
)

# 创建引擎
engine = Engine(data, strategy, broker=broker)

# 运行实盘
engine.live(dry_run=False)  # dry_run=False 为真实交易
```

#### XTP 实盘

```python
from dquant import Engine, XTPBroker, TopKStrategy

# 创建 XTP 接口
broker = XTPBroker(
    server='120.27.164.138',
    port=6001,
    account='YOUR_ACCOUNT',
    password='YOUR_PASSWORD',
)

# 创建引擎
engine = Engine(data, strategy, broker=broker)

# 运行实盘
engine.live(dry_run=False)
```

---

## 🛡️ 安全系统

### 订单验证

自动验证:
- 股票代码格式
- 买卖方向
- 价格范围 (0.01 - 10000)
- 数量 (100股整数倍)

### 资金检查

自动检查:
- 可用资金
- 交易成本 (佣金 + 印花税)
- 1% 安全边际

### 交易时间检查

自动检查:
- 交易日 (周一至周五)
- 交易时间:
  - 上午: 9:30 - 11:30
  - 下午: 13:00 - 15:00

### 日志记录

所有交易记录到:
```
logs/trading.log
```

格式:
```
2026-03-02 09:30:15 [INFO] [ORDER] BUY 600000.SH 1000@10.50
2026-03-02 09:30:16 [INFO] [TRADE] FILLED 600000.SH 1000@10.49
```

---

## 📊 监控和报告

### 实时监控

```python
from dquant import TradingMonitor

# 创建监控器
monitor = TradingMonitor(engine)

# 获取实时状态
status = monitor.get_status()

print(f"总资产: {status['total_asset']}")
print(f"持仓: {status['positions']}")
print(f"当日盈亏: {status['daily_pnl']}")
```

### 生成报告

```python
from dquant import ReportGenerator

# 生成日报
report = ReportGenerator(engine)
report.generate_daily_report()

# 生成周报
report.generate_weekly_report()

# 生成月报
report.generate_monthly_report()
```

---

## ⚠️ 风险提示

### 1. 测试充分

- ✅ 模拟测试至少 1-2 周
- ✅ 验证策略逻辑
- ✅ 验证风控系统
- ✅ 验证订单执行

### 2. 小资金开始

- 初始资金: 1-2 万
- 单笔交易: 10% 仓位
- 止损: 5% 日亏损

### 3. 人工监督

- 监控交易日志
- 定期检查持仓
- 异常情况及时干预

### 4. 系统稳定性

- 确保网络稳定
- 确保电力供应
- 准备应急预案

---

## 🔍 故障排查

### QMT 连接失败

```
错误: [QMT] 连接失败
```

解决方案:
1. 确认 QMT 客户端已启动
2. 确认路径配置正确
3. 确认账号已登录

### 订单被拒绝

```
错误: [ORDER] 订单被拒绝
```

可能原因:
- 资金不足
- 非交易时间
- 股票停牌
- 价格超出范围

### 策略不执行

```
错误: [STRATEGY] 策略未生成信号
```

检查:
- 数据是否正常
- 因子计算是否正确
- 策略参数是否合理

---

## 📞 技术支持

### 文档

- [API 文档](./API.md)
- [策略指南](./strategy_guide.md)
- [FAQ](./FAQ.md)

### 社区

- GitHub Issues: https://github.com/datacruiser/dquant/issues
- Discord: https://discord.com/invite/clawd

---

## 📝 最佳实践

### 1. 渐进式上线

```
第1周: 模拟测试
第2周: 小资金测试 (1-2万)
第3周: 观察和优化
第4周: 逐步增加资金
```

### 2. 风险控制

```python
# 严格的风控参数
safety = TradingSafety(
    max_position_pct=0.1,   # 单只股票 10%
    max_daily_loss=0.05,    # 日亏损 5%
    stop_loss_pct=0.03,     # 止损 3%
    max_total_position=0.8, # 总仓位 80%
)
```

### 3. 日志监控

```bash
# 实时查看日志
tail -f logs/trading.log

# 搜索错误
grep "ERROR" logs/trading.log

# 统计交易
grep "TRADE" logs/trading.log | wc -l
```

### 4. 定期回顾

- 每日检查交易日志
- 每周分析策略表现
- 每月优化策略参数

---

## 🎯 下一步

1. ✅ 完成模拟测试
2. ✅ 配置实盘环境
3. ✅ 小资金测试
4. ✅ 监控和优化
5. ✅ 逐步扩大规模

---

**⚠️ 重要提示: 量化交易存在风险，请谨慎投资！**

**祝您交易顺利！** 🚀
