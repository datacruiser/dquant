# Security Policy

## 支持的版本

dquant 仍在快速迭代，仅对最新主分支进行安全修复。

| 版本     | 支持状态           |
| -------- | ------------------ |
| 0.1.x    | :white_check_mark: |
| < 0.1    | :x:                |

## 报告漏洞

- 请 **不要** 在公开 issue 中提交安全漏洞。
- 通过 GitHub Private Vulnerability Reporting 提交报告，或在邮件中详细描述问题与复现路径。
- 收到报告后，会在 7 天内确认，并在修复后公开致谢（如愿意）。

## 已实现的安全机制

以下是 dquant 当前已经落地的安全设计；调用方在使用前应明确知晓这些边界。

### 1. 风控状态完整性（HMAC-SHA256）

- 位置：`dquant/risk.py`，`RiskManager.save_state()` / `restore_state()`。
- `RiskManager` 在持久化状态时附带 HMAC-SHA256 签名；`restore_state()` 校验失败会立即 `halt_trading = True`（fail-safe）。
- **生产环境强制要求** 通过 `DQUANT_RISK_SECRET` 环境变量提供签名密钥；未设置时会直接 `raise RuntimeError`，拒绝用默认密钥签名。
- 测试模式可通过参数显式 opt-in，避免误用默认密钥。

### 2. Webhook 通知的 SSRF 防护与脱敏

- 位置：`dquant/notify/dingtalk.py`、`dquant/notify/lark.py`、`dquant/notify/base.py`。
- 通知器对 webhook URL 做了协议+域名白名单（仅允许 `oapi.dingtalk.com` / `open.feishu.cn`）。
- 日志输出时 webhook URL 的查询串会被脱敏（替换为 `?***`），避免泄露签名密钥或 token。
- 钉钉/飞书的 HMAC-SHA256 签名统一封装在 `notify.base.sign_webhook_url`。

### 3. 凭据管理

- 密钥/口令通过环境变量读取，示例：
  - `XTPBrokerConfig`：`password_env` 字段指向含口令的环境变量名。
  - `DQUANT_RISK_SECRET`：风控状态签名密钥。
  - `TUSHARE_TOKEN`：Tushare 数据源 token。
- `.gitignore` 显式排除 `config.local.yaml`、`*.key`、`*.pem`、`secrets/`、`.env*` 等敏感文件。
- `DataManager._get_cache_key()` 显式过滤 `token / password / account / api_key / connection_string` 等敏感字段，确保凭据不会被写入缓存文件名或 meta.json。

### 4. SQL 注入防护

- 位置：`dquant/data/database_loader.py`。
- 表名/列名通过 `_IDENTIFIER_RE` 白名单过滤；查询参数一律使用参数化查询，不拼接 SQL 字符串。

### 5. 限流

- 位置：`dquant/data/rate_limiter.py`。
- 令牌桶限流器用于 AKShare / Tushare 等在线数据源，避免触发上游风控；线程安全实现。

## 安全边界与限制（请知悉）

dquant 是一个面向研究与个人量化的框架，**不**提供以下能力，使用者需要自行评估风险：

- 不内建账户级 RBAC 或审计日志的强一致性存储（`TradeJournal` 是 append-only JSONL，没有原子写入保证）。
- 不内建 mTLS / 双向证书认证；券商接口的传输安全由底层 SDK（XTP / QMT / xtquant）保证。
- `dquant.broker.simulator` 是模拟器，**不能**当作生产环境的可信记账引擎使用。
- 实盘下单前请务必使用 `dry_run=True` 进行端到端验证。

## 推荐部署实践

1. 在隔离的用户/容器中运行实盘循环，限制文件系统与网络访问范围。
2. 使用专用操作系统用户运行；通过 systemd / launchd 设置自动重启与资源限制。
3. 任何凭据通过环境变量或 secret manager 注入，**不要**写入仓库内的 YAML 模板。
4. 实盘前先在 Simulator 上跑通完整 `Engine.live(dry_run=True)` 链路，再切换到真实 broker。
