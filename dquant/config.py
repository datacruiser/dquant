"""
DQuant 配置管理

支持从文件、环境变量加载配置。
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from dquant.constants import (
    DEFAULT_COMMISSION,
    DEFAULT_INITIAL_CASH,
    DEFAULT_SLIPPAGE,
    DEFAULT_STAMP_DUTY,
)


@dataclass
class BacktestConfig:
    """回测配置"""

    initial_cash: float = DEFAULT_INITIAL_CASH
    commission_rate: float = DEFAULT_COMMISSION  # 佣金率 0.03%
    stamp_duty: float = DEFAULT_STAMP_DUTY  # 印花税 0.1%
    slippage: float = DEFAULT_SLIPPAGE  # 滑点 0.01%

    # 仓位限制
    max_position_pct: float = 0.1  # 单只股票最大仓位 10%
    max_cash_pct: float = 0.95  # 最大资金使用率 95%

    # 其他
    benchmark: str = "hs300"  # 基准


@dataclass
class DataConfig:
    """数据配置"""

    # 数据源
    default_source: str = "akshare"
    cache_dir: str = "./data/cache"

    # 缓存
    enable_cache: bool = True
    cache_expire_days: int = 7

    # 数据库
    db_url: Optional[str] = None
    mongo_url: Optional[str] = None

    # API Keys
    tushare_token: Optional[str] = None
    jq_username: Optional[str] = None
    jq_password: Optional[str] = None


@dataclass
class FactorConfig:
    """因子配置"""

    # 预处理
    standardize: bool = True
    winsorize: bool = True
    winsorize_limit: float = 0.01

    # 中性化
    neutralize: bool = True
    neutralize_industry: bool = True

    # 筛选
    min_ic: float = 0.02
    min_ir: float = 0.5
    max_correlation: float = 0.7


@dataclass
class MLConfig:
    """机器学习配置"""

    # 训练
    train_test_split: float = 0.8
    n_splits: int = 5  # 交叉验证折数

    # XGBoost
    xgboost_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 100,
            "objective": "reg:squarederror",
        }
    )

    # LightGBM
    lightgbm_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "max_depth": 6,
            "learning_rate": 0.05,
            "n_estimators": 100,
            "objective": "regression",
        }
    )


@dataclass
class LiveConfig:
    """实盘交易配置"""

    # 券商
    broker: str = "simulator"
    broker_config: Dict[str, Any] = field(default_factory=dict)

    # 交易循环
    interval: int = 60  # 轮询间隔 (秒)
    dry_run: bool = True  # 模拟运行

    # 风控
    max_drawdown: float = 0.15  # 最大回撤
    max_daily_loss: float = 0.03  # 单日最大亏损
    max_consecutive_errors: int = 10  # 连续错误上限

    # 仓位
    position_method: str = "equal_weight"
    max_single_pct: float = 0.1  # 单票最大仓位

    # 日志
    log_dir: str = "./logs"
    log_max_bytes: int = 10 * 1024 * 1024  # 10MB
    log_backup_count: int = 5

    # 审计
    journal_dir: str = "./trade_journal"


@dataclass
class LiveTradingConfig:
    """实盘交易运行时配置 (用于 Engine.live())"""

    dry_run: bool = True
    interval: int = 60  # 轮询间隔 (秒)
    symbols: Optional[List[str]] = None
    strategy_name: str = ""
    max_drawdown: float = 0.15  # 最大回撤
    max_daily_loss: float = 0.03  # 单日最大亏损
    max_consecutive_errors: int = 10  # 连续错误上限

    @classmethod
    def from_live_config(cls, config: LiveConfig, **overrides) -> "LiveTradingConfig":
        """从 LiveConfig 创建 LiveTradingConfig（共享默认值）"""
        return cls(
            dry_run=overrides.get("dry_run", config.dry_run),
            interval=overrides.get("interval", config.interval),
            max_drawdown=overrides.get("max_drawdown", config.max_drawdown),
            max_daily_loss=overrides.get("max_daily_loss", config.max_daily_loss),
            max_consecutive_errors=overrides.get(
                "max_consecutive_errors", config.max_consecutive_errors
            ),
            symbols=overrides.get("symbols"),
            strategy_name=overrides.get("strategy_name", ""),
        )


@dataclass
class XTPBrokerConfig:
    """XTP 券商连接配置"""

    server: str = ""
    port: int = 6001
    account: str = ""
    password: str = ""
    password_env: str = ""  # 环境变量名，优先使用
    client_id: int = 1
    timeout: int = 30


@dataclass
class DQuantConfig:
    """DQuant 总配置"""

    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    data: DataConfig = field(default_factory=DataConfig)
    factor: FactorConfig = field(default_factory=FactorConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    live: LiveConfig = field(default_factory=LiveConfig)

    @classmethod
    def from_file(cls, path) -> "DQuantConfig":
        """从文件加载配置"""
        p = Path(path) if not isinstance(path, Path) else path

        if not p.exists():
            raise FileNotFoundError(f"配置文件不存在: {p}")

        try:
            if p.suffix == ".json":
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                raise ValueError(f"不支持的配置格式: {p.suffix}")
        except json.JSONDecodeError as e:
            raise ValueError(f"配置文件 JSON 格式错误: {p} — {e}")

        return cls.from_dict(data)

    def _apply_config_section(self, section_name, section_data):
        """应用配置段"""
        section = getattr(self, section_name, None)
        if section is None:
            return

        for k, v in section_data.items():
            if hasattr(section, k):
                setattr(section, k, v)

    @classmethod
    def from_dict(cls, data: Dict) -> "DQuantConfig":
        """从字典创建配置"""
        config = cls()

        # 应用各配置段
        for section_name in ["backtest", "data", "factor", "ml", "live"]:
            if section_name in data:
                config._apply_config_section(section_name, data[section_name])

        return config

    @classmethod
    def from_env(cls) -> "DQuantConfig":
        """从环境变量加载配置"""
        config = cls()

        # 数据源配置
        if os.getenv("TUSHARE_TOKEN"):
            config.data.tushare_token = os.getenv("TUSHARE_TOKEN")

        if os.getenv("DB_URL"):
            config.data.db_url = os.getenv("DB_URL")

        if os.getenv("MONGO_URL"):
            config.data.mongo_url = os.getenv("MONGO_URL")

        # 回测配置
        cash_env = os.getenv("INITIAL_CASH")
        if cash_env:
            try:
                config.backtest.initial_cash = float(cash_env)
            except (ValueError, TypeError):
                raise ValueError(f"INITIAL_CASH 环境变量值无效: '{cash_env}'，应为数字")

        return config

    def to_dict(self) -> Dict:
        """转换为字典"""
        from dataclasses import asdict

        return asdict(self)

    def save(self, path) -> None:
        """保存配置到文件"""
        p = Path(path) if not isinstance(path, Path) else path
        p.parent.mkdir(parents=True, exist_ok=True)

        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


# 默认配置实例
default_config = DQuantConfig()
