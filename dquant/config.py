"""
DQuant 配置管理

支持从文件、环境变量加载配置。
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
import os
import json
from dquant.constants import DEFAULT_COMMISSION, DEFAULT_SLIPPAGE, DEFAULT_STAMP_DUTY, DEFAULT_INITIAL_CASH, MIN_SHARES, DEFAULT_WINDOW


@dataclass
class BacktestConfig:
    """回测配置"""
    initial_cash: float = 1000000.0
    commission_rate: float = DEFAULT_COMMISSION  # 佣金率 0.03%
    stamp_duty: float = DEFAULT_STAMP_DUTY        # 印花税 0.1%
    slippage: float = DEFAULT_STAMP_DUTY          # 滑点 0.1%

    # 仓位限制
    max_position_pct: float = 0.1    # 单只股票最大仓位 10%
    max_cash_pct: float = 0.95       # 最大资金使用率 95%

    # 其他
    benchmark: str = 'hs300'         # 基准


@dataclass
class DataConfig:
    """数据配置"""
    # 数据源
    default_source: str = 'akshare'
    cache_dir: str = './data/cache'

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
    n_splits: int = 5              # 交叉验证折数

    # XGBoost
    xgboost_params: Dict[str, Any] = field(default_factory=lambda: {
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_estimators': MIN_SHARES,
        'objective': 'reg:squarederror',
    })

    # LightGBM
    lightgbm_params: Dict[str, Any] = field(default_factory=lambda: {
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_estimators': MIN_SHARES,
        'objective': 'regression',
    })


@dataclass
class DQuantConfig:
    """DQuant 总配置"""
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    data: DataConfig = field(default_factory=DataConfig)
    factor: FactorConfig = field(default_factory=FactorConfig)
    ml: MLConfig = field(default_factory=MLConfig)

    @classmethod
    def from_file(cls, path: str) -> 'DQuantConfig':
        """从文件加载配置"""
        path = Path(path)

        if path.suffix == '.json':
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")

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
    def from_dict(cls, data: Dict) -> 'DQuantConfig':
        """从字典创建配置"""
        config = cls()

        # 应用各配置段
        for section_name in ['backtest', 'data', 'factor', 'ml']:
            if section_name in data:
                config._apply_config_section(section_name, data[section_name])

        return config

    @classmethod
    def from_env(cls) -> 'DQuantConfig':
        """从环境变量加载配置"""
        config = cls()

        # 数据源配置
        if os.getenv('TUSHARE_TOKEN'):
            config.data.tushare_token = os.getenv('TUSHARE_TOKEN')

        if os.getenv('DB_URL'):
            config.data.db_url = os.getenv('DB_URL')

        if os.getenv('MONGO_URL'):
            config.data.mongo_url = os.getenv('MONGO_URL')

        # 回测配置
        if os.getenv('INITIAL_CASH'):
            config.backtest.initial_cash = float(os.getenv('INITIAL_CASH'))

        return config

    def to_dict(self) -> Dict:
        """转换为字典"""
        from dataclasses import asdict
        return asdict(self)

    def save(self, path: str):
        """保存配置到文件"""
        path = Path(path)

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


# 默认配置实例
default_config = DQuantConfig()
