"""
DQuant 日志系统

提供统一的日志管理。
"""

import logging
import sys
from pathlib import Path
from typing import Optional

# 日志格式
DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DETAILED_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(funcName)s | %(message)s"


class DquantStreamHandler(logging.StreamHandler):
    """dquant 专用 StreamHandler，用于识别自身创建的 handler。"""

    pass


class DquantFileHandler(logging.FileHandler):
    """dquant 专用 FileHandler，用于识别自身创建的 handler。"""

    pass


# 级别名映射：含 stdlib setLevel 认可的 WARN 别名；未知值抛 ValueError 而不是静默降级为 INFO
_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARN": logging.WARNING,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}


def _resolve_level(level: str) -> int:
    """把级别字符串解析成 logging 常量；未知值显式抛 ValueError（对齐 stdlib 行为）。"""
    key = str(level).upper()
    if key not in _LEVEL_MAP:
        raise ValueError(
            f"Unknown log level: {level!r}. " f"Valid levels: {', '.join(sorted(set(_LEVEL_MAP)))}"
        )
    return _LEVEL_MAP[key]


def _build_file_handler(
    log_file: str,
    rotating: bool,
    max_bytes: int,
    backup_count: int,
    formatter: logging.Formatter,
) -> logging.FileHandler:
    """按配置构建文件 handler（rotating=True 时用 RotatingFileHandler）。"""
    if rotating:
        from logging.handlers import RotatingFileHandler

        handler: logging.FileHandler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
    else:
        handler = DquantFileHandler(log_file, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    return handler


def get_logger(
    name: str = "dquant",
    level: str = "INFO",
    log_file: Optional[str] = None,
    format_style: str = "simple",
    rotating: bool = False,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> logging.Logger:
    """
    获取 Logger 实例

    Args:
        name: logger 名称
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 日志文件路径
        format_style: 格式样式 (simple, detailed)
        rotating: 使用 RotatingFileHandler (适合实盘长时间运行)
        max_bytes: 单个日志文件最大字节数 (默认 10MB)
        backup_count: 保留的备份文件数

    Returns:
        Logger 实例

    Example:
        logger = get_logger('dquant.backtest')
        logger.info("开始回测")
        logger.warning("资金不足")
    """
    logger = logging.getLogger(name)

    resolved_level = _resolve_level(level)
    initialized = any(
        isinstance(h, (DquantStreamHandler, DquantFileHandler)) for h in logger.handlers
    )

    # 选择格式
    fmt = DEFAULT_FORMAT if format_style == "simple" else DETAILED_FORMAT
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    # 更新级别（允许后续调用调整级别）
    logger.setLevel(resolved_level)

    # 文件 handler：幂等挂载。注意用 FileHandler 基类判断 —— 它是
    # DquantFileHandler / RotatingFileHandler / TimedRotatingFileHandler
    # 以及 stdlib 裸 FileHandler 的公共父类，全部都能被识别，避免重复挂载。
    if log_file:
        existing_paths = {
            getattr(h, "baseFilename", None)
            for h in logger.handlers
            if isinstance(h, logging.FileHandler)
        }
        wanted = str(Path(log_file).resolve())
        if not existing_paths:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            logger.addHandler(
                _build_file_handler(log_file, rotating, max_bytes, backup_count, formatter)
            )
        elif wanted not in existing_paths:
            # 显式警告而不是静默丢弃：第一条文件配置继续生效
            logger.warning(
                "[logger] %s 已配置日志文件 %s，忽略新的 log_file=%s",
                name,
                sorted(p for p in existing_paths if p),
                log_file,
            )

    if initialized:
        return logger

    # 首次初始化：控制台 handler
    console_handler = DquantStreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 自带全套 handler 后不再向祖先 logger 传播：
    # 模块导入时预创建的父 logger（如 dquant.backtest）会让每条日志打两遍。
    logger.propagate = False

    return logger


class LoggerMixin:
    """
    Logger 混入类

    为类提供 logger 属性。

    Example:
        class MyStrategy(LoggerMixin):
            def run(self):
                self.logger.info("策略运行中")
    """

    @property
    def logger(self) -> logging.Logger:
        if not hasattr(self, "_logger"):
            self._logger = get_logger(f"dquant.{self.__class__.__name__}")
        return self._logger


# 预定义的 logger
backtest_logger = get_logger("dquant.backtest")
data_logger = get_logger("dquant.data")
strategy_logger = get_logger("dquant.strategy")
factor_logger = get_logger("dquant.factor")


def set_log_level(level: str):
    """
    设置全局日志级别（作用于 dquant 及所有已创建的 dquant.* 子 logger）

    get_logger 会给每个命名 logger 显式设置级别，子级自己的级别优先于父级，
    所以只设根 logger 不够 —— 必须同步刷新所有已存在的 dquant.* logger，
    否则 quiet_mode()/debug_mode() 对它们是静默 no-op。

    Args:
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL；WARN 别名亦可)
    """
    resolved = _resolve_level(level)
    logging.getLogger("dquant").setLevel(resolved)
    for name, child in logging.root.manager.loggerDict.items():
        if isinstance(child, logging.Logger) and (name == "dquant" or name.startswith("dquant.")):
            child.setLevel(resolved)


def quiet_mode():
    """静默模式 - 只显示 ERROR"""
    set_log_level("ERROR")


def debug_mode():
    """调试模式 - 显示所有日志"""
    set_log_level("DEBUG")
