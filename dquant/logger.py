"""
DQuant 日志系统

提供统一的日志管理。
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime


# 日志格式
DEFAULT_FORMAT = '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
DETAILED_FORMAT = '%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(funcName)s | %(message)s'


def get_logger(
    name: str = 'dquant',
    level: str = 'INFO',
    log_file: Optional[str] = None,
    format_style: str = 'simple',
) -> logging.Logger:
    """
    获取 Logger 实例
    
    Args:
        name: logger 名称
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: 日志文件路径
        format_style: 格式样式 (simple, detailed)
    
    Returns:
        Logger 实例
    
    Example:
        logger = get_logger('dquant.backtest')
        logger.info("开始回测")
        logger.warning("资金不足")
    """
    logger = logging.getLogger(name)
    
    # 如果已经有 handler，直接返回
    if logger.handlers:
        return logger
    
    # 设置级别
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL,
    }
    logger.setLevel(level_map.get(level.upper(), logging.INFO))
    
    # 选择格式
    fmt = DEFAULT_FORMAT if format_style == 'simple' else DETAILED_FORMAT
    formatter = logging.Formatter(fmt, datefmt='%Y-%m-%d %H:%M:%S')
    
    # 控制台 handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件 handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
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
        if not hasattr(self, '_logger'):
            self._logger = get_logger(f'dquant.{self.__class__.__name__}')
        return self._logger


# 预定义的 logger
backtest_logger = get_logger('dquant.backtest')
data_logger = get_logger('dquant.data')
strategy_logger = get_logger('dquant.strategy')
factor_logger = get_logger('dquant.factor')


def set_log_level(level: str):
    """
    设置全局日志级别
    
    Args:
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR)
    """
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
    }
    
    logging.getLogger('dquant').setLevel(level_map.get(level.upper(), logging.INFO))


def quiet_mode():
    """静默模式 - 只显示 ERROR"""
    set_log_level('ERROR')


def debug_mode():
    """调试模式 - 显示所有日志"""
    set_log_level('DEBUG')
