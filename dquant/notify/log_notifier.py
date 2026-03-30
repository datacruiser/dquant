"""
日志通知器

将通知写入 Python logger，作为默认回退通道。
"""

from dquant.notify.base import Notifier
from dquant.logger import get_logger


class LogNotifier(Notifier):
    """日志通知器"""

    def __init__(self, logger_name: str = "dquant.notify"):
        self._logger = get_logger(logger_name)

    def send(self, title: str, message: str, level: str = "INFO") -> bool:
        level_map = {
            "DEBUG": self._logger.debug,
            "INFO": self._logger.info,
            "WARNING": self._logger.warning,
            "ERROR": self._logger.error,
            "CRITICAL": self._logger.critical,
        }
        log_fn = level_map.get(level.upper(), self._logger.info)
        log_fn(f"[NOTIFY] {title}: {message}")
        return True
