"""
通知器基类
"""

from abc import ABC, abstractmethod


class Notifier(ABC):
    """通知器抽象基类"""

    @abstractmethod
    def send(self, title: str, message: str, level: str = "INFO") -> bool:
        """
        发送通知

        Args:
            title: 通知标题
            message: 通知内容
            level: 级别 (INFO, WARNING, ERROR, CRITICAL)

        Returns:
            是否发送成功
        """
        pass
