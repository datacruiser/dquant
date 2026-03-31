"""
通知告警系统

提供统一的通知接口，支持日志、钉钉等通道。
"""

from dquant.notify.base import Notifier
from dquant.notify.log_notifier import LogNotifier


def create_notifier(notifier_type: str = "log", **kwargs) -> Notifier:
    """
    创建通知器

    Args:
        notifier_type: 通知类型 ("log", "dingtalk")
        **kwargs: 通知器参数

    Returns:
        Notifier 实例
    """
    if notifier_type == "log":
        return LogNotifier(**kwargs)
    elif notifier_type == "dingtalk":
        from dquant.notify.dingtalk import DingTalkNotifier
        return DingTalkNotifier(**kwargs)
    else:
        raise ValueError(f"Unknown notifier type: {notifier_type}")
