"""
通知器基类
"""

import base64
import hashlib
import hmac
import time
import urllib.parse
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


def sign_webhook_url(webhook_url: str, secret: str, timestamp_ms: bool = False) -> str:
    """Compute HMAC signature and append to webhook URL.

    Args:
        webhook_url: Base webhook URL.
        secret: HMAC secret key.
        timestamp_ms: Use millisecond timestamp (DingTalk) vs seconds (Lark).

    Returns:
        URL with timestamp and sign query params appended.
    """
    if not secret:
        return webhook_url

    timestamp = str(int(time.time() * 1000) if timestamp_ms else int(time.time()))
    string_to_sign = f"{timestamp}\n{secret}"
    hmac_code = hmac.new(
        secret.encode("utf-8"),
        string_to_sign.encode("utf-8"),
        digestmod=hashlib.sha256,
    ).digest()
    sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))

    separator = "&" if "?" in webhook_url else "?"
    return f"{webhook_url}{separator}timestamp={timestamp}&sign={sign}"
