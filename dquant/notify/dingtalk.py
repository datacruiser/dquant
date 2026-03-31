"""
钉钉通知器

通过钉钉机器人 Webhook 发送通知，使用标准库 urllib（无外部依赖）。
"""

import json
import hmac
import hashlib
import base64
import os
import time
import urllib.request
import urllib.error
import urllib.parse
from typing import Optional

from dquant.notify.base import Notifier
from dquant.notify.log_notifier import LogNotifier
from dquant.logger import get_logger

logger = get_logger(__name__)


class DingTalkNotifier(Notifier):
    """
    钉钉机器人通知器

    通过 Webhook 发送 Markdown 格式消息。

    Usage:
        notifier = DingTalkNotifier(
            webhook_url="https://oapi.dingtalk.com/robot/send?access_token=...",
            secret="SEC...",
        )
        notifier.send("风控警报", "回撤超过阈值", "CRITICAL")
    """

    def __init__(
        self,
        webhook_url: str = "",
        secret: str = "",
        timeout: int = 5,
    ):
        self.webhook_url = webhook_url or os.getenv("DINGTALK_WEBHOOK", "")
        self.secret = secret or os.getenv("DINGTALK_SECRET", "")
        self.timeout = timeout
        self._fallback = LogNotifier()

    def send(self, title: str, message: str, level: str = "INFO") -> bool:
        if not self.webhook_url:
            self._fallback.send(title, message, level)
            return False

        try:
            url = self._build_url()
            payload = self._build_payload(title, message, level)
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
            )

            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                if result.get("errcode") == 0:
                    return True
                else:
                    logger.warning(f"[DingTalk] 发送失败: {result}")
                    self._fallback.send(title, f"[DingTalk FAILED] {message}", "ERROR")
                    return False

        except Exception as e:
            logger.warning(f"[DingTalk] 发送异常: {e}")
            self._fallback.send(title, f"[DingTalk FAILED] {message}", "ERROR")
            return False

    def _build_url(self) -> str:
        """构建带签名的 Webhook URL"""
        if not self.secret:
            return self.webhook_url

        timestamp = str(int(time.time() * 1000))
        string_to_sign = f"{timestamp}\n{self.secret}"
        hmac_code = hmac.new(
            self.secret.encode("utf-8"),
            string_to_sign.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).digest()
        sign = urllib.parse.quote_plus(base64.b64encode(hmac_code))

        separator = "&" if "?" in self.webhook_url else "?"
        return f"{self.webhook_url}{separator}timestamp={timestamp}&sign={sign}"

    def _build_payload(self, title: str, message: str, level: str) -> dict:
        """构建钉钉消息 payload"""
        level_emoji = {
            "INFO": "",
            "WARNING": "[WARNING] ",
            "ERROR": "[ERROR] ",
            "CRITICAL": "[CRITICAL] ",
        }
        prefix = level_emoji.get(level, "")
        return {
            "msgtype": "markdown",
            "markdown": {
                "title": f"{prefix}{title}",
                "text": f"### {prefix}{title}\n\n{message}",
            },
        }
