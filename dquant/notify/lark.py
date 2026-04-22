"""
飞书/Lark 通知器

通过飞书机器人 Webhook 发送通知，使用标准库 urllib（无外部依赖）。
支持签名校验（密钥模式）和 IP 白名单模式。
"""

import base64
import hashlib
import hmac
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

from dquant.logger import get_logger
from dquant.notify.base import Notifier
from dquant.notify.log_notifier import LogNotifier

logger = get_logger(__name__)


class LarkNotifier(Notifier):
    """
    飞书/Lark 机器人通知器

    通过 Webhook 发送交互式卡片消息。

    支持两种安全设置:
        1. 签名校验（secret）— 推荐
        2. IP 白名单 — 不需要 secret

    Usage:
        notifier = LarkNotifier(
            webhook_url="https://open.feishu.cn/open-apis/bot/v2/hook/...",
            secret="...",
        )
        notifier.send("风控警报", "回撤超过阈值", "CRITICAL")
    """

    _ALLOWED_PREFIXES = (
        "https://open.feishu.cn/open-apis/bot/v2/hook/",
        "https://open.larksuite.com/open-apis/bot/v2/hook/",
    )

    def __init__(
        self,
        webhook_url: str = "",
        secret: str = "",
        timeout: int = 5,
    ):
        self.webhook_url = webhook_url or os.getenv("LARK_WEBHOOK", "")
        self.secret = secret or os.getenv("LARK_SECRET", "")
        self.timeout = timeout
        self._fallback = LogNotifier()

    def send(self, title: str, message: str, level: str = "INFO") -> bool:
        if not self.webhook_url:
            self._fallback.send(title, message, level)
            return False

        if not self.webhook_url.startswith(self._ALLOWED_PREFIXES):
            logger.error(f"[Lark] webhook URL 不在白名单中，拒绝发送: {self.webhook_url.split('?')[0]}?***")
            self._fallback.send(title, f"[Lark BLOCKED] {message}", "ERROR")
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
                if result.get("code") == 0 or result.get("StatusCode") == 0:
                    return True
                else:
                    logger.warning(f"[Lark] 发送失败: {result}")
                    self._fallback.send(title, f"[Lark FAILED] {message}", "ERROR")
                    return False

        except Exception as e:
            logger.warning(f"[Lark] 发送异常: {e}")
            self._fallback.send(title, f"[Lark FAILED] {message}", "ERROR")
            return False

    def _build_url(self) -> str:
        """构建带签名的 Webhook URL（飞书 v2 签名方式）"""
        if not self.secret:
            return self.webhook_url

        timestamp = str(int(time.time()))
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
        """构建飞书交互式卡片消息 payload"""
        level_color = {
            "INFO": "blue",
            "WARNING": "orange",
            "ERROR": "red",
            "CRITICAL": "red",
        }
        level_tag = {
            "INFO": "ℹ️",
            "WARNING": "⚠️",
            "ERROR": "❌",
            "CRITICAL": "🚨",
        }

        color = level_color.get(level, "blue")
        tag = level_tag.get(level, "ℹ️")

        return {
            "msg_type": "interactive",
            "card": {
                "header": {
                    "title": {
                        "tag": "plain_text",
                        "content": f"{tag} {title}",
                    },
                    "template": color,
                },
                "elements": [
                    {
                        "tag": "markdown",
                        "content": message,
                    }
                ],
            },
        }
