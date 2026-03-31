"""
Phase 2 Step 4: 通知系统测试
"""

import os
import json
from unittest.mock import patch, MagicMock

import pytest

from dquant.notify import create_notifier
from dquant.notify.base import Notifier
from dquant.notify.log_notifier import LogNotifier
from dquant.notify.dingtalk import DingTalkNotifier


class TestNotifierBase:

    def test_cannot_instantiate_abc(self):
        """Notifier 是抽象类，不能直接实例化"""
        with pytest.raises(TypeError):
            Notifier()


class TestLogNotifier:

    def test_send_returns_true(self):
        notifier = LogNotifier()
        assert notifier.send("Test", "Hello") is True

    def test_send_different_levels(self):
        notifier = LogNotifier()
        assert notifier.send("Test", "info", "INFO") is True
        assert notifier.send("Test", "warning", "WARNING") is True
        assert notifier.send("Test", "error", "ERROR") is True
        assert notifier.send("Test", "critical", "CRITICAL") is True

    def test_is_notifier(self):
        notifier = LogNotifier()
        assert isinstance(notifier, Notifier)


class TestDingTalkNotifier:

    def test_no_webhook_falls_back(self):
        """无 webhook 时降级到日志"""
        notifier = DingTalkNotifier(webhook_url="", secret="")
        # Should not raise, falls back to LogNotifier
        result = notifier.send("Test", "Hello")
        assert result is False  # DingTalk failed, fallback used

    def test_with_webhook_env(self):
        """从环境变量读取 webhook"""
        with patch.dict(os.environ, {"DINGTALK_WEBHOOK": "https://example.com/webhook"}):
            notifier = DingTalkNotifier()
            assert notifier.webhook_url == "https://example.com/webhook"

    def test_build_payload(self):
        notifier = DingTalkNotifier()
        payload = notifier._build_payload("Test Title", "Test Message", "INFO")
        assert payload["msgtype"] == "markdown"
        assert "Test Title" in payload["markdown"]["title"]
        assert "Test Message" in payload["markdown"]["text"]

    def test_build_payload_error_level(self):
        notifier = DingTalkNotifier()
        payload = notifier._build_payload("Alert", "Something wrong", "ERROR")
        assert "[ERROR]" in payload["markdown"]["title"]

    def test_build_url_without_secret(self):
        notifier = DingTalkNotifier(webhook_url="https://example.com/hook")
        url = notifier._build_url()
        assert url == "https://example.com/hook"

    def test_build_url_with_secret(self):
        notifier = DingTalkNotifier(
            webhook_url="https://example.com/hook",
            secret="SEC_TEST",
        )
        url = notifier._build_url()
        assert "timestamp=" in url
        assert "sign=" in url

    def test_send_success(self):
        """模拟成功发送"""
        notifier = DingTalkNotifier(webhook_url="https://example.com/hook")

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps({"errcode": 0}).encode()
        mock_response.__enter__ = lambda s: mock_response
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = notifier.send("Test", "Hello")
            assert result is True

    def test_send_dingtalk_error(self):
        """模拟钉钉返回错误"""
        notifier = DingTalkNotifier(webhook_url="https://example.com/hook")

        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {"errcode": 1, "errmsg": "invalid token"}
        ).encode()
        mock_response.__enter__ = lambda s: mock_response
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("urllib.request.urlopen", return_value=mock_response):
            result = notifier.send("Test", "Hello")
            assert result is False

    def test_send_network_error_fallback(self):
        """模拟网络错误，降级到日志"""
        notifier = DingTalkNotifier(webhook_url="https://example.com/hook")

        with patch("urllib.request.urlopen", side_effect=Exception("Network error")):
            result = notifier.send("Test", "Hello")
            assert result is False


class TestCreateNotifier:

    def test_create_log(self):
        notifier = create_notifier("log")
        assert isinstance(notifier, LogNotifier)

    def test_create_dingtalk(self):
        notifier = create_notifier("dingtalk", webhook_url="https://example.com")
        assert isinstance(notifier, DingTalkNotifier)

    def test_create_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown notifier"):
            create_notifier("unknown")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
