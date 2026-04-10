"""
速率限制器测试
"""

import threading
import time

import pytest

from dquant.data.rate_limiter import RateLimiter


class TestRateLimiter:
    def test_basic_acquire(self):
        limiter = RateLimiter(max_calls=10, period=1.0)
        assert limiter.wait(timeout=1.0) is True

    def test_tokens_depleted(self):
        """消耗完令牌后应当等待"""
        limiter = RateLimiter(max_calls=5, period=1.0)
        # 消耗所有令牌
        for _ in range(5):
            assert limiter.wait(timeout=0.01) is True
        # 第6次应该超时（无令牌）
        assert limiter.wait(timeout=0.01) is False

    def test_token_refill(self):
        """令牌应该随时间补充"""
        limiter = RateLimiter(max_calls=5, period=1.0)
        for _ in range(5):
            limiter.wait(timeout=0.01)
        # 等待补充
        time.sleep(0.3)
        assert limiter.available_tokens > 0

    def test_concurrent_access(self):
        """多线程并发访问不应导致错误"""
        limiter = RateLimiter(max_calls=100, period=1.0)
        results = []
        errors = []

        def worker():
            try:
                ok = limiter.wait(timeout=2.0)
                results.append(ok)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 20

    def test_available_tokens_property(self):
        limiter = RateLimiter(max_calls=10, period=1.0)
        assert limiter.available_tokens == 10.0
        limiter.wait()
        assert limiter.available_tokens < 10.0

    def test_timeout_zero(self):
        """timeout=0 应该立即返回"""
        limiter = RateLimiter(max_calls=1, period=1.0)
        limiter.wait(timeout=0.01)
        # 已用完，立即检查
        result = limiter.wait(timeout=0.0)
        assert result is False
