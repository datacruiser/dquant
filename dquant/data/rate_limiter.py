"""
API 速率限制器

支持令牌桶（Token Bucket）算法，防止外部 API 调用超过频率限制。
"""

import threading
import time
from typing import Optional


class RateLimiter:
    """
    令牌桶速率限制器（线程安全）

    Usage:
        limiter = RateLimiter(max_calls=200, period=60)  # 200次/分钟
        limiter.wait()  # 阻塞等待直到获得令牌
        # ... 执行 API 调用 ...

    Args:
        max_calls: 时间窗口内最大调用次数
        period: 时间窗口（秒）
    """

    def __init__(self, max_calls: int = 200, period: float = 60.0):
        self.max_calls = max_calls
        self.period = period
        self._tokens = max_calls
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def _refill(self):
        """根据流逝时间补充令牌"""
        now = time.monotonic()
        elapsed = now - self._last_refill
        new_tokens = elapsed * (self.max_calls / self.period)
        if new_tokens > 0:
            self._tokens = min(self.max_calls, self._tokens + new_tokens)
            self._last_refill = now

    def wait(self, timeout: Optional[float] = None) -> bool:
        """
        阻塞等待直到获得一个令牌

        Args:
            timeout: 最大等待时间（秒），None 表示无限等待

        Returns:
            True 表示获得令牌，False 表示超时
        """
        deadline = None if timeout is None else time.monotonic() + timeout

        while True:
            with self._lock:
                self._refill()
                if self._tokens >= 1:
                    self._tokens -= 1
                    return True

            # 计算等待时间
            wait_time = (1 - self._tokens) * (self.period / self.max_calls)
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                wait_time = min(wait_time, remaining)

            time.sleep(wait_time)

    @property
    def available_tokens(self) -> float:
        """当前可用令牌数"""
        with self._lock:
            self._refill()
            return self._tokens
