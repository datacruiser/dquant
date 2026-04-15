"""
Phase 2 Step 5: 数据完整性校验测试
"""

import hashlib
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from dquant.data.data_manager import DataManager


class TestCacheKey:

    def test_different_symbol_lists_produce_different_keys(self):
        """不同 symbol 列表产生不同的缓存键"""
        dm = DataManager(cache_dir=None)
        key1 = dm._get_cache_key(
            "akshare", ["000001", "000002", "000003"], "2024-01-01", "2024-06-01", {}
        )
        key2 = dm._get_cache_key(
            "akshare", ["000001", "000002", "000004"], "2024-01-01", "2024-06-01", {}
        )
        assert key1 != key2

    def test_same_symbols_different_order_same_key(self):
        """相同 symbols 不同顺序产生相同缓存键"""
        dm = DataManager(cache_dir=None)
        key1 = dm._get_cache_key(
            "akshare", ["000003", "000001", "000002"], "2024-01-01", "2024-06-01", {}
        )
        key2 = dm._get_cache_key(
            "akshare", ["000001", "000002", "000003"], "2024-01-01", "2024-06-01", {}
        )
        assert key1 == key2

    def test_long_symbol_list_uses_hash(self):
        """长列表使用 hash"""
        dm = DataManager(cache_dir=None)
        # 200+ symbols
        symbols = [f"{i:06d}.SZ" for i in range(100)]
        key = dm._get_cache_key("akshare", symbols, "2024-01-01", "2024-06-01", {})
        # Key should be reasonable length
        assert len(key) < 300

    def test_single_symbol(self):
        """单 symbol 也能生成 key"""
        dm = DataManager(cache_dir=None)
        key = dm._get_cache_key("akshare", "000001.SZ", "2024-01-01", "2024-06-01", {})
        assert "000001.SZ" in key


class TestValidateAfterLoad:

    def test_validate_default_on(self):
        """默认启用验证"""
        dm = DataManager(cache_dir=None)
        assert dm.validate_after_load

    def test_validate_can_be_enabled(self):
        """可以启用验证"""
        dm = DataManager(cache_dir=None, validate_after_load=True)
        assert dm.validate_after_load


def test_live_trading_config_defaults():
    """LiveTradingConfig has expected default values."""
    from dquant.config import LiveTradingConfig

    cfg = LiveTradingConfig()
    assert cfg.dry_run is True
    assert cfg.interval == 60
    assert cfg.symbols is None
    assert cfg.strategy_name == ""
    assert cfg.max_drawdown == 0.15
    assert cfg.max_daily_loss == 0.03
    assert cfg.max_consecutive_errors == 10


def test_xtp_broker_config_defaults():
    """XTPBrokerConfig has expected default values."""
    from dquant.config import XTPBrokerConfig

    cfg = XTPBrokerConfig()
    assert cfg.server == "120.27.164.138"
    assert cfg.port == 6001
    assert cfg.account == ""
    assert cfg.password == ""
    assert cfg.password_env == ""
    assert cfg.client_id == 1
    assert cfg.timeout == 30


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
