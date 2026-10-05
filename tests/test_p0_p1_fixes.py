"""
针对 fix/p0-p1-code-review-fixes 分支修复点的回归测试。

覆盖：
- akshare / tushare loader 多股票因子串号（cross-contamination）
- DataManager 缓存键泄露凭据
- futures.py 保证金口径双计盈亏
- ml_factors._temporal_split 单日/小样本边界
- safety 股票代码正则放宽保护
- qlib_adapter 空壳 fit/predict
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# akshare / tushare 因子串号
# ---------------------------------------------------------------------------


def _build_panel():
    dates = pd.date_range("2024-01-01", periods=12, freq="D")
    df = pd.concat(
        [
            pd.DataFrame(
                {
                    "symbol": "000001.SZ",
                    "close": np.arange(1, 13, dtype=float),
                    "high": np.arange(1, 13, dtype=float) + 1,
                    "low": np.arange(1, 13, dtype=float) - 1,
                    "volume": np.arange(100, 124, 2, dtype=float),
                    "turnover": np.arange(1, 13, dtype=float),
                },
                index=dates,
            ),
            pd.DataFrame(
                {
                    "symbol": "600000.SH",
                    "close": np.arange(2, 26, 2, dtype=float),
                    "high": np.arange(2, 26, 2, dtype=float) + 1,
                    "low": np.arange(2, 26, 2, dtype=float) - 1,
                    "volume": np.arange(200, 248, 4, dtype=float),
                    "turnover": np.arange(2, 26, 2, dtype=float),
                },
                index=dates,
            ),
        ]
    )
    df.index.name = "date"
    return df.reset_index().set_index("date")


def test_akshare_turnover_ma_isolated_per_symbol():
    """AKShare 的 turnover_ma_5/10 必须按 symbol 隔离计算，不能串号。"""
    from dquant.data.akshare_loader import AKShareLoader

    df = _build_panel()
    loader = AKShareLoader.__new__(AKShareLoader)
    out = loader._calculate_factors(df)

    s1 = out[out["symbol"] == "000001.SZ"]["turnover_ma_5"].iloc[4]
    s2 = out[out["symbol"] == "600000.SH"]["turnover_ma_5"].iloc[4]
    # 000001 前 5 行 turnover=1..5，均值=3
    assert s1 == pytest.approx(3.0)
    # 600000 前 5 行 turnover=2,4,6,8,10，均值=6
    assert s2 == pytest.approx(6.0)


def test_tushare_extended_factors_isolated_per_symbol():
    """Tushare 扩展因子也必须按 symbol 隔离。"""
    from dquant.data.tushare_loader import TushareLoader

    df = _build_panel()
    loader = TushareLoader.__new__(TushareLoader)
    out = loader._calculate_factors(df)

    # 取一个明显不会有跨 symbol 干扰的点：ma_60 在 12 行样本里全是 NaN
    # 但 momentum_60 也是 NaN；改测 bias_60 不行，改测 ma_60 在前 11 行都 NaN
    sub = out[out["symbol"] == "000001.SZ"]
    # 12 行 < 60，所有 ma_60 应该都是 NaN，且不影响 close 原值
    assert sub["ma_60"].isna().all()
    # 关键：close 不能被另一个 symbol 覆盖
    assert sub["close"].tolist() == list(range(1, 13))

    # 判别性断言：12 行样本里 window=10 < 12，volume_ma_10 有真实值。
    # 000001 的 volume=100..122(step2)，第 10-12 行的 volume_ma_10 应为 [109, 111, 113]；
    # 若跨 symbol 串号（混入 600000 的 volume=200..244(step4)）会得到 [218, 222, 226]。
    # 只断言 ma_60 全 NaN 无法区分正确与串号实现（突变测试已证明）。
    v1 = out[out["symbol"] == "000001.SZ"]["volume_ma_10"].iloc[9:12].tolist()
    assert v1 == pytest.approx([109.0, 111.0, 113.0])
    v2 = out[out["symbol"] == "600000.SH"]["volume_ma_10"].iloc[9:12].tolist()
    assert v2 == pytest.approx([218.0, 222.0, 226.0])


# ---------------------------------------------------------------------------
# DataManager 缓存键
# ---------------------------------------------------------------------------


def test_cache_key_excludes_sensitive_kwargs():
    """token/password/account 等敏感字段不能进入缓存文件名。"""
    from dquant.data.data_manager import DataManager

    dm = DataManager.__new__(DataManager)
    key = dm._get_cache_key(
        source="tushare",
        symbols=["000001.SZ"],
        start="2024-01-01",
        end="2024-12-31",
        kwargs={
            "token": "SUPER_SECRET_TOKEN",
            "password": "dont_leak_me",
            "account": "u12345",
            "freq": "D",
        },
    )
    assert "SUPER_SECRET_TOKEN" not in key
    assert "dont_leak_me" not in key
    assert "u12345" not in key
    assert "token=" not in key
    assert "password=" not in key
    assert "account=" not in key
    # 非敏感字段仍保留，保证缓存命中正常
    assert "freq" in key


# ---------------------------------------------------------------------------
# futures.py 保证金口径
# ---------------------------------------------------------------------------


def test_futures_long_open_close_no_double_count():
    """IF 多 1 手 @100 → 平仓 @110，净变动应恰好等于盈亏 +3000。"""
    from dquant.futures import INDEX_FUTURES, FuturesAccount

    account = FuturesAccount(initial_capital=1_000_000.0)
    account._contracts = INDEX_FUTURES

    assert account.open_position(symbol="IF", direction="long", quantity=1, price=100.0)
    pnl = account.close_position(symbol="IF", direction="long", price=110.0)

    delta = account.cash - 1_000_000.0
    # IF multiplier=300, 价差 10, 手数 1
    assert pnl == pytest.approx(3000.0)
    assert delta == pytest.approx(3000.0)


def test_futures_short_open_close_correct_pnl():
    """IF 空 1 手 @100 → 平仓 @90，盈亏应为 +3000。"""
    from dquant.futures import INDEX_FUTURES, FuturesAccount

    account = FuturesAccount(initial_capital=1_000_000.0)
    account._contracts = INDEX_FUTURES

    account.open_position(symbol="IF", direction="short", quantity=1, price=100.0)
    pnl = account.close_position(symbol="IF", direction="short", price=90.0)
    delta = account.cash - 1_000_000.0
    assert pnl == pytest.approx(3000.0)
    assert delta == pytest.approx(3000.0)


# ---------------------------------------------------------------------------
# ml_factors._temporal_split 边界
# ---------------------------------------------------------------------------


def test_temporal_split_5_dates_leaves_one_test_date():
    from dquant.ai.ml_factors import _temporal_split

    dates = pd.to_datetime(
        ["2024-01-01"] * 2
        + ["2024-01-02"] * 2
        + ["2024-01-03"] * 2
        + ["2024-01-04"] * 2
        + ["2024-01-05"] * 2
    )
    data = pd.DataFrame({"x": range(10)}, index=dates)
    X = np.arange(10).reshape(-1, 1)
    y = np.arange(10)
    mask = np.ones(10, dtype=bool)

    Xt, yt = _temporal_split(data, X, y, mask, train_ratio=0.8)
    assert Xt.shape[0] == 8  # 4 dates × 2 symbols
    # 必须留至少 1 个测试日期 × 2 symbols = 2
    assert 10 - Xt.shape[0] == 2


def test_temporal_split_2_dates_keeps_one_train_one_test():
    from dquant.ai.ml_factors import _temporal_split

    dates = pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"])
    data = pd.DataFrame({"x": [0, 1, 2, 3]}, index=dates)
    X = np.arange(4).reshape(-1, 1)
    y = np.arange(4)
    mask = np.ones(4, dtype=bool)

    Xt, _ = _temporal_split(data, X, y, mask, train_ratio=0.8)
    assert Xt.shape[0] == 2  # 1 train date × 2 symbols


def test_temporal_split_1_date_falls_back_to_row_split():
    from dquant.ai.ml_factors import _temporal_split

    dates = pd.to_datetime(["2024-01-01"] * 4)
    data = pd.DataFrame({"x": [0, 1, 2, 3]}, index=dates)
    X = np.arange(4).reshape(-1, 1)
    y = np.arange(4)
    mask = np.ones(4, dtype=bool)

    Xt, _ = _temporal_split(data, X, y, mask, train_ratio=0.8)
    assert Xt.shape[0] == 3  # row-count fallback


# ---------------------------------------------------------------------------
# safety 股票代码正则
# ---------------------------------------------------------------------------


def test_safety_rejects_seven_digit_codes():
    from dquant.broker.safety import OrderValidator

    for sym in ["6000000.SH", "3000000.SZ", "0000000.SZ"]:
        valid, _ = OrderValidator.validate_symbol(sym)
        assert valid is False, f"{sym} 应当被拒绝"


def test_safety_accepts_six_digit_known_prefixes():
    from dquant.broker.safety import OrderValidator

    for sym in ["600000.SH", "688888.SH", "000001.SZ", "301234.SZ", "430010.BJ"]:
        valid, msg = OrderValidator.validate_symbol(sym)
        assert valid, f"{sym} 应当被接受，但得到 {msg!r}"


# ---------------------------------------------------------------------------
# qlib_adapter 不再静默
# ---------------------------------------------------------------------------


def test_qlib_adapter_fit_without_dataset_raises():
    """无 Qlib Dataset 时 fit() 必须显式抛错，不能 no-op + _is_fitted=True。"""
    from dquant.ai.qlib_adapter import QlibModelAdapter

    adapter = QlibModelAdapter(model_name="gbdt")
    # 跳过 _init_qlib（避免触发 qlib import，专注于 fit 的契约）
    adapter._qlib_initialized = True
    adapter._dataset = None

    df = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(NotImplementedError):
        adapter.fit(df)


def test_qlib_adapter_predict_fallback_raises():
    """predict() 在 model=None 时必须抛 NotImplementedError，不能返回脏数据。"""
    from dquant.ai.qlib_adapter import QlibModelAdapter

    adapter = QlibModelAdapter(model_name="gbdt")
    adapter._qlib_initialized = True
    adapter._is_fitted = True
    adapter._dataset = None
    adapter._model = None

    with pytest.raises(NotImplementedError):
        adapter.predict(pd.DataFrame({"x": [1, 2, 3]}))


# ---------------------------------------------------------------------------
# logger 二次配置（已经修过，这里锁住不退化）
# ---------------------------------------------------------------------------


def test_logger_second_call_with_log_file_attaches_file_handler():
    from dquant.logger import DquantFileHandler, get_logger

    with TemporaryDirectory() as d:
        log_path = Path(d) / "a.log"
        lg = get_logger("dquant.tests.reconfig_ok")
        lg2 = get_logger("dquant.tests.reconfig_ok", log_file=str(log_path))

        has_file = any(isinstance(h, DquantFileHandler) for h in lg2.handlers)
        assert has_file, "二次配置补挂 FileHandler 失败"

        lg2.info("smoke test for reconfigured logger")
        assert log_path.exists()


# ---------------------------------------------------------------------------
# TradeJournal 失败恢复顺序（已修，锁住）
# ---------------------------------------------------------------------------


def test_trade_journal_preserves_order_after_recovery():
    from dquant.broker.base import Order
    from dquant.broker.trade_journal import TradeJournal

    order1 = Order(symbol="000001.SZ", side="BUY", quantity=100, order_id="o1")
    order2 = Order(symbol="000002.SZ", side="BUY", quantity=100, order_id="o2")

    with TemporaryDirectory() as d:
        journal = TradeJournal(d)
        real_open = open
        state = {"first_fail": True}

        def flaky(*args, **kwargs):
            # 第一次写盘失败，第二次恢复
            if state["first_fail"] and len(args) >= 2 and args[1] == "a":
                state["first_fail"] = False
                raise OSError("disk full")
            return real_open(*args, **kwargs)

        with patch("builtins.open", side_effect=flaky):
            journal.record("ORDER_PLACED", order1)
            journal.record("ORDER_PLACED", order2)

        fp = next(Path(d).glob("*.jsonl"))
        ids = [
            json.loads(line)["order_id"]
            for line in fp.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert ids == ["o1", "o2"], f"审计顺序错乱: {ids}"
