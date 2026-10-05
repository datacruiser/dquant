"""
第 5 轮深度 code review 修复的回归测试。

覆盖：DQN 探索期经验缓冲 / 目标网络同步 / 错误显式传播、
logger 去重与级别解析、交易环境无效价格守卫、Tushare 因子除零与 momentum 口径。
"""

import logging

import numpy as np
import pandas as pd
import pytest

from dquant.logger import get_logger, quiet_mode, set_log_level


@pytest.fixture(autouse=True)
def _restore_dquant_levels():
    """快照并恢复所有 dquant* logger 的级别，避免用例间全局状态泄漏。"""
    before = {
        name: child.level
        for name, child in logging.root.manager.loggerDict.items()
        if isinstance(child, logging.Logger) and (name == "dquant" or name.startswith("dquant."))
    }
    yield
    for name, level in before.items():
        logging.getLogger(name).setLevel(level)


# ---------------------------------------------------------------------------
# logger：每条日志只输出一次
# ---------------------------------------------------------------------------


def test_child_logger_does_not_duplicate_through_parents(capsys):
    """子 logger 自带 handler 后不得再向父 logger 传播，避免一条日志打两遍。"""
    get_logger("dquant.round5.parent")
    child = get_logger("dquant.round5.parent.engine")
    child.info("ROUND5-UNIQUE-MESSAGE")
    assert capsys.readouterr().out.count("ROUND5-UNIQUE-MESSAGE") == 1


def test_get_logger_respects_preexisting_plain_file_handler(tmp_path):
    """已挂 stdlib 裸 FileHandler 的 logger 再请求同路径 log_file 不得重复挂载。"""
    log1 = tmp_path / "a.log"
    lg = logging.getLogger("dquant.round5.filehandler")
    plain = logging.FileHandler(log1)
    lg.addHandler(plain)
    try:
        get_logger("dquant.round5.filehandler", log_file=str(log1))
        file_handlers = [h for h in lg.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1
    finally:
        lg.removeHandler(plain)


def test_get_logger_warns_on_conflicting_log_file(tmp_path, capsys):
    """换一个 log_file 再配置时必须显式警告，而不是静默沿用旧文件。"""
    get_logger("dquant.round5.reconfig", log_file=str(tmp_path / "a.log"))
    get_logger("dquant.round5.reconfig", log_file=str(tmp_path / "b.log"))
    assert "忽略" in capsys.readouterr().out


def test_quiet_mode_silences_child_loggers(capsys):
    """quiet_mode 必须作用于 get_logger 创建的子 logger（它们自带显式级别）。"""
    child = get_logger("dquant.round5.quiet", level="INFO")
    child.info("ROUND5-LOUD-BEFORE")
    assert "ROUND5-LOUD-BEFORE" in capsys.readouterr().out

    quiet_mode()
    child.info("ROUND5-SHOULD-BE-SILENT")
    child.error("ROUND5-EXPECTED-ERROR")
    out = capsys.readouterr().out
    assert "ROUND5-SHOULD-BE-SILENT" not in out
    assert "ROUND5-EXPECTED-ERROR" in out


def test_set_log_level_accepts_critical_warn_and_rejects_unknown():
    """CRITICAL 不得静默降级为 INFO；WARN 别名合法；未知值显式抛错。"""
    set_log_level("CRITICAL")
    assert logging.getLogger("dquant").level == logging.CRITICAL

    set_log_level("WARN")  # stdlib setLevel 认可的别名
    assert logging.getLogger("dquant").level == logging.WARNING

    with pytest.raises(ValueError):
        set_log_level("BOGUS")

    with pytest.raises(ValueError):
        get_logger("dquant.round5.bogus", level="NOT_A_LEVEL")


# ---------------------------------------------------------------------------
# TradingEnvironment：无效价格守卫
# ---------------------------------------------------------------------------


def _env_panel(bbb_index, bbb_close):
    dates = pd.date_range("2024-01-01", periods=30)
    aaa = pd.DataFrame({"symbol": "AAA", "close": 10.0}, index=dates)
    bbb = pd.DataFrame({"symbol": "BBB", "close": bbb_close}, index=bbb_index)
    return pd.concat([aaa, bbb])


def test_env_sell_skips_zero_price():
    """缺失行情被零填充：零价卖出不得把持仓按 0 收入清空。"""
    from dquant.ai.rl_agents import TradingEnvironment

    dates = pd.date_range("2024-01-01", periods=30)
    # BBB 只有前 5 天有数据 → 第 6 天（首个 step 日）价格为 0
    env = TradingEnvironment(_env_panel(dates[:5], 5.0), n_stocks=2, lookback=5)
    env.reset()

    env.positions[1] = 1000.0
    cash_before = env.cash
    env.step(np.array([1, 0]))  # 卖出 BBB，但当日价格为 0

    assert env.positions[1] == 1000.0, "零价（停牌填充）不应把持仓清零"
    assert env.cash == cash_before, "零价卖出不应产生收入"


def test_env_buy_skips_nan_price():
    """NaN close 不得进入买入份额计算（int(NaN) 会 ValueError 崩溃）。"""
    from dquant.ai.rl_agents import TradingEnvironment

    dates = pd.date_range("2024-01-01", periods=30)
    closes = [5.0] * 5 + [np.nan]  # 第 6 天（首个 step 日）为 NaN
    env = TradingEnvironment(_env_panel(dates[:6], closes), n_stocks=2, lookback=5)
    env.reset()

    env.step(np.array([1, 2]))  # 买入 BBB，当日价格 NaN → 跳过，不得崩溃

    assert env.positions[1] == 0


# ---------------------------------------------------------------------------
# DQN：探索期经验必须入缓冲、目标网络必须周期性同步、错误必须显式传播
# ---------------------------------------------------------------------------


def _make_agent(**overrides):
    pytest.importorskip("torch")
    from dquant.ai.rl_agents import DQNAgent

    params = dict(n_stocks=2, batch_size=4, target_update_freq=1)
    params.update(overrides)
    return DQNAgent(**params)


def test_dqn_exploration_still_builds_model_and_buffers():
    """epsilon=1.0 必然走探索分支，但模型仍须构建、经验仍须入缓冲。"""
    agent = _make_agent(epsilon=1.0)
    state = np.linspace(1.0, 10.0, 24)

    action = agent.select_action(state, training=True)
    assert agent._model is not None, "探索分支不能阻止模型构建（否则 update 会丢经验）"

    agent.update((state, action, 0.01, state * 1.01, False))
    assert len(agent._buffer) == 1


def test_dqn_update_buffers_without_prior_select():
    """不经过 select_action 直接 update 也必须缓冲经验并构建模型。"""
    agent = _make_agent()
    state = np.linspace(1.0, 10.0, 24)

    agent.update((state, np.ones(2, dtype=int), 0.0, state, True))

    assert agent._model is not None
    assert len(agent._buffer) == 1


def test_dqn_target_model_syncs_during_training():
    """target 网络必须在训练中周期性同步，而不是永远停留在随机初始化。"""
    torch = pytest.importorskip("torch")
    agent = _make_agent(target_update_freq=1)
    state = np.linspace(1.0, 10.0, 24)

    for _ in range(6):  # batch_size=4 → 第 4 次起触发训练
        agent.update((state, np.array([1, 1]), 0.0, state, False))

    for key, param in agent._model.state_dict().items():
        target_param = agent._target_model.state_dict()[key]
        assert torch.allclose(param, target_param), f"target 网络未同步: {key}"


def test_dqn_select_action_shape_error_propagates():
    """输入维度不匹配必须显式抛错，而不是静默返回全 HOLD。"""
    agent = _make_agent()
    state = np.linspace(1.0, 10.0, 24)
    agent.select_action(state, training=False)  # 以 state_dim=24 构建模型

    with pytest.raises(RuntimeError):
        agent.select_action(np.ones(10), training=False)


# ---------------------------------------------------------------------------
# Tushare 扩展因子：除零守卫与 momentum 口径
# ---------------------------------------------------------------------------


def _single_symbol_panel(closes, high=None, low=None):
    dates = pd.date_range("2024-01-01", periods=len(closes))
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {
            "symbol": "AAA",
            "close": closes,
            "high": closes + 1 if high is None else high,
            "low": closes - 1 if low is None else low,
            "volume": 100.0,
            "turnover": 1.0,
        },
        index=dates,
    )


def test_tushare_price_position_degenerate_range_is_nan():
    """一字板（high_max == low_min）不得产生 inf；退化窗口统一置 NaN。"""
    from dquant.data.tushare_loader import TushareLoader

    n = 25
    closes = [10.0] * (n - 1) + [15.0]  # 最后一行 close 被抬出区间 → 无守卫时 (c-l)/0 = inf
    df = _single_symbol_panel(closes, high=[10.0] * n, low=[10.0] * n)

    out = TushareLoader.__new__(TushareLoader)._calculate_factors(df)

    assert not np.isinf(out["price_position_20"]).any()
    assert out["price_position_20"].isna().all()


def test_tushare_momentum_60_no_implicit_pad():
    """momentum_60 不得隐式 pad 内部 NaN（与 builtin_factors 的 fill_method=None 口径一致）。"""
    from dquant.data.tushare_loader import TushareLoader

    n = 70
    closes = np.arange(1.0, n + 1.0)
    closes[5] = np.nan  # 内部缺口
    df = _single_symbol_panel(closes)

    out = TushareLoader.__new__(TushareLoader)._calculate_factors(df)
    m = out["momentum_60"]

    # 隐式 pad（旧行为）会拿被 pad 的 close[5]=5 当分母：第 65 行 = 65/5-1 = 12
    # fill_method=None（正确口径）：第 65 行 NaN，第 66 行 = 67/7-1
    assert np.isnan(m.iloc[65])
    assert m.iloc[66] == pytest.approx(67.0 / 7.0 - 1.0)


def test_tushare_factors_accept_unnamed_index():
    """无名 DatetimeIndex 不应触发按 'date' 列排序的 KeyError。"""
    from dquant.data.tushare_loader import TushareLoader

    df = _single_symbol_panel(np.arange(1.0, 26.0))
    df = df.rename_axis(index=None)  # 抹掉索引名，复现无名索引场景

    out = TushareLoader.__new__(TushareLoader)._calculate_factors(df)
    assert "volume_ma_10" in out.columns
    assert len(out) == 25
