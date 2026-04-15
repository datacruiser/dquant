"""综合测试（修复版）。"""

import numpy as np
import pandas as pd

from dquant import (
    BacktestEngine,
    DQuantConfig,
    PositionSizer,
    RiskManager,
    StopLoss,
    default_config,
    format_money,
    format_percent,
    get_factor,
    get_logger,
    get_trading_days,
)
from dquant.strategy.base import BaseStrategy, Signal, SignalType


def create_test_data(days=100, symbols=None):
    """创建测试数据"""
    rng = np.random.default_rng(0)
    if symbols is None:
        symbols = ["000001.SZ", "600000.SH"]

    dates = pd.date_range("2023-01-01", periods=days, freq="D")
    data_list = []

    for symbol in symbols:
        for i, date in enumerate(dates):
            base_price = 10 + i * 0.05
            data_list.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": base_price,
                    "high": base_price * 1.02,
                    "low": base_price * 0.98,
                    "close": base_price * 1.01,
                    "volume": 1000000 + int(rng.normal(0, 100000)),
                }
            )

    return pd.DataFrame(data_list).set_index("date")


def test_factors():
    """关键因子至少返回标准结果结构。"""
    data = create_test_data()
    for name in ["momentum", "rsi", "volatility"]:
        factor = get_factor(name, window=20)
        factor.fit(data)
        result = factor.predict(data)
        assert list(result.columns) == ["symbol", "score"]


def test_risk():
    """风险管理基础组件返回合理结果。"""
    sizer = PositionSizer(method="equal_weight", total_value=1_000_000)
    positions = sizer.size(["000001.SZ", "600000.SH", "000002.SZ"])
    assert len(positions) == 3
    assert sum(positions.values()) <= sizer.total_value

    manager = RiskManager(max_drawdown=0.15)
    assert manager.max_drawdown == 0.15

    stop_price = StopLoss.fixed_stop(entry_price=100.0, stop_pct=0.1)
    assert stop_price == 90.0


class SimpleStrategy(BaseStrategy):
    def generate_signals(self, data):
        first_date = data.index.min()
        return [
            Signal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                strength=1.0,
                timestamp=first_date,
            )
            for symbol in data["symbol"].unique()
        ]


def test_backtest():
    """回测引擎可输出带绩效的结果对象。"""
    data = create_test_data()
    result = BacktestEngine(data, SimpleStrategy(), initial_cash=1_000_000).run()

    assert result.metrics.total_trades == 2
    assert result.metrics.total_return > 0
    assert len(result.portfolio.to_dataframe()) == len(data.index.unique())


def test_utils():
    """工具函数返回预期格式。"""
    days = get_trading_days("2023-01-01", "2023-12-31")
    assert len(days) > 0
    assert format_money(1234567.89) == "¥123.46万"
    assert format_percent(0.1234) == "12.34%"


def test_config():
    """配置对象可以正常构造。"""
    config = DQuantConfig()
    assert isinstance(config, DQuantConfig)
    assert type(default_config).__name__ == "DQuantConfig"


def test_logger():
    """日志工厂返回带名称的 logger。"""
    logger = get_logger("test")
    assert logger.name == "test"
