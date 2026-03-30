"""
绩效分析
"""

from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np

from dquant.constants import TRADING_DAYS_PER_YEAR


@dataclass
class Metrics:
    """
    回测绩效指标
    """
    total_return: float = 0.0  # 总收益率
    annual_return: float = 0.0  # 年化收益率
    sharpe: float = 0.0  # 夏普比率
    max_drawdown: float = 0.0  # 最大回撤
    win_rate: float = 0.0  # 胜率
    profit_factor: float = 0.0  # 盈亏比
    total_trades: int = 0  # 总交易次数
    volatility: float = 0.0  # 年化波动率
    calmar: float = 0.0  # 卡玛比率

    @classmethod
    def from_nav(cls, nav_series: pd.Series, rf: float = 0.03) -> "Metrics":
        """
        从净值序列计算绩效

        Args:
            nav_series: 净值序列
            rf: 无风险利率 (年化)
        """
        if len(nav_series) < 2:
            return cls()

        # 日收益率
        returns = nav_series.pct_change().dropna()

        # 总收益率
        total_return = nav_series.iloc[-1] / nav_series.iloc[0] - 1

        # 年化收益率
        trading_days = len(returns)
        annual_return = (1 + total_return) ** (TRADING_DAYS_PER_YEAR / trading_days) - 1 if trading_days > 0 else 0

        # 年化波动率
        volatility = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

        # 夏普比率
        rf_daily = rf / TRADING_DAYS_PER_YEAR
        excess_returns = returns - rf_daily
        excess_std = excess_returns.std()
        sharpe = excess_returns.mean() / excess_std * np.sqrt(TRADING_DAYS_PER_YEAR) if excess_std > 0 else 0

        # 最大回撤
        cummax = nav_series.cummax()
        drawdown = (nav_series - cummax) / cummax
        max_drawdown = abs(drawdown.min())

        # 卡玛比率
        calmar = annual_return / max_drawdown if max_drawdown > 0 else 0

        return cls(
            total_return=total_return,
            annual_return=annual_return,
            sharpe=sharpe,
            max_drawdown=max_drawdown,
            volatility=volatility,
            calmar=calmar,
        )

    def __repr__(self):
        return (
            f"Metrics(\n"
            f"  total_return={self.total_return:.2%}\n"
            f"  annual_return={self.annual_return:.2%}\n"
            f"  sharpe={self.sharpe:.2f}\n"
            f"  max_drawdown={self.max_drawdown:.2%}\n"
            f"  volatility={self.volatility:.2%}\n"
            f"  calmar={self.calmar:.2f}\n"
            f")"
        )

    def to_dict(self) -> dict:
        return {
            'total_return': self.total_return,
            'annual_return': self.annual_return,
            'sharpe': self.sharpe,
            'max_drawdown': self.max_drawdown,
            'volatility': self.volatility,
            'calmar': self.calmar,
        }
