"""
绩效分析
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

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
        annual_return = (
            (1 + total_return) ** (TRADING_DAYS_PER_YEAR / trading_days) - 1
            if trading_days > 0
            else 0
        )

        # 年化波动率
        volatility = returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)

        # 夏普比率
        rf_daily = rf / TRADING_DAYS_PER_YEAR
        excess_returns = returns - rf_daily
        excess_std = excess_returns.std()
        sharpe = (
            excess_returns.mean() / excess_std * np.sqrt(TRADING_DAYS_PER_YEAR)
            if excess_std > 0
            else 0
        )

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

    @classmethod
    def from_nav_and_trades(
        cls,
        nav_series: pd.Series,
        trades: pd.DataFrame,
        rf: float = 0.03,
    ) -> "Metrics":
        """
        从净值序列和交易记录计算绩效（含 win_rate/profit_factor）

        Args:
            nav_series: 净值序列
            trades: 交易记录 DataFrame，需包含 'pnl' 列
            rf: 无风险利率 (年化)
        """
        metrics = cls.from_nav(nav_series, rf=rf)

        if trades is not None and "pnl" in trades.columns and len(trades) > 0:
            pnl = trades["pnl"].dropna()
            if len(pnl) > 0:
                wins = pnl[pnl > 0]
                losses = pnl[pnl < 0]
                metrics.total_trades = len(pnl)
                metrics.win_rate = len(wins) / len(pnl) if len(pnl) > 0 else 0.0
                gross_profit = wins.sum()
                gross_loss = abs(losses.sum())
                metrics.profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        return metrics

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
            "total_return": self.total_return,
            "annual_return": self.annual_return,
            "sharpe": self.sharpe,
            "max_drawdown": self.max_drawdown,
            "volatility": self.volatility,
            "calmar": self.calmar,
        }
