"""
风险管理模块

提供仓位管理、风险控制、资金管理等功能。
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from dquant.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PositionLimit:
    """仓位限制"""

    max_single_pct: float = 0.1  # 单只股票最大仓位 10%
    max_sector_pct: float = 0.3  # 单个行业最大仓位 30%
    max_total_pct: float = 0.95  # 最大总仓位 95%
    min_cash_pct: float = 0.05  # 最小现金比例 5%


@dataclass
class RiskMetrics:
    """风险指标"""

    var_95: float = 0.0  # 95% VaR
    var_99: float = 0.0  # 99% VaR
    cvar_95: float = 0.0  # 95% CVaR
    beta: float = 0.0  # Beta
    tracking_error: float = 0.0  # 跟踪误差
    information_ratio: float = 0.0  # 信息比率


class PositionSizer:
    """
    仓位管理器

    决定每只股票的买入数量。

    Usage:
        sizer = PositionSizer(method='equal_weight', total_value=1000000)
        positions = sizer.size(['000001.SZ', '000002.SZ'], signals)
    """

    def __init__(
        self,
        method: str = "equal_weight",
        total_value: float = 1000000,
        limits: Optional[PositionLimit] = None,
    ):
        """
        Args:
            method: 仓位分配方法
                - equal_weight: 等权
                - signal_weight: 按信号强度加权
                - risk_parity: 风险平价
                - kelly: 凯利公式
            total_value: 总资金
            limits: 仓位限制
        """
        self.method = method
        self.total_value = total_value
        self.limits = limits or PositionLimit()

    def size(
        self,
        symbols: List[str],
        signals: Optional[Dict[str, float]] = None,
        volatilities: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        计算每只股票的目标仓位

        Args:
            symbols: 股票列表
            signals: 信号强度 {symbol: strength}
            volatilities: 波动率 {symbol: volatility}

        Returns:
            目标仓位金额 {symbol: value}
        """
        n = len(symbols)
        if n == 0:
            return {}

        if self.method == "equal_weight":
            return self._equal_weight(symbols)
        elif self.method == "signal_weight":
            return self._signal_weight(symbols, signals or {})
        elif self.method == "risk_parity":
            return self._risk_parity(symbols, volatilities or {})
        elif self.method == "kelly":
            return self._kelly(symbols, signals or {})
        else:
            return self._equal_weight(symbols)

    def _equal_weight(self, symbols: List[str]) -> Dict[str, float]:
        """等权分配"""
        max_position = self.total_value * self.limits.max_single_pct
        per_stock = min(self.total_value / len(symbols), max_position)

        return {symbol: per_stock for symbol in symbols}

    def _signal_weight(
        self,
        symbols: List[str],
        signals: Dict[str, float],
    ) -> Dict[str, float]:
        """按信号强度加权"""
        total_signal = sum(abs(signals.get(s, 0)) for s in symbols)

        if total_signal == 0:
            return self._equal_weight(symbols)

        positions = {}
        for symbol in symbols:
            signal = abs(signals.get(symbol, 0))
            weight = signal / total_signal
            value = self.total_value * weight
            value = min(value, self.total_value * self.limits.max_single_pct)
            positions[symbol] = value

        return positions

    def _risk_parity(
        self,
        symbols: List[str],
        volatilities: Dict[str, float],
    ) -> Dict[str, float]:
        """风险平价"""
        # 使用副本，避免修改调用方传入的字典
        volatilities = dict(volatilities)
        for symbol in symbols:
            if symbol not in volatilities:
                volatilities[symbol] = 0.02  # 默认 2% 日波动

        # 计算风险贡献权重
        inv_vol = {s: 1.0 / volatilities[s] for s in symbols}
        total_inv_vol = sum(inv_vol.values())

        positions = {}
        for symbol in symbols:
            weight = inv_vol[symbol] / total_inv_vol
            value = self.total_value * weight
            value = min(value, self.total_value * self.limits.max_single_pct)
            positions[symbol] = value

        return positions

    def _kelly(
        self,
        symbols: List[str],
        signals: Dict[str, float],
    ) -> Dict[str, float]:
        """凯利公式 (简化版)"""
        positions = {}

        for symbol in symbols:
            signal = signals.get(symbol, 0)

            # 简化: 用信号强度近似期望收益
            # 上界保护: win_rate 不超过 0.95
            win_rate = min(0.5 + signal * 0.1, 0.95)
            win_loss_ratio = 2.0  # 盈亏比

            # Kelly: f = (p * b - q) / b
            # p = 胜率, q = 1-p, b = 盈亏比
            kelly_pct = (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio
            # 限制范围 [0, max_single_pct]
            kelly_pct = max(0, min(kelly_pct, self.limits.max_single_pct))

            positions[symbol] = self.total_value * kelly_pct

        return positions


class RiskManager:
    """
    风险管理器

    监控和控制投资组合风险。
    """

    def __init__(
        self,
        limits: Optional[PositionLimit] = None,
        max_drawdown: float = 0.15,  # 最大回撤限制
        max_daily_loss: float = 0.03,  # 单日最大亏损
    ):
        self.limits = limits or PositionLimit()
        self.max_drawdown = max_drawdown
        self.max_daily_loss = max_daily_loss

        self.peak_value = 0.0
        self.current_drawdown = 0.0

        # 日亏损追踪
        self.daily_start_value: float = 0.0
        self.daily_start_date: Optional[str] = None
        self.halt_trading: bool = False

        # 持久化路径（可选）
        self._state_path: Optional[Path] = None

    def enable_persistence(self, state_path: str = "risk_state.json"):
        """
        启用风控状态持久化

        Args:
            state_path: 状态文件路径
        """
        self._state_path = Path(state_path)
        # 尝试恢复已有状态
        self.restore_state()

    def save_state(self):
        """保存当前风控状态到文件"""
        if self._state_path is None:
            return

        state = {
            "peak_value": self.peak_value,
            "current_drawdown": self.current_drawdown,
            "daily_start_value": self.daily_start_value,
            "daily_start_date": self.daily_start_date,
            "halt_trading": self.halt_trading,
        }
        try:
            with open(self._state_path, "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.error(f"[RiskManager] Failed to save state: {e}")

    def restore_state(self) -> bool:
        """
        从文件恢复风控状态

        Returns:
            是否成功恢复
        """
        if self._state_path is None or not self._state_path.exists():
            return False

        try:
            with open(self._state_path, "r") as f:
                state = json.load(f)

            self.peak_value = state.get("peak_value", 0.0)
            self.current_drawdown = state.get("current_drawdown", 0.0)
            self.daily_start_value = state.get("daily_start_value", 0.0)
            self.daily_start_date = state.get("daily_start_date")
            self.halt_trading = state.get("halt_trading", False)

            logger.info(
                f"[RiskManager] State restored: peak={self.peak_value:.2f}, "
                f"dd={self.current_drawdown:.2%}, halt={self.halt_trading}"
            )
            return True
        except Exception as e:
            logger.warning(f"[RiskManager] Failed to restore state: {e}")
            return False

    def check_position_limit(
        self,
        symbol: str,
        position_value: float,
        total_value: float,
    ) -> Tuple[bool, str]:
        """
        检查仓位限制

        Returns:
            (是否通过, 原因)
        """
        if total_value <= 0:
            return False, "总价值为零或负，无法计算仓位比例"
        pct = position_value / total_value

        if pct > self.limits.max_single_pct:
            return (
                False,
                f"单只股票仓位 {pct:.1%} 超过限制 {self.limits.max_single_pct:.1%}",
            )

        return True, "OK"

    def check_drawdown(
        self,
        current_value: float,
    ) -> Tuple[bool, float]:
        """
        检查回撤

        Returns:
            (是否触发风控, 当前回撤)
        """
        if current_value > self.peak_value:
            self.peak_value = current_value

        if self.peak_value > 0:
            self.current_drawdown = (self.peak_value - current_value) / self.peak_value
        else:
            self.current_drawdown = 0.0

        triggered = self.current_drawdown >= self.max_drawdown
        if triggered:
            self.halt_trading = True

        self.save_state()
        return triggered, self.current_drawdown

    def reset_daily_start(self, current_value: float, date_str: str):
        """
        每日开盘时重置日亏损基准

        Args:
            current_value: 当前组合价值
            date_str: 当日日期字符串 (YYYY-MM-DD)
        """
        if self.daily_start_date != date_str:
            self.daily_start_value = current_value
            self.daily_start_date = date_str
            self.save_state()

    def check_daily_loss(self, current_value: float) -> Tuple[bool, float]:
        """
        检查日亏损

        Returns:
            (是否触发日亏损风控, 当前日亏损比例)
        """
        if self.daily_start_value <= 0:
            return False, 0.0

        daily_loss = (self.daily_start_value - current_value) / self.daily_start_value
        triggered = daily_loss >= self.max_daily_loss

        if triggered:
            self.halt_trading = True

        self.save_state()
        return triggered, daily_loss

    def should_halt(self) -> bool:
        """综合判断是否应停止交易（回撤 + 日亏损）"""
        return self.halt_trading

    def calculate_var(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
    ) -> float:
        """
        计算 VaR (Value at Risk)

        Args:
            returns: 收益率序列
            confidence: 置信水平

        Returns:
            VaR (正值表示损失)
        """
        if len(returns) == 0:
            return 0.0

        alpha = 1 - confidence
        var = abs(returns.quantile(alpha))

        return var

    def calculate_cvar(
        self,
        returns: pd.Series,
        confidence: float = 0.95,
    ) -> float:
        """
        计算 CVaR (Conditional VaR)

        CVaR = E[Loss | Loss > VaR]
        """
        if len(returns) == 0:
            return 0.0

        alpha = 1 - confidence
        var = returns.quantile(alpha)

        # 计算超过 VaR 的平均损失
        tail_losses = returns[returns <= var]

        if len(tail_losses) == 0:
            return abs(var)

        return abs(tail_losses.mean())

    def calculate_risk_metrics(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> RiskMetrics:
        """
        计算风险指标
        """
        metrics = RiskMetrics()

        # VaR
        metrics.var_95 = self.calculate_var(returns, 0.95)
        metrics.var_99 = self.calculate_var(returns, 0.99)
        metrics.cvar_95 = self.calculate_cvar(returns, 0.95)

        # Beta
        if benchmark_returns is not None:
            covariance = returns.cov(benchmark_returns)
            variance = benchmark_returns.var()
            metrics.beta = covariance / variance if variance > 0 else 0

            # 跟踪误差
            excess_returns = returns - benchmark_returns
            metrics.tracking_error = excess_returns.std() * np.sqrt(252)

            # 信息比率
            if metrics.tracking_error > 0:
                metrics.information_ratio = excess_returns.mean() * 252 / metrics.tracking_error

        return metrics

    def should_stop_loss(
        self,
        current_value: float,
        initial_value: float,
    ) -> Tuple[bool, str]:
        """
        是否应该止损

        Returns:
            (是否止损, 原因)
        """
        if initial_value <= 0:
            return False, "初始价值为零或负"
        loss = (initial_value - current_value) / initial_value

        if loss >= self.max_drawdown:
            return True, f"触发止损: 亏损 {loss:.1%} >= {self.max_drawdown:.1%}"

        return False, "OK"


class StopLoss:
    """
    止损策略
    """

    @staticmethod
    def fixed_stop(
        entry_price: float,
        stop_pct: float = 0.05,
    ) -> float:
        """固定止损"""
        return entry_price * (1 - stop_pct)

    @staticmethod
    def trailing_stop(
        current_price: float,
        highest_price: float,
        trailing_pct: float = 0.1,
    ) -> float:
        """移动止损：基于最高价的回撤比例"""
        stop = highest_price * (1 - trailing_pct)
        # 确保 stop 不超过当前价格（否则无意义）
        return min(stop, current_price)

    @staticmethod
    def atr_stop(
        entry_price: float,
        atr: float,
        multiplier: float = 2.0,
    ) -> float:
        """ATR 止损"""
        stop = entry_price - multiplier * atr
        # 确保 stop 不为负数
        return max(stop, 0.0)

    @staticmethod
    def volatility_stop(
        entry_price: float,
        volatility: float,
        multiplier: float = 2.0,
    ) -> float:
        """波动率止损"""
        stop = entry_price * (1 - multiplier * volatility)
        # 确保 stop 不为负数
        return max(stop, 0.0)


class TakeProfit:
    """
    止盈策略
    """

    @staticmethod
    def fixed_profit(
        entry_price: float,
        profit_pct: float = 0.2,
    ) -> float:
        """固定止盈"""
        return entry_price * (1 + profit_pct)

    @staticmethod
    def risk_reward(
        entry_price: float,
        stop_price: float,
        ratio: float = 2.0,
    ) -> float:
        """风险回报比止盈"""
        risk = entry_price - stop_price
        return entry_price + risk * ratio
