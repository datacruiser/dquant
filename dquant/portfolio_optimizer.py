"""
多策略组合与组合优化器

支持多策略并行运行、信号合成、以及多种组合优化方法。
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from dquant.logger import get_logger
from dquant.strategy.base import BaseStrategy, Signal

logger = get_logger(__name__)


# ============================================================
# 多策略组合器
# ============================================================


class MultiStrategyPortfolio:
    """
    多策略组合器

    管理多个策略实例，合并信号后统一输出。

    Usage:
        msp = MultiStrategyPortfolio()
        msp.add_strategy('momentum', momentum_strategy, weight=0.6)
        msp.add_strategy('flow', flow_strategy, weight=0.4)
        signals = msp.generate_signals(data)
    """

    def __init__(self):
        self._strategies: Dict[str, dict] = {}

    def add_strategy(
        self,
        name: str,
        strategy: BaseStrategy,
        weight: float = 1.0,
        signal_filter: Optional[str] = None,
    ):
        """
        添加策略

        Args:
            name: 策略名称
            strategy: 策略实例
            weight: 权重（用于信号合成）
            signal_filter: 信号类型过滤 ('buy', 'sell', None=全部)
        """
        if weight < 0:
            raise ValueError(f"Strategy weight must be >= 0, got {weight}")
        self._strategies[name] = {
            "strategy": strategy,
            "weight": weight,
            "signal_filter": signal_filter,
        }

    def remove_strategy(self, name: str):
        """移除策略"""
        self._strategies.pop(name, None)

    @property
    def strategy_names(self) -> List[str]:
        return list(self._strategies.keys())

    def generate_signals(self, data: pd.DataFrame) -> List[Signal]:
        """
        生成合成信号

        对每个策略独立生成信号，然后按权重合成。同股票同方向的信号合并为加权强度。
        """
        # 收集所有策略的信号
        all_signals: Dict[str, List[Signal]] = {}

        for name, config in self._strategies.items():
            strategy = config["strategy"]
            weight = config["weight"]
            filt = config["signal_filter"]

            try:
                raw_signals = strategy.generate_signals(data)
            except Exception as e:
                logger.warning(f"[MultiStrategy] Strategy '{name}' failed: {e}")
                continue

            for sig in raw_signals:
                # 过滤信号类型
                if filt == "buy" and not sig.is_buy:
                    continue
                if filt == "sell" and not sig.is_sell:
                    continue

                key = f"{sig.symbol}_{sig.signal_type.value}"
                weighted_sig = Signal(
                    symbol=sig.symbol,
                    signal_type=sig.signal_type,
                    strength=sig.strength * weight,
                    price=sig.price,
                    timestamp=sig.timestamp,
                    metadata={**(sig.metadata or {}), "source": name, "raw_strength": sig.strength},
                )

                if key not in all_signals:
                    all_signals[key] = []
                all_signals[key].append(weighted_sig)

        # 合并信号
        merged = []
        for key, signals in all_signals.items():
            if len(signals) == 1:
                merged.append(signals[0])
            else:
                total_weight = sum(s.strength for s in signals)
                # 使用强度最高的信号的元数据
                best = max(signals, key=lambda s: s.strength)
                merged.append(
                    Signal(
                        symbol=best.symbol,
                        signal_type=best.signal_type,
                        strength=total_weight,
                        price=best.price,
                        timestamp=best.timestamp,
                        metadata={
                            **best.metadata,
                            "n_sources": len(signals),
                            "sources": [s.metadata.get("source", "?") for s in signals],
                        },
                    )
                )

        return merged


# ============================================================
# 组合优化器
# ============================================================


@dataclass
class OptimizationResult:
    """优化结果"""

    weights: Dict[str, float]
    expected_return: float = 0.0
    expected_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    method: str = ""


class PortfolioOptimizer:
    """
    组合优化器

    支持多种优化方法：
    - equal_weight: 等权
    - mean_variance: 均值方差优化（最大 Sharpe）
    - risk_parity: 风险平价
    - min_variance: 最小方差
    - black_litterman: Black-Litterman（需要 views）

    Usage:
        optimizer = PortfolioOptimizer(returns_df)
        result = optimizer.optimize(method='risk_parity')
        print(result.weights)
    """

    def __init__(self, returns: pd.DataFrame):
        """
        Args:
            returns: 收益率 DataFrame (index=date, columns=symbols)
        """
        self.returns = returns
        self.symbols = list(returns.columns)
        self.n_assets = len(self.symbols)

        # 预计算
        self._mean_returns = returns.mean()
        self._cov_matrix = returns.cov()

    def optimize(self, method: str = "equal_weight", **kwargs) -> OptimizationResult:
        """
        执行优化

        Args:
            method: 优化方法
            **kwargs: 方法特定参数

        Returns:
            OptimizationResult
        """
        if method == "equal_weight":
            return self._equal_weight()
        elif method == "mean_variance":
            return self._mean_variance(**kwargs)
        elif method == "risk_parity":
            return self._risk_parity()
        elif method == "min_variance":
            return self._min_variance()
        elif method == "black_litterman":
            return self._black_litterman(**kwargs)
        else:
            raise ValueError(f"Unknown optimization method: {method}")

    def _portfolio_metrics(self, weights: np.ndarray) -> Tuple[float, float, float]:
        """计算组合指标"""
        ret = np.dot(weights, self._mean_returns) * 252
        vol = np.sqrt(np.dot(weights, np.dot(self._cov_matrix.values * 252, weights)))
        sharpe = ret / vol if vol > 0 else 0.0
        return ret, vol, sharpe

    def _equal_weight(self) -> OptimizationResult:
        """等权分配"""
        w = 1.0 / self.n_assets
        weights = {s: w for s in self.symbols}
        ret, vol, sharpe = self._portfolio_metrics(np.full(self.n_assets, w))
        return OptimizationResult(
            weights=weights, expected_return=ret,
            expected_volatility=vol, sharpe_ratio=sharpe, method="equal_weight",
        )

    def _risk_parity(self) -> OptimizationResult:
        """
        风险平价（逆波动率加权）

        每个资产的风险贡献相等: w_i ∝ 1/σ_i
        """
        vols = self.returns.std()
        inv_vol = 1.0 / vols.replace(0, np.nan).fillna(1e-10)
        total = inv_vol.sum()
        weights_arr = (inv_vol / total).values
        weights = dict(zip(self.symbols, weights_arr))
        ret, vol, sharpe = self._portfolio_metrics(weights_arr)
        return OptimizationResult(
            weights=weights, expected_return=ret,
            expected_volatility=vol, sharpe_ratio=sharpe, method="risk_parity",
        )

    def _mean_variance(self, risk_free: float = 0.02) -> OptimizationResult:
        """
        均值方差优化（最大 Sharpe，数值搜索）

        使用网格搜索避免 scipy 依赖。
        """
        if self.n_assets <= 5:
            # 小规模：网格搜索
            best_sharpe = -np.inf
            best_weights = np.ones(self.n_assets) / self.n_assets

            # 生成随机权重组合
            np.random.seed(42)
            for _ in range(5000):
                raw = np.random.dirichlet(np.ones(self.n_assets))
                ret, vol, sharpe = self._portfolio_metrics(raw)
                adj_sharpe = (ret - risk_free) / vol if vol > 0 else 0
                if adj_sharpe > best_sharpe:
                    best_sharpe = adj_sharpe
                    best_weights = raw

            weights = dict(zip(self.symbols, best_weights))
            ret, vol, sharpe = self._portfolio_metrics(best_weights)
            return OptimizationResult(
                weights=weights, expected_return=ret,
                expected_volatility=vol, sharpe_ratio=sharpe, method="mean_variance",
            )
        else:
            # 大规模：退化为风险平价
            logger.info("[Optimizer] Too many assets for grid search, falling back to risk_parity")
            return self._risk_parity()

    def _min_variance(self) -> OptimizationResult:
        """
        最小方差组合

        使用迭代收缩方法：从等权开始，逐步降低方差。
        """
        # 简化实现：基于协方差矩阵逆的最小方差
        try:
            cov = self._cov_matrix.values
            inv_cov = np.linalg.pinv(cov)
            ones = np.ones(self.n_assets)
            w = inv_cov @ ones
            w = w / w.sum()
            # 确保权重非负
            w = np.maximum(w, 0.001)
            w = w / w.sum()
        except np.linalg.LinAlgError:
            w = np.ones(self.n_assets) / self.n_assets

        weights = dict(zip(self.symbols, w))
        ret, vol, sharpe = self._portfolio_metrics(w)
        return OptimizationResult(
            weights=weights, expected_return=ret,
            expected_volatility=vol, sharpe_ratio=sharpe, method="min_variance",
        )

    def _black_litterman(
        self,
        views: Optional[Dict[str, float]] = None,
        tau: float = 0.05,
        risk_aversion: float = 2.5,
    ) -> OptimizationResult:
        """
        Black-Litterman 模型

        将市场均衡与投资者主观观点结合。

        Args:
            views: 主观观点 {symbol: 预期超额收益}
            tau: 观点不确定性参数 (0-1)
            risk_aversion: 风险厌恶系数
        """
        # 市场均衡（等权作为先验）
        pi = self._mean_returns.values * 252  # 年化均衡收益
        Sigma = self._cov_matrix.values * 252  # 年化协方差

        if views:
            # 构建观点矩阵
            P = np.zeros(self.n_assets)
            Q = np.zeros(1)
            Omega = np.zeros((1, 1))

            for i, sym in enumerate(self.symbols):
                if sym in views:
                    P[i] = 1.0
                    Q[0] = views[sym]
                    Omega[0, 0] = tau

            # Black-Litterman 后验
            try:
                tau_Sigma = tau * Sigma
                tau_Sigma_P = tau_Sigma @ P
                Omega_inv = (
                    1.0 / Omega[0, 0] if Omega[0, 0] > 0 else 0
                )
                mu_bl = pi + tau_Sigma_P * Omega_inv * (Q[0] - P @ pi)
            except np.linalg.LinAlgError:
                mu_bl = pi
        else:
            mu_bl = pi

        # 优化权重
        try:
            inv_sigma = np.linalg.pinv(risk_aversion * Sigma)
            w = inv_sigma @ mu_bl
            w = np.maximum(w, 0.001)
            w = w / w.sum()
        except np.linalg.LinAlgError:
            w = np.ones(self.n_assets) / self.n_assets

        weights = dict(zip(self.symbols, w))
        ret, vol, sharpe = self._portfolio_metrics(w)
        return OptimizationResult(
            weights=weights, expected_return=ret,
            expected_volatility=vol, sharpe_ratio=sharpe, method="black_litterman",
        )
