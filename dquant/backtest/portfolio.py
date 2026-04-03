"""
组合管理
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
from dquant.constants import DEFAULT_COMMISSION, DEFAULT_SLIPPAGE, DEFAULT_STAMP_DUTY, DEFAULT_INITIAL_CASH, MIN_SHARES, DEFAULT_WINDOW
from dquant.logger import get_logger
from dquant.constants import DEFAULT_INITIAL_CASH as DEFAULT_INITIAL_CASH_CONST

logger = get_logger(__name__)


@dataclass
class Position:
    """持仓"""
    symbol: str
    shares: float
    avg_cost: float
    current_price: float = 0.0
    timestamp: Optional[datetime] = None

    @property
    def market_value(self) -> float:
        return self.shares * self.current_price

    @property
    def profit(self) -> float:
        return (self.current_price - self.avg_cost) * self.shares

    @property
    def profit_pct(self) -> float:
        if self.avg_cost == 0:
            return 0
        return (self.current_price - self.avg_cost) / self.avg_cost


@dataclass
class Portfolio:
    """
    投资组合

    管理持仓、现金、净值等。
    """
    initial_cash: float = DEFAULT_INITIAL_CASH
    cash: float = field(default=0.0)
    positions: Dict[str, Position] = field(default_factory=dict)
    nav_history: List[float] = field(default_factory=list)
    timestamp_history: List[datetime] = field(default_factory=list)

    def __post_init__(self):
        if self.cash == 0:
            self.cash = self.initial_cash

    @property
    def total_value(self) -> float:
        """总资产"""
        return self.cash + sum(p.market_value for p in self.positions.values())

    @property
    def nav(self) -> float:
        """净值"""
        return self.total_value / self.initial_cash

    def update_prices(self, prices: Dict[str, float], timestamp: datetime = None):
        """更新持仓价格"""
        # 防止同一日期重复追加 NAV
        if self.timestamp_history and self.timestamp_history[-1] == timestamp:
            return

        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].current_price = price

        # 记录净值
        self.nav_history.append(self.nav)
        self.timestamp_history.append(timestamp)

    def buy(self, symbol: str, shares: float, price: float, commission: float = 0):
        """买入"""
        # 整手买入
        shares = int(shares // MIN_SHARES) * MIN_SHARES
        if shares <= 0:
            return

        cost = shares * price * (1 + commission)

        if cost > self.cash:
            # 调整为可买入的最大整手数量
            max_shares = int(self.cash / (price * (1 + commission)) // MIN_SHARES) * MIN_SHARES
            if max_shares <= 0:
                return
            shares = max_shares
            cost = shares * price * (1 + commission)

        self.cash -= cost

        if symbol in self.positions:
            pos = self.positions[symbol]
            total_shares = pos.shares + shares
            pos.avg_cost = (pos.avg_cost * pos.shares + price * shares) / total_shares
            pos.shares = total_shares
        else:
            self.positions[symbol] = Position(
                symbol=symbol,
                shares=shares,
                avg_cost=price,
                current_price=price,
            )

    def sell(self, symbol: str, shares: float, price: float, commission: float = 0, stamp_duty: float = DEFAULT_STAMP_DUTY):
        """卖出（含印花税）"""
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]
        shares = min(shares, pos.shares)

        revenue = shares * price * (1 - commission - stamp_duty)
        self.cash += revenue
        pos.shares -= shares

        if pos.shares < MIN_SHARES:
            del self.positions[symbol]

    def rebalance(
        self,
        target_weights: Dict[str, float],
        prices: Dict[str, float],
        commission: float = 0,
    ):
        """
        再平衡到目标权重

        Args:
            target_weights: {symbol: weight} 目标权重
            prices: {symbol: price} 当前价格
            commission: 手续费率
        """
        total = self.total_value

        # 计算目标市值
        target_values = {s: total * w for s, w in target_weights.items()}

        # 先卖出
        for symbol in list(self.positions.keys()):
            if symbol not in target_weights:
                # 清仓
                pos = self.positions[symbol]
                self.sell(symbol, pos.shares, prices.get(symbol, pos.current_price), commission)

        # 卖出后重新计算总资产（现金已变化）
        total = self.total_value

        # 重新计算目标市值（基于卖出后的总资产）
        target_values = {s: total * w for s, w in target_weights.items()}

        # 再买入调整
        for symbol, target_value in target_values.items():
            if symbol not in prices:
                continue

            current_value = 0
            if symbol in self.positions:
                current_value = self.positions[symbol].market_value

            diff = target_value - current_value

            if diff > 0:
                # 需要买入
                shares = diff / prices[symbol]
                self.buy(symbol, shares, prices[symbol], commission)
            elif diff < 0:
                # 需要卖出
                shares = -diff / prices[symbol]
                self.sell(symbol, shares, prices[symbol], commission)

    def to_dataframe(self) -> pd.DataFrame:
        """转换为 DataFrame"""
        return pd.DataFrame({
            'timestamp': self.timestamp_history,
            'nav': self.nav_history,
        }).set_index('timestamp')

    def apply_corporate_action(self, action: "CorporateAction"):
        """
        应用公司行动（存根 — 仅日志记录）

        回测使用前复权数据，无需实际处理分红/送股。
        实盘模式需要数据源提供公司行动数据后实现完整逻辑。
        """
        logger.info(
            f"[CorporateAction] {action.action_type} for {action.symbol} "
            f"on {action.ex_date}: amount={action.amount}, ratio={action.ratio}"
        )


@dataclass
class CorporateAction:
    """
    公司行动事件（存根）

    用于记录分红、送股、拆股等事件。
    回测模式使用前复权数据，不需要实际处理。
    实盘模式需要数据源支持。
    """
    symbol: str
    action_type: str  # 'dividend', 'split', 'bonus_shares'
    ex_date: str
    amount: float = 0.0       # 每股分红金额 (dividend) 或拆股比例 (split)
    ratio: float = 1.0        # 送股比例 (bonus_shares: 0.1 = 每10股送1股)
    metadata: Optional[Dict] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
