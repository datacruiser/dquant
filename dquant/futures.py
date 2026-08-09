"""
期货支持基础框架

提供期货合约数据模型、保证金计算、做空机制接口。
为后续完整期货回测奠定基础。
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

from dquant.logger import get_logger

logger = get_logger(__name__)


class FuturesType(Enum):
    """期货品种类型"""

    INDEX = "index"  # 股指期货 (IF/IC/IM/IH)
    COMMODITY = "commodity"  # 商品期货
    TREASURY = "treasury"  # 国债期货 (T/TF/TS)


class MarginType(Enum):
    """保证金类型"""

    FIXED = "fixed"  # 固定比例
    TIERED = "tiered"  # 分级保证金


@dataclass
class FuturesContract:
    """
    期货合约

    定义一个期货合约的基本属性。
    """

    symbol: str  # 合约代码, e.g., "IF2401"
    underlying: str  # 标的代码, e.g., "IF" (沪深300股指)
    futures_type: FuturesType = FuturesType.INDEX
    multiplier: float = 300.0  # 合约乘数 (IF=300, IC=200, 螺纹钢=10)
    margin_rate: float = 0.12  # 保证金比例 (默认12%)
    tick_size: float = 0.2  # 最小变动价位
    list_date: str = ""  # 上市日期
    expire_date: str = ""  # 到期日期

    @property
    def is_index_futures(self) -> bool:
        return self.futures_type == FuturesType.INDEX

    def margin_required(self, price: float, direction: str = "long") -> float:
        """
        计算保证金需求

        Args:
            price: 合约价格
            direction: 方向 (long/short)

        Returns:
            保证金金额
        """
        notional = price * self.multiplier
        return notional * self.margin_rate

    def notional_value(self, price: float) -> float:
        """计算合约名义价值"""
        return price * self.multiplier

    def tick_value(self) -> float:
        """每个 tick 的价值"""
        return self.tick_size * self.multiplier


@dataclass
class FuturesPosition:
    """
    期货持仓

    支持多空双向持仓。
    """

    contract: FuturesContract
    direction: str = "long"  # long / short
    quantity: int = 0  # 手数
    entry_price: float = 0.0
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    margin_used: float = 0.0  # 按开仓价锁定的累计保证金；不在 update_price 中重算

    def update_price(self, price: float):
        """更新当前价格和未实现盈亏。

        注意：``margin_used`` 是按开仓价锁定的累计保证金，**不**会随价格变动重算，
        以避免“开仓扣减按 A 价、平仓返还按 B 价”的双计盈亏错误。
        """
        self.current_price = price
        price_diff = price - self.entry_price
        if self.direction == "short":
            price_diff = -price_diff
        self.unrealized_pnl = price_diff * self.contract.multiplier * self.quantity

    @property
    def notional(self) -> float:
        return self.contract.notional_value(self.current_price) * self.quantity


# A 股股指期货合约模板
INDEX_FUTURES = {
    "IF": FuturesContract(
        symbol="IF",
        underlying="IF",
        futures_type=FuturesType.INDEX,
        multiplier=300.0,
        margin_rate=0.12,
        tick_size=0.2,
    ),
    "IC": FuturesContract(
        symbol="IC",
        underlying="IC",
        futures_type=FuturesType.INDEX,
        multiplier=200.0,
        margin_rate=0.12,
        tick_size=0.2,
    ),
    "IM": FuturesContract(
        symbol="IM",
        underlying="IM",
        futures_type=FuturesType.INDEX,
        multiplier=200.0,
        margin_rate=0.15,
        tick_size=0.2,
    ),
    "IH": FuturesContract(
        symbol="IH",
        underlying="IH",
        futures_type=FuturesType.INDEX,
        multiplier=300.0,
        margin_rate=0.12,
        tick_size=0.2,
    ),
}

# 商品期货合约模板（部分）
COMMODITY_FUTURES = {
    "RB": FuturesContract(
        symbol="RB",
        underlying="RB",
        futures_type=FuturesType.COMMODITY,
        multiplier=10.0,
        margin_rate=0.10,
        tick_size=1.0,
    ),  # 螺纹钢
    "CU": FuturesContract(
        symbol="CU",
        underlying="CU",
        futures_type=FuturesType.COMMODITY,
        multiplier=5.0,
        margin_rate=0.10,
        tick_size=10.0,
    ),  # 沪铜
    "AU": FuturesContract(
        symbol="AU",
        underlying="AU",
        futures_type=FuturesType.COMMODITY,
        multiplier=1000.0,
        margin_rate=0.10,
        tick_size=0.02,
    ),  # 黄金
}


class FuturesAccount:
    """
    期货账户

    管理期货账户的资金、持仓、保证金。
    """

    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        contracts: Optional[Dict[str, FuturesContract]] = None,
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, FuturesPosition] = {}
        self._contracts = contracts or {**INDEX_FUTURES, **COMMODITY_FUTURES}

    @property
    def total_margin_used(self) -> float:
        return sum(p.margin_used for p in self.positions.values())

    @property
    def available_margin(self) -> float:
        return self.cash - self.total_margin_used

    @property
    def total_equity(self) -> float:
        """总权益 = 现金 + 未实现盈亏"""
        return self.cash + sum(p.unrealized_pnl for p in self.positions.values())

    @property
    def margin_usage_ratio(self) -> float:
        """保证金使用率"""
        equity = self.total_equity
        return self.total_margin_used / equity if equity > 0 else 0

    def open_position(
        self,
        symbol: str,
        direction: str,
        quantity: int,
        price: float,
    ) -> bool:
        """
        开仓

        Args:
            symbol: 合约代码 (e.g., "IF")
            direction: 方向 ("long" / "short")
            quantity: 手数
            price: 开仓价格

        Returns:
            是否成功
        """
        contract = self._contracts.get(symbol)
        if contract is None:
            logger.warning(f"[FuturesAccount] Unknown contract: {symbol}")
            return False

        margin = contract.margin_required(price) * quantity
        if margin > self.available_margin:
            logger.warning(
                f"[FuturesAccount] Insufficient margin: need {margin:.0f}, "
                f"available {self.available_margin:.0f}"
            )
            return False

        # 如果已有同方向持仓，合并：保证金按加权累计锁定（始终基于开仓价）
        key = f"{symbol}_{direction}"
        if key in self.positions:
            pos = self.positions[key]
            total_qty = pos.quantity + quantity
            pos.entry_price = (pos.entry_price * pos.quantity + price * quantity) / total_qty
            # 累加新开仓部分按其开仓价锁定的保证金
            pos.margin_used += margin
            pos.quantity = total_qty
            pos.update_price(price)
        else:
            pos = FuturesPosition(
                contract=contract,
                direction=direction,
                quantity=quantity,
                entry_price=price,
            )
            pos.margin_used = margin  # 锁定开仓时的保证金
            pos.update_price(price)
            self.positions[key] = pos

        self.cash -= margin
        logger.info(f"[FuturesAccount] Opened {direction} {symbol} x{quantity} @ {price:.2f}")
        return True

    def close_position(
        self,
        symbol: str,
        direction: str,
        quantity: Optional[int] = None,
        price: float = 0.0,
    ) -> Optional[float]:
        """
        平仓

        Args:
            symbol: 合约代码
            direction: 方向
            quantity: 平仓手数 (None=全部)
            price: 平仓价格

        Returns:
            平仓盈亏 (None=失败)
        """
        key = f"{symbol}_{direction}"
        pos = self.positions.get(key)
        if pos is None:
            return None

        close_qty = quantity or pos.quantity
        close_qty = min(close_qty, pos.quantity)

        price_diff = price - pos.entry_price
        if direction == "short":
            price_diff = -price_diff
        realized_pnl = price_diff * pos.contract.multiplier * close_qty

        # 返还保证金：按“剩余累计锁定保证金 / 当前持仓手数 * 平仓手数”算，
        # 保证与开仓时扣减的口径完全一致，避免双计盈亏。
        if pos.quantity > 0:
            margin_return = pos.margin_used * (close_qty / pos.quantity)
        else:
            margin_return = 0.0
        pos.margin_used -= margin_return
        self.cash += margin_return + realized_pnl

        pos.quantity -= close_qty
        if pos.quantity <= 0:
            del self.positions[key]
        else:
            pos.update_price(price)

        logger.info(
            f"[FuturesAccount] Closed {direction} {symbol} x{close_qty} @ {price:.2f}, "
            f"PnL={realized_pnl:.0f}"
        )
        return realized_pnl

    def mark_to_market(self, prices: Dict[str, float]):
        """
        逐日盯市 (Mark to Market)

        更新所有持仓的当前价格和未实现盈亏。

        Args:
            prices: {symbol: current_price}
        """
        for key, pos in self.positions.items():
            symbol = key.split("_")[0]
            if symbol in prices:
                pos.update_price(prices[symbol])

    def check_margin_call(self, maintenance_ratio: float = 0.15) -> bool:
        """
        检查是否触发追保

        追保条件: 总权益 < 维持保证金 (= 总持仓保证金 × 维持保证金比例)
        即当亏损导致权益不足以覆盖维持保证金时触发。

        Args:
            maintenance_ratio: 维持保证金占总保证金的比例

        Returns:
            是否触发追保
        """
        required_equity = self.total_margin_used * maintenance_ratio
        return self.total_equity < required_equity
