"""
事件驱动回测引擎

支持逐笔成交、滑点、市场冲击等高级功能。
"""

from dquant.logger import get_logger

logger = get_logger(__name__)

from typing import Optional, List, Dict, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from collections import defaultdict
import pandas as pd
import numpy as np
from dquant.constants import DEFAULT_COMMISSION, DEFAULT_SLIPPAGE, DEFAULT_STAMP_DUTY, DEFAULT_INITIAL_CASH, MIN_SHARES, DEFAULT_WINDOW


class EventType(Enum):
    """事件类型"""
    MARKET = "market"          # 市场数据
    SIGNAL = "signal"          # 交易信号
    ORDER = "order"            # 订单
    FILL = "fill"              # 成交
    POSITION = "position"      # 持仓更新
    RISK = "risk"              # 风险事件


@dataclass
class Event:
    """事件基类"""
    type: EventType
    timestamp: datetime


class MarketEvent(Event):
    """市场事件"""
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int

    def __init__(self, timestamp, symbol, open, high, low, close, volume):
        super().__init__(EventType.MARKET, timestamp)
        self.symbol = symbol
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume


class SignalEvent(Event):
    """信号事件"""
    symbol: str
    signal_type: str  # 'LONG', 'SHORT', 'EXIT'
    strength: float = 1.0

    def __init__(self, timestamp, symbol, signal_type, strength=1.0):
        super().__init__(EventType.SIGNAL, timestamp)
        self.symbol = symbol
        self.signal_type = signal_type
        self.strength = strength


class OrderEvent(Event):
    """订单事件"""
    symbol: str
    order_type: str  # 'MARKET', 'LIMIT'
    side: str        # 'BUY', 'SELL'
    quantity: int
    price: Optional[float] = None  # 限价单价格
    order_id: str = ""

    def __init__(self, timestamp, symbol, order_type, side, quantity, price=None):
        super().__init__(EventType.ORDER, timestamp)
        self.symbol = symbol
        self.order_type = order_type
        self.side = side
        self.quantity = quantity
        self.price = price
        self.order_id = f"{symbol}_{timestamp.timestamp()}"


class FillEvent(Event):
    """成交事件"""
    symbol: str
    side: str
    quantity: int
    fill_price: float
    commission: float
    slippage: float
    order_id: str

    def __init__(self, timestamp, symbol, side, quantity, fill_price, commission, slippage, order_id):
        super().__init__(EventType.FILL, timestamp)
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.fill_price = fill_price
        self.commission = commission
        self.slippage = slippage
        self.order_id = order_id


class SlippageModel:
    """
    滑点模型

    计算实际成交价与理论价格的差异。
    """

    @staticmethod
    def fixed_slippage(price: float, slippage_pct: float = DEFAULT_STAMP_DUTY) -> float:
        """固定滑点"""
        return price * slippage_pct

    @staticmethod
    def volume_based_slippage(
        price: float,
        volume: int,
        avg_volume: int,
        base_slippage: float = DEFAULT_STAMP_DUTY,
    ) -> float:
        """
        基于成交量的滑点

        成交量越大，滑点越大。
        """
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1
        slippage = base_slippage * (1 + np.log1p(volume_ratio))
        return price * slippage

    @staticmethod
    def market_impact_slippage(
        price: float,
        order_size: int,
        daily_volume: int,
        impact_coefficient: float = 0.1,
    ) -> float:
        """
        市场冲击滑点

        基于订单占日成交量的比例计算滑点。
        """
        if daily_volume == 0:
            return price * DEFAULT_STAMP_DUTY

        participation_rate = order_size / daily_volume
        impact = impact_coefficient * np.sqrt(participation_rate)

        return price * impact

    @staticmethod
    def volatility_slippage(
        price: float,
        volatility: float,
        multiplier: float = 1.0,
    ) -> float:
        """
        波动率滑点

        波动率越大，滑点越大。
        """
        return price * volatility * multiplier


class ExecutionHandler:
    """
    执行处理器

    处理订单执行、滑点计算等。
    """

    def __init__(
        self,
        slippage_model: str = 'fixed',
        slippage_pct: float = DEFAULT_STAMP_DUTY,
        commission_rate: float = DEFAULT_COMMISSION,
    ):
        self.slippage_model = slippage_model
        self.slippage_pct = slippage_pct
        self.commission_rate = commission_rate

    def execute_order(
        self,
        order: OrderEvent,
        market_data: Optional[Dict] = None,
    ) -> FillEvent:
        """
        执行订单

        Args:
            order: 订单事件
            market_data: 市场数据 (包含 price, volume 等)

        Returns:
            成交事件
        """
        # 确定基准价格
        if order.order_type == 'MARKET':
            # 市价单：使用当前价格
            base_price = market_data.get('close', 0) if market_data else 0
        else:
            # 限价单：使用限价
            base_price = order.price or 0

        # 计算滑点
        if self.slippage_model == 'fixed':
            slippage = SlippageModel.fixed_slippage(base_price, self.slippage_pct)
        elif self.slippage_model == 'volume':
            volume = market_data.get('volume', 0) if market_data else 0
            avg_volume = market_data.get('avg_volume', volume) if market_data else volume
            slippage = SlippageModel.volume_based_slippage(
                base_price, volume, avg_volume, self.slippage_pct
            )
        elif self.slippage_model == 'impact':
            daily_volume = market_data.get('volume', 0) if market_data else 0
            slippage = SlippageModel.market_impact_slippage(
                base_price, order.quantity, daily_volume
            )
        else:
            slippage = base_price * self.slippage_pct

        # 计算成交价
        if order.side == 'BUY':
            # 买入：向上滑动
            fill_price = base_price + slippage
        else:
            # 卖出：向下滑动
            fill_price = base_price - slippage

        # 计算佣金
        commission = fill_price * order.quantity * self.commission_rate

        return FillEvent(
            timestamp=order.timestamp,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            fill_price=fill_price,
            commission=commission,
            slippage=slippage,
            order_id=order.order_id,
        )


class EventDrivenBacktest:
    """
    事件驱动回测引擎

    Usage:
        engine = EventDrivenBacktest(initial_cash=1000000)
        engine.add_data(data)
        engine.add_strategy(strategy)
        engine.run()
    """

    def __init__(
        self,
        initial_cash: float = 1000000,
        slippage_model: str = 'fixed',
        slippage_pct: float = DEFAULT_STAMP_DUTY,
        commission_rate: float = DEFAULT_COMMISSION,
    ):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions = defaultdict(int)
        self.trades = []

        # 事件队列
        self.events = []
        self.continue_backtest = True

        # 组件
        self.execution_handler = ExecutionHandler(
            slippage_model=slippage_model,
            slippage_pct=slippage_pct,
            commission_rate=commission_rate,
        )

        # 回调
        self.market_handlers = []
        self.signal_handlers = []
        self.fill_handlers = []

        # 数据
        self.data = None
        self.current_bar = None

    def add_data(self, data: pd.DataFrame):
        """添加数据"""
        self.data = data

    def add_strategy(self, strategy: Any):
        """添加策略"""
        if hasattr(strategy, 'on_market'):
            self.market_handlers.append(strategy.on_market)

        if hasattr(strategy, 'on_fill'):
            self.fill_handlers.append(strategy.on_fill)

    def on_market(self, handler: Callable):
        """注册市场事件处理器"""
        self.market_handlers.append(handler)

    def on_fill(self, handler: Callable):
        """注册成交事件处理器"""
        self.fill_handlers.append(handler)

    def _generate_market_events(self):
        """生成市场事件"""
        if self.data is None:
            return

        for idx, row in self.data.iterrows():
            event = MarketEvent(
                timestamp=idx,
                symbol=row.get('symbol', ''),
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume'],
            )
            yield event

    def _process_event(self, event: Event):
        """处理事件"""
        if event.type == EventType.MARKET:
            self._on_market(event)
        elif event.type == EventType.SIGNAL:
            self._on_signal(event)
        elif event.type == EventType.ORDER:
            self._on_order(event)
        elif event.type == EventType.FILL:
            self._on_fill(event)

    def _on_market(self, event: MarketEvent):
        """处理市场事件"""
        self.current_bar = event

        # 调用策略
        for handler in self.market_handlers:
            handler(event)

    def _on_signal(self, event: SignalEvent):
        """处理信号事件"""
        # 生成订单
        if event.signal_type == 'LONG':
            # 买入
            order = OrderEvent(
                timestamp=event.timestamp,
                symbol=event.symbol,
                order_type='MARKET',
                side='BUY',
                quantity=MIN_SHARES,  # 简化：固定数量
            )
            self.events.append(order)

        elif event.signal_type == 'EXIT':
            # 平仓
            if self.positions[event.symbol] > 0:
                order = OrderEvent(
                    timestamp=event.timestamp,
                    symbol=event.symbol,
                    order_type='MARKET',
                    side='SELL',
                    quantity=self.positions[event.symbol],
                )
                self.events.append(order)

    def _on_order(self, event: OrderEvent):
        """处理订单事件"""
        # 执行订单
        market_data = {
            'close': self.current_bar.close if self.current_bar else 0,
            'volume': self.current_bar.volume if self.current_bar else 0,
        }

        fill = self.execution_handler.execute_order(event, market_data)
        self.events.append(fill)

    def _on_fill(self, event: FillEvent):
        """处理成交事件"""
        # 更新持仓和资金
        if event.side == 'BUY':
            self.positions[event.symbol] += event.quantity
            self.cash -= event.fill_price * event.quantity + event.commission
        else:
            self.positions[event.symbol] -= event.quantity
            self.cash += event.fill_price * event.quantity - event.commission

        # 记录交易
        self.trades.append(event)

        # 调用回调
        for handler in self.fill_handlers:
            handler(event)

    def run(self):
        """运行回测"""
        logger.info("开始事件驱动回测...")

        # 生成市场事件
        for market_event in self._generate_market_events():
            # 添加市场事件到队列
            self.events.append(market_event)

            # 处理所有事件
            while self.events:
                event = self.events.pop(0)
                self._process_event(event)

        # 计算绩效
        total_value = self.cash

        # 计算持仓价值
        if self.current_bar:
            for symbol, quantity in self.positions.items():
                total_value += quantity * self.current_bar.close

        total_return = (total_value - self.initial_cash) / self.initial_cash

        logger.info("\n回测完成!")
        logger.info(f"  总收益率: {total_return:.2%}")
        logger.info(f"  交易次数: {len(self.trades)}")
        logger.info(f"  最终资金: ¥{self.cash:,.0f}")
        logger.info(f"  总价值:   ¥{total_value:,.0f}")

        return {
            'total_return': total_return,
            'trades': len(self.trades),
            'final_cash': self.cash,
            'total_value': total_value,
        }
