"""
XTP 券商接口 (中泰证券极速交易接口)

注意: 需要申请 XTP 接口权限
"""

import os
import threading
from copy import deepcopy
from datetime import datetime
from typing import Dict, Optional

from dquant.broker.base import BaseBroker, Order, OrderResult
from dquant.broker.safety import TradingSafety, log_error
from dquant.broker.simulator import Simulator
from dquant.config import XTPBrokerConfig
from dquant.constants import DEFAULT_INITIAL_CASH
from dquant.logger import get_logger

logger = get_logger(__name__)


class XTPBroker(BaseBroker):
    """
    XTP 券商接口

    中泰证券 XTP 极速交易接口封装。

    使用前需要:
    1. 向中泰证券申请 XTP 接口权限
    2. 下载 XTP API (Python 版本)
    3. 配置账户信息（建议通过环境变量传递密码）

    Usage:
        broker = XTPBroker(
            server='120.27.164.138',
            port=6001,
            account='your_account',
            password_env='XTP_PASSWORD',  # 推荐方式
        )

        broker.connect()

        # 查询
        account = broker.get_account()
        positions = broker.get_positions()

        # 下单
        order = Order(symbol='000001.SZ', side='BUY', quantity=MIN_SHARES, price=10.0)
        result = broker.place_order(order)
    """

    def __init__(
        self,
        config: Optional["XTPBrokerConfig"] = None,
        /,
        # Backward-compatible individual params
        server: str = "",  # XTP 服务器地址，需通过配置或环境变量提供
        port: int = 6001,
        account: str = "",
        password: str = "",
        password_env: str = "",  # 环境变量名，优先使用
        client_id: int = 1,
        timeout: int = 30,
        **kwargs,
    ):
        super().__init__(name="XTP")

        if config is not None:
            server = config.server
            port = config.port
            account = config.account
            password = config.password
            password_env = config.password_env
            client_id = config.client_id
            timeout = config.timeout

        self.server = server
        self.port = port
        self.account = account
        # 优先从环境变量读取密码，避免明文存储
        self._password = os.getenv(password_env, password) if password_env else password
        self.client_id = client_id
        self.timeout = timeout

        self._api = None
        self._connected = False
        self._orders: Dict[str, dict] = {}
        self._lock = threading.Lock()

        # 交易安全控制
        self.safety = TradingSafety(
            enable_time_check=kwargs.get("enable_time_check", True),
            enable_fund_check=kwargs.get("enable_fund_check", True),
            enable_order_validation=kwargs.get("enable_order_validation", True),
            enable_position_check=kwargs.get("enable_position_check", True),
        )

    def __repr__(self):
        return f"XTPBroker(server={self.server}, account={self.account}, connected={self._connected})"

    def connect(self, **kwargs) -> bool:
        """连接 XTP 服务器"""
        try:
            # 尝试导入 XTP API
            try:
                from xtp import XTPAPI
            except ImportError:
                logger.error("[XTP] XTP API not installed")
                logger.error("[XTP] 请联系中泰证券获取 XTP Python SDK")
                return False

            # 创建 API 实例
            self._api = XTPAPI()

            # 设置回调
            self._setup_callbacks()

            # 登录
            result = self._api.login(
                self.server,
                self.port,
                self.account,
                self._password,
                self.client_id,
            )

            if result:
                self._connected = True
                logger.info(f"[XTP] Connected to {self.server}:{self.port}")
                return True
            else:
                logger.error("[XTP] Login failed")
                return False

        except Exception as e:
            logger.error(f"[XTP] Connect error: {e}")
            return False

    def _setup_callbacks(self):
        """设置回调函数"""
        if self._api is None:
            return

        # 注册回调 (实际 API 调用可能不同)
        try:
            # XTP API 回调注册
            if hasattr(self._api, "register_callback"):
                self._api.register_callback(
                    on_order=self._handle_order_event,
                    on_trade=self._handle_trade_event,
                    on_quote=self._handle_quote_event,
                )
            else:
                logger.warning("[XTP] Callback registration not supported by API")

        except Exception as e:
            logger.error(f"[XTP] Failed to setup callbacks: {e}")
            logger.warning("[XTP] Continuing without callbacks...")

    def _handle_order_event(self, event, error):
        """订单事件回调"""
        try:
            if error:
                logger.error(f"[XTP] Order error: {error}")
                return

            order_id = str(event.order_xt_id)
            status = self._map_order_status(event.order_status)
            logger.info(f"[XTP] Order {order_id} status: {status}")

            filled_qty = self._extract_filled_quantity(event)

            with self._lock:
                cached = self._orders.setdefault(order_id, {"filled": 0})
                if filled_qty is not None:
                    total_quantity = cached.get("quantity")
                    if total_quantity:
                        filled_qty = min(filled_qty, total_quantity)
                    cached["filled"] = filled_qty
                cached["status"] = status

        except Exception as e:
            logger.error(f"[XTP] Order callback error: {e}")

    def _handle_trade_event(self, event, error):
        """成交事件回调"""
        try:
            if error:
                logger.error(f"[XTP] Trade error: {error}")
                return

            order_id = str(event.order_xt_id)
            symbol = event.stock_code
            quantity = int(getattr(event, "quantity", getattr(event, "traded_quantity", 0)))
            price = float(getattr(event, "price", getattr(event, "traded_price", 0.0)))

            logger.info(f"[XTP] Trade: {symbol} x {quantity} @ {price}")

            with self._lock:
                cached = self._orders.setdefault(
                    order_id,
                    {
                        "symbol": symbol,
                        "filled": 0,
                        "filled_price": 0.0,
                        "status": "PENDING",
                    },
                )
                total_quantity = cached.get("quantity")
                prev_filled = cached.get("filled", 0)
                prev_price = cached.get("filled_price", 0.0)

                new_filled = prev_filled + quantity
                if total_quantity:
                    new_filled = min(new_filled, total_quantity)

                # 计算加权成交均价 (VWAP)
                effective_quantity = new_filled - prev_filled
                if new_filled > 0 and effective_quantity > 0:
                    total_cost = (prev_filled * prev_price) + (effective_quantity * price)
                    new_vwap = total_cost / new_filled
                else:
                    new_vwap = prev_price

                cached["symbol"] = symbol
                cached["filled"] = new_filled
                cached["filled_price"] = new_vwap

                if total_quantity and new_filled >= total_quantity:
                    cached["status"] = "FILLED"
                elif new_filled > 0 and cached.get("status") == "PENDING":
                    cached["status"] = "PARTIAL_FILLED"

        except Exception as e:
            logger.error(f"[XTP] Trade callback error: {e}")

    def _handle_quote_event(self, quote):
        """行情推送回调"""
        try:
            _ = quote.stock_code
            _ = quote.last_price
        except Exception as e:
            logger.error(f"[XTP] Quote callback error: {e}")

    def _get_cached_order(self, order_id: str) -> Optional[dict]:
        """获取本地缓存的订单快照。"""
        with self._lock:
            cached = self._orders.get(order_id)
            return deepcopy(cached) if cached is not None else None

    def _extract_filled_quantity(self, event) -> Optional[int]:
        """从订单/成交事件中提取累计成交量。"""
        # 收紧到 XTP 官方 SDK 常见的累计成交量字段
        for attr in (
            "qty_traded",  # XTPOrderInfo 官方字段
            "traded_volume",  # 常见 Python 封装衍生字段
        ):
            value = getattr(event, attr, None)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return None
        return None

    def _resolve_order_side(self, side_value, cached: dict) -> str:
        """解析订单方向；未知值优先沿用本地缓存，避免误判成 SELL。"""
        if side_value == 1:
            return "BUY"
        if side_value == 2:
            return "SELL"

        cached_side = cached.get("side")
        if cached_side in {"BUY", "SELL"}:
            return cached_side
        return "UNKNOWN"

    def _build_order_from_cache(self, order_id: str, cached: Optional[dict]) -> Optional[Order]:
        """将本地缓存转换为 Order。"""
        if cached is None:
            return None

        return Order(
            symbol=cached.get("symbol", ""),
            side=cached.get("side", "BUY"),
            quantity=cached.get("quantity", 0),
            price=cached.get("price"),
            order_id=order_id,
            filled_quantity=cached.get("filled", 0),
            filled_price=cached.get("filled_price", 0.0),
            status=cached.get("status", "PENDING"),
        )

    def _refresh_cached_order_from_api(self, order_id: str) -> Optional[Order]:
        """通过 API 刷新本地订单缓存。"""
        if self._api is None:
            return None

        orders = self._api.query_orders()
        for order_info in orders:
            api_order_id = getattr(
                order_info,
                "order_id",
                getattr(order_info, "order_xt_id", None),
            )
            if str(api_order_id) != order_id:
                continue

            filled_qty = getattr(order_info, "traded_volume", 0)
            status_value = getattr(
                order_info,
                "status",
                getattr(order_info, "order_status", None),
            )
            status = (
                self._map_order_status(status_value)
                if isinstance(status_value, int)
                else status_value or "UNKNOWN"
            )

            with self._lock:
                cached = self._orders.setdefault(order_id, {"filled": 0})
                side_value = getattr(order_info, "side", getattr(order_info, "order_side", 0))
                side = self._resolve_order_side(side_value, cached)
                cached.update(
                    {
                        "symbol": getattr(
                            order_info,
                            "ticker",
                            getattr(order_info, "stock_code", cached.get("symbol", "")),
                        ),
                        "side": side,
                        "quantity": getattr(
                            order_info,
                            "quantity",
                            getattr(order_info, "order_volume", cached.get("quantity", 0)),
                        ),
                        "price": getattr(order_info, "price", cached.get("price")),
                        "filled": filled_qty,
                        "status": status,
                    }
                )
                refreshed = deepcopy(cached)

            return self._build_order_from_cache(order_id, refreshed)

        return None

    def disconnect(self) -> bool:
        """断开连接"""
        if self._api and self._connected:
            try:
                self._api.logout()
                self._connected = False
                logger.info("[XTP] Disconnected")
                return True
            except Exception as e:
                logger.error(f"[XTP] Disconnect error: {e}")
                return False
        return True

    def get_account(self) -> dict:
        """获取账户信息"""
        if not self._connected:
            return {}

        try:
            # 查询资金
            asset = self._api.query_asset()
            return {
                "cash": asset.total_asset - asset.market_value,
                "total_value": asset.total_asset,
                "market_value": asset.market_value,
                "available": asset.buying_power,
            }
        except Exception as e:
            logger.error(f"[XTP] Query account error: {e}")
            return {}

    def get_positions(self) -> Dict[str, dict]:
        """获取持仓"""
        if not self._connected:
            return {}

        try:
            positions = self._api.query_positions()
            result = {}

            for pos in positions:
                result[pos.ticker] = {
                    "symbol": pos.ticker,
                    "quantity": pos.total_qty,
                    "available": pos.sellable_qty,
                    "avg_cost": pos.avg_price,
                    "current_price": pos.market_price,
                    "profit": pos.unrealized_pnl,
                }

            return result
        except Exception as e:
            logger.error(f"[XTP] Query positions error: {e}")
            return {}

    def place_order(self, order: Order) -> OrderResult:
        """下单"""
        if not self._connected:
            return OrderResult(
                order_id="",
                symbol=order.symbol,
                side=order.side,
                filled_quantity=0,
                filled_price=0,
                commission=0,
                timestamp=datetime.now(),
                status="REJECTED",
            )

        try:
            # 安全检查
            account_info = self.get_account()
            positions = self.get_positions()
            estimated_price = None

            if order.order_type == "MARKET" and order.price is None:
                md = self.get_market_data(order.symbol)
                if md and "price" in md:
                    estimated_price = md["price"]

            valid, msg = self.safety.check_order(
                order,
                available_cash=account_info.get("cash", 0),
                positions=positions,
                estimated_price=estimated_price,
            )

            if not valid:
                log_error(
                    "PLACE_ORDER",
                    Exception(msg),
                    {
                        "symbol": order.symbol,
                        "side": order.side,
                        "quantity": order.quantity,
                    },
                )
                return OrderResult(
                    order_id="",
                    symbol=order.symbol,
                    side=order.side,
                    filled_quantity=0,
                    filled_price=0,
                    commission=0,
                    timestamp=datetime.now(),
                    status="REJECTED",
                )

            # 转换订单类型
            # XTP: 23=市价单, 24=限价单
            price_type = 23 if order.order_type == "MARKET" else 24

            # 转换买卖方向: 'BUY'/'SELL' -> 1/2
            side_map = {"BUY": 1, "SELL": 2}
            xtp_side = side_map.get(order.side.upper(), 0)
            if xtp_side == 0:
                return OrderResult(
                    order_id="",
                    symbol=order.symbol,
                    side=order.side,
                    filled_quantity=0,
                    filled_price=0,
                    commission=0,
                    timestamp=datetime.now(),
                    status="REJECTED",
                )

            # 下单
            api_result = self._api.place_order(
                ticker=order.symbol,
                side=xtp_side,
                quantity=order.quantity,
                price=order.price or 0,
                price_type=price_type,
            )

            order.order_id = str(api_result.order_id)
            order.status = "PENDING"
            with self._lock:
                self._orders[order.order_id] = {
                    "symbol": order.symbol,
                    "side": order.side,
                    "quantity": order.quantity,
                    "price": order.price,
                    "filled": 0,
                    "filled_price": 0,
                    "status": "PENDING",
                }

            return OrderResult(
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                filled_quantity=0,  # 待成交回报更新
                filled_price=0,
                commission=0,
                timestamp=datetime.now(),
                status="PENDING",
            )

        except Exception as e:
            logger.error(f"[XTP] Place order error: {e}")
            return OrderResult(
                order_id="",
                symbol=order.symbol,
                side=order.side,
                filled_quantity=0,
                filled_price=0,
                commission=0,
                timestamp=datetime.now(),
                status="REJECTED",
            )

    def cancel_order(self, order_id: str) -> bool:
        """撤单"""
        if not self._connected:
            return False

        try:
            self._api.cancel_order(int(order_id))
            with self._lock:
                cached = self._orders.setdefault(order_id, {"filled": 0})
                if cached.get("status") != "FILLED":
                    cached["status"] = "CANCELLED"
            return True
        except Exception as e:
            logger.error(f"[XTP] Cancel order error: {e}")
            return False

    def get_order_status(self, order_id: str) -> Optional[Order]:
        """查询订单状态"""
        if not self._connected:
            return None

        cached_order = self._build_order_from_cache(order_id, self._get_cached_order(order_id))

        try:
            refreshed_order = self._refresh_cached_order_from_api(order_id)
            return refreshed_order or cached_order
        except Exception as e:
            logger.error(f"[XTP] Query order error: {e}")
            return cached_order

    def get_market_data(self, symbol: str) -> dict:
        """获取实时行情"""
        if not self._connected:
            return {}

        try:
            quote = self._api.query_quote(symbol)
            return {
                "symbol": symbol,
                "price": quote.last_price,
                "bid": quote.bid_price,
                "ask": quote.ask_price,
                "volume": quote.volume,
                "timestamp": datetime.now(),
            }
        except Exception as e:
            logger.error(f"[XTP] Query market data error: {e}")
            return {}

    def _map_order_status(self, xtp_status: int) -> str:
        """映射订单状态"""
        status_map = {
            0: "PENDING",
            1: "PARTIAL_FILLED",
            2: "FILLED",
            3: "CANCELLED",
            4: "REJECTED",
        }
        return status_map.get(xtp_status, "UNKNOWN")


class XTPSimulator(XTPBroker):
    """
    XTP 模拟交易

    在没有 XTP 接口权限时使用模拟交易。
    通过组合 Simulator 实现，避免代码重复。
    """

    def __init__(self, initial_cash: float = DEFAULT_INITIAL_CASH, **kwargs):
        # 忽略连接参数，不调用 XTPBroker.__init__
        super().__init__(**kwargs)
        self.name = "XTPSimulator"
        # 组合：内部委托给 Simulator
        self._sim = Simulator(
            initial_cash=initial_cash,
            order_id_prefix="XTPSIM",
            apply_slippage=False,
            validate_orders=False,
            adjust_lots=False,
            strict_sell=True,
        )

    # ---------- 连接管理 ----------
    def connect(self, **kwargs) -> bool:
        logger.info(f"[{self.name}] Connected (simulated)")
        self._connected = True
        self._sim.connect()
        return True

    def disconnect(self) -> bool:
        self._connected = False
        return True

    # ---------- 账户 & 持仓 ----------
    def get_account(self) -> dict:
        total_value = self._sim.cash + sum(
            p["quantity"] * p.get("price", 0) for p in self._sim.positions.values()
        )
        return {
            "cash": self._sim.cash,
            "total_value": total_value,
            "market_value": total_value - self._sim.cash,
            "available": self._sim.cash,
        }

    def get_positions(self) -> Dict[str, dict]:
        return deepcopy(self._sim.positions)

    # ---------- 交易 ----------
    def place_order(self, order: Order) -> OrderResult:
        if not self._connected:
            return OrderResult(
                order_id="",
                symbol=order.symbol,
                side=order.side,
                filled_quantity=0,
                filled_price=0,
                commission=0,
                timestamp=datetime.now(),
                status="REJECTED",
            )
        return self._sim.place_order(order)

    def cancel_order(self, order_id: str) -> bool:
        return self._sim.cancel_order(order_id)

    def get_order_status(self, order_id: str) -> Optional[Order]:
        return self._sim.get_order_status(order_id)

    # ---------- 行情 ----------
    def get_market_data(self, symbol: str) -> dict:
        return self._sim.get_market_data(symbol)

    # ---------- 属性代理（保持兼容） ----------
    @property
    def initial_cash(self) -> float:
        return self._sim.initial_cash

    @property
    def cash(self) -> float:
        return self._sim.cash

    @cash.setter
    def cash(self, value: float):
        self._sim.cash = value

    @property
    def positions(self) -> Dict[str, dict]:
        return self._sim.positions

    @positions.setter
    def positions(self, value: Dict[str, dict]):
        self._sim.positions = value

    @property
    def orders(self) -> Dict[str, Order]:
        return self._sim.orders

    @orders.setter
    def orders(self, value: Dict[str, Order]):
        self._sim.orders = value
