"""
XTP 券商接口 (中泰证券极速交易接口)

注意: 需要申请 XTP 接口权限
"""

from typing import Dict, Optional, List
from datetime import datetime
import queue

from dquant.broker.base import BaseBroker, Order, OrderResult
from dquant.constants import DEFAULT_COMMISSION, DEFAULT_SLIPPAGE, DEFAULT_STAMP_DUTY, DEFAULT_INITIAL_CASH, MIN_SHARES, DEFAULT_WINDOW
from dquant.broker.safety import TradingSafety, log_trade, log_error


class XTPBroker(BaseBroker):
    """
    XTP 券商接口

    中泰证券 XTP 极速交易接口封装。

    使用前需要:
    1. 向中泰证券申请 XTP 接口权限
    2. 下载 XTP API (Python 版本)
    3. 配置账户信息

    Usage:
        broker = XTPBroker(
            server='120.27.164.138',
            port=6001,
            account='your_account',
            password='your_password',
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
        server: str = '120.27.164.138',  # XTP 服务器地址
        port: int = 6001,
        account: str = '',
        password: str = '',
        client_id: int = 1,
        timeout: int = 30,
        **kwargs
    ):
        super().__init__(name="XTP")
        self.server = server
        self.port = port
        self.account = account
        self.password = password
        self.client_id = client_id
        self.timeout = timeout

        self._api = None
        self._connected = False
        self._callback_queue = queue.Queue()

        # 交易安全控制
        self.safety = TradingSafety(
            enable_time_check=kwargs.get('enable_time_check', True),
            enable_fund_check=kwargs.get('enable_fund_check', True),
            enable_order_validation=kwargs.get('enable_order_validation', True),
            enable_position_check=kwargs.get('enable_position_check', True),
        )

    def connect(self, **kwargs) -> bool:
        """连接 XTP 服务器"""
        try:
            # 尝试导入 XTP API
            try:
                from xtp import XTPAPI, XTPProtocol
            except ImportError:
                print("[XTP] XTP API not installed")
                print("[XTP] 请联系中泰证券获取 XTP Python SDK")
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
                self.password,
                self.client_id,
            )

            if result:
                self._connected = True
                print(f"[XTP] Connected to {self.server}:{self.port}")
                return True
            else:
                print(f"[XTP] Login failed")
                return False

        except Exception as e:
            print(f"[XTP] Connect error: {e}")
            return False

    def _setup_callbacks(self):
        """设置回调函数"""
        if self._api is None:
            return

        # 定义回调处理函数
        def on_order_event(event, error):
            """订单事件回调"""
            try:
                if error:
                    print(f"[XTP] Order error: {error}")
                else:
                    order_id = event.order_xt_id
                    status = event.order_status
                    print(f"[XTP] Order {order_id} status: {status}")

                    # 更新订单状态
                    if order_id in self._orders:
                        self._orders[order_id]['status'] = status

            except Exception as e:
                print(f"[XTP] Order callback error: {e}")

        def on_trade_event(event, error):
            """成交事件回调"""
            try:
                if error:
                    print(f"[XTP] Trade error: {error}")
                else:
                    order_id = event.order_xt_id
                    symbol = event.stock_code
                    quantity = event.traded_quantity
                    price = event.traded_price

                    print(f"[XTP] Trade: {symbol} x {quantity} @ {price}")

                    # 记录成交
                    if order_id in self._orders:
                        self._orders[order_id]['filled'] = quantity

            except Exception as e:
                print(f"[XTP] Trade callback error: {e}")

        def on_quote_event(quote):
            """行情推送回调"""
            try:
                symbol = quote.stock_code
                price = quote.last_price
                # 可以在这里更新实时价格
                # self._prices[symbol] = price
            except Exception as e:
                print(f"[XTP] Quote callback error: {e}")

        # 注册回调 (实际 API 调用可能不同)
        try:
            # XTP API 回调注册
            # 注意: 实际的 XTP API 可能使用不同的方法名
            if hasattr(self._api, 'register_callback'):
                self._api.register_callback(
                    on_order=on_order_event,
                    on_trade=on_trade_event,
                    on_quote=on_quote_event,
                )
            else:
                # 如果 API 不支持 register_callback，使用其他方式
                print("[XTP] Callback registration not supported by API")

        except Exception as e:
            print(f"[XTP] Failed to setup callbacks: {e}")
            print("[XTP] Continuing without callbacks...")

    def disconnect(self) -> bool:
        """断开连接"""
        if self._api and self._connected:
            try:
                self._api.logout()
                self._connected = False
                print("[XTP] Disconnected")
                return True
            except Exception as e:
                print(f"[XTP] Disconnect error: {e}")
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
                'cash': asset.total_asset - asset.market_value,
                'total_value': asset.total_asset,
                'market_value': asset.market_value,
                'available': asset.buying_power,
            }
        except Exception as e:
            print(f"[XTP] Query account error: {e}")
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
                    'symbol': pos.ticker,
                    'quantity': pos.total_qty,
                    'available': pos.sellable_qty,
                    'avg_cost': pos.avg_price,
                    'current_price': pos.market_price,
                    'profit': pos.unrealized_pnl,
                }

            return result
        except Exception as e:
            print(f"[XTP] Query positions error: {e}")
            return {}

    def place_order(self, order: Order) -> OrderResult:
        """下单"""
        if not self._connected:
            return OrderResult(
                order_id='',
                symbol=order.symbol,
                side=order.side,
                filled_quantity=0,
                filled_price=0,
                commission=0,
                timestamp=datetime.now(),
                status='REJECTED',
            )

        try:
            # 转换订单类型
            # XTP: 23=市价单, 24=限价单
            price_type = 23 if order.order_type == 'MARKET' else 24

            # 下单
            result = self._api.place_order(
                ticker=order.symbol,
                side=order.side,  # 1=买, 2=卖
                quantity=order.quantity,
                price=order.price or 0,
                price_type=price_type,
            )

            order.order_id = str(result.order_id)
            order.status = 'PENDING'

            return OrderResult(
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                filled_quantity=0,  # 待成交回报更新
                filled_price=0,
                commission=0,
                timestamp=datetime.now(),
                status='PENDING',
            )

        except Exception as e:
            print(f"[XTP] Place order error: {e}")
            return OrderResult(
                order_id='',
                symbol=order.symbol,
                side=order.side,
                filled_quantity=0,
                filled_price=0,
                commission=0,
                timestamp=datetime.now(),
                status='REJECTED',
            )

    def cancel_order(self, order_id: str) -> bool:
        """撤单"""
        if not self._connected:
            return False

        try:
            self._api.cancel_order(int(order_id))
            return True
        except Exception as e:
            print(f"[XTP] Cancel order error: {e}")
            return False

    def get_order_status(self, order_id: str) -> Order:
        """查询订单状态"""
        if not self._connected:
            return None

        try:
            orders = self._api.query_orders()
            for o in orders:
                if str(o.order_id) == order_id:
                    return Order(
                        symbol=o.ticker,
                        side='BUY' if o.side == 1 else 'SELL',
                        quantity=o.quantity,
                        price=o.price,
                        order_id=str(o.order_id),
                        status=self._map_order_status(o.status),
                    )
            return None
        except Exception as e:
            print(f"[XTP] Query order error: {e}")
            return None

    def get_market_data(self, symbol: str) -> dict:
        """获取实时行情"""
        if not self._connected:
            return {}

        try:
            quote = self._api.query_quote(symbol)
            return {
                'symbol': symbol,
                'price': quote.last_price,
                'bid': quote.bid_price,
                'ask': quote.ask_price,
                'volume': quote.volume,
                'timestamp': datetime.now(),
            }
        except Exception as e:
            print(f"[XTP] Query market data error: {e}")
            return {}

    def _map_order_status(self, xtp_status: int) -> str:
        """映射订单状态"""
        status_map = {
            0: 'PENDING',
            1: 'PARTIAL_FILLED',
            2: 'FILLED',
            3: 'CANCELLED',
            4: 'REJECTED',
        }
        return status_map.get(xtp_status, 'UNKNOWN')


class XTPSimulator(XTPBroker):
    """
    XTP 模拟交易

    在没有 XTP 接口权限时使用模拟交易。
    """

    def __init__(self, initial_cash: float = DEFAULT_INITIAL_CASH, **kwargs):
        # 忽略连接参数
        super().__init__(**kwargs)
        self.name = "XTPSimulator"
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, dict] = {}
        self.orders: Dict[str, Order] = {}

    def connect(self, **kwargs) -> bool:
        print(f"[{self.name}] Connected (simulated)")
        self._connected = True
        return True

    def disconnect(self) -> bool:
        self._connected = False
        return True

    def get_account(self) -> dict:
        total_value = self.cash + sum(
            p['quantity'] * p.get('price', 0)
            for p in self.positions.values()
        )
        return {
            'cash': self.cash,
            'total_value': total_value,
            'market_value': total_value - self.cash,
            'available': self.cash,
        }
