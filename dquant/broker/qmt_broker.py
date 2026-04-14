"""
miniQMT 券商接口 (迅投 QMT)

注意: 需要开通 QMT 交易权限
"""

import json
import os
import re
import subprocess
import sys
from datetime import datetime
from typing import Dict

from dquant.broker.base import BaseBroker, Order, OrderResult
from dquant.broker.safety import TradingSafety, log_error
from dquant.broker.simulator import Simulator
from dquant.constants import DEFAULT_INITIAL_CASH
from dquant.logger import get_logger

logger = get_logger(__name__)


class QMTBroker(BaseBroker):
    """
    miniQMT 券商接口

    迅投 QMT 迷你交易接口封装。

    使用前需要:
    1. 开通 QMT 交易权限
    2. 安装 QMT 客户端
    3. 配置 miniQMT

    Usage:
        broker = QMTBroker(
            qmt_path='C:/国金证券QMT/userdata_mini',
            account='your_account',
        )

        broker.connect()

        # 下单
        order = Order(symbol='000001.SZ', side='BUY', quantity=MIN_SHARES)
        result = broker.place_order(order)
    """

    def __init__(self, qmt_path: str = "", account: str = "", **kwargs):
        super().__init__(name="QMT")
        self.qmt_path = qmt_path
        self.account = account

        self._connected = False

        # 交易安全控制
        self.safety = TradingSafety(
            enable_time_check=kwargs.get("enable_time_check", True),
            enable_fund_check=kwargs.get("enable_fund_check", True),
            enable_order_validation=kwargs.get("enable_order_validation", True),
            enable_position_check=kwargs.get("enable_position_check", True),
        )

    def connect(self, **kwargs) -> bool:
        """连接 QMT"""
        if not self.qmt_path:
            logger.error("[QMT] QMT path not configured")
            return False

        if not os.path.exists(self.qmt_path):
            logger.error(f"[QMT] QMT path not found: {self.qmt_path}")
            return False

        self._connected = True
        logger.info(f"[QMT] Connected to {self.qmt_path}")
        return True

    def disconnect(self) -> bool:
        """断开连接"""
        self._connected = False
        return True

    def _call_qmt(self, func_name: str, params: dict) -> dict:
        """调用 QMT 函数（通过环境变量和 stdin 安全传参）"""
        if not self._connected:
            return {"error": "not connected"}

        # 安全检查：func_name 只允许合法 Python 标识符
        if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", func_name):
            return {"error": f"invalid function name: {func_name}"}

        # 通过环境变量传递路径和函数名，通过 stdin 传递参数，避免命令注入
        script = """
import sys
import json
import os

qmt_path = os.environ.get('DQ_QMT_PATH', '')
if qmt_path:
    sys.path.insert(0, qmt_path)

func_name = os.environ.get('DQ_QMT_FUNC', '')
params = json.loads(sys.stdin.read())

# 白名单：仅允许调用的 xttrade 函数
_ALLOWED = {
    'download_history_data', 'get_stock_list_in_sector',
    'get_instrument_detail', 'get_instruments',
    'get_trading_dates', 'get_trading_calendar',
    'get_market_data', 'get_market_data_ex',
    'get_full_tick', 'get_l2_quote', 'get_l2_order_book', 'get_l2_order_queue',
    'get_financial_data',
    'subscribe_quote', 'unsubscribe_quote', 'subscribe_whole_quote',
    'query_stock_orders', 'query_stock_trades',
    'query_credit_detail', 'query_credit_orders', 'query_credit_trades',
    'query_option_orders', 'query_option_trades',
    'query_etf_orders', 'query_etf_trades',
    'stock_order', 'stock_cancel',
    'query_account', 'query_position', 'query_asset',
}
if func_name not in _ALLOWED:
    print(json.dumps({'error': f'function not allowed: {func_name}'}))
    sys.exit(0)

from xtquant import xttrade

# 初始化
xttrade.connect()

# 调用函数
func = getattr(xttrade, func_name, None)
if func is None:
    print(json.dumps({'error': f'function not found: {func_name}'}))
else:
    result = func(**params)
    print(json.dumps(result))
"""

        try:
            env = os.environ.copy()
            env["DQ_QMT_PATH"] = self.qmt_path
            env["DQ_QMT_FUNC"] = func_name

            proc_result = subprocess.run(
                [sys.executable, "-c", script],
                input=json.dumps(params),
                capture_output=True,
                text=True,
                env=env,
            )

            if proc_result.returncode == 0:
                return json.loads(proc_result.stdout)
            else:
                return {"error": proc_result.stderr}

        except Exception as e:
            return {"error": str(e)}

    def get_account(self) -> dict:
        """获取账户信息"""
        if not self._connected:
            return {}

        try:
            from xtquant import xttrade

            _ = xttrade.get_stock_account(self.account)
            asset = xttrade.get_asset(self.account)

            return {
                "cash": asset.cash,
                "total_value": asset.total_asset,
                "market_value": asset.market_value,
                "available": asset.cash,
            }
        except ImportError:
            logger.error("[QMT] xtquant not installed")
            return {}
        except Exception as e:
            logger.error(f"[QMT] Query account error: {e}")
            return {}

    def get_positions(self) -> Dict[str, dict]:
        """获取持仓"""
        if not self._connected:
            return {}

        try:
            from xtquant import xttrade

            positions = xttrade.get_stock_positions(self.account)
            result = {}

            for pos in positions:
                result[pos.stock_code] = {
                    "symbol": pos.stock_code,
                    "quantity": pos.volume,
                    "available": pos.can_use_volume,
                    "avg_cost": pos.open_price,
                    "current_price": (pos.market_value / pos.volume if pos.volume > 0 else 0),
                }

            return result
        except Exception as e:
            logger.error(f"[QMT] Query positions error: {e}")
            return {}

    def place_order(self, order: Order) -> OrderResult:
        """
        下单 (带安全检查)

        Args:
            order: 订单对象

        Returns:
            OrderResult: 订单结果
        """
        # 1. 连接检查
        if not self._connected:
            log_error("PLACE_ORDER", Exception("未连接到QMT"), {"symbol": order.symbol})
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
            # 2. 安全检查
            account_info = self.get_account()
            positions = self.get_positions()

            # 市价单需要 estimated_price 做资金检查
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

            # 3. 调用QMT下单
            from xtquant import xttrade

            # 下单
            # order_type: 23=市价, 24=限价
            order_type = 23 if order.order_type == "MARKET" else 24

            # price_type: 1=限价, 2=市价（最优五档即时成交剩余撤销）, 5=市价（最优五档即时成交剩余转限价）
            price_type = 2 if order.order_type == "MARKET" else 1

            api_result = xttrade.order_stock(
                account=self.account,
                stock_code=order.symbol,
                order_type=order_type,
                order_volume=order.quantity,
                price_type=price_type,
                price=order.price or 0,
            )

            if api_result > 0:
                order.order_id = str(api_result)
                order.status = "PENDING"

                order_result = OrderResult(
                    order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side,
                    filled_quantity=0,
                    filled_price=0,
                    commission=0,
                    timestamp=datetime.now(),
                    status="PENDING",
                )
                return order_result
            else:
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

        except Exception as e:
            logger.error(f"[QMT] Place order error: {e}")
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
            from xtquant import xttrade

            result = xttrade.cancel_order(self.account, int(order_id))
            return result > 0
        except Exception as e:
            logger.error(f"[QMT] Cancel order error: {e}")
            return False

    def get_order_status(self, order_id: str) -> Order:
        """查询订单状态"""
        if not self._connected:
            return None

        try:
            from xtquant import xttrade

            orders = xttrade.get_stock_orders(self.account)
            for o in orders:
                if str(o.order_id) == order_id:
                    side = "BUY" if getattr(o, "order_side", 0) == 23 else "SELL"
                    filled_qty = getattr(o, "traded_volume", 0)
                    return Order(
                        symbol=o.stock_code,
                        side=side,
                        quantity=o.order_volume,
                        price=o.price,
                        order_id=str(o.order_id),
                        filled_quantity=filled_qty,
                        status=(
                            "FILLED"
                            if filled_qty >= o.order_volume
                            else "PARTIAL_FILLED" if filled_qty > 0 else "PENDING"
                        ),
                    )
            return None
        except Exception as e:
            logger.error(f"[QMT] Query order error: {e}")
            return None

    def get_market_data(self, symbol: str) -> dict:
        """获取实时行情"""
        try:
            from xtquant import xtdata

            quote = xtdata.get_full_tick([symbol])
            if symbol in quote:
                q = quote[symbol]
                return {
                    "symbol": symbol,
                    "price": q["lastPrice"],
                    "bid": q["bidPrice"][0] if q["bidPrice"] else 0,
                    "ask": q["askPrice"][0] if q["askPrice"] else 0,
                    "volume": q["volume"],
                    "timestamp": datetime.now(),
                }
            return {}
        except Exception as e:
            logger.error(f"[QMT] Query market data error: {e}")
            return {}


class QMTSimulator(QMTBroker):
    """
    QMT 模拟交易

    通过组合 Simulator 实现，避免代码重复。
    """

    def __init__(self, initial_cash: float = DEFAULT_INITIAL_CASH, **kwargs):
        super().__init__(**kwargs)
        self.name = "QMTSimulator"
        # 组合：内部委托给 Simulator
        self._sim = Simulator(
            initial_cash=initial_cash,
            order_id_prefix="SIM",
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
        return dict(self._sim.positions)

    # ---------- 交易 ----------
    def place_order(self, order: Order) -> OrderResult:
        """模拟下单（不需要 xtquant）"""
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

    def get_order_status(self, order_id: str) -> Order:
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
