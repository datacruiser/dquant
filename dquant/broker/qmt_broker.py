"""
miniQMT 券商接口 (迅投 QMT)

注意: 需要开通 QMT 交易权限
"""

from typing import Dict, Optional, List
from datetime import datetime
import subprocess
import json
import os
import re
from dquant.constants import DEFAULT_INITIAL_CASH

from dquant.broker.base import BaseBroker, Order, OrderResult
from dquant.broker.safety import TradingSafety, log_trade, log_error


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

    def __init__(
        self,
        qmt_path: str = '',
        account: str = '',
        **kwargs
    ):
        super().__init__(name="QMT")
        self.qmt_path = qmt_path
        self.account = account

        self._connected = False

        # 交易安全控制
        self.safety = TradingSafety(
            enable_time_check=kwargs.get('enable_time_check', True),
            enable_fund_check=kwargs.get('enable_fund_check', True),
            enable_order_validation=kwargs.get('enable_order_validation', True),
            enable_position_check=kwargs.get('enable_position_check', True),
        )


    def connect(self, **kwargs) -> bool:
        """连接 QMT"""
        if not self.qmt_path:
            print("[QMT] QMT path not configured")
            return False

        if not os.path.exists(self.qmt_path):
            print(f"[QMT] QMT path not found: {self.qmt_path}")
            return False

        self._connected = True
        print(f"[QMT] Connected to {self.qmt_path}")
        return True

    def disconnect(self) -> bool:
        """断开连接"""
        self._connected = False
        return True

    def _call_qmt(self, func_name: str, params: dict) -> dict:
        """调用 QMT 函数（通过环境变量和 stdin 安全传参）"""
        if not self._connected:
            return {'error': 'not connected'}

        # 安全检查：func_name 只允许合法 Python 标识符
        if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', func_name):
            return {'error': f'invalid function name: {func_name}'}

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
            env['DQ_QMT_PATH'] = self.qmt_path
            env['DQ_QMT_FUNC'] = func_name

            result = subprocess.run(
                ['python', '-c', script],
                input=json.dumps(params),
                capture_output=True,
                text=True,
                env=env,
            )

            if result.returncode == 0:
                return json.loads(result.stdout)
            else:
                return {'error': result.stderr}

        except Exception as e:
            return {'error': str(e)}

    def get_account(self) -> dict:
        """获取账户信息"""
        if not self._connected:
            return {}

        try:
            from xtquant import xttrade

            account_info = xttrade.get_stock_account(self.account)
            asset = xttrade.get_asset(self.account)

            return {
                'cash': asset.cash,
                'total_value': asset.total_asset,
                'market_value': asset.market_value,
                'available': asset.cash,
            }
        except ImportError:
            print("[QMT] xtquant not installed")
            return {}
        except Exception as e:
            print(f"[QMT] Query account error: {e}")
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
                    'symbol': pos.stock_code,
                    'quantity': pos.volume,
                    'available': pos.can_use_volume,
                    'avg_cost': pos.open_price,
                    'current_price': pos.market_value / pos.volume if pos.volume > 0 else 0,
                }

            return result
        except Exception as e:
            print(f"[QMT] Query positions error: {e}")
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
            # 2. 安全检查
            account_info = self.get_account()
            positions = self.get_positions()

            valid, msg = self.safety.check_order(
                order,
                available_cash=account_info.get('cash', 0),
                positions=positions,
            )

            if not valid:
                log_error("PLACE_ORDER", Exception(msg), {
                    "symbol": order.symbol,
                    "side": order.side,
                    "quantity": order.quantity,
                })
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

            # 3. 调用QMT下单
            from xtquant import xttrade

            # 下单
            # order_type: 23=市价, 24=限价
            order_type = 23 if order.order_type == 'MARKET' else 24

            result = xttrade.order_stock(
                account=self.account,
                stock_code=order.symbol,
                order_type=order_type,
                order_volume=order.quantity,
                price_type=1,  # 1=限价, 2=市价
                price=order.price or 0,
            )

            if result > 0:
                order.order_id = str(result)
                order.status = 'PENDING'

                result = OrderResult(
                    order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side,
                    filled_quantity=0,
                    filled_price=0,
                    commission=0,
                    timestamp=datetime.now(),
                    status='PENDING',
                )
            else:
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

        except Exception as e:
            print(f"[QMT] Place order error: {e}")
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
            from xtquant import xttrade

            result = xttrade.cancel_order(self.account, int(order_id))
            return result > 0
        except Exception as e:
            print(f"[QMT] Cancel order error: {e}")
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
                    return Order(
                        symbol=o.stock_code,
                        side='BUY' if o.order_type == 23 else 'SELL',
                        quantity=o.order_volume,
                        price=o.price,
                        order_id=str(o.order_id),
                        status='FILLED' if o.traded_volume > 0 else 'PENDING',
                    )
            return None
        except Exception as e:
            print(f"[QMT] Query order error: {e}")
            return None

    def get_market_data(self, symbol: str) -> dict:
        """获取实时行情"""
        try:
            from xtquant import xtdata

            quote = xtdata.get_full_tick([symbol])
            if symbol in quote:
                q = quote[symbol]
                return {
                    'symbol': symbol,
                    'price': q['lastPrice'],
                    'bid': q['bidPrice'][0] if q['bidPrice'] else 0,
                    'ask': q['askPrice'][0] if q['askPrice'] else 0,
                    'volume': q['volume'],
                    'timestamp': datetime.now(),
                }
            return {}
        except Exception as e:
            print(f"[QMT] Query market data error: {e}")
            return {}


class QMTSimulator(QMTBroker):
    """
    QMT 模拟交易
    """

    def __init__(self, initial_cash: float = DEFAULT_INITIAL_CASH, **kwargs):
        super().__init__(**kwargs)
        self.name = "QMTSimulator"
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
