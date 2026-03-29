"""
实盘交易安全模块

提供订单验证、资金检查、交易时间检查等安全功能
"""

import re
import logging
import os
from datetime import datetime, time
from typing import Optional, Tuple
from dquant.broker.base import Order
from dquant.constants import MIN_SHARES
from dquant.calendar import is_trading_day as _calendar_is_trading_day


# 配置日志
import os

logger = logging.getLogger('dquant.trading')
logger.setLevel(logging.INFO)

# 创建文件处理器
if not logger.handlers:
    # 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 文件输出（路径可通过环境变量配置）
    _log_path = os.environ.get('DQ_TRADING_LOG', 'logs/trading.log')
    try:
        file_handler = logging.FileHandler(_log_path, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)

        # 格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    except (IOError, OSError):
        # 如果无法创建文件，只使用控制台
        logger.addHandler(console_handler)


class OrderValidator:
    """订单验证器"""

    # 股票代码格式
    SYMBOL_PATTERNS = {
        'SH': r'^6\d{5}$',      # 上海主板: 600000-689999
        'SZ': r'^(00|30)\d{4}$', # 深圳: 000001-002999, 300001-301999
        'BJ': r'^(4|8)\d{5}$',   # 北交所: 430001-873999
    }

    @staticmethod
    def validate_symbol(symbol: str) -> Tuple[bool, str]:
        """
        验证股票代码格式

        Args:
            symbol: 股票代码 (如 '000001.SZ')

        Returns:
            (是否有效, 错误信息)
        """
        if not symbol:
            return False, "股票代码不能为空"

        # 分离代码和市场
        parts = symbol.split('.')
        if len(parts) != 2:
            return False, f"股票代码格式错误: {symbol} (应为: CODE.MARKET)"

        code, market = parts
        market = market.upper()

        # 检查市场
        if market not in OrderValidator.SYMBOL_PATTERNS:
            return False, f"不支持的市场: {market}"

        # 检查代码格式
        pattern = OrderValidator.SYMBOL_PATTERNS[market]
        if not re.match(pattern, code):
            return False, f"股票代码格式错误: {code} (不符合{market}市场规则)"

        return True, ""

    @staticmethod
    def validate_quantity(quantity: int, symbol: str = '') -> Tuple[bool, str]:
        """
        验证交易数量

        Args:
            quantity: 交易数量
            symbol: 股票代码 (可选，用于特殊检查)

        Returns:
            (是否有效, 错误信息)
        """
        if quantity is None:
            return False, "交易数量不能为空"

        if not isinstance(quantity, (int, float)):
            return False, f"交易数量类型错误: {type(quantity)}"

        if quantity <= 0:
            return False, f"交易数量必须大于0: {quantity}"

        # 检查是否为100股的整数倍
        if quantity % MIN_SHARES != 0:
            return False, f"交易数量必须是{MIN_SHARES}股的整数倍: {quantity}"

        # 检查最大数量 (单笔不超过100万股)
        MAX_SHARES = 1000000
        if quantity > MAX_SHARES:
            return False, f"单笔交易数量不能超过{MAX_SHARES}股: {quantity}"

        return True, ""

    @staticmethod
    def validate_price(price: Optional[float], order_type: str) -> Tuple[bool, str]:
        """
        验证交易价格

        Args:
            price: 交易价格
            order_type: 订单类型 ('MARKET' 或 'LIMIT')

        Returns:
            (是否有效, 错误信息)
        """
        # 市价单不需要价格
        if order_type == 'MARKET':
            return True, ""

        # 限价单必须有价格
        if price is None:
            return False, "限价单必须指定价格"

        if not isinstance(price, (int, float)):
            return False, f"价格类型错误: {type(price)}"

        if price <= 0:
            return False, f"价格必须大于0: {price}"

        # 检查价格范围 (0.01 - 10000)
        if price < 0.01 or price > 10000:
            return False, f"价格超出合理范围: {price}"

        return True, ""

    @staticmethod
    def validate_side(side: str) -> Tuple[bool, str]:
        """
        验证交易方向

        Args:
            side: 交易方向 ('BUY' 或 'SELL')

        Returns:
            (是否有效, 错误信息)
        """
        if not side:
            return False, "交易方向不能为空"

        side = side.upper()
        if side not in ['BUY', 'SELL']:
            return False, f"交易方向错误: {side} (应为 BUY 或 SELL)"

        return True, ""

    @staticmethod
    def validate_order(order: Order) -> Tuple[bool, str]:
        """
        验证订单所有参数

        Args:
            order: 订单对象

        Returns:
            (是否有效, 错误信息)
        """
        # 验证股票代码
        valid, msg = OrderValidator.validate_symbol(order.symbol)
        if not valid:
            return False, f"股票代码无效: {msg}"

        # 验证交易方向
        valid, msg = OrderValidator.validate_side(order.side)
        if not valid:
            return False, f"交易方向无效: {msg}"

        # 验证数量
        valid, msg = OrderValidator.validate_quantity(order.quantity, order.symbol)
        if not valid:
            return False, f"交易数量无效: {msg}"

        # 验证价格
        valid, msg = OrderValidator.validate_price(order.price, order.order_type)
        if not valid:
            return False, f"价格无效: {msg}"

        return True, ""


class FundChecker:
    """资金检查器"""

    @staticmethod
    def check_buy_fund(
        price: float,
        quantity: int,
        available_cash: float,
        commission_rate: float = 0.0003,
        slippage: float = 0.0,
    ) -> Tuple[bool, str, float]:
        """
        检查买入资金是否充足

        Args:
            price: 买入价格
            quantity: 买入数量
            available_cash: 可用资金
            commission_rate: 佣金费率 (默认万分之三)
            slippage: 滑点 (默认0)

        Returns:
            (是否充足, 错误信息, 需要的资金)
        """
        # 计算需要的资金
        # 考虑滑点 (向上取2个tick)
        actual_price = price * (1 + slippage)

        # 买入金额
        amount = actual_price * quantity

        # 佣金 (最低5元)
        commission = max(amount * commission_rate, 5.0)

        # 过户费 (万分之二，仅上海)
        transfer_fee = amount * 0.00002

        # 总需要资金
        total_need = amount + commission + transfer_fee

        # 预留安全边际 (1%)
        total_need_safe = total_need * 1.01

        if available_cash < total_need_safe:
            return False, (
                f"资金不足: 需要 {total_need_safe:.2f} 元 "
                f"(含费用 {commission + transfer_fee:.2f} 元 + 1%安全边际), "
                f"可用 {available_cash:.2f} 元"
            ), total_need_safe

        return True, "", total_need

    @staticmethod
    def check_sell_position(
        symbol: str,
        quantity: int,
        positions: dict,
    ) -> Tuple[bool, str]:
        """
        检查卖出持仓是否充足

        Args:
            symbol: 股票代码
            quantity: 卖出数量
            positions: 持仓字典 {'symbol': {'quantity': int, 'available': int}}

        Returns:
            (是否充足, 错误信息)
        """
        if symbol not in positions:
            return False, f"没有持仓: {symbol}"

        pos = positions[symbol]
        available = pos.get('available', pos.get('quantity', 0))

        if quantity > available:
            return False, (
                f"可用持仓不足: 需要 {quantity} 股, "
                f"可用 {available} 股"
            )

        return True, ""


class TradingTimeChecker:
    """交易时间检查器"""

    # A股交易时间
    MORNING_OPEN = time(9, 30)
    MORNING_CLOSE = time(11, 30)
    AFTERNOON_OPEN = time(13, 0)
    AFTERNOON_CLOSE = time(15, 0)

    @staticmethod
    def is_trading_day(dt: Optional[datetime] = None) -> bool:
        """
        检查是否为交易日 (使用交易日历，含节假日判断)

        Args:
            dt: 日期时间 (默认当前时间)

        Returns:
            是否为交易日
        """
        if dt is None:
            dt = datetime.now()

        return _calendar_is_trading_day(dt)

    @staticmethod
    def is_trading_time(dt: Optional[datetime] = None) -> Tuple[bool, str]:
        """
        检查是否在交易时间

        Args:
            dt: 日期时间 (默认当前时间)

        Returns:
            (是否在交易时间, 状态信息)
        """
        if dt is None:
            dt = datetime.now()

        # 检查是否为交易日
        if not TradingTimeChecker.is_trading_day(dt):
            return False, f"非交易日: {dt.strftime('%Y-%m-%d %A')}"

        current_time = dt.time()

        # 上午盘: 9:30 - 11:30
        if TradingTimeChecker.MORNING_OPEN <= current_time <= TradingTimeChecker.MORNING_CLOSE:
            return True, "上午交易时间 (9:30-11:30)"

        # 下午盘: 13:00 - 15:00
        if TradingTimeChecker.AFTERNOON_OPEN <= current_time <= TradingTimeChecker.AFTERNOON_CLOSE:
            return True, "下午交易时间 (13:00-15:00)"

        # 非交易时间
        return False, f"非交易时间: {current_time.strftime('%H:%M')} (交易时间: 9:30-11:30, 13:00-15:00)"

    @staticmethod
    def check_can_trade(dt: Optional[datetime] = None) -> Tuple[bool, str]:
        """
        检查是否可以交易

        Args:
            dt: 日期时间 (默认当前时间)

        Returns:
            (是否可以交易, 状态信息)
        """
        return TradingTimeChecker.is_trading_time(dt)


class TradingSafety:
    """交易安全总控"""

    def __init__(
        self,
        enable_time_check: bool = True,
        enable_fund_check: bool = True,
        enable_order_validation: bool = True,
        enable_position_check: bool = True,
    ):
        """
        初始化交易安全控制

        Args:
            enable_time_check: 是否启用交易时间检查
            enable_fund_check: 是否启用资金检查
            enable_order_validation: 是否启用订单验证
            enable_position_check: 是否启用持仓检查
        """
        self.enable_time_check = enable_time_check
        self.enable_fund_check = enable_fund_check
        self.enable_order_validation = enable_order_validation
        self.enable_position_check = enable_position_check

        logger.info("交易安全控制初始化完成")
        logger.info(f"  - 交易时间检查: {'启用' if enable_time_check else '禁用'}")
        logger.info(f"  - 资金检查: {'启用' if enable_fund_check else '禁用'}")
        logger.info(f"  - 订单验证: {'启用' if enable_order_validation else '禁用'}")
        logger.info(f"  - 持仓检查: {'启用' if enable_position_check else '禁用'}")

    def check_order(
        self,
        order: Order,
        available_cash: float = 0,
        positions: dict = None,
    ) -> Tuple[bool, str]:
        """
        综合检查订单

        Args:
            order: 订单对象
            available_cash: 可用资金
            positions: 持仓字典

        Returns:
            (是否通过检查, 错误信息)
        """
        logger.info(f"开始检查订单: {order.symbol} {order.side} {order.quantity}股")

        # 1. 订单验证
        if self.enable_order_validation:
            valid, msg = OrderValidator.validate_order(order)
            if not valid:
                logger.error(f"订单验证失败: {msg}")
                return False, msg
            logger.info("✓ 订单验证通过")

        # 2. 交易时间检查
        if self.enable_time_check:
            can_trade, msg = TradingTimeChecker.check_can_trade()
            if not can_trade:
                logger.error(f"交易时间检查失败: {msg}")
                return False, msg
            logger.info(f"✓ 交易时间检查通过: {msg}")

        # 3. 资金/持仓检查
        if order.side.upper() == 'BUY':
            if self.enable_fund_check and order.price:
                valid, msg, _ = FundChecker.check_buy_fund(
                    order.price,
                    order.quantity,
                    available_cash,
                )
                if not valid:
                    logger.error(f"资金检查失败: {msg}")
                    return False, msg
                logger.info("✓ 资金检查通过")

        elif order.side.upper() == 'SELL':
            if self.enable_position_check and positions:
                valid, msg = FundChecker.check_sell_position(
                    order.symbol,
                    order.quantity,
                    positions,
                )
                if not valid:
                    logger.error(f"持仓检查失败: {msg}")
                    return False, msg
                logger.info("✓ 持仓检查通过")

        logger.info("✓ 所有检查通过，可以下单")
        return True, ""


def log_trade(
    order: Order,
    result,
    action: str = "PLACE_ORDER",
):
    """
    记录交易日志

    Args:
        order: 订单对象
        result: 订单结果
        action: 动作类型
    """
    log_msg = (
        f"[{action}] "
        f"股票: {order.symbol}, "
        f"方向: {order.side}, "
        f"数量: {order.quantity}股, "
        f"价格: {order.price or '市价'}, "
        f"订单ID: {result.order_id}, "
        f"状态: {result.status}"
    )

    if result.status == 'FILLED':
        log_msg += f", 成交价: {result.filled_price}, 成交量: {result.filled_quantity}"
        logger.info(log_msg)
    elif result.status == 'REJECTED':
        logger.warning(log_msg)
    else:
        logger.info(log_msg)


def log_error(
    action: str,
    error: Exception,
    context: dict = None,
):
    """
    记录错误日志

    Args:
        action: 动作类型
        error: 异常对象
        context: 上下文信息
    """
    log_msg = f"[{action}] 错误: {str(error)}"

    if context:
        log_msg += f", 上下文: {context}"

    logger.error(log_msg, exc_info=True)
