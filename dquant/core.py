"""
核心引擎 - 整合数据、策略、回测、实盘
"""

import signal
import time
from typing import Optional, Dict, Any, Union, List
from datetime import datetime
import pandas as pd

from dquant.data.base import DataSource
from dquant.strategy.base import BaseStrategy, Signal, SignalType
from dquant.backtest.engine import BacktestEngine
from dquant.backtest.result import BacktestResult
from dquant.backtest.portfolio import Portfolio
from dquant.broker.base import BaseBroker, Order, OrderResult
from dquant.broker.simulator import Simulator
from dquant.broker.safety import TradingTimeChecker
from dquant.broker.trade_journal import TradeJournal
from dquant.broker.order_tracker import OrderTracker
from dquant.notify import create_notifier
from dquant.notify.base import Notifier
from dquant.risk import RiskManager, PositionSizer, PositionLimit
from dquant.constants import DEFAULT_COMMISSION, DEFAULT_SLIPPAGE, DEFAULT_STAMP_DUTY, DEFAULT_INITIAL_CASH, MIN_SHARES, DEFAULT_WINDOW
from dquant.constants import BROKER_MAX_RECONNECT, BROKER_RECONNECT_DELAY, BROKER_RECONNECT_BACKOFF
from dquant.logger import get_logger
from dquant.calendar import is_trading_day

logger = get_logger(__name__)


class Engine:
    """
    量化交易引擎 - 整合回测与实盘

    Usage:
        # 回测模式
        engine = Engine(data, strategy)
        result = engine.backtest(start='2020-01-01', end='2024-01-01')

        # 实盘模式
        engine = Engine(data, strategy, broker='xtp')
        engine.live()
    """

    def __init__(
        self,
        data: DataSource,
        strategy: BaseStrategy,
        broker: Optional[Union[str, BaseBroker]] = None,
        initial_cash: float = DEFAULT_INITIAL_CASH,
    ):
        """
        初始化引擎

        Args:
            data: 数据源
            strategy: 交易策略
            broker: 券商接口 (None=模拟, 'xtp', 'qmt', 或自定义Broker实例)
            initial_cash: 初始资金
        """
        self.data = data
        self.strategy = strategy
        self.initial_cash = initial_cash

        # 初始化券商
        if broker is None:
            self.broker = Simulator(initial_cash=initial_cash)
        elif isinstance(broker, str):
            self.broker = self._init_broker(broker, initial_cash)
        else:
            self.broker = broker

        self._backtest_engine: Optional[BacktestEngine] = None

    def _init_broker(self, broker_name: str, initial_cash: float) -> BaseBroker:
        """初始化券商接口"""
        if broker_name == 'simulator':
            return Simulator(initial_cash=initial_cash)
        elif broker_name == 'xtp':
            # XTP 接口实现
            try:
                from dquant.broker.xtp_broker import XTPBroker
                return XTPBroker(initial_cash=initial_cash)
            except ImportError:
                raise ImportError(
                    "XTP SDK not installed.\n"
                    "Please install XTP SDK and configure:\n"
                    "1. Download from https://xtp.zts.com.cn\n"
                    "2. Install: pip install xtp-python-api\n"
                    "3. Configure in config.py"
                )
        elif broker_name == 'qmt':
            # miniQMT 接口实现
            try:
                from dquant.broker.qmt_broker import QMTBroker
                return QMTBroker(initial_cash=initial_cash)
            except ImportError:
                raise ImportError(
                    "miniQMT SDK not installed.\n"
                    "Please install miniQMT and configure:\n"
                    "1. Download from broker\n"
                    "2. Configure in config.py"
                )
        else:
            raise ValueError(f"Unknown broker: {broker_name}")

    def backtest(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
        commission: float = DEFAULT_COMMISSION,
        slippage: float = DEFAULT_SLIPPAGE,
        benchmark: Optional[str] = None,
    ) -> "BacktestResult":
        """
        运行回测

        Args:
            start: 开始日期 (YYYY-MM-DD)
            end: 结束日期 (YYYY-MM-DD)
            commission: 手续费率
            slippage: 滑点
            benchmark: 基准 (如 '000300.SH')

        Returns:
            BacktestResult: 回测结果
        """
        # 获取数据
        df = self.data.load()

        # 过滤日期范围
        if start:
            df = df[df.index >= pd.to_datetime(start)]
        if end:
            df = df[df.index <= pd.to_datetime(end)]

        # 初始化回测引擎
        self._backtest_engine = BacktestEngine(
            data=df,
            strategy=self.strategy,
            initial_cash=self.initial_cash,
            commission=commission,
            slippage=slippage,
            benchmark=benchmark,
        )

        # 运行回测
        return self._backtest_engine.run()

    def live(
        self,
        dry_run: bool = True,
        interval: int = 60,
        symbols: Optional[List[str]] = None,
        strategy_name: str = "",
        max_drawdown: float = 0.15,
        max_daily_loss: float = 0.03,
        max_consecutive_errors: int = 10,
        notifier: Optional[Notifier] = None,
        **kwargs
    ) -> None:
        """
        启动实盘交易

        Args:
            dry_run: 是否模拟运行 (不实际下单)
            interval: 循环间隔（秒），默认 60
            symbols: 交易标的列表 (None 则从 strategy 获取)
            strategy_name: 策略名称 (用于审计日志)
            max_drawdown: 最大回撤限制
            max_daily_loss: 单日最大亏损
            max_consecutive_errors: 连续错误上限 (超过则停止)
            notifier: 通知器 (None 则用 LogNotifier)
        """
        # ---- 初始化 ----
        if dry_run:
            logger.info("[LIVE] Running in dry-run mode (模拟运行)")
        else:
            logger.info("[LIVE] Running in live mode (实盘运行)")

        # 连接 broker
        if not self.broker.connect(**kwargs):
            logger.error("[LIVE] Broker 连接失败，退出")
            return

        logger.info(f"[LIVE] Connected to broker: {self.broker.name}")

        # 风控 & 审计
        risk_mgr = RiskManager(max_drawdown=max_drawdown, max_daily_loss=max_daily_loss)
        journal = TradeJournal()
        tracker = OrderTracker()
        sizer = PositionSizer(method='equal_weight', total_value=self.initial_cash)
        time_checker = TradingTimeChecker()

        # 信号处理：优雅关机
        self._running = True

        def _shutdown_handler(signum, frame):
            logger.info(f"[LIVE] 收到信号 {signum}，准备优雅关机...")
            self._running = False

        original_sigint = signal.getsignal(signal.SIGINT)
        original_sigterm = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGINT, _shutdown_handler)
        signal.signal(signal.SIGTERM, _shutdown_handler)

        consecutive_errors = 0
        last_date_str = ""

        try:
            while self._running:
                loop_start = time.time()

                try:
                    now = datetime.now()
                    date_str = now.strftime("%Y-%m-%d")

                    # 0. 检查 broker 连接
                    if not self.broker.is_connected():
                        if not self._try_reconnect(**kwargs):
                            time.sleep(interval)
                            continue

                    # 1. 检查交易日
                    if not is_trading_day(now):
                        logger.debug(f"[LIVE] 非交易日: {date_str}")
                        time.sleep(interval)
                        continue

                    # 2. 检查交易时间
                    if not time_checker.is_trading_time():
                        logger.debug(f"[LIVE] 非交易时间: {now.strftime('%H:%M:%S')}")
                        time.sleep(interval)
                        continue

                    # 3. 新一天 → 重置日亏损基准
                    if date_str != last_date_str:
                        account = self.broker.get_account()
                        total_value = account.get('total_value', account.get('cash', self.initial_cash))
                        risk_mgr.reset_daily_start(total_value, date_str)
                        last_date_str = date_str
                        logger.info(f"[LIVE] 新交易日: {date_str}, 组合价值: {total_value:,.0f}")

                    # 4. 获取实时行情
                    realtime_df = self._fetch_realtime_data(symbols)
                    if realtime_df is None or realtime_df.empty:
                        logger.warning("[LIVE] 获取实时行情失败，跳过本轮")
                        time.sleep(interval)
                        continue

                    # 5. 生成信号
                    # 检测策略是否 override 了 on_bar
                    use_on_bar = type(self.strategy).on_bar is not BaseStrategy.on_bar

                    if use_on_bar:
                        # 逐行调用 on_bar，收集非 None 信号
                        signals = []
                        for idx, row in realtime_df.iterrows():
                            sig = self.strategy.on_bar(row)
                            if sig is not None:
                                signals.append(sig)
                    else:
                        signals = self.strategy.generate_signals(realtime_df)

                    if not signals:
                        logger.debug("[LIVE] 无交易信号")
                        time.sleep(interval)
                        continue

                    buy_signals = [s for s in signals if s.is_buy]
                    sell_signals = [s for s in signals if s.is_sell]

                    # 6. 获取账户 & 持仓
                    account = self.broker.get_account()
                    positions = self.broker.get_positions()
                    current_value = account.get('total_value', account.get('cash', self.initial_cash))
                    available_cash = account.get('cash', 0)

                    # 7. 风控检查
                    risk_mgr.check_drawdown(current_value)
                    risk_mgr.check_daily_loss(current_value)

                    if risk_mgr.should_halt():
                        logger.warning("[LIVE] 触发风控 halt，停止交易")
                        break

                    # 8. 执行卖出信号
                    for sig in sell_signals:
                        result = self._execute_sell(
                            sig, positions, dry_run, journal, strategy_name,
                        )
                        if result and result.status in ('PENDING', 'PARTIAL_FILLED'):
                            tracker.add(sig, result)

                    # 9. 轮询未完成订单
                    if tracker.has_pending() and not dry_run:
                        self._poll_pending_orders(tracker, journal, strategy_name)

                    # 10. 执行买入信号 (等权仓位)
                    if buy_signals:
                        self._execute_buys(
                            buy_signals, available_cash, sizer,
                            dry_run, journal, strategy_name,
                        )

                    # 10. 更新持仓价格
                    self._update_position_prices(realtime_df)

                    consecutive_errors = 0

                except Exception as e:
                    consecutive_errors += 1
                    logger.error(f"[LIVE] 循环异常 ({consecutive_errors}/{max_consecutive_errors}): {e}")

                    if consecutive_errors >= max_consecutive_errors:
                        logger.error(f"[LIVE] 连续错误达到 {max_consecutive_errors} 次，停止交易")
                        break

                # 控制循环频率
                elapsed = time.time() - loop_start
                sleep_time = max(0, interval - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        finally:
            # 优雅关机：取消所有 pending 订单
            if tracker.has_pending():
                logger.info(f"[LIVE] 关机：取消 {len(tracker.get_pending())} 个 pending 订单")
                cancelled = tracker.cancel_all(self.broker)
                for order_id in cancelled:
                    journal.record("CANCELLED_ON_SHUTDOWN", Order(
                        symbol='', side='', quantity=0, order_id=order_id,
                    ), None, strategy_name=strategy_name)

            # 清理
            signal.signal(signal.SIGINT, original_sigint)
            signal.signal(signal.SIGTERM, original_sigterm)
            self.broker.disconnect()
            logger.info("[LIVE] 交易会话结束")

    def _fetch_realtime_data(self, symbols: Optional[List[str]]) -> Optional[pd.DataFrame]:
        """获取实时行情数据"""
        try:
            from dquant.data.akshare_loader import AKShareRealTime
            return AKShareRealTime.get_realtime_quotes(symbols=symbols)
        except ImportError:
            logger.warning("[LIVE] akshare 未安装，尝试通过 data source 获取")
            try:
                return self.data.load()
            except Exception as e:
                logger.error(f"[LIVE] 数据获取失败: {e}")
                return None
        except Exception as e:
            logger.error(f"[LIVE] 实时行情获取失败: {e}")
            return None

    def _execute_sell(
        self,
        signal: Signal,
        positions: Dict[str, dict],
        dry_run: bool,
        journal: TradeJournal,
        strategy_name: str,
    ) -> Optional[OrderResult]:
        """执行卖出信号"""
        symbol = signal.symbol
        if symbol not in positions:
            logger.debug(f"[LIVE] 无持仓，跳过卖出: {symbol}")
            return None

        pos = positions[symbol]
        quantity = pos.get('quantity', pos.get('available', 0))
        if quantity <= 0:
            return None

        # 整手处理
        quantity = (quantity // MIN_SHARES) * MIN_SHARES
        if quantity <= 0:
            return None

        order = Order(
            symbol=symbol,
            side='SELL',
            quantity=quantity,
            price=signal.price,
            order_type='LIMIT' if signal.price else 'MARKET',
        )

        if dry_run:
            logger.info(f"[LIVE][DRY-RUN] 卖出: {symbol} x {quantity}")
            return None

        result = self.broker.place_order(order)
        journal.record("ORDER_PLACED", order, result, strategy_name=strategy_name,
                       signal_info=signal.to_dict())

        if result.status in ('FILLED', 'PARTIAL_FILLED'):
            logger.info(f"[LIVE] 卖出成交: {symbol} x {result.filled_quantity} @ {result.filled_price:.2f}")
        elif result.status == 'REJECTED':
            logger.warning(f"[LIVE] 卖出被拒: {symbol} — {result.status}")

        return result

    def _execute_buys(
        self,
        buy_signals: List[Signal],
        available_cash: float,
        sizer: PositionSizer,
        dry_run: bool,
        journal: TradeJournal,
        strategy_name: str,
    ) -> None:
        """执行买入信号 (等权仓位分配)"""
        if not buy_signals:
            return

        symbols = [s.symbol for s in buy_signals]
        position_values = sizer.size(symbols)

        for sig in buy_signals:
            target_value = position_values.get(sig.symbol, 0)
            if target_value <= 0:
                continue

            price = sig.price or 10.0  # 需要价格计算股数
            if price <= 0:
                continue

            # 计算买入数量 (整手)
            quantity = int(target_value / price // MIN_SHARES) * MIN_SHARES
            if quantity <= 0:
                logger.debug(f"[LIVE] 资金不足，跳过买入: {sig.symbol}")
                continue

            order = Order(
                symbol=sig.symbol,
                side='BUY',
                quantity=quantity,
                price=sig.price,
                order_type='LIMIT' if sig.price else 'MARKET',
            )

            if dry_run:
                logger.info(f"[LIVE][DRY-RUN] 买入: {sig.symbol} x {quantity}")
                continue

            result = self.broker.place_order(order)
            journal.record("ORDER_PLACED", order, result, strategy_name=strategy_name,
                           signal_info=sig.to_dict())

            if result.status in ('FILLED', 'PARTIAL_FILLED'):
                logger.info(f"[LIVE] 买入成交: {sig.symbol} x {result.filled_quantity} @ {result.filled_price:.2f}")
            elif result.status == 'REJECTED':
                logger.warning(f"[LIVE] 买入被拒: {sig.symbol} — {result.status}")

    def _try_reconnect(self, **kwargs) -> bool:
        """
        尝试重新连接 broker（指数退避）

        Returns:
            True 如果重连成功
        """
        for attempt in range(BROKER_MAX_RECONNECT):
            delay = BROKER_RECONNECT_DELAY * (BROKER_RECONNECT_BACKOFF ** attempt)
            logger.warning(
                f"[LIVE] Broker 断线，{delay:.1f}s 后尝试重连 "
                f"({attempt + 1}/{BROKER_MAX_RECONNECT})"
            )
            time.sleep(delay)

            try:
                if self.broker.connect(**kwargs):
                    logger.info("[LIVE] Broker 重连成功")
                    return True
            except Exception as e:
                logger.error(f"[LIVE] Broker 重连失败: {e}")

        logger.error(f"[LIVE] Broker 重连 {BROKER_MAX_RECONNECT} 次后仍失败")
        return False

    def _update_position_prices(self, realtime_df: pd.DataFrame) -> None:
        """用最新价更新持仓价格（通用版）"""
        # Simulator: 直接更新
        if isinstance(self.broker, Simulator):
            if 'symbol' not in realtime_df.columns or 'price' not in realtime_df.columns:
                return
            price_map = dict(zip(realtime_df['symbol'], realtime_df['price']))
            self.broker.update_prices(price_map)
            return

        # 其他 broker: 通过 get_market_data 逐个更新持仓估值
        positions = self.broker.get_positions()
        if not positions:
            return

        for symbol in positions:
            try:
                md = self.broker.get_market_data(symbol)
                if md and 'price' in md:
                    # 实盘 broker 内部维护持仓状态，此处仅做风控估值
                    logger.debug(
                        f"[LIVE] 持仓估值: {symbol} price={md['price']:.2f}"
                    )
            except Exception as e:
                logger.debug(f"[LIVE] 获取 {symbol} 行情失败: {e}")

    def _poll_pending_orders(
        self,
        tracker: OrderTracker,
        journal: TradeJournal,
        strategy_name: str,
    ) -> None:
        """
        轮询未完成订单，处理超时

        超时订单自动取消并记录审计日志。
        """
        # 轮询状态更新
        for order_id, tracked in tracker.get_pending().items():
            try:
                updated = self.broker.get_order_status(order_id)
                if updated:
                    tracker.update(order_id, updated)
            except Exception as e:
                logger.error(f"[LIVE] 轮询订单状态失败: {order_id} — {e}")

        # 取消超时订单
        timed_out = tracker.get_timed_out()
        for tracked in timed_out:
            try:
                ok = self.broker.cancel_order(tracked.order.order_id)
                if ok:
                    logger.warning(
                        f"[LIVE] 超时取消订单: {tracked.order.order_id} "
                        f"{tracked.order.symbol} {tracked.order.side}"
                    )
                    journal.record("CANCELLED_TIMEOUT", tracked.order, tracked.result,
                                   strategy_name=strategy_name)
                tracker.remove(tracked.order.order_id)
            except Exception as e:
                logger.error(f"[LIVE] 取消超时订单失败: {tracked.order.order_id} — {e}")

    def optimize(
        self,
        param_grid: Dict[str, list],
        metric: str = 'sharpe',
        **backtest_kwargs
    ) -> Dict[str, Any]:
        """
        参数优化

        Args:
            param_grid: 参数网格 {'param_name': [value1, value2, ...]}
            metric: 优化目标 ('sharpe', 'return', 'max_drawdown')

        Returns:
            最优参数和结果
        """
        from itertools import product

        best_score = -float('inf')
        best_params = None
        best_result = None

        # 保存原始参数，出错时可恢复
        original_params = {
            name: getattr(self.strategy, name)
            for name in param_grid
            if hasattr(self.strategy, name)
        }

        # 生成所有参数组合
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        for values in product(*param_values):
            params = dict(zip(param_names, values))

            # 更新策略参数
            for name, value in params.items():
                if hasattr(self.strategy, name):
                    setattr(self.strategy, name, value)

            try:
                # 运行回测
                result = self.backtest(**backtest_kwargs)

                # 获取评分
                score = getattr(result.metrics, metric, 0)

                if score > best_score:
                    best_score = score
                    best_params = params
                    best_result = result
            except Exception:
                continue

        # 恢复策略为最优参数（而非最后一个网格点的参数）
        if best_params:
            for name, value in best_params.items():
                if hasattr(self.strategy, name):
                    setattr(self.strategy, name, value)

        return {
            'best_params': best_params,
            'best_score': best_score,
            'best_result': best_result,
        }


