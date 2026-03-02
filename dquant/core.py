"""
核心引擎 - 整合数据、策略、回测、实盘
"""

from typing import Optional, Dict, Any, Union
from datetime import datetime
import pandas as pd

from dquant.data.base import DataSource
from dquant.strategy.base import BaseStrategy
from dquant.backtest.engine import BacktestEngine
from dquant.backtest.result import BacktestResult
from dquant.backtest.portfolio import Portfolio
from dquant.broker.base import BaseBroker
from dquant.broker.simulator import Simulator
from dquant.constants import DEFAULT_COMMISSION, DEFAULT_SLIPPAGE, DEFAULT_STAMP_DUTY, DEFAULT_INITIAL_CASH, MIN_SHARES, DEFAULT_WINDOW


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
        **kwargs
    ) -> None:
        """
        启动实盘交易

        Args:
            dry_run: 是否模拟运行 (不实际下单)
            **kwargs: 券商配置参数
        """
        if dry_run:
            print("[LIVE] Running in dry-run mode (模拟运行)")
        else:
            print("[LIVE] Running in live mode (实盘运行)")
            if not isinstance(self.broker, Simulator):
                print(f"[LIVE] Connected to broker: {self.broker.name}")

        # 实盘循环
        import time
        from datetime import datetime, time as dt_time

        print(f"[LIVE] Starting live trading loop (interval: {interval}s)")

        while True:
            try:
                loop_start = time.time()

                # 检查交易时间
                now = datetime.now()
                current_time = now.time()

                # A股交易时间: 9:30-11:30, 13:00-15:00
                morning_start = dt_time(9, 30)
                morning_end = dt_time(11, 30)
                afternoon_start = dt_time(13, 0)
                afternoon_end = dt_time(15, 0)

                is_trading_time = (
                    (morning_start <= current_time <= morning_end) or
                    (afternoon_start <= current_time <= afternoon_end)
                )

                if not is_trading_time:
                    print(f"[LIVE] Outside trading hours: {current_time}")
                    time.sleep(interval)
                    continue

                # 1. 获取最新数据
                print(f"[LIVE] Fetching latest data...")
                # TODO: 实现数据获取 (需要数据源支持)

                # 2. 生成信号
                print(f"[LIVE] Generating signals...")
                # TODO: 实现信号生成 (需要策略支持)

                # 3. 执行交易
                print(f"[LIVE] Executing trades...")
                # TODO: 实现交易执行 (需要券商接口支持)

                # 4. 更新持仓
                print(f"[LIVE] Updating positions...")
                # TODO: 实现持仓更新

                # 控制循环频率
                elapsed = time.time() - loop_start
                sleep_time = max(0, interval - elapsed)

                if sleep_time > 0:
                    time.sleep(sleep_time)

            except KeyboardInterrupt:
                print("\n[LIVE] Stopped by user")
                break
            except Exception as e:
                print(f"[LIVE] Error: {e}")
                time.sleep(interval)

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

        # 生成所有参数组合
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())

        for values in product(*param_values):
            params = dict(zip(param_names, values))

            # 更新策略参数
            for name, value in params.items():
                if hasattr(self.strategy, name):
                    setattr(self.strategy, name, value)

            # 运行回测
            result = self.backtest(**backtest_kwargs)

            # 获取评分
            score = getattr(result.metrics, metric, 0)

            if score > best_score:
                best_score = score
                best_params = params
                best_result = result

        return {
            'best_params': best_params,
            'best_score': best_score,
            'best_result': best_result,
        }


