"""
资金流数据加载器

基于 AKShare 的免费资金流数据
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Union
from datetime import datetime
import logging

from dquant.data.base import DataSource

logger = logging.getLogger(__name__)


class MoneyFlowLoader(DataSource):
    """
    资金流数据加载器

    使用 AKShare 获取个股资金流数据。

    数据字段:
    - close: 收盘价
    - change_pct: 涨跌幅
    - main_net_inflow: 主力净流入
    - main_net_inflow_pct: 主力净流入占比
    - super_large_net_inflow: 超大单净流入
    - large_net_inflow: 大单净流入
    - medium_net_inflow: 中单净流入
    - small_net_inflow: 小单净流入

    Usage:
        loader = MoneyFlowLoader(symbols=['000001', '600000'])
        df = loader.load()
    """

    def __init__(
        self,
        symbols: Union[str, List[str]],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)

        if isinstance(symbols, str):
            symbols = [symbols]

        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date

    def _get_market(self, symbol: str) -> str:
        """判断市场"""
        code = symbol.split('.')[0]
        if code.startswith('6'):
            return 'sh'
        elif code.startswith(('0', '3')):
            return 'sz'
        else:
            return 'sh'  # 默认

    def load(self) -> pd.DataFrame:
        """
        加载资金流数据

        Returns:
            DataFrame with columns:
            - date: 日期 (index)
            - symbol: 股票代码
            - close: 收盘价
            - change_pct: 涨跌幅
            - main_net_inflow: 主力净流入
            - main_net_inflow_pct: 主力净流入占比
            - super_large_net_inflow: 超大单净流入
            - large_net_inflow: 大单净流入
            - medium_net_inflow: 中单净流入
            - small_net_inflow: 小单净流入
        """
        try:
            import akshare as ak
        except ImportError:
            logger.error("请安装 akshare: pip install akshare")
            raise ImportError("akshare not installed")

        all_data = []

        for symbol in self.symbols:
            try:
                # 清理股票代码
                code = symbol.split('.')[0]
                market = self._get_market(symbol)

                logger.info(f"获取 {symbol} 资金流数据...")

                # 调用 AKShare 接口
                df = ak.stock_individual_fund_flow(stock=code, market=market)

                # 重命名列
                df = df.rename(columns={
                    '日期': 'date',
                    '收盘价': 'close',
                    '涨跌幅': 'change_pct',
                    '主力净流入-净额': 'main_net_inflow',
                    '主力净流入-净占比': 'main_net_inflow_pct',
                    '超大单净流入-净额': 'super_large_net_inflow',
                    '超大单净流入-净占比': 'super_large_net_inflow_pct',
                    '大单净流入-净额': 'large_net_inflow',
                    '大单净流入-净占比': 'large_net_inflow_pct',
                    '中单净流入-净额': 'medium_net_inflow',
                    '中单净流入-净占比': 'medium_net_inflow_pct',
                    '小单净流入-净额': 'small_net_inflow',
                    '小单净流入-净占比': 'small_net_inflow_pct',
                })

                # 添加股票代码
                df['symbol'] = symbol

                # 转换日期
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')

                # 过滤日期范围
                if self.start_date:
                    df = df[df.index >= pd.to_datetime(self.start_date)]
                if self.end_date:
                    df = df[df.index <= pd.to_datetime(self.end_date)]

                all_data.append(df)
                logger.info(f"✓ {symbol}: 获取 {len(df)} 条记录")

            except Exception as e:
                logger.error(f"✗ {symbol}: {e}")
                continue

        if not all_data:
            raise ValueError("没有获取到任何数据")

        # 合并所有数据
        result = pd.concat(all_data, axis=0)
        result = result.sort_index()

        logger.info(f"总计获取 {len(result)} 条记录")

        return result


class MockMoneyFlowLoader(DataSource):
    """
    模拟资金流数据 (用于测试)

    当没有安装 AKShare 或无法获取真实数据时使用。
    """

    def __init__(
        self,
        symbols: Union[str, List[str]],
        start_date: str = '2023-01-01',
        end_date: str = '2023-12-31',
        **kwargs
    ):
        super().__init__(**kwargs)

        if isinstance(symbols, str):
            symbols = [symbols]

        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date

    def load(self) -> pd.DataFrame:
        """生成模拟资金流数据"""
        np.random.seed(42)

        dates = pd.date_range(self.start_date, self.end_date, freq='B')
        all_data = []

        for symbol in self.symbols:
            price = 10 + np.random.randn() * 5

            for date in dates:
                # 生成价格
                ret = np.random.randn() * 0.02
                price *= (1 + ret)

                # 生成资金流数据 (单位: 万元)
                main_flow = np.random.randn() * 1000
                super_large = np.random.randn() * 500
                large = np.random.randn() * 300
                medium = np.random.randn() * 200
                small = -(main_flow + super_large + large + medium)  # 总和为0

                all_data.append({
                    'date': date,
                    'symbol': symbol,
                    'close': price,
                    'change_pct': ret * 100,
                    'main_net_inflow': main_flow,
                    'main_net_inflow_pct': main_flow / (price * 100) * 100,
                    'super_large_net_inflow': super_large,
                    'super_large_net_inflow_pct': super_large / (price * 100) * 100,
                    'large_net_inflow': large,
                    'large_net_inflow_pct': large / (price * 100) * 100,
                    'medium_net_inflow': medium,
                    'medium_net_inflow_pct': medium / (price * 100) * 100,
                    'small_net_inflow': small,
                    'small_net_inflow_pct': small / (price * 100) * 100,
                })

        df = pd.DataFrame(all_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

        return df.sort_index()
