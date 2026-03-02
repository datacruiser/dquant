"""
数据源基类
"""

from abc import ABC, abstractmethod
from typing import Optional, List
import pandas as pd


class DataSource(ABC):
    """
    数据源基类

    所有数据源都需要实现 load() 方法返回标准格式的 DataFrame。

    标准格式:
        index: DatetimeIndex (交易日期)
        columns:
            - symbol: 股票代码
            - open: 开盘价
            - high: 最高价
            - low: 最低价
            - close: 收盘价
            - volume: 成交量
            - (可选) 其他因子列
    """

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ):
        self.symbols = symbols
        self.start = start
        self.end = end

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """加载数据，返回标准格式 DataFrame"""
        pass

    def validate(self, df: pd.DataFrame) -> bool:
        """验证数据格式"""
        required_cols = ['symbol', 'open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        return True
