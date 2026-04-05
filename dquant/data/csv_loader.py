"""
CSV 数据加载器
"""

from typing import List, Union

import pandas as pd

from dquant.data.base import DataSource
from dquant.logger import get_logger

logger = get_logger(__name__)


class CSVLoader(DataSource):
    """
    CSV 文件数据加载器

    支持单文件或多文件加载，自动合并。

    Usage:
        # 单文件
        loader = CSVLoader("data/stocks.csv")
        df = loader.load()

        # 多文件
        loader = CSVLoader(["data/000001.csv", "data/000002.csv"])
        df = loader.load()
    """

    def __init__(
        self,
        path: Union[str, List[str]],
        date_col: str = "date",
        symbol_col: str = "symbol",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.path = path if isinstance(path, list) else [path]
        self.date_col = date_col
        self.symbol_col = symbol_col

    def load(self) -> pd.DataFrame:
        """加载 CSV 文件"""
        dfs = []

        for p in self.path:
            try:
                df = pd.read_csv(p)
            except FileNotFoundError:
                logger.error(f"CSV 文件不存在: {p}")
                raise
            except pd.errors.EmptyDataError:
                logger.error(f"CSV 文件为空: {p}")
                raise
            except Exception as e:
                logger.error(f"读取 CSV 文件失败: {p} — {e}")
                raise

            # 转换日期列
            if self.date_col in df.columns:
                df[self.date_col] = pd.to_datetime(df[self.date_col])
                df = df.set_index(self.date_col)

            dfs.append(df)

        # 合并所有数据
        result = pd.concat(dfs, axis=0)

        # 过滤日期范围
        if self.start:
            result = result[result.index >= pd.to_datetime(self.start)]
        if self.end:
            result = result[result.index <= pd.to_datetime(self.end)]

        # 过滤股票
        if self.symbols:
            result = result[result[self.symbol_col].isin(self.symbols)]

        # 验证格式
        self.validate(result)

        return result.sort_index()
