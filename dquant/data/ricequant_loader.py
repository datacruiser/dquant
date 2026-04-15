"""
米筐 RiceQuant 数据加载器

米筐是另一个流行的量化平台。
需要注册账号: https://www.ricequant.com/
"""

from datetime import datetime
from typing import List, Optional, Union

import pandas as pd

from dquant.constants import normalize_symbol
from dquant.data.base import DataSource
from dquant.data.factors_utils import calculate_common_factors
from dquant.logger import get_logger

logger = get_logger(__name__)


class RiceQuantLoader(DataSource):
    """
    米筐数据加载器

    Usage:
        loader = RiceQuantLoader(
            symbols='hs300',
            token='your_token',
        )
        df = loader.load()
    """

    INDEX_MAP = {
        "hs300": "000300.XSHG",
        "zz500": "000905.XSHG",
        "zz1000": "000852.XSHG",
        "sz50": "000016.XSHG",
    }

    def __init__(
        self,
        symbols: Union[str, List[str]] = "hs300",
        start: Optional[str] = None,
        end: Optional[str] = None,
        token: Optional[str] = None,
        include_factors: bool = True,
    ):
        super().__init__(symbols=symbols, start=start, end=end)
        self.token = token
        self.include_factors = include_factors

    def load(self) -> pd.DataFrame:
        """加载数据"""
        try:
            import rqdatac as rq
        except ImportError:
            raise ImportError("rqdatac not installed. Run: pip install rqdatac")

        # 初始化
        import os

        token = self.token or os.environ.get("RICEQUANT_TOKEN")
        if token:
            rq.init(token)

        # 获取股票列表
        symbol_list = self._get_symbol_list(rq)

        # 批量获取数据
        all_data = []

        # 米筐支持批量获取
        try:
            df = rq.get_price(
                symbol_list,
                start_date=self.start or "2020-01-01",
                end_date=self.end or datetime.now().strftime("%Y-%m-%d"),
                frequency="1d",
                fields=["open", "high", "low", "close", "volume", "total_turnover"],
            )

            if df is not None and len(df) > 0:
                # 重置索引
                df = df.reset_index()
                df = df.rename(
                    columns={
                        "order_book_id": "symbol",
                        "date": "date",
                        "total_turnover": "amount",
                    }
                )

                # 标准化股票代码
                df["symbol"] = df["symbol"].str.replace(".XSHG", ".SH").str.replace(".XSHE", ".SZ")
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")

                all_data.append(df)

        except Exception as e:
            logger.warning(f"  [RiceQuant] Error: {e}")

        if not all_data:
            raise ValueError("No data loaded")

        result = pd.concat(all_data, axis=0)

        # 计算因子
        if self.include_factors:
            result = self._calculate_factors(result)

        self.validate(result)

        return result.sort_index()

    def _get_symbol_list(self, rq) -> List[str]:
        """获取股票代码列表"""
        if isinstance(self.symbols, list):
            return [self._normalize_symbol(s) for s in self.symbols]

        if self.symbols in self.INDEX_MAP:
            index_code = self.INDEX_MAP[self.symbols]
            try:
                df = rq.index_components(index_code)
                return df.index.tolist() if hasattr(df, "index") else []
            except Exception as e:
                logger.warning(f"Failed to get index components for {index_code}: {e}")

        if self.symbols == "all":
            df = rq.all_instruments(type="CS")
            return df["order_book_id"].tolist()

        return [self._normalize_symbol(self.symbols)]

    def _normalize_symbol(self, symbol: str) -> str:
        """标准化股票代码 → RiceQuant 格式 (.XSHG/.XSHE)"""
        std = normalize_symbol(symbol)
        return std.replace(".SH", ".XSHG").replace(".SZ", ".XSHE")

    def _calculate_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术因子"""
        return calculate_common_factors(df)
