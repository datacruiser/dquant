"""
AKShare 数据加载器
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from dquant.data.base import DataSource
from dquant.data.factors_utils import calculate_common_factors
from dquant.data.rate_limiter import RateLimiter
from dquant.logger import get_logger

logger = get_logger(__name__)


class AKShareLoader(DataSource):
    """
    AKShare 数据加载器

    支持 A 股实时和历史数据。

    Usage:
        # 获取沪深300成分股
        loader = AKShareLoader(symbols='hs300')
        df = loader.load()

        # 获取指定股票
        loader = AKShareLoader(symbols=['000001', '000002'])
        df = loader.load()

        # 获取全市场
        loader = AKShareLoader(symbols='all')
        df = loader.load()
    """

    # 常用指数成分股
    INDEX_SYMBOLS = {
        "hs300": "000300.SH",  # 沪深300
        "zz500": "000905.SH",  # 中证500
        "zz1000": "000852.SH",  # 中证1000
        "sz50": "000016.SH",  # 上证50
    }

    def __init__(
        self,
        symbols: Union[str, List[str]] = "hs300",
        start: Optional[str] = None,
        end: Optional[str] = None,
        freq: str = "d",  # d=日线, w=周线, m=月线
        adjust: str = "qfq",  # qfq=前复权, hfq=后复权, None=不复权
        include_factors: bool = True,  # 是否计算因子
        max_workers: int = 5,  # 并发加载数
        rate_limit: int = 10,  # 每秒最大请求数
    ):
        super().__init__(symbols=symbols, start=start, end=end)
        self.freq = freq
        self.adjust = adjust
        self.include_factors = include_factors
        self.max_workers = max_workers
        self._rate_limiter = RateLimiter(max_calls=rate_limit, period=1.0)

    def load(self) -> pd.DataFrame:
        """加载数据（并发）"""
        try:
            import akshare as ak  # noqa: F401
        except ImportError:
            raise ImportError("akshare not installed. Run: pip install akshare")

        # 获取股票列表
        symbol_list = self._get_symbol_list()
        logger.info(f"[AKShare] 开始加载 {len(symbol_list)} 只股票 (workers={self.max_workers})")

        all_data = []
        failed = []
        completed = 0

        def _fetch_one(symbol):
            """获取单个股票数据（带速率限制）"""
            self._rate_limiter.wait()
            return symbol, self._get_stock_data(symbol)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(_fetch_one, s): s for s in symbol_list}

            for future in as_completed(futures):
                completed += 1
                try:
                    symbol, df = future.result()
                    if df is not None and len(df) > 0:
                        all_data.append(df)
                    else:
                        failed.append(symbol)
                except Exception:
                    sym = futures[future]
                    logger.warning(f"[AKShare] 并发加载 {sym} 失败")
                    failed.append(sym)

                if completed % 50 == 0:
                    logger.info(f"[AKShare] 已加载 {completed}/{len(symbol_list)} 只股票")

        if failed:
            logger.warning(f"[AKShare] 加载失败: {len(failed)} 只")

        if not all_data:
            raise ValueError("No data loaded")

        # 合并数据
        result = pd.concat(all_data, axis=0, ignore_index=False)

        # 过滤日期
        if self.start:
            result = result[result.index >= pd.to_datetime(self.start)]
        if self.end:
            result = result[result.index <= pd.to_datetime(self.end)]

        # 计算因子
        if self.include_factors:
            result = self._calculate_factors(result)

        # 验证格式
        self.validate(result)

        logger.info(f"[AKShare] 加载完成: {len(result)} 行, {result['symbol'].nunique()} 只股票")
        return result.sort_index()

    def _get_symbol_list(self) -> List[str]:
        """获取股票代码列表"""
        import akshare as ak

        if isinstance(self.symbols, list):
            return self.symbols

        if self.symbols == "all":
            # 全市场 A 股
            df = ak.stock_zh_a_spot_em()
            return df["代码"].tolist()

        if self.symbols in self.INDEX_SYMBOLS:
            # 指数成分股
            index_name = self.symbols
            if index_name == "hs300":
                df = ak.index_stock_cons_weight_csindex(symbol="000300")
                return df["成分券代码"].tolist()
            elif index_name == "zz500":
                df = ak.index_stock_cons_weight_csindex(symbol="000905")
                return df["成分券代码"].tolist()
            elif index_name == "zz1000":
                df = ak.index_stock_cons_weight_csindex(symbol="000852")
                return df["成分券代码"].tolist()
            elif index_name == "sz50":
                df = ak.index_stock_cons_weight_csindex(symbol="000016")
                return df["成分券代码"].tolist()

        # 单只股票
        if self.symbols is None:
            raise ValueError("symbols 参数不能为 None")
        return [self.symbols]

    def _get_stock_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """获取单只股票数据"""
        import akshare as ak

        try:
            # 东方财富日线
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period=self.freq,
                start_date=self.start or "20200101",
                end_date=self.end or datetime.now().strftime("%Y%m%d"),
                adjust=self.adjust,
            )

            if df is None or len(df) == 0:
                return None

            # 标准化列名
            df = df.rename(
                columns={
                    "日期": "date",
                    "开盘": "open",
                    "最高": "high",
                    "最低": "low",
                    "收盘": "close",
                    "成交量": "volume",
                    "成交额": "amount",
                    "振幅": "amplitude",
                    "涨跌幅": "pct_change",
                    "涨跌额": "change",
                    "换手率": "turnover",
                }
            )

            df["date"] = pd.to_datetime(df["date"])
            df["symbol"] = symbol

            # 添加后缀
            if symbol.startswith("6"):
                df["symbol"] = symbol + ".SH"
            elif symbol.startswith(("4", "8")):
                df["symbol"] = symbol + ".BJ"
            else:
                df["symbol"] = symbol + ".SZ"

            df = df.set_index("date")

            return df

        except Exception:
            logger.warning(f"[AKShare] _get_stock_data 加载 {symbol} 失败")
            return None

    def _calculate_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术因子"""
        # 通用因子：momentum 5/10/20, volatility 5/10/20, ma 5/10/20, bias 5/10/20, volume_ma_5, volume_ratio
        df = calculate_common_factors(df)

        # AKShare 特有因子：换手率均线
        if "turnover" in df.columns:
            for symbol, group in df.groupby("symbol"):
                group = group.sort_index()
                df.loc[group.index, "turnover_ma_5"] = group["turnover"].rolling(5).mean()
                df.loc[group.index, "turnover_ma_10"] = group["turnover"].rolling(10).mean()

        return df


class AKShareRealTime:
    """
    AKShare 实时数据
    """

    @staticmethod
    def get_realtime_quotes(symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """获取实时行情"""
        try:
            import akshare as ak
        except ImportError:
            raise ImportError("akshare not installed")

        df = ak.stock_zh_a_spot_em()

        if symbols:
            df = df[df["代码"].isin(symbols)]

        df = df.rename(
            columns={
                "代码": "symbol",
                "名称": "name",
                "最新价": "price",
                "涨跌幅": "pct_change",
                "涨跌额": "change",
                "成交量": "volume",
                "成交额": "amount",
                "最高": "high",
                "最低": "low",
                "今开": "open",
                "昨收": "pre_close",
            }
        )

        return df

    @staticmethod
    def get_realtime_quote(symbol: str) -> dict:
        """获取单只股票实时行情"""
        df = AKShareRealTime.get_realtime_quotes([symbol])
        if len(df) > 0:
            return df.iloc[0].to_dict()
        return {}
