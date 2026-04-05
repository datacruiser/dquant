"""
Tushare 数据加载器

Tushare 是国内最流行的免费金融数据接口之一。
需要注册获取 token: https://tushare.pro/
"""

from datetime import datetime
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from dquant.constants import BATCH_SIZE, normalize_symbol
from dquant.data.base import DataSource
from dquant.logger import get_logger

logger = get_logger(__name__)


class TushareLoader(DataSource):
    """
    Tushare 数据加载器

    支持 A 股日线、分钟线、财务数据等。

    Usage:
        # 方式1: 设置环境变量 TUSHARE_TOKEN
        loader = TushareLoader(symbols='hs300')

        # 方式2: 直接传入 token
        loader = TushareLoader(symbols='hs300', token='your_token')

        df = loader.load()
    """

    # 常用指数代码
    INDEX_MAP = {
        "hs300": "000300.SH",
        "zz500": "000905.SH",
        "zz1000": "000852.SH",
        "sz50": "000016.SH",
        "sz100": "000043.SH",
        "cyb50": "399673.SZ",
        "kc50": "000688.SH",
    }

    def __init__(
        self,
        symbols: Union[str, List[str]] = "hs300",
        start: Optional[str] = None,
        end: Optional[str] = None,
        token: Optional[str] = None,
        freq: str = "D",  # D=日线, W=周线, M=月线, 1/5/15/30/60=分钟线
        adj: str = "qfq",  # qfq=前复权, hfq=后复权, None=不复权
        include_factors: bool = True,
        include_financial: bool = False,  # 是否包含财务数据
    ):
        super().__init__(symbols=symbols, start=start, end=end)
        self.token = token
        self.freq = freq
        self.adj = adj
        self.include_factors = include_factors
        self.include_financial = include_financial

        self._pro = None

    def _init_tushare(self):
        """初始化 Tushare"""
        if self._pro is not None:
            return

        try:
            import tushare as ts
        except ImportError:
            raise ImportError("tushare not installed. Run: pip install tushare")

        # 设置 token
        if self.token:
            ts.set_token(self.token)
        else:
            # 尝试从环境变量获取
            import os

            token = os.environ.get("TUSHARE_TOKEN")
            if token:
                ts.set_token(token)
            else:
                raise ValueError(
                    "Tushare token not found. "
                    "Set TUSHARE_TOKEN environment variable or pass token parameter."
                )

        self._pro = ts.pro_api()
        print("[Tushare] API initialized")

    def _load_single_symbol(self, symbol):
        """加载单个股票数据"""
        try:
            df = self._pro.daily(
                ts_code=symbol,
                start_date=self.start.replace("-", ""),
                end_date=self.end.replace("-", ""),
            )

            if df is None or len(df) == 0:
                return None

            df = df.rename(
                columns={
                    "trade_date": "date",
                    "vol": "volume",
                }
            )

            df["symbol"] = df["ts_code"]
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
            df = df.sort_index()

            return df[["symbol", "open", "high", "low", "close", "volume"]]

        except Exception as e:
            logger.debug(f"Failed to load symbol {symbol}: {e}")
            return None

    def load(self) -> pd.DataFrame:
        """加载数据"""
        self._init_tushare()

        # 获取股票列表
        symbol_list = self._get_symbol_list()

        all_data = []
        failed = []

        for i, symbol in enumerate(symbol_list):
            try:
                df = self._get_stock_data(symbol)
                if df is not None and len(df) > 0:
                    all_data.append(df)

                if (i + 1) % 50 == 0:
                    print(f"  [Tushare] 已加载 {i + 1}/{len(symbol_list)} 只股票")

            except Exception as e:
                failed.append((symbol, str(e)))
                continue

        if failed:
            print(f"  [Tushare] 加载失败: {len(failed)} 只")

        if not all_data:
            raise ValueError("No data loaded")

        # 合并
        result = pd.concat(all_data, axis=0, ignore_index=False)

        # 过滤日期
        if self.start:
            result = result[result.index >= pd.to_datetime(self.start)]
        if self.end:
            result = result[result.index <= pd.to_datetime(self.end)]

        # 计算因子
        if self.include_factors:
            result = self._calculate_factors(result)

        # 添加财务数据
        if self.include_financial:
            result = self._add_financial_data(result)

        self.validate(result)

        return result.sort_index()

    def _get_symbol_list(self) -> List[str]:
        """获取股票代码列表"""
        if isinstance(self.symbols, list):
            return [self._normalize_symbol(s) for s in self.symbols]

        if self.symbols in self.INDEX_MAP:
            # 获取指数成分股
            index_code = self.INDEX_MAP[self.symbols]
            return self._get_index_constituents(index_code)

        if self.symbols == "all":
            # 全市场
            return self._get_all_stocks()

        # 单只股票
        return [self._normalize_symbol(self.symbols)]

    def _normalize_symbol(self, symbol: str) -> str:
        """标准化股票代码"""
        # Tushare 返回纯数字代码，需要加后缀
        code = symbol.replace(".SH", "").replace(".SZ", "").replace(".BJ", "")
        return normalize_symbol(code)

    def _get_index_constituents(self, index_code: str) -> List[str]:
        """获取指数成分股"""
        try:
            df = self._pro.index_weight(
                index_code=index_code, start_date=self.start or "20200101"
            )
            return df["con_code"].unique().tolist()
        except Exception as e:
            # 备用方案：返回部分股票
            logger.warning(
                f"Failed to get constituents for {index_code}: {e}, using sample"
            )
            return self._get_all_stocks()[:50]

    def _get_all_stocks(self) -> List[str]:
        """获取全部 A 股"""
        df = self._pro.stock_basic(exchange="", list_status="L")
        return (df["ts_code"]).tolist()

    def _get_stock_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """获取单只股票数据"""
        try:
            # 日线数据
            if self.freq == "D":
                df = self._pro.daily(
                    ts_code=symbol,
                    start_date=(
                        self.start.replace("-", "") if self.start else "20200101"
                    ),
                    end_date=(
                        self.end.replace("-", "")
                        if self.end
                        else datetime.now().strftime("%Y%m%d")
                    ),
                )

                # 复权因子
                if self.adj:
                    adj_df = self._pro.adj_factor(ts_code=symbol)
                    df = df.merge(adj_df, on="trade_date", how="left")
                    df["adj_factor"] = df["adj_factor"].fillna(1)

                    if self.adj == "qfq":
                        df["open"] = df["open"] * df["adj_factor"]
                        df["high"] = df["high"] * df["adj_factor"]
                        df["low"] = df["low"] * df["adj_factor"]
                        df["close"] = df["close"] * df["adj_factor"]
                    elif self.adj == "hfq":
                        # 后复权：从最新日期往前累积复权因子
                        df = df.sort_values("trade_date")
                        last_factor = df["adj_factor"].iloc[-1]
                        hfq_factor = last_factor / df["adj_factor"]
                        df["open"] = df["open"] * hfq_factor
                        df["high"] = df["high"] * hfq_factor
                        df["low"] = df["low"] * hfq_factor
                        df["close"] = df["close"] * hfq_factor

            # 分钟线数据
            elif self.freq in ["1", "5", "15", "30", "60"]:
                df = self._pro.stk_mins(
                    ts_code=symbol,
                    freq=self.freq + "min",
                    start_date=self.start,
                    end_date=self.end,
                )

            else:
                df = self._pro.daily(ts_code=symbol)

            if df is None or len(df) == 0:
                return None

            # 标准化
            df = df.rename(
                columns={
                    "trade_date": "date",
                    "vol": "volume",
                    "amount": "amount",
                    "pct_chg": "pct_change",
                }
            )

            df["date"] = pd.to_datetime(df["date"])
            df["symbol"] = symbol
            df = df.set_index("date")

            return df

        except Exception as e:
            logger.debug(f"Failed to get stock data for {symbol}: {e}")
            return None

    def _calculate_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术因子"""
        results = []

        for symbol, group in df.groupby("symbol"):
            group = group.sort_index()

            # 动量因子
            group["momentum_5"] = group["close"].pct_change(5)
            group["momentum_10"] = group["close"].pct_change(10)
            group["momentum_20"] = group["close"].pct_change(20)
            group["momentum_60"] = group["close"].pct_change(60)

            # 波动率
            returns = group["close"].pct_change()
            group["volatility_5"] = returns.rolling(5).std()
            group["volatility_10"] = returns.rolling(10).std()
            group["volatility_20"] = returns.rolling(20).std()

            # 均线
            for window in [5, 10, 20, 60]:
                group[f"ma_{window}"] = group["close"].rolling(window).mean()
                group[f"bias_{window}"] = (
                    group["close"] - group[f"ma_{window}"]
                ) / group[f"ma_{window}"]

            # 成交量因子
            group["volume_ma_5"] = group["volume"].rolling(5).mean()
            group["volume_ma_10"] = group["volume"].rolling(10).mean()
            group["volume_ratio"] = group["volume"] / group["volume_ma_5"].replace(
                0, np.nan
            )

            # 价格位置
            group["price_position_20"] = (
                group["close"] - group["low"].rolling(20).min()
            ) / (group["high"].rolling(20).max() - group["low"].rolling(20).min())

            results.append(group)

        return pd.concat(results)

    def _add_financial_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加财务数据"""
        # 获取所有股票代码
        symbols = df["symbol"].unique()

        try:
            # 获取财务指标
            fin_df = self._pro.fina_indicator(
                ts_code=",".join(symbols[:BATCH_SIZE]),  # 限制数量
                start_date=self.start.replace("-", "") if self.start else "20200101",
            )

            if fin_df is not None and len(fin_df) > 0:
                # 合并财务数据
                # 简化处理：只保留最新财务数据
                fin_df = fin_df.sort_values("end_date").groupby("ts_code").last()

                # TODO: 合并到主数据
                pass
        except Exception as e:
            logger.warning(f"Failed to add financial data: {e}")

        return df

    def get_realtime_quotes(self, symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """获取实时行情"""
        self._init_tushare()

        if symbols is None:
            symbols = self._get_symbol_list()[:BATCH_SIZE]  # 限制数量

        try:
            df = self._pro.daily(
                ts_code=",".join(symbols),
                trade_date=datetime.now().strftime("%Y%m%d"),
            )
            return df
        except Exception as e:
            logger.warning(f"Failed to get realtime quotes: {e}")
            return pd.DataFrame()


class TushareFinancial:
    """
    Tushare 财务数据接口
    """

    def __init__(self, token: Optional[str] = None):
        self.token = token
        self._pro = None

    def _init(self):
        if self._pro is None:
            import tushare as ts

            if self.token:
                ts.set_token(self.token)
            self._pro = ts.pro_api()

    def get_balance_sheet(self, symbol: str) -> pd.DataFrame:
        """获取资产负债表"""
        self._init()
        return self._pro.balancesheet(ts_code=symbol)

    def get_income_statement(self, symbol: str) -> pd.DataFrame:
        """获取利润表"""
        self._init()
        return self._pro.income(ts_code=symbol)

    def get_cash_flow(self, symbol: str) -> pd.DataFrame:
        """获取现金流量表"""
        self._init()
        return self._pro.cashflow(ts_code=symbol)

    def get_financial_indicator(self, symbol: str) -> pd.DataFrame:
        """获取财务指标"""
        self._init()
        return self._pro.fina_indicator(ts_code=symbol)

    def get_daily_basic(self, symbol: str) -> pd.DataFrame:
        """获取每日指标 (PE/PB/PS等)"""
        self._init()
        return self._pro.daily_basic(ts_code=symbol)
