"""
Yahoo Finance 数据加载器

支持全球股票、指数、ETF、外汇、加密货币等。
"""

from datetime import datetime
from typing import List, Optional, Union

import pandas as pd

from dquant.data.base import DataSource
from dquant.data.factors_utils import (
    calculate_bollinger,
    calculate_common_factors,
    calculate_macd,
    calculate_rsi,
)


class YahooLoader(DataSource):
    """
    Yahoo Finance 数据加载器

    支持美股、港股、A股(部分)、ETF、指数、加密货币等。

    Usage:
        # 美股
        loader = YahooLoader(symbols=['AAPL', 'MSFT', 'GOOGL'])

        # 港股
        loader = YahooLoader(symbols=['0700.HK', '9988.HK'])

        # A股 (部分)
        loader = YahooLoader(symbols=['600519.SS', '000001.SZ'])

        # ETF
        loader = YahooLoader(symbols=['SPY', 'QQQ', 'GLD'])

        # 加密货币
        loader = YahooLoader(symbols=['BTC-USD', 'ETH-USD'])

        df = loader.load()
    """

    # 常用标的
    POPULAR_SYMBOLS = {
        # 美股指数
        "sp500": "^GSPC",
        "nasdaq": "^IXIC",
        "dow": "^DJI",
        "russell": "^RUT",
        # 港股指数
        "hsi": "^HSI",
        "hscei": "^HSCE",
        # A股指数
        "sse": "000001.SS",
        "csi300": "000300.SS",
        # 大宗商品
        "gold": "GC=F",
        "silver": "SI=F",
        "oil": "CL=F",
        "natural_gas": "NG=F",
        # 加密货币
        "btc": "BTC-USD",
        "eth": "ETH-USD",
        # 美股龙头
        "mag7": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"],
    }

    def __init__(
        self,
        symbols: Union[str, List[str]],
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d",  # 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
        include_factors: bool = True,
        auto_adjust: bool = True,  # 自动复权
    ):
        super().__init__(symbols=symbols, start=start, end=end)
        self.interval = interval
        self.include_factors = include_factors
        self.auto_adjust = auto_adjust

    def load(self) -> pd.DataFrame:
        """加载数据"""
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance not installed. Run: pip install yfinance")

        # 处理预设标的
        symbol_list = self._get_symbol_list()

        all_data = []
        failed = []

        for symbol in symbol_list:
            try:
                df = self._get_data(yf, symbol)
                if df is not None and len(df) > 0:
                    all_data.append(df)
            except Exception as e:
                failed.append((symbol, str(e)))

        if failed:
            print(f"  [Yahoo] 加载失败: {len(failed)} 只")

        if not all_data:
            raise ValueError("No data loaded")

        result = pd.concat(all_data, axis=0, ignore_index=False)

        # 计算因子
        if self.include_factors:
            result = self._calculate_factors(result)

        self.validate(result)

        return result.sort_index()

    def _get_symbol_list(self) -> List[str]:
        """获取标的列表"""
        if isinstance(self.symbols, list):
            return self.symbols

        if self.symbols in self.POPULAR_SYMBOLS:
            symbols = self.POPULAR_SYMBOLS[self.symbols]
            if isinstance(symbols, list):
                return symbols
            return [symbols]

        return [self.symbols]

    def _get_data(self, yf, symbol: str) -> Optional[pd.DataFrame]:
        """获取单只标的数据"""
        try:
            ticker = yf.Ticker(symbol)

            # 下载历史数据
            df = ticker.history(
                start=self.start or "2020-01-01",
                end=self.end or datetime.now().strftime("%Y-%m-%d"),
                interval=self.interval,
                auto_adjust=self.auto_adjust,
            )

            if df is None or len(df) == 0:
                return None

            # 标准化列名
            df = df.rename(
                columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume",
                    "Dividends": "dividend",
                    "Stock Splits": "stock_splits",
                }
            )

            df["symbol"] = symbol
            df.index.name = "date"

            return df

        except Exception:
            return None

    def _calculate_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术因子"""
        # 通用因子：momentum 5/10/20, volatility 5/10/20, ma 5/10/20, bias 5/10/20,
        # volume_ma_5, volume_ratio
        df = calculate_common_factors(df)

        # Yahoo 扩展因子：额外的窗口 + RSI / MACD / Bollinger
        def _add_yahoo_factors(group: pd.DataFrame) -> pd.DataFrame:
            group = group.sort_index()
            close = group["close"]
            returns = close.pct_change()

            # 扩展波动率窗口
            group["volatility_60"] = returns.rolling(60).std()

            # 扩展均线窗口
            for w in [50, 200]:
                group[f"ma_{w}"] = close.rolling(w).mean()
                group[f"bias_{w}"] = (close - group[f"ma_{w}"]) / group[f"ma_{w}"]

            # RSI / MACD / Bollinger
            group["rsi_14"] = calculate_rsi(close, 14)
            macd, signal, hist = calculate_macd(close)
            group["macd"] = macd
            group["macd_signal"] = signal
            group["macd_hist"] = hist
            upper, middle, lower = calculate_bollinger(close)
            group["bollinger_upper"] = upper
            group["bollinger_middle"] = middle
            group["bollinger_lower"] = lower
            return group

        return df.groupby("symbol", group_keys=False).apply(_add_yahoo_factors)


class YahooRealTime:
    """
    Yahoo Finance 实时数据
    """

    @staticmethod
    def get_quote(symbol: str) -> dict:
        """获取实时行情"""
        try:
            import yfinance as yf

            ticker = yf.Ticker(symbol)
            info = ticker.info

            return {
                "symbol": symbol,
                "name": info.get("shortName", ""),
                "price": info.get("currentPrice", info.get("regularMarketPrice", 0)),
                "change": info.get("regularMarketChange", 0),
                "change_pct": info.get("regularMarketChangePercent", 0),
                "volume": info.get("regularMarketVolume", 0),
                "market_cap": info.get("marketCap", 0),
                "pe": info.get("trailingPE", 0),
                "pb": info.get("priceToBook", 0),
            }
        except Exception:
            return {}

    @staticmethod
    def get_quotes(symbols: List[str]) -> pd.DataFrame:
        """批量获取行情"""
        results = []
        for symbol in symbols:
            quote = YahooRealTime.get_quote(symbol)
            if quote:
                results.append(quote)
        return pd.DataFrame(results)

    @staticmethod
    def get_fx_rate(from_currency: str, to_currency: str = "USD") -> float:
        """获取汇率"""
        try:
            import yfinance as yf

            pair = f"{from_currency}{to_currency}=X"
            ticker = yf.Ticker(pair)
            return ticker.history(period="1d")["Close"].iloc[-1]
        except Exception:
            return 0
