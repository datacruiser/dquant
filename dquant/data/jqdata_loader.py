"""
聚宽 JoinQuant 数据加载器

聚宽是国内流行的量化平台，提供丰富的金融数据。
需要注册账号: https://www.joinquant.com/
"""

from typing import Optional, List, Union
from datetime import datetime
import pandas as pd
import numpy as np

from dquant.data.base import DataSource


class JQDataLoader(DataSource):
    """
    聚宽数据加载器

    支持 A 股、期货、基金、指数等数据。

    Usage:
        loader = JQDataLoader(
            symbols='hs300',
            account='your_phone',
            password='your_password',
        )
        df = loader.load()
    """

    # 指数代码映射
    INDEX_MAP = {
        'hs300': '000300.XSHG',
        'zz500': '000905.XSHG',
        'zz1000': '000852.XSHG',
        'sz50': '000016.XSHG',
    }

    def __init__(
        self,
        symbols: Union[str, List[str]] = 'hs300',
        start: Optional[str] = None,
        end: Optional[str] = None,
        account: Optional[str] = None,
        password: Optional[str] = None,
        freq: str = 'daily',  # daily, minute
        fq: str = 'pre',  # pre=前复权, post=后复权, None=不复权
        include_factors: bool = True,
    ):
        super().__init__(symbols=symbols, start=start, end=end)
        self.account = account
        self.password = password
        self.freq = freq
        self.fq = fq
        self.include_factors = include_factors

        self._jq = None

    def _init_jqdata(self):
        """初始化聚宽数据"""
        if self._jq is not None:
            return

        try:
            import jqdatasdk as jq
        except ImportError:
            raise ImportError("jqdatasdk not installed. Run: pip install jqdatasdk")

        # 登录
        import os
        account = self.account or os.environ.get('JQDATA_ACCOUNT')
        password = self.password or os.environ.get('JQDATA_PASSWORD')

        if not account or not password:
            raise ValueError(
                "JQData account not found. "
                "Set JQDATA_ACCOUNT and JQDATA_PASSWORD environment variables."
            )

        jq.auth(account, password)
        self._jq = jq
        print("[JQData] Authenticated")

    def load(self) -> pd.DataFrame:
        """加载数据"""
        self._init_jqdata()

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
                    print(f"  [JQData] 已加载 {i+1}/{len(symbol_list)} 只股票")

            except Exception as e:
                failed.append((symbol, str(e)))

        if failed:
            print(f"  [JQData] 加载失败: {len(failed)} 只")

        if not all_data:
            raise ValueError("No data loaded")

        result = pd.concat(all_data, axis=0, ignore_index=False)

        # 计算因子
        if self.include_factors:
            result = self._calculate_factors(result)

        self.validate(result)

        return result.sort_index()

    def _get_symbol_list(self) -> List[str]:
        """获取股票代码列表"""
        if isinstance(self.symbols, list):
            return [self._normalize_symbol(s) for s in self.symbols]

        if self.symbols in self.INDEX_MAP:
            index_code = self.INDEX_MAP[self.symbols]
            return self._get_index_constituents(index_code)

        if self.symbols == 'all':
            return self._get_all_stocks()

        return [self._normalize_symbol(self.symbols)]

    def _normalize_symbol(self, symbol: str) -> str:
        """标准化股票代码"""
        symbol = symbol.replace('.SH', '.XSHG').replace('.SZ', '.XSHE')
        if '.' not in symbol:
            if symbol.startswith('6'):
                return symbol + '.XSHG'
            else:
                return symbol + '.XSHE'
        return symbol

    def _get_index_constituents(self, index_code: str) -> List[str]:
        """获取指数成分股"""
        try:
            df = self._jq.get_index_stocks(index_code, date=self.end)
            return df['code'].tolist()
        except Exception:
            return self._get_all_stocks()[:50]

    def _get_all_stocks(self) -> List[str]:
        """获取全部 A 股"""
        df = self._jq.get_all_securities(types=['stock'])
        return df.index.tolist()

    def _get_stock_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """获取单只股票数据"""
        try:
            # 日线数据
            df = self._jq.get_price(
                symbol,
                start_date=self.start or '2020-01-01',
                end_date=self.end or datetime.now().strftime('%Y-%m-%d'),
                frequency=self.freq,
                fq=self.fq,
            )

            if df is None or len(df) == 0:
                return None

            # 标准化
            df = df.reset_index()
            df = df.rename(columns={
                'time': 'date',
                'money': 'amount',
            })

            df['date'] = pd.to_datetime(df['date'])
            df['symbol'] = symbol.replace('.XSHG', '.SH').replace('.XSHE', '.SZ')
            df = df.set_index('date')

            return df

        except Exception as e:
            return None

    def _calculate_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术因子"""
        results = []

        for symbol, group in df.groupby('symbol'):
            group = group.sort_index()

            # 动量
            group['momentum_5'] = group['close'].pct_change(5)
            group['momentum_10'] = group['close'].pct_change(10)
            group['momentum_20'] = group['close'].pct_change(20)

            # 波动率
            returns = group['close'].pct_change()
            group['volatility_20'] = returns.rolling(20).std()

            # 均线
            group['ma_5'] = group['close'].rolling(5).mean()
            group['ma_10'] = group['close'].rolling(10).mean()
            group['ma_20'] = group['close'].rolling(20).mean()

            results.append(group)

        return pd.concat(results)

    def get_factor(self, symbol: str, factor_name: str) -> pd.DataFrame:
        """获取聚宽因子"""
        self._init_jqdata()

        try:
            df = self._jq.get_factor_values(
                [symbol],
                [factor_name],
                start_date=self.start,
                end_date=self.end,
            )
            return df
        except Exception:
            return pd.DataFrame()


class JQDataFactor:
    """
    聚宽因子数据
    """

    def __init__(self, account: str = None, password: str = None):
        self.account = account
        self.password = password
        self._jq = None

    def _init(self):
        if self._jq is None:
            import jqdatasdk as jq
            import os
            account = self.account or os.environ.get('JQDATA_ACCOUNT')
            password = self.password or os.environ.get('JQDATA_PASSWORD')
            jq.auth(account, password)
            self._jq = jq

    def get_financial_indicator(self, symbol: str, indicator: str) -> pd.Series:
        """获取财务指标"""
        self._init()
        # 聚宽支持的财务因子
        # indicator: roe, inc_revenue_year_on_year, inc_net_profit_year_on_year, etc.
        df = self._jq.financial.run_query(
            self._jq.financial_indicator
            .filter(self._jq.financial_indicator.code == symbol)
        )
        return df[indicator] if indicator in df.columns else pd.Series()

    def get_valuation(self, symbol: str) -> pd.DataFrame:
        """获取估值数据 (PE/PB/PS等)"""
        self._init()
        df = self._jq.get_factor_values(
            [symbol],
            ['pe_ratio', 'pb_ratio', 'ps_ratio', 'market_cap'],
            start_date='2020-01-01',
            end_date=datetime.now().strftime('%Y-%m-%d'),
        )
        return df
