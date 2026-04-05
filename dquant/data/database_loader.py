"""
数据库数据加载器

支持 MySQL、PostgreSQL、SQLite 等数据库。
"""

import re
from typing import List, Optional

import pandas as pd

from dquant.data.base import DataSource

# SQL 标识符白名单校验
_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


class DatabaseLoader(DataSource):
    """
    数据库数据加载器

    从 SQL 数据库加载行情数据。

    Usage:
        # MySQL
        loader = DatabaseLoader(
            connection_string='mysql://user:pass@localhost/stock',
            table='daily_quotes',
            symbols=['000001', '000002'],
        )

        # SQLite
        loader = DatabaseLoader(
            connection_string='sqlite:///data/stock.db',
            table='daily_quotes',
        )

        df = loader.load()
    """

    def __init__(
        self,
        connection_string: str,
        table: str = "daily_quotes",
        symbols: Optional[List[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        date_col: str = "date",
        symbol_col: str = "symbol",
        include_factors: bool = True,
    ):
        super().__init__(symbols=symbols, start=start, end=end)
        self.connection_string = connection_string
        self.table = table
        self.date_col = date_col
        self.symbol_col = symbol_col
        self.include_factors = include_factors

        self._engine = None

    def _init_db(self):
        """初始化数据库连接"""
        if self._engine is not None:
            return

        try:
            from sqlalchemy import create_engine
        except ImportError:
            raise ImportError("sqlalchemy not installed. Run: pip install sqlalchemy")

        self._engine = create_engine(self.connection_string)

    def load(self) -> pd.DataFrame:
        """加载数据"""
        self._init_db()

        # 校验 SQL 标识符（表名、列名）
        for name, label in [
            (self.table, "table"),
            (self.date_col, "date_col"),
            (self.symbol_col, "symbol_col"),
        ]:
            if not _IDENTIFIER_RE.match(name):
                raise ValueError(f"Invalid SQL identifier for {label}: {name}")

        # 使用参数化查询防止 SQL 注入
        from sqlalchemy import text

        query_parts = [f"SELECT * FROM {self.table} WHERE 1=1"]
        params = {}

        if self.start:
            query_parts.append(f" AND {self.date_col} >= :start")
            params["start"] = self.start
        if self.end:
            query_parts.append(f" AND {self.date_col} <= :end")
            params["end"] = self.end
        if self.symbols:
            placeholders = ", ".join([f":sym_{i}" for i in range(len(self.symbols))])
            query_parts.append(f" AND {self.symbol_col} IN ({placeholders})")
            for i, symbol in enumerate(self.symbols):
                params[f"sym_{i}"] = symbol

        query = text("".join(query_parts))

        # 执行查询
        df = pd.read_sql(query, self._engine, params=params)

        if len(df) == 0:
            raise ValueError("No data loaded")

        # 标准化
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        df = df.set_index(self.date_col)

        # 计算因子
        if self.include_factors:
            df = self._calculate_factors(df)

        self.validate(df)

        return df.sort_index()

    def _calculate_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术因子"""
        results = []

        for symbol, group in df.groupby(self.symbol_col):
            group = group.sort_index()

            if "close" in group.columns:
                group["momentum_5"] = group["close"].pct_change(5)
                group["momentum_10"] = group["close"].pct_change(10)
                group["momentum_20"] = group["close"].pct_change(20)

                returns = group["close"].pct_change()
                group["volatility_20"] = returns.rolling(20).std()

                group["ma_5"] = group["close"].rolling(5).mean()
                group["ma_10"] = group["close"].rolling(10).mean()
                group["ma_20"] = group["close"].rolling(20).mean()

            results.append(group)

        return pd.concat(results)


class MongoLoader(DataSource):
    """
    MongoDB 数据加载器

    Usage:
        loader = MongoLoader(
            connection_string='mongodb://localhost:27017',
            database='stock',
            collection='daily_quotes',
            symbols=['000001'],
        )
        df = loader.load()
    """

    def __init__(
        self,
        connection_string: str = "mongodb://localhost:27017",
        database: str = "stock",
        collection: str = "daily_quotes",
        symbols: Optional[List[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        include_factors: bool = True,
    ):
        super().__init__(symbols=symbols, start=start, end=end)
        self.connection_string = connection_string
        self.database = database
        self.collection = collection
        self.include_factors = include_factors

        self._client = None
        self._db = None

    def _init_mongo(self):
        """初始化 MongoDB 连接"""
        if self._client is not None:
            return

        try:
            from pymongo import MongoClient
        except ImportError:
            raise ImportError("pymongo not installed. Run: pip install pymongo")

        self._client = MongoClient(self.connection_string)
        self._db = self._client[self.database]

    def load(self) -> pd.DataFrame:
        """加载数据"""
        self._init_mongo()

        # 构建查询
        query = {}

        if self.start or self.end:
            query["date"] = {}
            if self.start:
                query["date"]["$gte"] = self.start
            if self.end:
                query["date"]["$lte"] = self.end

        if self.symbols:
            query["symbol"] = {"$in": self.symbols}

        # 查询
        cursor = self._db[self.collection].find(query)

        # 转换为 DataFrame
        df = pd.DataFrame(list(cursor))

        if len(df) == 0:
            raise ValueError("No data loaded")

        # 处理 MongoDB _id
        if "_id" in df.columns:
            df = df.drop(columns=["_id"])

        # 标准化日期
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")

        # 计算因子
        if self.include_factors:
            df = self._calculate_factors(df)

        self.validate(df)

        return df.sort_index()

    def _calculate_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术因子"""
        results = []

        for symbol, group in df.groupby("symbol"):
            group = group.sort_index()

            if "close" in group.columns:
                group["momentum_5"] = group["close"].pct_change(5)
                group["momentum_10"] = group["close"].pct_change(10)
                group["momentum_20"] = group["close"].pct_change(20)

                returns = group["close"].pct_change()
                group["volatility_20"] = returns.rolling(20).std()

                group["ma_5"] = group["close"].rolling(5).mean()
                group["ma_10"] = group["close"].rolling(10).mean()
                group["ma_20"] = group["close"].rolling(20).mean()

            results.append(group)

        return pd.concat(results)
