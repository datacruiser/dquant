"""
DQuant 数据模块

支持多种数据源:
- csv: CSV 文件
- akshare: AKShare 实时数据 (免费)
- tushare: Tushare 数据 (需要 token)
- yahoo: Yahoo Finance (全球市场)
- jqdata: 聚宽数据 (需要账号)
- ricequant: 米筐数据 (需要 token)
- tdx: 通达信本地数据
- sql: SQL 数据库
- mongodb: MongoDB 数据库
"""

from dquant.data.base import DataSource
from dquant.data.csv_loader import CSVLoader
from dquant.data.akshare_loader import AKShareLoader, AKShareRealTime
from dquant.data.tushare_loader import TushareLoader, TushareFinancial
from dquant.data.yahoo_loader import YahooLoader, YahooRealTime
from dquant.data.jqdata_loader import JQDataLoader, JQDataFactor
from dquant.data.ricequant_loader import RiceQuantLoader
from dquant.data.tdx_loader import TDXLoader, TDXBlockLoader
from dquant.data.database_loader import DatabaseLoader, MongoLoader
from dquant.data.data_manager import (
    DataManager,
    DataSourceRegistry,
    load_data,
)

__all__ = [
    # Base
    "DataSource",

    # Loaders
    "CSVLoader",
    "AKShareLoader",
    "AKShareRealTime",
    "TushareLoader",
    "TushareFinancial",
    "YahooLoader",
    "YahooRealTime",
    "JQDataLoader",
    "JQDataFactor",
    "RiceQuantLoader",
    "TDXLoader",
    "TDXBlockLoader",
    "DatabaseLoader",
    "MongoLoader",

    # Manager
    "DataManager",
    "DataSourceRegistry",
    "load_data",
]

from dquant.data.money_flow_loader import MoneyFlowLoader, MockMoneyFlowLoader
