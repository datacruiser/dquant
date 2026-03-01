"""
数据源使用示例

展示 DQuant 支持的所有数据源及其使用方法。
"""

import sys
sys.path.insert(0, '/Users/datacruiser/github/dquant')

import pandas as pd


def example_csv():
    """CSV 数据源"""
    print("\n" + "="*60)
    print("CSV 数据源")
    print("="*60)
    
    from dquant import CSVLoader
    
    # 加载 CSV 文件
    # loader = CSVLoader("data/stocks.csv")
    # df = loader.load()
    
    print("用法:")
    print("  loader = CSVLoader('data/stocks.csv')")
    print("  df = loader.load()")
    print("")
    print("支持:")
    print("  - 单文件或多文件加载")
    print("  - 自动日期解析")
    print("  - 日期范围过滤")


def example_akshare():
    """AKShare 数据源 (免费)"""
    print("\n" + "="*60)
    print("AKShare 数据源 (免费)")
    print("="*60)
    
    try:
        from dquant import AKShareLoader, AKShareRealTime
        
        print("用法:")
        print("  # 沪深300成分股")
        print("  loader = AKShareLoader(symbols='hs300', start='2022-01-01')")
        print("  df = loader.load()")
        print("")
        print("  # 指定股票")
        print("  loader = AKShareLoader(symbols=['000001', '000002'])")
        print("")
        print("  # 实时行情")
        print("  quotes = AKShareRealTime.get_realtime_quotes(['000001'])")
        print("")
        
        # 尝试获取实时行情
        try:
            quotes = AKShareRealTime.get_realtime_quotes(['000001', '600000'])
            print("实时行情示例:")
            print(quotes[['symbol', 'name', 'price', 'pct_change']].head())
        except Exception as e:
            print(f"  (需要安装 akshare: pip install akshare)")
        
    except ImportError:
        print("  需要安装: pip install akshare")


def example_tushare():
    """Tushare 数据源"""
    print("\n" + "="*60)
    print("Tushare 数据源")
    print("="*60)
    
    print("用法:")
    print("  # 设置 token")
    print("  loader = TushareLoader(symbols='hs300', token='your_token')")
    print("  df = loader.load()")
    print("")
    print("  # 或设置环境变量 TUSHARE_TOKEN")
    print("  loader = TushareLoader(symbols='hs300')")
    print("")
    print("支持:")
    print("  - 日线/分钟线数据")
    print("  - 前复权/后复权")
    print("  - 财务数据")
    print("")
    print("注册地址: https://tushare.pro/")


def example_yahoo():
    """Yahoo Finance 数据源"""
    print("\n" + "="*60)
    print("Yahoo Finance 数据源 (全球市场)")
    print("="*60)
    
    print("用法:")
    print("  # 美股")
    print("  loader = YahooLoader(symbols=['AAPL', 'MSFT', 'GOOGL'])")
    print("")
    print("  # 港股")
    print("  loader = YahooLoader(symbols=['0700.HK', '9988.HK'])")
    print("")
    print("  # A股 (部分)")
    print("  loader = YahooLoader(symbols=['600519.SS'])")
    print("")
    print("  # ETF")
    print("  loader = YahooLoader(symbols=['SPY', 'QQQ'])")
    print("")
    print("  # 加密货币")
    print("  loader = YahooLoader(symbols=['BTC-USD', 'ETH-USD'])")
    print("")
    print("  # 预设标的")
    print("  loader = YahooLoader(symbols='sp500')  # 标普500")
    print("  loader = YahooLoader(symbols='mag7')   # 美股七巨头")
    print("")
    print("需要安装: pip install yfinance")


def example_jqdata():
    """聚宽数据源"""
    print("\n" + "="*60)
    print("聚宽数据源")
    print("="*60)
    
    print("用法:")
    print("  loader = JQDataLoader(")
    print("      symbols='hs300',")
    print("      account='your_phone',")
    print("      password='your_password',")
    print("  )")
    print("  df = loader.load()")
    print("")
    print("支持:")
    print("  - A股全市场数据")
    print("  - 期货数据")
    print("  - 因子数据")
    print("")
    print("注册地址: https://www.joinquant.com/")


def example_ricequant():
    """米筐数据源"""
    print("\n" + "="*60)
    print("米筐数据源")
    print("="*60)
    
    print("用法:")
    print("  loader = RiceQuantLoader(")
    print("      symbols='hs300',")
    print("      token='your_token',")
    print("  )")
    print("  df = loader.load()")
    print("")
    print("注册地址: https://www.ricequant.com/")


def example_tdx():
    """通达信本地数据"""
    print("\n" + "="*60)
    print("通达信本地数据")
    print("="*60)
    
    print("用法:")
    print("  loader = TDXLoader(")
    print("      tdx_path='C:/通达信/vipdoc',")
    print("      market='sz',  # sz, sh")
    print("      symbols=['000001', '000002'],")
    print("  )")
    print("  df = loader.load()")
    print("")
    print("支持:")
    print("  - 日线数据 (.day)")
    print("  - 5分钟线 (.lc5)")
    print("  - 1分钟线 (.lc1)")
    print("  - 无需网络，读取本地文件")


def example_database():
    """数据库数据源"""
    print("\n" + "="*60)
    print("数据库数据源")
    print("="*60)
    
    print("SQL 数据库:")
    print("  # MySQL")
    print("  loader = DatabaseLoader(")
    print("      connection_string='mysql://user:pass@localhost/stock',")
    print("      table='daily_quotes',")
    print("  )")
    print("")
    print("  # SQLite")
    print("  loader = DatabaseLoader(")
    print("      connection_string='sqlite:///data/stock.db',")
    print("      table='daily_quotes',")
    print("  )")
    print("")
    print("MongoDB:")
    print("  loader = MongoLoader(")
    print("      connection_string='mongodb://localhost:27017',")
    print("      database='stock',")
    print("      collection='daily_quotes',")
    print("  )")


def example_data_manager():
    """数据管理器"""
    print("\n" + "="*60)
    print("数据管理器")
    print("="*60)
    
    from dquant import DataManager, load_data
    
    print("统一数据管理:")
    print("  dm = DataManager(cache_dir='./cache')")
    print("")
    print("  # 加载数据 (自动缓存)")
    print("  df = dm.load(")
    print("      source='akshare',")
    print("      symbols='hs300',")
    print("      start='2022-01-01',")
    print("  )")
    print("")
    print("  # 增量更新")
    print("  df = dm.update(source='akshare', symbols='hs300')")
    print("")
    print("  # 批量加载")
    print("  dfs = dm.load_batch([")
    print("      {'source': 'akshare', 'symbols': 'hs300'},")
    print("      {'source': 'tushare', 'symbols': 'zz500'},")
    print("  ])")
    print("")
    print("快捷函数:")
    print("  df = load_data('akshare', symbols='hs300')")
    print("")
    
    # 列出可用数据源
    from dquant import DataSourceRegistry
    print("已注册数据源:")
    for source in DataSourceRegistry.list_sources():
        print(f"  - {source}")


def main():
    print("="*60)
    print("DQuant 数据源示例")
    print("="*60)
    
    example_csv()
    example_akshare()
    example_tushare()
    example_yahoo()
    example_jqdata()
    example_ricequant()
    example_tdx()
    example_database()
    example_data_manager()
    
    print("\n" + "="*60)
    print("示例完成!")
    print("="*60)


if __name__ == '__main__':
    main()
