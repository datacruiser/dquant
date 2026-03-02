#!/usr/bin/env python3
"""
DQuant 数据源使用示例

本示例展示如何使用不同的数据源加载股票数据。
"""

from dquant import (
    AKShareLoader,
    YahooLoader,
    TushareLoader,
    CSVLoader,
    DataManager,
    load_data,
)
import pandas as pd


def example_akshare():
    """AKShare 数据源示例 (免费 A 股数据)"""
    print("\n" + "=" * 60)
    print("1. AKShare 数据源 (免费 A 股)")
    print("=" * 60)

    # 加载沪深300成分股数据
    loader = AKShareLoader(symbols='hs300', start='2023-01-01', end='2023-12-31')
    df = loader.load()

    print(f"✓ 加载成功: {len(df)} 条记录")
    print(f"✓ 股票数量: {df['symbol'].nunique()}")
    print(f"✓ 数据列: {list(df.columns)}")
    print("\n数据预览:")
    print(df.head())

    return df


def example_yahoo():
    """Yahoo Finance 数据源示例 (美股/全球市场)"""
    print("\n" + "=" * 60)
    print("2. Yahoo Finance 数据源 (美股)")
    print("=" * 60)

    # 加载美股数据
    loader = YahooLoader(
        symbols=['AAPL', 'MSFT', 'GOOGL'],
        start='2023-01-01',
        end='2023-12-31'
    )
    df = loader.load()

    print(f"✓ 加载成功: {len(df)} 条记录")
    print(f"✓ 股票数量: {df['symbol'].nunique()}")
    print("\n数据预览:")
    print(df.head())

    return df


def example_tushare():
    """Tushare 数据源示例 (专业金融数据)"""
    print("\n" + "=" * 60)
    print("3. Tushare 数据源 (需要 Token)")
    print("=" * 60)

    # 需要有效的 Tushare Token
    # 可从 https://tushare.pro/ 获取
    token = "YOUR_TUSHARE_TOKEN"

    print("⚠️  需要 Tushare Token 才能使用")
    print("   获取地址: https://tushare.pro/")

    # 示例代码 (需要 token)
    # loader = TushareLoader(
    #     symbols='hs300',
    #     start='2023-01-01',
    #     end='2023-12-31',
    #     token=token
    # )
    # df = loader.load()

    return None


def example_csv():
    """CSV 数据源示例 (本地数据)"""
    print("\n" + "=" * 60)
    print("4. CSV 数据源 (本地文件)")
    print("=" * 60)

    # 创建示例 CSV 数据
    print("创建示例 CSV 数据...")

    import numpy as np
    sample_data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=100),
        'symbol': '600000.SH',
        'open': 10.0 + np.random.randn(100) * 0.1,
        'high': 10.2 + np.random.randn(100) * 0.1,
        'low': 9.8 + np.random.randn(100) * 0.1,
        'close': 10.0 + np.random.randn(100) * 0.1,
        'volume': 1000000 + np.random.randint(0, 100000, 100),
    })

    # 保存到临时文件
    csv_file = '/tmp/sample_stock_data.csv'
    sample_data.to_csv(csv_file, index=False)
    print(f"✓ 示例数据已保存到: {csv_file}")

    # 从 CSV 加载
    loader = CSVLoader(filepath=csv_file)
    df = loader.load()

    print(f"✓ 加载成功: {len(df)} 条记录")
    print("\n数据预览:")
    print(df.head())

    return df


def example_data_manager():
    """DataManager 数据管理器示例 (统一接口 + 缓存)"""
    print("\n" + "=" * 60)
    print("5. DataManager 数据管理器 (统一接口 + 缓存)")
    print("=" * 60)

    # 创建数据管理器
    dm = DataManager(cache_dir='./data/cache')

    # 加载数据 (自动缓存)
    print("第一次加载 (从网络)...")
    df1 = dm.load(source='akshare', symbols='hs300', start='2023-01-01', end='2023-01-31')
    print(f"✓ 加载成功: {len(df1)} 条记录")

    # 再次加载 (从缓存)
    print("\n第二次加载 (从缓存)...")
    df2 = dm.load(source='akshare', symbols='hs300', start='2023-01-01', end='2023-01-31')
    print(f"✓ 加载成功: {len(df2)} 条记录")

    # 增量更新
    print("\n增量更新...")
    df3 = dm.update(source='akshare', symbols='hs300')
    print(f"✓ 更新成功: {len(df3)} 条记录")

    return dm


def example_quick_load():
    """快捷加载函数示例"""
    print("\n" + "=" * 60)
    print("6. 快捷加载函数")
    print("=" * 60)

    # 使用快捷函数加载数据
    df = load_data('akshare', symbols='hs300', start='2023-01-01', end='2023-01-31')

    print(f"✓ 加载成功: {len(df)} 条记录")
    print(f"✓ 股票数量: {df['symbol'].nunique()}")

    return df


def example_multiple_sources():
    """多数据源组合示例"""
    print("\n" + "=" * 60)
    print("7. 多数据源组合")
    print("=" * 60)

    # A 股数据
    a_stock = load_data('akshare', symbols='hs300', start='2023-01-01', end='2023-01-31')

    # 美股数据
    us_stock = load_data('yahoo', symbols=['AAPL', 'MSFT'], start='2023-01-01', end='2023-01-31')

    print(f"✓ A 股数据: {len(a_stock)} 条")
    print(f"✓ 美股数据: {len(us_stock)} 条")

    # 合并数据
    combined = pd.concat([a_stock, us_stock], ignore_index=True)
    print(f"✓ 合并后: {len(combined)} 条")

    return combined


def example_realtime():
    """实时数据示例 (WebSocket)"""
    print("\n" + "=" * 60)
    print("8. 实时数据 (WebSocket)")
    print("=" * 60)

    print("⚠️  实时数据功能需要特定数据源支持")
    print("   示例:")

    code = '''
from dquant import RealtimeDataLoader

# 创建实时数据加载器
loader = RealtimeDataLoader(
    source='akshare',
    symbols=['600000.SH', '000001.SZ']
)

# 订阅实时数据
def on_data(data):
    print(f"实时数据: {data['symbol']} - {data['close']}")

loader.subscribe(callback=on_data)

# 开始接收数据
loader.start()
'''

    print(code)


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("DQuant 数据源使用示例")
    print("=" * 60)

    # 运行示例
    try:
        example_akshare()
    except Exception as e:
        print(f"❌ AKShare 示例失败: {e}")

    try:
        example_yahoo()
    except Exception as e:
        print(f"❌ Yahoo 示例失败: {e}")

    example_tushare()

    try:
        example_csv()
    except Exception as e:
        print(f"❌ CSV 示例失败: {e}")

    try:
        example_data_manager()
    except Exception as e:
        print(f"❌ DataManager 示例失败: {e}")

    try:
        example_quick_load()
    except Exception as e:
        print(f"❌ 快捷加载示例失败: {e}")

    try:
        example_multiple_sources()
    except Exception as e:
        print(f"❌ 多数据源示例失败: {e}")

    example_realtime()

    print("\n" + "=" * 60)
    print("✅ 所有示例执行完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
