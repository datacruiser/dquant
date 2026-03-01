"""
DQuant 完整示例

展示框架的所有主要功能:
1. 数据加载 (AKShare)
2. ML 因子策略
3. 回测与绩效分析
4. 可视化
5. 实盘接口 (模拟)
"""

import sys
sys.path.insert(0, '/Users/datacruiser/github/dquant')

import pandas as pd
import numpy as np
from datetime import datetime


def example_1_basic_backtest():
    """示例1: 基础回测"""
    print("\n" + "="*60)
    print("示例1: 基础回测")
    print("="*60)
    
    from dquant import Engine, TopKStrategy
    from dquant.data.base import DataSource
    
    # 生成模拟数据
    class MockDataSource(DataSource):
        def load(self):
            np.random.seed(42)
            dates = pd.date_range('2022-01-01', '2023-12-31', freq='B')
            symbols = [f'{i:06d}.SH' for i in range(1, 51)]
            
            data = []
            for symbol in symbols:
                price = 10 + np.random.randn() * 5
                for date in dates:
                    ret = np.random.randn() * 0.02
                    price *= (1 + ret)
                    data.append({
                        'date': date,
                        'symbol': symbol,
                        'open': price * (1 + np.random.randn() * 0.01),
                        'high': price * 1.01,
                        'low': price * 0.99,
                        'close': price,
                        'volume': int(np.random.exponential(1000000)),
                    })
            
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            return df
    
    # 创建策略和引擎
    data = MockDataSource()
    strategy = TopKStrategy(factor_name='close', top_k=10)  # 使用价格作为因子(示例)
    engine = Engine(data=data, strategy=strategy, initial_cash=1_000_000)
    
    # 运行回测
    result = engine.backtest(start='2022-06-01', end='2023-12-31')
    
    print(f"\n回测结果:")
    print(result.metrics)


def example_2_ml_factor():
    """示例2: ML 因子策略"""
    print("\n" + "="*60)
    print("示例2: ML 因子策略")
    print("="*60)
    
    from dquant import Engine, MLFactorStrategy, MomentumFactor
    
    # 使用动量因子 (不需要训练)
    factor = MomentumFactor(window=20)
    
    # 创建策略
    strategy = MLFactorStrategy(factor=factor, top_k=10)
    
    print(f"因子名称: {factor.name}")
    print(f"策略名称: {strategy.name}")
    print("ML 因子策略创建成功")


def example_3_akshare_data():
    """示例3: AKShare 数据"""
    print("\n" + "="*60)
    print("示例3: AKShare 数据 (需要安装 akshare)")
    print("="*60)
    
    try:
        from dquant import AKShareLoader, AKShareRealTime
        
        # 获取实时行情
        print("获取实时行情...")
        quotes = AKShareRealTime.get_realtime_quotes(['000001', '600000'])
        print(quotes[['symbol', 'name', 'price', 'pct_change']].head())
        
    except ImportError:
        print("akshare 未安装，运行: pip install akshare")


def example_4_visualization():
    """示例4: 可视化"""
    print("\n" + "="*60)
    print("示例4: 可视化 (需要安装 matplotlib)")
    print("="*60)
    
    from dquant import BacktestPlotter, Metrics
    from dquant.backtest.portfolio import Portfolio
    
    # 创建模拟回测结果
    portfolio = Portfolio(initial_cash=1_000_000)
    
    # 模拟净值曲线
    dates = pd.date_range('2022-01-01', '2023-12-31', freq='D')
    np.random.seed(42)
    returns = np.random.randn(len(dates)) * 0.01
    
    for i, (date, ret) in enumerate(zip(dates, returns)):
        portfolio.nav_history.append(1 + np.cumsum(returns[:i+1])[-1])
        portfolio.timestamp_history.append(date)
    
    # 计算绩效
    nav_series = pd.Series(portfolio.nav_history, index=portfolio.timestamp_history)
    metrics = Metrics.from_nav(nav_series)
    
    print(f"绩效指标:")
    print(metrics)
    
    # 创建绘图器
    plotter = BacktestPlotter(
        type('Result', (), {
            'portfolio': portfolio,
            'trades': pd.DataFrame(),
            'metrics': metrics,
        })()
    )
    
    print("\n可视化方法:")
    print("  - plotter.plot_nav()       # 净值曲线")
    print("  - plotter.plot_drawdown()  # 回撤曲线")
    print("  - plotter.plot_monthly_returns()  # 月度收益热力图")
    print("  - plotter.plot_yearly_returns()   # 年度收益柱状图")


def example_5_broker():
    """示例5: 券商接口"""
    print("\n" + "="*60)
    print("示例5: 券商接口")
    print("="*60)
    
    from dquant import Simulator, Order
    
    # 创建模拟券商
    broker = Simulator(initial_cash=1_000_000)
    broker.connect()
    
    # 查询账户
    account = broker.get_account()
    print(f"账户信息: {account}")
    
    # 下单
    order = Order(symbol='000001.SZ', side='BUY', quantity=1000, price=10.0)
    result = broker.place_order(order)
    print(f"\n下单结果: {result}")
    
    # 查询持仓
    positions = broker.get_positions()
    print(f"持仓: {positions}")


def example_6_optimization():
    """示例6: 参数优化"""
    print("\n" + "="*60)
    print("示例6: 参数优化")
    print("="*60)
    
    from dquant import Engine, TopKStrategy
    from dquant.data.base import DataSource
    
    # 模拟数据
    class MockDataSource(DataSource):
        def load(self):
            np.random.seed(42)
            dates = pd.date_range('2022-01-01', '2023-12-31', freq='B')
            symbols = [f'{i:06d}.SH' for i in range(1, 21)]
            
            data = []
            for symbol in symbols:
                price = 10
                for date in dates:
                    price *= (1 + np.random.randn() * 0.02)
                    data.append({
                        'date': date,
                        'symbol': symbol,
                        'open': price,
                        'high': price * 1.01,
                        'low': price * 0.99,
                        'close': price,
                        'volume': 1000000,
                    })
            
            df = pd.DataFrame(data).set_index('date')
            return df
    
    data = MockDataSource()
    strategy = TopKStrategy(factor_name='close', top_k=5)
    engine = Engine(data=data, strategy=strategy)
    
    # 参数优化
    print("运行参数优化...")
    result = engine.optimize(
        param_grid={'top_k': [3, 5, 10]},
        metric='sharpe',
        start='2022-06-01',
        end='2023-06-01',
    )
    
    print(f"\n最优参数: {result['best_params']}")
    print(f"最优夏普: {result['best_score']:.2f}")


def main():
    print("="*60)
    print("DQuant 完整示例")
    print("="*60)
    
    example_1_basic_backtest()
    example_2_ml_factor()
    example_3_akshare_data()
    example_4_visualization()
    example_5_broker()
    example_6_optimization()
    
    print("\n" + "="*60)
    print("示例运行完成!")
    print("="*60)


if __name__ == '__main__':
    main()
