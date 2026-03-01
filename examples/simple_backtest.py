"""
简单回测示例
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 添加路径
import sys
sys.path.insert(0, '/Users/datacruiser/github/dquant')

from dquant import Engine
from dquant.data.base import DataSource
from dquant.strategy.base import BaseStrategy, Signal, SignalType
from dquant.ai.base import MomentumFactor


class SimpleDataSource(DataSource):
    """简单数据源 - 生成模拟数据"""
    
    def __init__(self, n_stocks: int = 50, n_days: int = 500):
        super().__init__()
        self.n_stocks = n_stocks
        self.n_days = n_days
    
    def load(self) -> pd.DataFrame:
        """生成模拟股票数据"""
        np.random.seed(42)
        
        dates = pd.date_range(
            start='2022-01-01',
            periods=self.n_days,
            freq='B'  # 工作日
        )
        
        symbols = [f'{i:06d}.SH' for i in range(1, self.n_stocks + 1)]
        
        data = []
        for symbol in symbols:
            # 模拟价格
            price = 10 + np.random.randn() * 5
            prices = [price]
            
            for _ in range(len(dates) - 1):
                ret = np.random.randn() * 0.02  # 2% 日波动
                price = price * (1 + ret)
                prices.append(price)
            
            for i, (date, close) in enumerate(zip(dates, prices)):
                open_price = close * (1 + np.random.randn() * 0.01)
                high = max(open_price, close) * (1 + abs(np.random.randn() * 0.005))
                low = min(open_price, close) * (1 - abs(np.random.randn() * 0.005))
                volume = int(np.random.exponential(1000000))
                
                data.append({
                    'date': date,
                    'symbol': symbol,
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': close,
                    'volume': volume,
                    # 因子
                    'momentum': (close / prices[max(0, i-20)] - 1) if i >= 20 else 0,
                    'volatility': np.std(prices[max(0, i-20):i+1]) / close if i >= 20 else 0,
                })
        
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
        
        return df


class SimpleMomentumStrategy(BaseStrategy):
    """简单动量策略"""
    
    def __init__(self, top_k: int = 10, rebalance_freq: int = 5):
        super().__init__(name="SimpleMomentum")
        self.top_k = top_k
        self.rebalance_freq = rebalance_freq
    
    def generate_signals(self, data: pd.DataFrame) -> list:
        """生成信号 - 选取动量 TopK"""
        signals = []
        
        for date, group in data.groupby(data.index):
            if 'momentum' not in group.columns:
                continue
            
            # 选动量最大的 TopK
            top_stocks = group.nlargest(self.top_k, 'momentum')
            
            for _, row in top_stocks.iterrows():
                signal = Signal(
                    symbol=row['symbol'],
                    signal_type=SignalType.BUY,
                    strength=1.0 / self.top_k,
                    timestamp=date,
                    metadata={'momentum': row['momentum']}
                )
                signals.append(signal)
        
        return signals


def main():
    print("=" * 50)
    print("DQuant 简单回测示例")
    print("=" * 50)
    
    # 1. 加载数据
    print("\n[1] 加载数据...")
    data_source = SimpleDataSource(n_stocks=50, n_days=500)
    data = data_source.load()
    print(f"    数据形状: {data.shape}")
    print(f"    日期范围: {data.index.min()} ~ {data.index.max()}")
    print(f"    股票数量: {data['symbol'].nunique()}")
    
    # 2. 创建策略
    print("\n[2] 创建策略...")
    strategy = SimpleMomentumStrategy(top_k=10, rebalance_freq=5)
    print(f"    策略名称: {strategy.name}")
    print(f"    TopK: {strategy.top_k}")
    
    # 3. 运行回测
    print("\n[3] 运行回测...")
    engine = Engine(data=data_source, strategy=strategy, initial_cash=1_000_000)
    result = engine.backtest(start='2022-06-01', end='2023-12-31')
    
    # 4. 输出结果
    print("\n[4] 回测结果:")
    print(result.metrics)
    
    # 5. 交易记录
    if len(result.trades) > 0:
        print(f"\n[5] 交易记录 (共 {len(result.trades)} 笔):")
        print(result.trades.head(10))
    
    print("\n" + "=" * 50)
    print("回测完成!")
    print("=" * 50)


if __name__ == '__main__':
    main()
