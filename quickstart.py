#!/usr/bin/env python3
"""
DQuant 快速开始

5 分钟上手 DQuant 量化框架。
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
import numpy as np


def step1_hello_dquant():
    """步骤1: Hello DQuant"""
    print("\n" + "="*60)
    print("步骤1: Hello DQuant")
    print("="*60)
    
    from dquant import list_factors, get_factor
    
    factors = list_factors()
    print(f"✓ DQuant 加载成功")
    print(f"✓ 已注册 {len(factors)} 个因子")


def step2_builtin_factors():
    """步骤2: 使用内置因子"""
    print("\n" + "="*60)
    print("步骤2: 内置因子")
    print("="*60)
    
    from dquant import get_factor, list_factors
    
    factors = list_factors()
    print(f"✓ 已注册 {len(factors)} 个因子")
    print(f"  示例: {', '.join(factors[:5])}...")
    
    # 创建因子
    momentum = get_factor('momentum', window=20)
    print(f"\n✓ 创建因子: {momentum.name}")


def step3_generate_data():
    """步骤3: 生成测试数据"""
    print("\n" + "="*60)
    print("步骤3: 生成测试数据")
    print("="*60)
    
    np.random.seed(42)
    
    dates = pd.date_range('2023-01-01', periods=100, freq='B')
    symbols = ['000001.SZ', '000002.SZ', '600000.SH']
    
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
                'high': price * 1.02,
                'low': price * 0.98,
                'close': price,
                'volume': int(np.random.exponential(1000000)),
            })
    
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    
    print(f"✓ 生成测试数据: {len(df)} 行")
    print(f"  股票: {symbols}")
    print(f"  日期: {dates[0].date()} 到 {dates[-1].date()}")
    
    return df


def step4_calculate_factors(data):
    """步骤4: 计算因子"""
    print("\n" + "="*60)
    print("步骤4: 计算因子")
    print("="*60)
    
    from dquant import get_factor
    
    # 计算动量因子
    momentum = get_factor('momentum', window=20)
    mom_result = momentum.predict(data)
    print(f"✓ 动量因子: mean={mom_result['score'].mean():.4f}")
    
    # 计算 RSI 因子
    rsi = get_factor('rsi', window=14)
    rsi_result = rsi.predict(data)
    print(f"✓ RSI 因子: mean={rsi_result['score'].mean():.4f}")
    
    # 计算波动率因子
    volatility = get_factor('volatility', window=20)
    vol_result = volatility.predict(data)
    print(f"✓ 波动率因子: mean={vol_result['score'].mean():.4f}")


def step5_combine_factors(data):
    """步骤5: 组合因子"""
    print("\n" + "="*60)
    print("步骤5: 组合因子")
    print("="*60)
    
    from dquant import FactorCombiner, get_factor
    
    combiner = FactorCombiner()
    combiner.add_factor('momentum', get_factor('momentum', window=20))
    combiner.add_factor('volatility', get_factor('volatility', window=20))
    
    combiner.fit(data)
    
    combined = combiner.combine(method='equal')
    print(f"✓ 组合因子: {len(combined)} 个数据点")
    print(f"  均值: {combined['score'].mean():.4f}")
    print(f"  标准差: {combined['score'].std():.4f}")


def step6_simple_backtest(data):
    """步骤6: 简单回测"""
    print("\n" + "="*60)
    print("步骤6: 简单回测")
    print("="*60)
    
    from dquant import BacktestEngine
    from dquant.strategy.base import BaseStrategy, Signal, SignalType
    
    class SimpleStrategy(BaseStrategy):
        def generate_signals(self, data):
            signals = []
            
            for symbol in data['symbol'].unique():
                # 简单策略：动量 > 0 则买入
                symbol_data = data[data['symbol'] == symbol].sort_index()
                
                if len(symbol_data) < 20:
                    continue
                
                last = symbol_data.iloc[-1]
                mom = (last['close'] / symbol_data.iloc[-20]['close'] - 1)
                
                if mom > 0:
                    signals.append(Signal(
                        symbol=symbol,
                        signal_type=SignalType.BUY,
                        strength=mom,
                    ))
            
            return signals
    
    strategy = SimpleStrategy()
    engine = BacktestEngine(
        data=data,
        strategy=strategy,
        initial_cash=1000000,
        commission=0.0003,
    )
    
    result = engine.run()
    
    print(f"✓ 回测完成")
    print(f"  总收益率: {result.metrics.total_return*100:.2f}%")
    print(f"  最大回撤: {result.metrics.max_drawdown*100:.2f}%")


def step7_utils():
    """步骤7: 工具函数"""
    print("\n" + "="*60)
    print("步骤7: 工具函数")
    print("="*60)
    
    from dquant import format_money, format_percent, sharpe_ratio
    import numpy as np
    
    # 格式化金额
    print("✓ 金额格式化:")
    print(f"  {format_money(12345678)}")
    print(f"  {format_money(1234567890)}")
    
    # 格式化百分比
    print("\n✓ 百分比格式化:")
    print(f"  {format_percent(0.1234)}")
    
    # 计算 Sharpe
    returns = pd.Series(np.random.randn(100) * 0.02)
    sharpe = sharpe_ratio(returns)
    print(f"\n✓ Sharpe 比率: {sharpe:.2f}")


def step8_config():
    """步骤8: 配置管理"""
    print("\n" + "="*60)
    print("步骤8: 配置管理")
    print("="*60)
    
    from dquant import DQuantConfig
    
    config = DQuantConfig()
    
    print(f"✓ 默认配置:")
    print(f"  初始资金: {config.backtest.initial_cash:,.0f}")
    print(f"  佣金率: {config.backtest.commission_rate*100:.2f}%")
    print(f"  缓存目录: {config.data.cache_dir}")


def main():
    print("="*60)
    print("DQuant 快速开始")
    print("="*60)
    
    step1_hello_dquant()
    step2_builtin_factors()
    data = step3_generate_data()
    step4_calculate_factors(data)
    step5_combine_factors(data)
    step6_simple_backtest(data)
    step7_utils()
    step8_config()
    
    print("\n" + "="*60)
    print("✓ 快速开始完成!")
    print("="*60)
    print()
    print("下一步:")
    print("  - 查看 examples/ 目录了解更多示例")
    print("  - 阅读 README.md 了解详细文档")
    print("  - 查看 tests/ 目录了解测试用例")
    print()


if __name__ == '__main__':
    main()
