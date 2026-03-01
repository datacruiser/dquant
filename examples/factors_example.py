"""
因子使用示例

展示 DQuant 34 个内置因子和因子组合功能。
"""

import sys
sys.path.insert(0, '/Users/datacruiser/github/dquant')

import pandas as pd
import numpy as np


def generate_mock_data(n_stocks: int = 50, n_days: int = 500) -> pd.DataFrame:
    """生成模拟数据"""
    np.random.seed(42)
    
    dates = pd.date_range('2022-01-01', periods=n_days, freq='B')
    symbols = [f'{i:06d}.SH' for i in range(1, n_stocks + 1)]
    
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
    return df.set_index('date')


def example_all_factors():
    """示例: 所有内置因子"""
    print("\n" + "="*60)
    print("DQuant 内置因子库 (34个)")
    print("="*60)
    
    from dquant import get_factor, list_factors
    
    data = generate_mock_data(20, 200)
    factors = list_factors()
    
    # 分类
    categories = {
        '动量类': ['momentum', 'reversal', 'acc_momentum'],
        '波动率类': ['volatility', 'atr', 'skewness', 'kurtosis', 'max_drawdown'],
        '技术指标': ['rsi', 'macd', 'bollinger', 'trend', 'kdj', 'cci', 'williams_r'],
        '成交量': ['volume_ratio', 'turnover_rate', 'obv', 'vwap'],
        '价格形态': ['price_position', 'gap', 'intraday', 'overnight'],
        '均线': ['ma_slope', 'ma_cross', 'bias'],
        '基本面': ['pe', 'pb', 'roe', 'revenue_growth', 'profit_growth', 'market_cap'],
        '情绪': ['money_flow', 'amihud'],
    }
    
    for category, factor_list in categories.items():
        print(f"\n{category}:")
        for name in factor_list:
            if name in factors:
                try:
                    factor = get_factor(name)
                    result = factor.predict(data)
                    if len(result) > 0:
                        print(f"  {name:18s}: mean={result['score'].mean():8.4f}, std={result['score'].std():8.4f}")
                except Exception as e:
                    print(f"  {name:18s}: (需要额外数据)")


def example_technical_factors():
    """示例: 技术指标因子"""
    print("\n" + "="*60)
    print("技术指标因子")
    print("="*60)
    
    from dquant import get_factor
    
    data = generate_mock_data(20, 200)
    
    # RSI
    print("\nRSI 因子:")
    rsi = get_factor('rsi', window=14)
    result = rsi.predict(data)
    print(f"  形状: {result.shape}")
    print(f"  范围: [{result['score'].min():.2f}, {result['score'].max():.2f}]")
    
    # MACD
    print("\nMACD 因子:")
    macd = get_factor('macd')
    result = macd.predict(data)
    print(f"  形状: {result.shape}")
    print(f"  均值: {result['score'].mean():.4f}")
    
    # KDJ
    print("\nKDJ 因子:")
    kdj = get_factor('kdj', n=9)
    result = kdj.predict(data)
    print(f"  形状: {result.shape}")
    print(f"  均值: {result['score'].mean():.4f}")
    
    # CCI
    print("\nCCI 因子:")
    cci = get_factor('cci', window=14)
    result = cci.predict(data)
    print(f"  形状: {result.shape}")
    print(f"  均值: {result['score'].mean():.4f}")


def example_volatility_factors():
    """示例: 波动率因子"""
    print("\n" + "="*60)
    print("波动率因子")
    print("="*60)
    
    from dquant import get_factor
    
    data = generate_mock_data(20, 200)
    
    # 波动率
    print("\n波动率因子:")
    vol = get_factor('volatility', window=20)
    result = vol.predict(data)
    print(f"  均值: {result['score'].mean():.4f}")
    
    # ATR
    print("\nATR 因子:")
    atr = get_factor('atr', window=14)
    result = atr.predict(data)
    print(f"  均值: {result['score'].mean():.4f}")
    
    # 偏度
    print("\n偏度因子:")
    skew = get_factor('skewness', window=20)
    result = skew.predict(data)
    print(f"  均值: {result['score'].mean():.4f}")
    
    # 峰度
    print("\n峰度因子:")
    kurt = get_factor('kurtosis', window=20)
    result = kurt.predict(data)
    print(f"  均值: {result['score'].mean():.4f}")


def example_factor_combination():
    """示例: 因子组合"""
    print("\n" + "="*60)
    print("因子组合")
    print("="*60)
    
    from dquant import FactorCombiner, get_factor
    
    data = generate_mock_data(30, 300)
    
    # 创建组合器
    combiner = FactorCombiner()
    
    # 添加多个因子
    combiner.add_factor('momentum', get_factor('momentum', 20))
    combiner.add_factor('reversal', get_factor('reversal', 5))
    combiner.add_factor('volatility', get_factor('volatility', 20))
    combiner.add_factor('rsi', get_factor('rsi', 14))
    combiner.add_factor('macd', get_factor('macd'))
    combiner.add_factor('kdj', get_factor('kdj'))
    
    # 计算
    print("\n计算因子值...")
    combiner.fit(data)
    
    # 权重摘要
    print("\n因子权重摘要:")
    summary = combiner.get_weights_summary()
    print(summary.to_string(index=False))
    
    # 组合
    print("\n等权组合:")
    combined = combiner.combine(method='equal')
    print(f"  形状: {combined.shape}")
    print(f"  均值: {combined['score'].mean():.4f}")
    
    # 因子相关性
    print("\n因子相关性矩阵:")
    corr = combiner.get_factor_correlation()
    print(corr.round(2).to_string())


def example_custom_weights():
    """示例: 自定义权重"""
    print("\n" + "="*60)
    print("自定义权重组合")
    print("="*60)
    
    from dquant import FactorCombiner, get_factor
    
    data = generate_mock_data(30, 300)
    
    combiner = FactorCombiner()
    combiner.add_factor('momentum', get_factor('momentum', 20))
    combiner.add_factor('reversal', get_factor('reversal', 5))
    combiner.add_factor('volatility', get_factor('volatility', 20))
    
    combiner.fit(data)
    
    # 自定义权重
    custom_weights = {
        'momentum': 0.5,
        'reversal': 0.3,
        'volatility': 0.2,
    }
    
    print("\n自定义权重:")
    for name, weight in custom_weights.items():
        print(f"  {name}: {weight}")
    
    combined = combiner.combine(method='equal', weights=custom_weights)
    print(f"\n组合结果:")
    print(f"  形状: {combined.shape}")
    print(f"  均值: {combined['score'].mean():.4f}")


def example_combined_factor():
    """示例: 组合因子类"""
    print("\n" + "="*60)
    print("组合因子类")
    print("="*60)
    
    from dquant import CombinedFactor, get_factor
    
    data = generate_mock_data(30, 300)
    
    # 创建组合因子
    combined = CombinedFactor(
        factors={
            'momentum': get_factor('momentum', 20),
            'reversal': get_factor('reversal', 5),
            'volatility': get_factor('volatility', 20),
            'rsi': get_factor('rsi', 14),
        },
        weights={
            'momentum': 0.3,
            'reversal': 0.2,
            'volatility': 0.2,
            'rsi': 0.3,
        },
        combine_method='equal',
    )
    
    print("\n训练组合因子...")
    combined.fit(data)
    
    predictions = combined.predict(data)
    print(f"\n预测结果:")
    print(f"  形状: {predictions.shape}")
    print(f"  均值: {predictions['score'].mean():.4f}")
    print(f"  标准差: {predictions['score'].std():.4f}")


def main():
    print("="*60)
    print("DQuant 因子示例")
    print("="*60)
    
    example_all_factors()
    example_technical_factors()
    example_volatility_factors()
    example_factor_combination()
    example_custom_weights()
    example_combined_factor()
    
    print("\n" + "="*60)
    print("示例完成!")
    print("="*60)


if __name__ == '__main__':
    main()
