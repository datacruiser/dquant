"""
风险管理示例

展示如何使用 DQuant 的风险管理功能。
"""

import sys
sys.path.insert(0, '/Users/datacruiser/github/dquant')

import pandas as pd
import numpy as np


def generate_mock_data():
    """生成模拟数据"""
    np.random.seed(42)
    
    dates = pd.date_range('2023-01-01', periods=200, freq='B')
    symbols = ['000001.SZ', '000002.SZ', '600000.SH', '600519.SH']
    
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


def example_position_sizing():
    """示例: 仓位管理"""
    print("\n" + "="*60)
    print("仓位管理")
    print("="*60)
    
    from dquant.risk import PositionSizer, PositionLimit
    
    symbols = ['000001.SZ', '000002.SZ', '600000.SH', '600519.SH']
    total_value = 1000000
    
    # 1. 等权分配
    print("\n1. 等权分配:")
    sizer = PositionSizer(method='equal_weight', total_value=total_value)
    positions = sizer.size(symbols)
    
    for symbol, value in positions.items():
        print(f"  {symbol}: ¥{value:,.0f} ({value/total_value:.1%})")
    
    # 2. 按信号强度加权
    print("\n2. 信号强度加权:")
    signals = {
        '000001.SZ': 0.8,
        '000002.SZ': 0.6,
        '600000.SH': 0.4,
        '600519.SH': 0.2,
    }
    
    sizer = PositionSizer(method='signal_weight', total_value=total_value)
    positions = sizer.size(symbols, signals=signals)
    
    for symbol, value in positions.items():
        signal = signals.get(symbol, 0)
        print(f"  {symbol}: ¥{value:,.0f} ({value/total_value:.1%}) [信号={signal}]")
    
    # 3. 风险平价
    print("\n3. 风险平价:")
    volatilities = {
        '000001.SZ': 0.025,  # 高波动
        '000002.SZ': 0.020,
        '600000.SH': 0.015,
        '600519.SH': 0.010,  # 低波动
    }
    
    sizer = PositionSizer(method='risk_parity', total_value=total_value)
    positions = sizer.size(symbols, volatilities=volatilities)
    
    for symbol, value in positions.items():
        vol = volatilities.get(symbol, 0.02)
        print(f"  {symbol}: ¥{value:,.0f} ({value/total_value:.1%}) [波动率={vol:.1%}]")
    
    # 4. 仓位限制
    print("\n4. 仓位限制:")
    limits = PositionLimit(
        max_single_pct=0.15,  # 单只最大 15%
        max_total_pct=0.90,   # 最大总仓位 90%
    )
    
    sizer = PositionSizer(
        method='equal_weight',
        total_value=total_value,
        limits=limits,
    )
    positions = sizer.size(symbols[:2])  # 只买 2 只
    
    for symbol, value in positions.items():
        print(f"  {symbol}: ¥{value:,.0f} ({value/total_value:.1%})")


def example_risk_metrics():
    """示例: 风险指标"""
    print("\n" + "="*60)
    print("风险指标")
    print("="*60)
    
    from dquant.risk import RiskManager
    
    # 生成收益率序列
    np.random.seed(42)
    returns = pd.Series(np.random.randn(100) * 0.02)
    
    # 基准收益率
    benchmark = pd.Series(np.random.randn(100) * 0.015)
    
    manager = RiskManager()
    
    # VaR
    var_95 = manager.calculate_var(returns, confidence=0.95)
    var_99 = manager.calculate_var(returns, confidence=0.99)
    
    print(f"\nVaR (95%): {var_95:.2%}")
    print(f"VaR (99%): {var_99:.2%}")
    
    # CVaR
    cvar_95 = manager.calculate_cvar(returns, confidence=0.95)
    print(f"\nCVaR (95%): {cvar_95:.2%}")
    
    # 风险指标
    metrics = manager.calculate_risk_metrics(returns, benchmark)
    
    print(f"\n综合风险指标:")
    print(f"  Beta: {metrics.beta:.2f}")
    print(f"  跟踪误差: {metrics.tracking_error:.2%}")
    print(f"  信息比率: {metrics.information_ratio:.2f}")


def example_drawdown_control():
    """示例: 回撤控制"""
    print("\n" + "="*60)
    print("回撤控制")
    print("="*60)
    
    from dquant.risk import RiskManager
    
    manager = RiskManager(max_drawdown=0.15)
    
    # 模拟净值曲线
    np.random.seed(42)
    initial_value = 1000000
    values = [initial_value]
    
    for i in range(100):
        ret = np.random.randn() * 0.02
        values.append(values[-1] * (1 + ret))
    
    # 检查回撤
    triggered_count = 0
    
    for i, value in enumerate(values):
        triggered, drawdown = manager.check_drawdown(value)
        
        if triggered:
            print(f"\n⚠️  第 {i} 天触发风控!")
            print(f"  当前净值: ¥{value:,.0f}")
            print(f"  当前回撤: {drawdown:.2%}")
            triggered_count += 1
            
            if triggered_count >= 3:
                break
    
    if triggered_count == 0:
        print("\n✓ 回撤控制正常")
        print(f"  最大回撤: {manager.current_drawdown:.2%}")


def example_stop_loss():
    """示例: 止损策略"""
    print("\n" + "="*60)
    print("止损策略")
    print("="*60)
    
    from dquant.risk import StopLoss
    
    entry_price = 100.0
    current_price = 105.0
    highest_price = 110.0
    atr = 3.0
    volatility = 0.02
    
    # 1. 固定止损
    stop1 = StopLoss.fixed_stop(entry_price, stop_pct=0.05)
    print(f"\n1. 固定止损 (5%):")
    print(f"  入场价: ¥{entry_price:.2f}")
    print(f"  止损价: ¥{stop1:.2f}")
    
    # 2. 移动止损
    stop2 = StopLoss.trailing_stop(current_price, highest_price, trailing_pct=0.1)
    print(f"\n2. 移动止损 (10%):")
    print(f"  当前价: ¥{current_price:.2f}")
    print(f"  最高价: ¥{highest_price:.2f}")
    print(f"  止损价: ¥{stop2:.2f}")
    
    # 3. ATR 止损
    stop3 = StopLoss.atr_stop(entry_price, atr, multiplier=2.0)
    print(f"\n3. ATR 止损 (2x ATR):")
    print(f"  入场价: ¥{entry_price:.2f}")
    print(f"  ATR: ¥{atr:.2f}")
    print(f"  止损价: ¥{stop3:.2f}")
    
    # 4. 波动率止损
    stop4 = StopLoss.volatility_stop(entry_price, volatility, multiplier=2.0)
    print(f"\n4. 波动率止损 (2x Vol):")
    print(f"  入场价: ¥{entry_price:.2f}")
    print(f"  波动率: {volatility:.1%}")
    print(f"  止损价: ¥{stop4:.2f}")


def example_take_profit():
    """示例: 止盈策略"""
    print("\n" + "="*60)
    print("止盈策略")
    print("="*60)
    
    from dquant.risk import TakeProfit
    
    entry_price = 100.0
    stop_price = 95.0
    
    # 1. 固定止盈
    target1 = TakeProfit.fixed_profit(entry_price, profit_pct=0.2)
    print(f"\n1. 固定止盈 (20%):")
    print(f"  入场价: ¥{entry_price:.2f}")
    print(f"  止盈价: ¥{target1:.2f}")
    
    # 2. 风险回报比止盈
    target2 = TakeProfit.risk_reward(entry_price, stop_price, ratio=2.0)
    print(f"\n2. 风险回报比 (2:1):")
    print(f"  入场价: ¥{entry_price:.2f}")
    print(f"  止损价: ¥{stop_price:.2f}")
    print(f"  止盈价: ¥{target2:.2f}")
    print(f"  风险: ¥{entry_price - stop_price:.2f}")
    print(f"  回报: ¥{target2 - entry_price:.2f}")


def main():
    print("="*60)
    print("DQuant 风险管理示例")
    print("="*60)
    
    example_position_sizing()
    example_risk_metrics()
    example_drawdown_control()
    example_stop_loss()
    example_take_profit()
    
    print("\n" + "="*60)
    print("示例完成!")
    print("="*60)


if __name__ == '__main__':
    main()
