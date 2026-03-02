#!/usr/bin/env python3
"""
DQuant 策略使用示例

本示例展示如何使用不同的策略进行回测和实盘交易。
"""

from dquant import (
    Engine,
    TopKStrategy,
    MLFactorStrategy,
    MoneyFlowStrategy,
    SmartFlowStrategy,
    MomentumFactor,
    VolatilityFactor,
    RSIFactor,
    XGBoostFactor,
    BacktestPlotter,
)
from dquant.ai.money_flow_factors import (
    MediumFlowFactor,
    MainForceFactor,
    SmartFlowFactor,
)
import pandas as pd


def example_topk_strategy():
    """TopK 策略示例 (简单的因子选股策略)"""
    print("\n" + "=" * 60)
    print("1. TopK 策略 (简单因子选股)")
    print("=" * 60)

    # 创建因子
    momentum = MomentumFactor(window=20)

    # 创建策略
    strategy = TopKStrategy(
        factor=momentum,
        top_k=10,  # 选择前10只股票
        rebalance_freq=5,  # 每5天调仓
    )

    print("✓ 策略创建成功")
    print(f"  - 因子: 动量因子 (20日)")
    print(f"  - 选股数量: {strategy.top_k}")
    print(f"  - 调仓频率: {strategy.rebalance_freq} 天")

    return strategy


def example_ml_factor_strategy():
    """ML 因子策略示例 (机器学习因子选股)"""
    print("\n" + "=" * 60)
    print("2. ML 因子策略 (机器学习选股)")
    print("=" * 60)

    # 创建 ML 因子
    ml_factor = XGBoostFactor(
        features=['momentum_20', 'volatility_20', 'rsi_14'],
        target='return_5d',
        params={
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
        }
    )

    # 创建策略
    strategy = MLFactorStrategy(
        factor=ml_factor,
        top_k=10,
        rebalance_freq=5,
    )

    print("✓ ML 策略创建成功")
    print(f"  - 模型: XGBoost")
    print(f"  - 特征: {ml_factor.features}")
    print(f"  - 目标: {ml_factor.target}")

    return strategy


def example_money_flow_strategy():
    """资金流策略示例 (中户资金流)"""
    print("\n" + "=" * 60)
    print("3. 资金流策略 (中户资金流)")
    print("=" * 60)

    # 创建资金流因子
    medium_flow = MediumFlowFactor(window=20)

    # 创建策略
    strategy = MoneyFlowStrategy(
        factor=medium_flow,
        top_k=10,
        rebalance_freq=5,
    )

    print("✓ 资金流策略创建成功")
    print(f"  - 因子: 中户资金流 (20日)")
    print(f"  - 选股数量: {strategy.top_k}")

    return strategy


def example_smart_flow_strategy():
    """聪明钱策略示例 (主力资金)"""
    print("\n" + "=" * 60)
    print("4. 聪明钱策略 (主力资金)")
    print("=" * 60)

    # 创建聪明钱因子
    smart_flow = SmartFlowFactor(
        main_weight=0.5,
        medium_weight=0.3,
        retail_weight=0.2,
    )

    # 创建策略
    strategy = SmartFlowStrategy(
        factor=smart_flow,
        top_k=10,
        rebalance_freq=5,
    )

    print("✓ 聪明钱策略创建成功")
    print(f"  - 主力权重: {smart_flow.main_weight}")
    print(f"  - 中户权重: {smart_flow.medium_weight}")
    print(f"  - 散户权重: {smart_flow.retail_weight}")

    return strategy


def example_combined_strategy():
    """组合策略示例 (多因子组合)"""
    print("\n" + "=" * 60)
    print("5. 组合策略 (多因子组合)")
    print("=" * 60)

    from dquant import FactorCombiner

    # 创建多个因子
    momentum = MomentumFactor(window=20)
    volatility = VolatilityFactor(window=20)
    rsi = RSIFactor(window=14)

    # 创建组合器
    combiner = FactorCombiner()
    combiner.add_factor('momentum', momentum)
    combiner.add_factor('volatility', volatility)
    combiner.add_factor('rsi', rsi)

    # 计算组合因子 (等权)
    combined_factor = combiner.combine(method='equal')

    # 创建策略
    strategy = TopKStrategy(
        factor=combined_factor,
        top_k=10,
        rebalance_freq=5,
    )

    print("✓ 组合策略创建成功")
    print(f"  - 因子数量: 3")
    print(f"  - 组合方式: 等权")

    return strategy


def example_backtest():
    """回测示例"""
    print("\n" + "=" * 60)
    print("6. 回测示例")
    print("=" * 60)

    from dquant import AKShareLoader

    # 加载数据
    print("加载数据...")
    data = AKShareLoader(symbols='hs300', start='2023-01-01', end='2023-12-31').load()

    # 创建策略
    strategy = TopKStrategy(
        factor=MomentumFactor(window=20),
        top_k=10,
        rebalance_freq=5,
    )

    # 创建回测引擎
    engine = Engine(data, strategy)

    # 运行回测
    print("运行回测...")
    result = engine.backtest(start='2023-06-01', end='2023-12-31')

    # 显示结果
    print("\n回测结果:")
    print(f"  - 总收益率: {result.metrics['total_return']:.2%}")
    print(f"  - 年化收益率: {result.metrics['annual_return']:.2%}")
    print(f"  - 夏普比率: {result.metrics['sharpe_ratio']:.2f}")
    print(f"  - 最大回撤: {result.metrics['max_drawdown']:.2%}")

    return result


def example_backtest_visualization():
    """回测可视化示例"""
    print("\n" + "=" * 60)
    print("7. 回测可视化")
    print("=" * 60)

    from dquant import AKShareLoader

    # 加载数据
    data = AKShareLoader(symbols='hs300', start='2023-01-01', end='2023-12-31').load()

    # 创建策略
    strategy = TopKStrategy(
        factor=MomentumFactor(window=20),
        top_k=10,
    )

    # 回测
    engine = Engine(data, strategy)
    result = engine.backtest(start='2023-06-01', end='2023-12-31')

    # 可视化
    plotter = BacktestPlotter(result)

    print("生成图表...")
    print("  - 净值曲线: plotter.plot_nav()")
    print("  - 回撤曲线: plotter.plot_drawdown()")
    print("  - 月度收益: plotter.plot_monthly_returns()")
    print("  - 年度收益: plotter.plot_yearly_returns()")
    print("  - 全部图表: plotter.plot_all()")

    # 实际生成图表 (取消注释运行)
    # plotter.plot_all()

    return plotter


def example_live_trading():
    """实盘交易示例"""
    print("\n" + "=" * 60)
    print("8. 实盘交易")
    print("=" * 60)

    print("⚠️  实盘交易需要配置券商接口")
    print("   示例代码:")

    code = '''
from dquant import Engine, QMTBroker, TopKStrategy

# 创建券商接口
broker = QMTBroker(
    qmt_path='C:/中航证券QMT/userdata_mini',
    account='YOUR_ACCOUNT',
)

# 创建策略
strategy = TopKStrategy(
    factor=MomentumFactor(window=20),
    top_k=10,
)

# 创建引擎
engine = Engine(data, strategy, broker=broker)

# 运行实盘
engine.live(dry_run=False)  # dry_run=True 为模拟
'''

    print(code)


def example_risk_management():
    """风险管理示例"""
    print("\n" + "=" * 60)
    print("9. 风险管理")
    print("=" * 60)

    from dquant.broker.safety import TradingSafety

    # 创建安全控制器
    safety = TradingSafety(
        max_position_pct=0.1,  # 单只股票最大仓位 10%
        max_daily_loss=0.05,   # 日最大亏损 5%
        stop_loss_pct=0.03,    # 止损 3%
    )

    print("✓ 风控参数设置:")
    print(f"  - 单只股票最大仓位: {safety.max_position_pct:.1%}")
    print(f"  - 日最大亏损: {safety.max_daily_loss:.1%}")
    print(f"  - 止损比例: {safety.stop_loss_pct:.1%}")

    return safety


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("DQuant 策略使用示例")
    print("=" * 60)

    # 运行示例
    example_topk_strategy()
    example_ml_factor_strategy()
    example_money_flow_strategy()
    example_smart_flow_strategy()
    example_combined_strategy()

    try:
        example_backtest()
    except Exception as e:
        print(f"❌ 回测示例失败: {e}")

    try:
        example_backtest_visualization()
    except Exception as e:
        print(f"❌ 可视化示例失败: {e}")

    example_live_trading()
    example_risk_management()

    print("\n" + "=" * 60)
    print("✅ 所有示例执行完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
