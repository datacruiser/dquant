#!/usr/bin/env python3
"""
DQuant 回测使用示例

本示例展示如何使用回测引擎进行策略回测和绩效分析。
"""

from dquant import (
    Engine,
    AKShareLoader,
    TopKStrategy,
    MLFactorStrategy,
    MomentumFactor,
    VolatilityFactor,
    XGBoostFactor,
    BacktestPlotter,
)
import pandas as pd


def example_basic_backtest():
    """基础回测示例"""
    print("\n" + "=" * 60)
    print("1. 基础回测")
    print("=" * 60)

    # 加载数据
    print("加载数据...")
    data = AKShareLoader(symbols='hs300', start='2022-01-01', end='2023-12-31').load()

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
    result = engine.backtest(start='2022-06-01', end='2023-12-31')

    # 显示结果
    print("\n回测结果:")
    for key, value in result.metrics.items():
        if isinstance(value, float):
            print(f"  - {key}: {value:.4f}")

    return result


def example_ml_backtest():
    """ML 策略回测示例"""
    print("\n" + "=" * 60)
    print("2. ML 策略回测")
    print("=" * 60)

    # 加载数据
    print("加载数据...")
    data = AKShareLoader(symbols='hs300', start='2022-01-01', end='2023-12-31').load()

    # 分割训练集和测试集
    train_data = data[data.index < '2023-01-01']
    test_data = data[data.index >= '2023-01-01']

    # 创建 ML 因子
    ml_factor = XGBoostFactor(
        features=['momentum_20', 'volatility_20'],
        target='return_5d',
    )

    # 训练因子
    print("训练 ML 因子...")
    ml_factor.fit(train_data)

    # 创建策略
    strategy = MLFactorStrategy(
        factor=ml_factor,
        top_k=10,
        rebalance_freq=5,
    )

    # 创建回测引擎
    engine = Engine(test_data, strategy)

    # 运行回测
    print("运行回测...")
    result = engine.backtest(start='2023-01-01', end='2023-12-31')

    # 显示结果
    print("\n回测结果:")
    for key, value in result.metrics.items():
        if isinstance(value, float):
            print(f"  - {key}: {value:.4f}")

    return result


def example_parameter_tuning():
    """参数调优示例"""
    print("\n" + "=" * 60)
    print("3. 参数调优")
    print("=" * 60)

    # 加载数据
    print("加载数据...")
    data = AKShareLoader(symbols='hs300', start='2022-01-01', end='2023-12-31').load()

    # 测试不同的参数组合
    windows = [10, 20, 30, 60]
    top_ks = [5, 10, 15, 20]

    results = []

    for window in windows:
        for top_k in top_ks:
            # 创建策略
            strategy = TopKStrategy(
                factor=MomentumFactor(window=window),
                top_k=top_k,
                rebalance_freq=5,
            )

            # 回测
            engine = Engine(data, strategy)
            result = engine.backtest(start='2022-06-01', end='2023-12-31')

            # 记录结果
            results.append({
                'window': window,
                'top_k': top_k,
                'sharpe': result.metrics['sharpe_ratio'],
                'return': result.metrics['total_return'],
                'drawdown': result.metrics['max_drawdown'],
            })

    # 显示最佳参数
    df_results = pd.DataFrame(results)
    best = df_results.loc[df_results['sharpe'].idxmax()]

    print("\n最佳参数:")
    print(f"  - 窗口期: {best['window']}")
    print(f"  - 选股数: {best['top_k']}")
    print(f"  - 夏普比率: {best['sharpe']:.2f}")
    print(f"  - 总收益率: {best['return']:.2%}")
    print(f"  - 最大回撤: {best['drawdown']:.2%}")

    return df_results


def example_benchmark_comparison():
    """基准对比示例"""
    print("\n" + "=" * 60)
    print("4. 基准对比")
    print("=" * 60)

    # 加载数据
    print("加载数据...")
    data = AKShareLoader(symbols='hs300', start='2022-01-01', end='2023-12-31').load()

    # 创建策略
    strategy = TopKStrategy(
        factor=MomentumFactor(window=20),
        top_k=10,
    )

    # 回测
    engine = Engine(data, strategy, benchmark='hs300')
    result = engine.backtest(start='2022-06-01', end='2023-12-31')

    # 显示对比结果
    print("\n策略 vs 基准:")
    print(f"  - 策略收益率: {result.metrics['total_return']:.2%}")
    print(f"  - 基准收益率: {result.metrics['benchmark_return']:.2%}")
    print(f"  - 超额收益: {result.metrics['excess_return']:.2%}")
    print(f"  - 信息比率: {result.metrics['information_ratio']:.2f}")

    return result


def example_visualization():
    """可视化示例"""
    print("\n" + "=" * 60)
    print("5. 可视化")
    print("=" * 60)

    # 加载数据
    data = AKShareLoader(symbols='hs300', start='2022-01-01', end='2023-12-31').load()

    # 创建策略
    strategy = TopKStrategy(
        factor=MomentumFactor(window=20),
        top_k=10,
    )

    # 回测
    engine = Engine(data, strategy)
    result = engine.backtest(start='2022-06-01', end='2023-12-31')

    # 创建可视化
    plotter = BacktestPlotter(result)

    print("生成图表:")
    print("  1. 净值曲线")
    print("  2. 回撤曲线")
    print("  3. 月度收益热力图")
    print("  4. 年度收益柱状图")
    print("  5. 持仓分布")
    print("  6. 换手率")

    # 实际生成 (取消注释运行)
    # plotter.plot_all()

    # 保存图表
    # plotter.save('./output/backtest_report.html')

    return plotter


def example_multi_strategy():
    """多策略组合示例"""
    print("\n" + "=" * 60)
    print("6. 多策略组合")
    print("=" * 60)

    # 加载数据
    print("加载数据...")
    data = AKShareLoader(symbols='hs300', start='2022-01-01', end='2023-12-31').load()

    # 创建多个策略
    strategies = {
        'momentum': TopKStrategy(factor=MomentumFactor(window=20), top_k=10),
        'volatility': TopKStrategy(factor=VolatilityFactor(window=20), top_k=10),
    }

    # 分别回测
    results = {}
    for name, strategy in strategies.items():
        engine = Engine(data, strategy)
        result = engine.backtest(start='2022-06-01', end='2023-12-31')
        results[name] = result

        print(f"\n策略 {name}:")
        print(f"  - 收益率: {result.metrics['total_return']:.2%}")
        print(f"  - 夏普比率: {result.metrics['sharpe_ratio']:.2f}")
        print(f"  - 最大回撤: {result.metrics['max_drawdown']:.2%}")

    return results


def example_walk_forward():
    """滚动回测示例 (Walk-Forward Analysis)"""
    print("\n" + "=" * 60)
    print("7. 滚动回测 (Walk-Forward)")
    print("=" * 60)

    # 加载数据
    print("加载数据...")
    data = AKShareLoader(symbols='hs300', start='2020-01-01', end='2023-12-31').load()

    # 设置滚动参数
    train_period = 252  # 1年训练期
    test_period = 63    # 3个月测试期

    results = []
    dates = data.index.unique()

    for i in range(0, len(dates) - train_period - test_period, test_period):
        # 分割数据
        train_start = dates[i]
        train_end = dates[i + train_period]
        test_start = dates[i + train_period]
        test_end = dates[i + train_period + test_period]

        train_data = data[(data.index >= train_start) & (data.index <= train_end)]
        test_data = data[(data.index >= test_start) & (data.index <= test_end)]

        # 创建策略
        strategy = TopKStrategy(
            factor=MomentumFactor(window=20),
            top_k=10,
        )

        # 回测
        engine = Engine(test_data, strategy)
        result = engine.backtest(start=test_start, end=test_end)

        results.append({
            'period': f"{test_start.strftime('%Y-%m')} - {test_end.strftime('%Y-%m')}",
            'return': result.metrics['total_return'],
            'sharpe': result.metrics['sharpe_ratio'],
        })

    # 显示结果
    df_results = pd.DataFrame(results)

    print("\n滚动回测结果:")
    print(df_results.to_string(index=False))

    print(f"\n平均收益: {df_results['return'].mean():.2%}")
    print(f"平均夏普: {df_results['sharpe'].mean():.2f}")

    return df_results


def example_risk_metrics():
    """风险指标示例"""
    print("\n" + "=" * 60)
    print("8. 风险指标")
    print("=" * 60)

    # 加载数据
    data = AKShareLoader(symbols='hs300', start='2022-01-01', end='2023-12-31').load()

    # 创建策略
    strategy = TopKStrategy(
        factor=MomentumFactor(window=20),
        top_k=10,
    )

    # 回测
    engine = Engine(data, strategy)
    result = engine.backtest(start='2022-06-01', end='2023-12-31')

    # 显示风险指标
    print("\n风险指标:")
    risk_metrics = {
        '最大回撤': result.metrics['max_drawdown'],
        '年化波动率': result.metrics['annual_volatility'],
        '下行波动率': result.metrics['downside_volatility'],
        'VaR (95%)': result.metrics['var_95'],
        'CVaR (95%)': result.metrics['cvar_95'],
        '夏普比率': result.metrics['sharpe_ratio'],
        '索提诺比率': result.metrics['sortino_ratio'],
        '卡玛比率': result.metrics['calmar_ratio'],
    }

    for name, value in risk_metrics.items():
        print(f"  - {name}: {value:.4f}")

    return result


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("DQuant 回测使用示例")
    print("=" * 60)

    # 运行示例
    try:
        example_basic_backtest()
    except Exception as e:
        print(f"❌ 基础回测失败: {e}")

    try:
        example_ml_backtest()
    except Exception as e:
        print(f"❌ ML 回测失败: {e}")

    try:
        example_parameter_tuning()
    except Exception as e:
        print(f"❌ 参数调优失败: {e}")

    try:
        example_benchmark_comparison()
    except Exception as e:
        print(f"❌ 基准对比失败: {e}")

    try:
        example_visualization()
    except Exception as e:
        print(f"❌ 可视化失败: {e}")

    try:
        example_multi_strategy()
    except Exception as e:
        print(f"❌ 多策略失败: {e}")

    try:
        example_walk_forward()
    except Exception as e:
        print(f"❌ 滚动回测失败: {e}")

    try:
        example_risk_metrics()
    except Exception as e:
        print(f"❌ 风险指标失败: {e}")

    print("\n" + "=" * 60)
    print("✅ 所有示例执行完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()
