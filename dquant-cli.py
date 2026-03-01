#!/usr/bin/env python3
"""
DQuant 命令行工具

Usage:
    python dquant-cli.py --help
    python dquant-cli.py factors
    python dquant-cli.py test
    python dquant-cli.py info
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def cmd_factors(args):
    """列出所有因子"""
    from dquant import list_factors, get_factor
    
    factors = list_factors()
    
    print(f"\n已注册 {len(factors)} 个因子:\n")
    
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
        print(f"{category}:")
        for name in factor_list:
            if name in factors:
                factor = get_factor(name)
                print(f"  - {name:20s} ({factor.name})")
        print()


def cmd_test(args):
    """运行测试"""
    import subprocess
    
    print("\n运行基础测试...")
    result = subprocess.run(
        [sys.executable, 'tests/test_basic.py'],
        cwd=Path(__file__).parent,
    )
    
    print("\n运行因子测试...")
    result = subprocess.run(
        [sys.executable, 'tests/test_factors.py'],
        cwd=Path(__file__).parent,
    )


def cmd_info(args):
    """显示项目信息"""
    print("\n" + "="*60)
    print("DQuant - 轻量级AI量化框架")
    print("="*60)
    
    from dquant import list_factors
    
    print(f"\n版本: 0.1.0")
    print(f"因子数量: {len(list_factors())}")
    print(f"Python: {sys.version.split()[0]}")
    
    print("\n核心模块:")
    print("  - 数据源: 10+ (AKShare, Tushare, Yahoo, ...)")
    print("  - 因子库: 34 个内置因子")
    print("  - 回测引擎: 向量化回测")
    print("  - 机器学习: XGBoost, LightGBM")
    print("  - 强化学习: DQN, PPO")
    print("  - 可视化: Matplotlib, Seaborn")
    
    print("\n快速开始:")
    print("  python quickstart.py")
    print("  python examples/simple_backtest.py")
    
    print("\n文档:")
    print("  README.md - 项目说明")
    print("  INSTALL.md - 安装指南")
    print("  CHANGELOG.md - 更新日志")
    
    print()


def cmd_version(args):
    """显示版本"""
    print("DQuant v0.1.0")


def cmd_run(args):
    """运行快速开始"""
    import subprocess
    
    subprocess.run(
        [sys.executable, 'quickstart.py'],
        cwd=Path(__file__).parent,
    )


def main():
    parser = argparse.ArgumentParser(
        description='DQuant 命令行工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # factors
    parser_factors = subparsers.add_parser('factors', help='列出所有因子')
    parser_factors.set_defaults(func=cmd_factors)
    
    # test
    parser_test = subparsers.add_parser('test', help='运行测试')
    parser_test.set_defaults(func=cmd_test)
    
    # info
    parser_info = subparsers.add_parser('info', help='显示项目信息')
    parser_info.set_defaults(func=cmd_info)
    
    # version
    parser_version = subparsers.add_parser('version', help='显示版本')
    parser_version.set_defaults(func=cmd_version)
    
    # run
    parser_run = subparsers.add_parser('run', help='运行快速开始')
    parser_run.set_defaults(func=cmd_run)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
    else:
        args.func(args)


if __name__ == '__main__':
    main()
