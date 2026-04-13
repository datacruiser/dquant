#!/usr/bin/env python3
"""
DQuant 命令行工具

Usage:
    python dquant-cli.py --help
    python dquant-cli.py factors
    python dquant-cli.py backtest --strategy momentum --start 2023-01-01
    python dquant-cli.py test
    python dquant-cli.py info
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def _build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        description="DQuant - 轻量级AI量化框架 CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    sub = parser.add_subparsers(dest="command", help="可用命令")

    # ---- factors ----
    p = sub.add_parser("factors", help="列出所有已注册因子")
    p.add_argument("--category", help="筛选因子类别")

    # ---- backtest ----
    p = sub.add_parser("backtest", help="运行回测")
    p.add_argument("--data", default="csv", help="数据源 (csv/akshare/tushare)")
    p.add_argument("--symbols", default="hs300_sample", help="股票代码或指数名称")
    p.add_argument("--start", default="2023-01-01", help="开始日期")
    p.add_argument("--end", default=None, help="结束日期")
    p.add_argument("--cash", type=float, default=1_000_000, help="初始资金")
    p.add_argument("--strategy", default="buyhold", help="策略名称")
    p.add_argument("--output", default=None, help="输出报告路径 (HTML)")
    p.add_argument("--no-price-limit", action="store_true", help="禁用涨跌停限制")

    # ---- test ----
    p = sub.add_parser("test", help="运行测试")
    p.add_argument("--verbose", "-v", action="store_true")

    # ---- info ----
    sub.add_parser("info", help="显示项目信息")

    # ---- version ----
    sub.add_parser("version", help="显示版本")

    # ---- run ----
    sub.add_parser("run", help="运行快速开始教程")

    return parser


def cmd_factors(args):
    from dquant import get_factor, list_factors

    factors = list_factors()
    print(f"\n已注册 {len(factors)} 个因子:\n")

    categories = {
        "动量类": ["momentum", "reversal", "acc_momentum"],
        "波动率类": ["volatility", "atr", "skewness", "kurtosis", "max_drawdown"],
        "技术指标": ["rsi", "macd", "bollinger", "trend", "kdj", "cci", "williams_r"],
        "成交量": ["volume_ratio", "turnover_rate", "obv", "vwap"],
        "价格形态": ["price_position", "gap", "intraday", "overnight"],
        "均线": ["ma_slope", "ma_cross", "bias"],
        "基本面": ["pe", "pb", "roe", "revenue_growth", "profit_growth", "market_cap"],
        "情绪": ["money_flow", "amihud"],
    }

    for cat, factor_list in categories.items():
        if args.category and args.category != cat:
            continue
        print(f"{cat}:")
        for name in factor_list:
            if name in factors:
                f = get_factor(name)
                print(f"  - {name:20s} ({f.name})")
        print()


def cmd_backtest(args):
    """运行回测"""
    import subprocess

    print(f"\n[Backtest] 策略={args.strategy}, 数据={args.data}, "
          f"起始={args.start}, 资金={args.cash:,.0f}")
    print("[Backtest] 正在加载数据...")

    # 使用 quickstart 示例作为基础
    example_path = Path(__file__).parent / "examples" / "simple_backtest.py"
    if example_path.exists():
        subprocess.run([sys.executable, str(example_path)], cwd=Path(__file__).parent)
    else:
        print("示例文件不存在，请先创建回测脚本")


def cmd_test(args):
    import subprocess

    verbose = ["-v"] if args.verbose else []
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", *verbose],
        cwd=Path(__file__).parent,
    )
    sys.exit(result.returncode)


def cmd_info(args):
    from dquant import list_factors

    print("\n" + "=" * 60)
    print("DQuant - 轻量级AI量化框架")
    print("=" * 60)

    print(f"\n版本: 0.1.0")
    print(f"因子数量: {len(list_factors())}")
    print(f"Python: {sys.version.split()[0]}")

    print("\n核心模块:")
    print("  - 数据源: 10+ (AKShare, Tushare, Yahoo, ...)")
    print("  - 因子库: 68+ 内置因子")
    print("  - 回测引擎: 向量化 + 事件驱动")
    print("  - 涨跌停: A股主板±10%, 北交所±30%")
    print("  - Walk-Forward: 时序交叉验证")
    print("  - 机器学习: XGBoost, LightGBM")
    print("  - 强化学习: DQN, PPO")

    print("\n快速开始:")
    print("  python dquant-cli.py run")
    print("  python examples/simple_backtest.py")

    print()


def cmd_version(args):
    print("DQuant v0.1.0")


def cmd_run(args):
    import subprocess

    subprocess.run(
        [sys.executable, "quickstart.py"],
        cwd=Path(__file__).parent,
    )


def main():
    parser = _build_parser()
    args = parser.parse_args()

    commands = {
        "factors": cmd_factors,
        "backtest": cmd_backtest,
        "test": cmd_test,
        "info": cmd_info,
        "version": cmd_version,
        "run": cmd_run,
    }

    if args.command is None:
        parser.print_help()
    else:
        handler = commands.get(args.command)
        if handler:
            handler(args)
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
