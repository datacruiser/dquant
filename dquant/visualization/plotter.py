"""
回测可视化
"""

from typing import Optional

import pandas as pd

from dquant.backtest.result import BacktestResult
from dquant.logger import get_logger

logger = get_logger(__name__)


class BacktestPlotter:
    """
    回测结果可视化

    支持多种图表类型:
    - 净值曲线
    - 回撤曲线
    - 持仓分布
    - 月度收益热力图
    - 年度收益柱状图

    Usage:
        plotter = BacktestPlotter(result)
        plotter.plot_nav()
        plotter.plot_drawdown()
        plotter.plot_monthly_returns()
    """

    def __init__(self, result: BacktestResult):
        self.result = result
        self.portfolio = result.portfolio
        self.trades = result.trades
        self.metrics = result.metrics

        # 净值序列
        self.nav_series = pd.Series(
            self.portfolio.nav_history, index=self.portfolio.timestamp_history
        )

    def plot_nav(
        self,
        benchmark: Optional[pd.Series] = None,
        title: str = "策略净值曲线",
        figsize: tuple = (12, 6),
        save_path: Optional[str] = None,
    ):
        """
        绘制净值曲线

        Args:
            benchmark: 基准净值序列
            title: 图表标题
            figsize: 图表大小
            save_path: 保存路径
        """
        try:
            import matplotlib.dates as mdates
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("[Plotter] matplotlib not installed. Run: pip install matplotlib")
            return

        fig, ax = plt.subplots(figsize=figsize)

        # 策略净值
        ax.plot(
            self.nav_series.index,
            self.nav_series.values,
            label="策略",
            linewidth=2,
            color="#2E86AB",
        )

        # 基准净值
        if benchmark is not None:
            # 归一化基准
            benchmark = benchmark / benchmark.iloc[0]
            ax.plot(
                benchmark.index,
                benchmark.values,
                label="基准",
                linewidth=1.5,
                color="#A23B72",
                alpha=0.8,
            )

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("日期", fontsize=11)
        ax.set_ylabel("净值", fontsize=11)
        ax.legend(loc="upper left", fontsize=10)
        ax.grid(True, alpha=0.3)

        # 格式化日期
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.xticks(rotation=45)

        # 添加关键指标
        textstr = (
            f"年化收益: {self.metrics.annual_return:.2%}\n"
            f"夏普: {self.metrics.sharpe:.2f}\n"
            f"最大回撤: {self.metrics.max_drawdown:.2%}"
        )
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        ax.text(
            0.02,
            0.98,
            textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=props,
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"[Plotter] Saved to {save_path}")

        plt.show()

    def plot_drawdown(
        self,
        title: str = "回撤曲线",
        figsize: tuple = (12, 4),
        save_path: Optional[str] = None,
    ):
        """绘制回撤曲线"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("[Plotter] matplotlib not installed")
            return

        # 计算回撤
        cummax = self.nav_series.cummax()
        drawdown = (self.nav_series - cummax) / cummax

        fig, ax = plt.subplots(figsize=figsize)

        ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.5, color="#E74C3C")
        ax.plot(drawdown.index, drawdown.values, color="#E74C3C", linewidth=1)

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("日期", fontsize=11)
        ax.set_ylabel("回撤", fontsize=11)
        ax.grid(True, alpha=0.3)

        # 标注最大回撤
        max_dd_date = drawdown.idxmin()
        max_dd_value = drawdown.min()
        ax.annotate(
            f"最大回撤: {max_dd_value:.2%}",
            xy=(max_dd_date, max_dd_value),
            xytext=(max_dd_date, max_dd_value + 0.05),
            fontsize=10,
            arrowprops=dict(arrowstyle="->", color="black"),
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        plt.show()

    def plot_monthly_returns(
        self,
        title: str = "月度收益热力图",
        figsize: tuple = (12, 6),
        save_path: Optional[str] = None,
    ):
        """绘制月度收益热力图"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            logger.warning("[Plotter] matplotlib/seaborn not installed")
            return

        # 计算月度收益
        monthly_returns = self.nav_series.resample("M").last().pct_change()
        monthly_returns = monthly_returns.dropna()

        # 构建热力图数据
        df = pd.DataFrame(
            {
                "year": monthly_returns.index.year,
                "month": monthly_returns.index.month,
                "return": monthly_returns.values,
            }
        )

        pivot = df.pivot(index="year", columns="month", values="return")
        pivot.columns = [
            "1月",
            "2月",
            "3月",
            "4月",
            "5月",
            "6月",
            "7月",
            "8月",
            "9月",
            "10月",
            "11月",
            "12月",
        ]

        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(
            pivot,
            annot=True,
            fmt=".2%",
            cmap="RdYlGn",
            center=0,
            linewidths=0.5,
            ax=ax,
            cbar_kws={"label": "收益率"},
        )

        ax.set_title(title, fontsize=14, fontweight="bold")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        plt.show()

    def plot_yearly_returns(
        self,
        title: str = "年度收益",
        figsize: tuple = (10, 5),
        save_path: Optional[str] = None,
    ):
        """绘制年度收益柱状图"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("[Plotter] matplotlib not installed")
            return

        # 计算年度收益
        yearly_returns = self.nav_series.resample("Y").last().pct_change()
        yearly_returns = yearly_returns.dropna()

        fig, ax = plt.subplots(figsize=figsize)

        colors = ["#2ECC71" if r > 0 else "#E74C3C" for r in yearly_returns.values]

        bars = ax.bar(yearly_returns.index.year, yearly_returns.values, color=colors)

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("年份", fontsize=11)
        ax.set_ylabel("收益率", fontsize=11)
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.grid(True, alpha=0.3, axis="y")

        # 添加数值标签
        for bar, val in zip(bars, yearly_returns.values):
            height = bar.get_height()
            ax.annotate(
                f"{val:.2%}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                ha="center",
                va="bottom" if height > 0 else "top",
                fontsize=10,
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        plt.show()

    def plot_position_distribution(
        self,
        date: Optional[pd.Timestamp] = None,
        title: str = "持仓分布",
        figsize: tuple = (10, 6),
        save_path: Optional[str] = None,
    ):
        """绘制持仓分布饼图"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("[Plotter] matplotlib not installed")
            return

        if not self.portfolio.positions:
            logger.warning("[Plotter] No positions")
            return

        positions = self.portfolio.positions

        labels = list(positions.keys())
        sizes = [p.market_value for p in positions.values()]

        fig, ax = plt.subplots(figsize=figsize)

        ax.pie(
            sizes,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90,
            colors=plt.cm.Set3.colors,
        )

        ax.set_title(title, fontsize=14, fontweight="bold")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        plt.show()

    def plot_all(self, save_dir: Optional[str] = None):
        """绘制所有图表"""
        print("\n[Plotter] 绘制净值曲线...")
        self.plot_nav(save_path=f"{save_dir}/nav.png" if save_dir else None)

        print("[Plotter] 绘制回撤曲线...")
        self.plot_drawdown(save_path=f"{save_dir}/drawdown.png" if save_dir else None)

        print("[Plotter] 绘制月度收益...")
        self.plot_monthly_returns(save_path=f"{save_dir}/monthly.png" if save_dir else None)

        print("[Plotter] 绘制年度收益...")
        self.plot_yearly_returns(save_path=f"{save_dir}/yearly.png" if save_dir else None)


def plot_backtest(result: BacktestResult, kind: str = "nav", **kwargs):
    """
    快捷绘图函数

    Args:
        result: 回测结果
        kind: 图表类型 ('nav', 'drawdown', 'monthly', 'yearly', 'all')
        **kwargs: 传递给对应绘图方法的参数
    """
    plotter = BacktestPlotter(result)

    if kind == "nav":
        plotter.plot_nav(**kwargs)
    elif kind == "drawdown":
        plotter.plot_drawdown(**kwargs)
    elif kind == "monthly":
        plotter.plot_monthly_returns(**kwargs)
    elif kind == "yearly":
        plotter.plot_yearly_returns(**kwargs)
    elif kind == "all":
        plotter.plot_all(**kwargs)
    else:
        raise ValueError(f"Unknown plot kind: {kind}")
