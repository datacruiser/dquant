"""
因子分析工具

提供因子 IC/IR 分析、因子收益预测能力分析、因子衰减分析等。
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from dquant.logger import get_logger

logger = get_logger(__name__)


@dataclass
class FactorAnalysisResult:
    """因子分析结果"""

    ic_mean: float = 0.0  # IC 均值
    ic_std: float = 0.0  # IC 标准差
    ir: float = 0.0  # IR (IC 均值 / IC 标准差)
    ic_positive_ratio: float = 0.0  # IC 为正的比例

    # 分组收益
    group_returns: Optional[pd.DataFrame] = None
    top_bottom_spread: float = 0.0  # 多空收益差

    # IC 时间序列
    ic_series: Optional[pd.Series] = None


class FactorAnalyzer:
    """
    因子分析器

    分析因子的预测能力和稳定性。

    Usage:
        analyzer = FactorAnalyzer()

        # 分析单个因子
        result = analyzer.analyze(factor_scores, forward_returns)

        # 分析多个因子
        results = analyzer.analyze_multiple(factors_dict, forward_returns)
    """

    def __init__(
        self,
        n_groups: int = 5,  # 分组数
        ic_method: str = "spearman",  # IC 计算方法
    ):
        self.n_groups = n_groups
        self.ic_method = ic_method

    def analyze(
        self,
        factor_scores: pd.DataFrame,
        forward_returns: pd.Series,
        dates: Optional[pd.DatetimeIndex] = None,
    ) -> FactorAnalysisResult:
        """
        分析单个因子

        Args:
            factor_scores: 因子值 (index=date, columns=['symbol', 'score'])
            forward_returns: 未来收益 (index=date, columns=['symbol', 'return'])
            dates: 分析的日期范围

        Returns:
            FactorAnalysisResult
        """
        result = FactorAnalysisResult()

        # 计算 IC 序列
        ic_series = self._calculate_ic_series(factor_scores, forward_returns, dates)
        result.ic_series = ic_series

        # IC 统计
        if len(ic_series) > 0:
            result.ic_mean = ic_series.mean()
            result.ic_std = ic_series.std()
            result.ir = result.ic_mean / result.ic_std if result.ic_std > 0 else 0
            result.ic_positive_ratio = (ic_series > 0).sum() / len(ic_series)

        # 分组收益分析
        group_returns = self._calculate_group_returns(factor_scores, forward_returns, dates)
        result.group_returns = group_returns

        # 多空收益
        if group_returns is not None and len(group_returns.columns) >= 2:
            top_col = group_returns.columns[0]  # 第一组 (因子值最大)
            bottom_col = group_returns.columns[-1]  # 最后一组 (因子值最小)
            result.top_bottom_spread = (
                group_returns[top_col].mean() - group_returns[bottom_col].mean()
            )

        return result

    def _calculate_ic_series(
        self,
        factor_scores: pd.DataFrame,
        forward_returns: pd.Series,
        dates: Optional[pd.DatetimeIndex],
    ) -> pd.Series:
        """计算 IC 时间序列（向量化）"""
        if dates is None:
            dates = factor_scores.index.unique()

        method = self.ic_method if self.ic_method in ("spearman", "pearson") else "pearson"

        # 准备合并的 DataFrame: date x symbol
        factor_df = factor_scores.set_index("symbol", append=True)["score"]
        factor_df.index.names = ["date", "symbol"]
        factor_df = factor_df.rename("factor")

        if isinstance(forward_returns.index, pd.DatetimeIndex):
            returns_df = forward_returns
            returns_df.index.names = ["date"]
            merged = factor_df.to_frame().join(returns_df.rename("return"), how="inner").dropna()
        elif isinstance(forward_returns.index, pd.MultiIndex):
            # (date, symbol) MultiIndex — 直接按 (date, symbol) 对齐
            ret_aligned = forward_returns.copy()
            ret_aligned.index.names = ["date", "symbol"]
            ret_aligned = ret_aligned.rename("return")
            merged = factor_df.to_frame().join(ret_aligned, how="inner").dropna()
        else:
            raise ValueError("forward_returns 必须有 DatetimeIndex 或 (date, symbol) MultiIndex")

        if len(merged) < 5:
            return pd.Series(dtype=float)

        # 按日期分组计算相关系数
        def _group_ic(group):
            if len(group) < 5:
                return np.nan
            return group["factor"].corr(group["return"], method=method)

        ic_series = merged.groupby(level=0).apply(_group_ic).dropna()
        if isinstance(ic_series.index, pd.MultiIndex):
            ic_series = ic_series.droplevel(0)

        # 过滤只保留指定日期范围
        if len(dates) > 0:
            ic_series = ic_series.reindex(dates).dropna()

        return ic_series

    def _get_day_factors(self, factor_scores, date):
        """获取当天的因子数据并分组"""
        day_factors = factor_scores[factor_scores.index == date].copy()

        if len(day_factors) == 0:
            return None

        try:
            day_factors["group"] = pd.qcut(
                day_factors["score"], self.n_groups, labels=False, duplicates="drop"
            )
        except ValueError:
            # 样本不足或全部相同，无法分组
            return None

        return day_factors

    def _get_day_returns(self, forward_returns, date):
        """获取当天的收益数据"""
        if isinstance(forward_returns.index, pd.DatetimeIndex):
            return forward_returns[forward_returns.index == date]

        try:
            return forward_returns.loc[date]
        except (KeyError, ValueError):
            logger.debug("[FactorAnalysis] 计算因子相关性失败")
            return None

    def _calculate_group_avg_return(self, day_factors, day_returns, group_id):
        """计算单个分组的平均收益"""
        group_symbols = day_factors[day_factors["group"] == group_id]["symbol"]

        if len(group_symbols) == 0:
            return None

        if isinstance(day_returns, pd.Series):
            return day_returns[group_symbols].mean()

        return 0

    def _calculate_group_returns(
        self,
        factor_scores: pd.DataFrame,
        forward_returns: pd.Series,
        dates: Optional[pd.DatetimeIndex],
    ) -> Optional[pd.DataFrame]:
        """计算分组收益"""
        if dates is None:
            dates = factor_scores.index.unique()

        group_returns_list = []

        for date in dates:
            # 获取当天因子数据
            day_factors = self._get_day_factors(factor_scores, date)
            if day_factors is None:
                continue

            # 获取收益数据
            day_returns = self._get_day_returns(forward_returns, date)
            if day_returns is None or len(day_returns) == 0:
                continue

            # 计算每组平均收益
            group_rets = {}
            actual_groups = sorted(day_factors["group"].dropna().unique())
            for group_id in actual_groups:
                group_ret = self._calculate_group_avg_return(day_factors, day_returns, group_id)
                if group_ret is not None:
                    group_rets[f"G{int(group_id) + 1}"] = group_ret

            if group_rets:
                group_rets["date"] = date
                group_returns_list.append(group_rets)

        if not group_returns_list:
            return None

        df = pd.DataFrame(group_returns_list)
        df = df.set_index("date")

        return df

    def analyze_multiple(
        self,
        factors: Dict[str, pd.DataFrame],
        forward_returns: pd.Series,
        dates: Optional[pd.DatetimeIndex] = None,
    ) -> pd.DataFrame:
        """
        分析多个因子

        Args:
            factors: {因子名: 因子值 DataFrame}

        Returns:
            分析结果汇总 DataFrame
        """
        results = []

        for name, factor_scores in factors.items():
            result = self.analyze(factor_scores, forward_returns, dates)

            results.append(
                {
                    "factor": name,
                    "ic_mean": result.ic_mean,
                    "ic_std": result.ic_std,
                    "ir": result.ir,
                    "ic_positive_ratio": result.ic_positive_ratio,
                    "top_bottom_spread": result.top_bottom_spread,
                }
            )

        return pd.DataFrame(results)

    def factor_decay(
        self,
        factor_scores: pd.DataFrame,
        returns: pd.DataFrame,
        max_periods: int = 20,
    ) -> pd.Series:
        """
        因子衰减分析

        分析因子对不同持有期的预测能力。

        Args:
            returns: 收益率数据 (index=date, columns=['symbol', 'return'])
            max_periods: 最大持有期

        Returns:
            {持有期: IC}
        """
        decay = {}

        for period in range(1, max_periods + 1):
            # 计算 period 天后的收益
            forward_returns = self._calculate_forward_returns(returns, period)

            # 计算 IC
            result = self.analyze(factor_scores, forward_returns)
            decay[period] = result.ic_mean

        return pd.Series(decay)

    def _calculate_forward_returns(
        self,
        returns: pd.DataFrame,
        period: int,
    ) -> pd.Series:
        """
        计算未来 N 期收益

        forward_return[T] = sum(r[T+1], r[T+2], ..., r[T+period])
        即从 T+1 到 T+period 的累计日收益（近似多期复利）。
        """

        def calc_forward(group):
            group = group.sort_index()
            # rolling(period).sum() 在位置 i 给出 sum(r[i-period+1]..r[i])
            # 在位置 T+period 给出 sum(r[T+1]..r[T+period])
            # shift(-period) 将其移到位置 T
            return group.rolling(period).sum().shift(-period)

        result = returns.groupby("symbol")["return"].apply(calc_forward)
        # groupby.apply 返回 MultiIndex (symbol, date)，重排为 (date, symbol) 以便对齐
        if isinstance(result.index, pd.MultiIndex):
            result = result.reorder_levels([1, 0]).sort_index()
        return result


class FactorReport:
    """
    因子分析报告
    """

    def __init__(self, analyzer: FactorAnalyzer):
        self.analyzer = analyzer

    def generate(
        self,
        factor_name: str,
        factor_scores: pd.DataFrame,
        forward_returns: pd.Series,
    ) -> str:
        """生成报告"""
        result = self.analyzer.analyze(factor_scores, forward_returns)

        report = []
        report.append("=" * 60)
        report.append(f"因子分析报告: {factor_name}")
        report.append("=" * 60)

        # IC 统计
        report.append("\nIC 统计:")
        report.append(f"  IC 均值:     {result.ic_mean:.4f}")
        report.append(f"  IC 标准差:   {result.ic_std:.4f}")
        report.append(f"  IR:          {result.ir:.4f}")
        report.append(f"  IC > 0 比例: {result.ic_positive_ratio:.1%}")

        # 评级
        if abs(result.ir) > 0.5:
            rating = "优秀 ⭐⭐⭐"
        elif abs(result.ir) > 0.3:
            rating = "良好 ⭐⭐"
        elif abs(result.ir) > 0.1:
            rating = "一般 ⭐"
        else:
            rating = "较差"

        report.append(f"\n因子评级: {rating}")

        # 多空收益
        if result.top_bottom_spread != 0:
            report.append(f"\n多空收益差: {result.top_bottom_spread:.2%}")

        # 分组收益
        if result.group_returns is not None:
            report.append("\n分组平均收益:")
            for col in result.group_returns.columns:
                avg_ret = result.group_returns[col].mean()
                report.append(f"  {col}: {avg_ret:.2%}")

        report.append("\n" + "=" * 60)

        return "\n".join(report)
