"""
数据验证和清洗

提供数据质量检查、异常值处理、缺失值处理等功能。
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class DataValidator:
    """
    数据验证器

    检查数据质量。
    """

    @staticmethod
    def check_required_columns(
        data: pd.DataFrame,
        required: List[str],
    ) -> Tuple[bool, List[str]]:
        """
        检查必需列

        Returns:
            (是否通过, 缺失的列)
        """
        missing = [col for col in required if col not in data.columns]
        return len(missing) == 0, missing

    @staticmethod
    def check_missing_values(
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
    ) -> Dict[str, int]:
        """
        检查缺失值

        Returns:
            {列名: 缺失数量}
        """
        cols = columns or data.columns
        return {col: data[col].isna().sum() for col in cols}

    @staticmethod
    def check_duplicates(
        data: pd.DataFrame,
        subset: Optional[List[str]] = None,
    ) -> int:
        """
        检查重复行

        Args:
            subset: 用于判断重复的列

        Returns:
            重复行数
        """
        return data.duplicated(subset=subset).sum()

    @staticmethod
    def check_price_validity(
        data: pd.DataFrame,
    ) -> Tuple[int, Dict[str, Any]]:
        """
        检查价格有效性

        Returns:
            (问题数量, 问题详情)
        """
        issues = {}
        count = 0

        # 负价格
        for col in ["open", "high", "low", "close"]:
            if col in data.columns:
                neg_count = (data[col] <= 0).sum()
                if neg_count > 0:
                    issues[f"{col}_negative"] = neg_count
                    count += neg_count

        # high < low
        if "high" in data.columns and "low" in data.columns:
            invalid = (data["high"] < data["low"]).sum()
            if invalid > 0:
                issues["high_lt_low"] = invalid
                count += invalid

        # open/close 不在 [low, high] 范围
        for col in ["open", "close"]:
            if col in data.columns:
                out_of_range = ((data[col] < data["low"]) | (data[col] > data["high"])).sum()
                if out_of_range > 0:
                    issues[f"{col}_out_of_range"] = out_of_range
                    count += out_of_range

        return count, issues

    @staticmethod
    def check_volume_validity(
        data: pd.DataFrame,
    ) -> int:
        """检查成交量有效性"""
        if "volume" not in data.columns:
            return 0

        return (data["volume"] < 0).sum()

    @staticmethod
    def check_date_continuity(
        data: pd.DataFrame,
        date_col: str = "date",
        freq: str = "B",
    ) -> Tuple[int, List[Any]]:
        """
        检查日期连续性

        Args:
            freq: 频率 (B=工作日, D=日)

        Returns:
            (缺失日期数, 缺失日期列表)
        """
        if date_col not in data.columns and data.index.name == date_col:
            dates = data.index
        else:
            dates = pd.to_datetime(data[date_col])

        full_range = pd.date_range(dates.min(), dates.max(), freq=freq)
        missing = full_range.difference(dates)

        return len(missing), missing.tolist()

    def _run_check(self, check_name, data, results):
        """执行单个检查"""
        if check_name == "columns":
            required = ["symbol", "open", "high", "low", "close", "volume"]
            passed, missing = self.check_required_columns(data, required)
            if not passed:
                results["issues"]["missing_columns"] = missing
                results["passed"] = False

        elif check_name == "missing":
            missing = self.check_missing_values(data)
            total_missing = sum(missing.values())
            if total_missing > 0:
                results["issues"]["missing_values"] = missing

        elif check_name == "duplicates":
            dups = self.check_duplicates(data, subset=["symbol"])
            if dups > 0:
                results["issues"]["duplicates"] = dups

        elif check_name == "price":
            count, issues = self.check_price_validity(data)
            if count > 0:
                results["issues"]["price_issues"] = issues
                results["passed"] = False

        elif check_name == "volume":
            invalid = self.check_volume_validity(data)
            if invalid > 0:
                results["issues"]["invalid_volume"] = invalid
                results["passed"] = False

    def validate(
        self,
        data: pd.DataFrame,
        checks: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        综合验证

        Args:
            checks: 检查项目 ['columns', 'missing', 'duplicates', 'price', 'volume', 'dates']
        """
        checks = checks or ["columns", "missing", "duplicates", "price", "volume"]

        results = {
            "passed": True,
            "issues": {},
        }

        # 执行所有检查
        for check in checks:
            self._run_check(check, data, results)

        return results


class DataCleaner:
    """
    数据清洗器

    处理异常值、缺失值等。
    """

    @staticmethod
    def remove_outliers(
        data: pd.DataFrame,
        columns: List[str],
        method: str = "iqr",
        threshold: float = 3.0,
    ) -> pd.DataFrame:
        """
        移除异常值

        Args:
            columns: 要处理的列
            method: 'iqr' 或 'zscore'
            threshold: 阈值
        """
        df = data.copy()

        for col in columns:
            if col not in df.columns:
                continue

            if method == "iqr":
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1

                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR

                df = df[(df[col] >= lower) & (df[col] <= upper)]

            elif method == "zscore":
                mean = df[col].mean()
                std = df[col].std()

                if std > 0:
                    zscore = (df[col] - mean) / std
                    df = df[abs(zscore) <= threshold]

        return df

    @staticmethod
    def fill_missing(
        data: pd.DataFrame,
        method: str = "ffill",
        value: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        填充缺失值

        Args:
            method: 'ffill', 'bfill', 'mean', 'median', 'zero', 'value'
            value: 当 method='value' 时使用的值
        """
        df = data.copy()

        if method == "ffill":
            df = df.ffill()
        elif method == "bfill":
            df = df.bfill()
        elif method == "mean":
            df = df.fillna(df.mean())
        elif method == "median":
            df = df.fillna(df.median())
        elif method == "zero":
            df = df.fillna(0)
        elif method == "value" and value is not None:
            df = df.fillna(value)

        return df

    @staticmethod
    def remove_duplicates(
        data: pd.DataFrame,
        subset: Optional[List[str]] = None,
        keep: str = "last",
    ) -> pd.DataFrame:
        """
        移除重复行

        Args:
            keep: 'first', 'last', False
        """
        return data.drop_duplicates(subset=subset, keep=keep)

    @staticmethod
    def fix_price_anomalies(
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        修复价格异常

        - 修正 high < low
        - 修正价格超出范围
        """
        df = data.copy()

        if "high" in df.columns and "low" in df.columns:
            # 修正 high < low
            mask = df["high"] < df["low"]
            df.loc[mask, ["high", "low"]] = df.loc[mask, ["low", "high"]].values

        # 修正 open/close 超出范围
        for col in ["open", "close"]:
            if col in df.columns:
                # 超过 high
                mask = df[col] > df["high"]
                df.loc[mask, col] = df.loc[mask, "high"]

                # 低于 low
                mask = df[col] < df["low"]
                df.loc[mask, col] = df.loc[mask, "low"]

        return df

    @staticmethod
    def normalize_volume(
        data: pd.DataFrame,
        method: str = "log",
    ) -> pd.DataFrame:
        """
        成交量标准化

        Args:
            method: 'log', 'rank', 'zscore'
        """
        df = data.copy()

        if "volume" not in df.columns:
            return df

        if method == "log":
            df["volume"] = np.log1p(df["volume"])
        elif method == "rank":
            df["volume"] = df["volume"].rank(pct=True)
        elif method == "zscore":
            mean = df["volume"].mean()
            std = df["volume"].std()
            if std > 0:
                df["volume"] = (df["volume"] - mean) / std

        return df

    def clean(
        self,
        data: pd.DataFrame,
        remove_outliers_cols: Optional[List[str]] = None,
        fill_missing_method: str = "ffill",
        fix_prices: bool = True,
    ) -> pd.DataFrame:
        """
        综合清洗
        """
        df = data.copy()

        # 移除重复
        df = self.remove_duplicates(df)

        # 移除异常值
        if remove_outliers_cols:
            df = self.remove_outliers(df, remove_outliers_cols)

        # 修复价格
        if fix_prices:
            df = self.fix_price_anomalies(df)

        # 填充缺失
        df = self.fill_missing(df, method=fill_missing_method)

        return df


class DataQualityReport:
    """
    数据质量报告
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.validator = DataValidator()

    def generate(self) -> str:
        """生成报告"""
        report = []
        report.append("=" * 60)
        report.append("数据质量报告")
        report.append("=" * 60)

        # 基本信息
        report.append(f"\n数据形状: {self.data.shape[0]} 行 × {self.data.shape[1]} 列")

        if self.data.index.name:
            report.append(f"索引: {self.data.index.name}")

        # 验证结果
        validation = self.validator.validate(self.data)

        report.append(f"\n验证结果: {'✓ 通过' if validation['passed'] else '✗ 失败'}")

        if validation["issues"]:
            report.append("\n问题详情:")
            for issue, detail in validation["issues"].items():
                report.append(f"  - {issue}: {detail}")

        # 缺失值统计
        missing = self.validator.check_missing_values(self.data)
        total_missing = sum(missing.values())

        if total_missing > 0:
            report.append(f"\n缺失值总数: {total_missing}")
            for col, count in missing.items():
                if count > 0:
                    report.append(f"  - {col}: {count} ({count / len(self.data):.1%})")
        else:
            report.append("\n✓ 无缺失值")

        # 重复行
        dups = self.validator.check_duplicates(self.data)
        if dups > 0:
            report.append(f"\n重复行: {dups}")
        else:
            report.append("\n✓ 无重复行")

        # 价格问题
        count, issues = self.validator.check_price_validity(self.data)
        if count > 0:
            report.append(f"\n价格问题: {count}")
            for issue, cnt in issues.items():
                report.append(f"  - {issue}: {cnt}")
        else:
            report.append("\n✓ 价格数据正常")

        report.append("\n" + "=" * 60)

        return "\n".join(report)
