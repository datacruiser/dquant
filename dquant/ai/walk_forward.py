"""
Walk-Forward 验证

时间序列专用的交叉验证，避免前视偏差。
支持滚动窗口（Rolling）和扩展窗口（Expanding）两种模式。
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from dquant.logger import get_logger

logger = get_logger(__name__)


@dataclass
class WalkForwardSplit:
    """单个 Walk-Forward 分割"""

    train_start: int
    train_end: int
    test_start: int
    test_end: int
    purge_gap: int = 0  # 训练集和测试集之间的间隔（防止标签泄露）


@dataclass
class WalkForwardResult:
    """Walk-Forward 验证结果"""

    fold_results: List[dict] = field(default_factory=list)
    mean_score: float = 0.0
    std_score: float = 0.0
    sharpe_ratio: float = 0.0
    n_folds: int = 0

    def summary(self) -> str:
        lines = [
            f"Walk-Forward 验证结果 ({self.n_folds} folds)",
            f"  平均得分: {self.mean_score:.4f}",
            f"  得分标准差: {self.std_score:.4f}",
            f"  Sharpe Ratio: {self.sharpe_ratio:.4f}",
        ]
        for i, fr in enumerate(self.fold_results):
            lines.append(
                f"  Fold {i + 1}: train=[{fr['train_start']},{fr['train_end']}], "
                f"test=[{fr['test_start']},{fr['test_end']}], "
                f"score={fr['score']:.4f}"
            )
        return "\n".join(lines)


class WalkForwardValidator:
    """
    Walk-Forward 时序交叉验证器

    适用于量化因子的模型验证，严格按照时间顺序分割数据。

    Usage:
        wf = WalkForwardValidator(
            n_splits=5,
            train_ratio=0.7,
            purge_gap=5,  # 5天 purge 防止标签泄露
        )
        result = wf.validate(data, factor)

    Args:
        n_splits: 分割次数
        train_ratio: 每次分割中训练集占比
        purge_gap: 训练集与测试集之间的间隔（行数）
        expanding: True 使用扩展窗口，False 使用滚动窗口
    """

    def __init__(
        self,
        n_splits: int = 5,
        train_ratio: float = 0.7,
        purge_gap: int = 5,
        expanding: bool = False,
    ):
        if n_splits < 1:
            raise ValueError("n_splits must be >= 1")
        if not (0 < train_ratio < 1):
            raise ValueError("train_ratio must be in (0, 1)")

        self.n_splits = n_splits
        self.train_ratio = train_ratio
        self.purge_gap = purge_gap
        self.expanding = expanding

    def split(self, n_samples: int) -> List[WalkForwardSplit]:
        """
        生成 Walk-Forward 分割

        Args:
            n_samples: 总样本数

        Returns:
            WalkForwardSplit 列表
        """
        if n_samples < 10:
            raise ValueError(f"Need at least 10 samples, got {n_samples}")

        splits = []
        test_size = n_samples // (self.n_splits + 1)
        if test_size < 1:
            test_size = 1

        for i in range(self.n_splits):
            # 测试集终点
            test_end = n_samples - (self.n_splits - i - 1) * test_size
            test_start = test_end - test_size

            if test_start < 0:
                break

            # 训练集
            if self.expanding:
                # 扩展窗口：从开头开始
                train_start = 0
            else:
                # 滚动窗口：与测试集大小成比例
                train_size = int(test_start * self.train_ratio)
                train_start = max(0, test_start - train_size)

            train_end = test_start - self.purge_gap
            if train_end <= train_start:
                # purge gap 太大，跳过
                continue

            splits.append(
                WalkForwardSplit(
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    purge_gap=self.purge_gap,
                )
            )

        return splits

    def validate(
        self,
        data: pd.DataFrame,
        factor,
        score_fn=None,
    ) -> WalkForwardResult:
        """
        执行 Walk-Forward 验证

        Args:
            data: 带特征和目标列的 DataFrame
            factor: 实现了 fit() 和 predict() 的因子对象
            score_fn: 评分函数 (y_true, y_pred) -> float，默认使用 Spearman IC

        Returns:
            WalkForwardResult
        """
        from scipy import stats

        n_samples = len(data)
        splits = self.split(n_samples)

        if not splits:
            logger.warning("[WalkForward] No valid splits generated")
            return WalkForwardResult()

        if score_fn is None:
            def score_fn(y_true, y_pred):
                if len(y_true) < 3:
                    return 0.0
                corr, _ = stats.spearmanr(y_true, y_pred)
                return corr if not np.isnan(corr) else 0.0

        fold_results = []
        scores = []

        for i, split in enumerate(splits):
            train_data = data.iloc[split.train_start: split.train_end]
            test_data = data.iloc[split.test_start: split.test_end]

            if len(train_data) < 10 or len(test_data) < 1:
                continue

            try:
                # 训练
                factor.fit(train_data)
                # 预测
                preds = factor.predict(test_data)

                # 评分
                if "score" in preds.columns and factor.target in test_data.columns:
                    merged = preds.join(test_data[[factor.target]], how="inner")
                    if len(merged) > 0:
                        score = score_fn(merged[factor.target], merged["score"])
                    else:
                        score = 0.0
                else:
                    score = 0.0

                scores.append(score)
                fold_results.append(
                    {
                        "fold": i + 1,
                        "train_start": split.train_start,
                        "train_end": split.train_end,
                        "test_start": split.test_start,
                        "test_end": split.test_end,
                        "train_size": len(train_data),
                        "test_size": len(test_data),
                        "score": score,
                    }
                )

            except Exception as e:
                logger.debug(f"[WalkForward] Fold {i + 1} failed: {e}")
                fold_results.append(
                    {
                        "fold": i + 1,
                        "train_start": split.train_start,
                        "train_end": split.train_end,
                        "test_start": split.test_start,
                        "test_end": split.test_end,
                        "train_size": len(train_data),
                        "test_size": len(test_data),
                        "score": 0.0,
                        "error": str(e),
                    }
                )

        scores = [s for s in scores if not np.isnan(s)]

        result = WalkForwardResult(
            fold_results=fold_results,
            mean_score=float(np.mean(scores)) if scores else 0.0,
            std_score=float(np.std(scores)) if scores else 0.0,
            n_folds=len(fold_results),
        )

        if len(scores) > 1 and result.std_score > 0:
            result.sharpe_ratio = result.mean_score / result.std_score

        return result
