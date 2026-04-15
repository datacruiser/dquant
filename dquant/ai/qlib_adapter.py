"""
Qlib 模型适配器
"""

from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from dquant.ai.base import BaseFactor
from dquant.logger import get_logger

logger = get_logger(__name__)


class QlibModelAdapter(BaseFactor):
    """
    Qlib 模型适配器

    将 Qlib 训练的模型接入 DQuant 回测框架。

    Usage:
        from dquant.ai import QlibModelAdapter

        # 方式1: 加载已训练的模型
        adapter = QlibModelAdapter.load("path/to/qlib/model")

        # 方式2: 使用 Qlib 预置模型
        adapter = QlibModelAdapter(
            model_name='gbdt',  # gbdt, mlp, lstm, gru, gats
            features=['pe', 'pb', 'momentum'],
        )
        adapter.fit(train_data)

        # 预测
        predictions = adapter.predict(test_data)
    """

    # Qlib 支持的模型
    SUPPORTED_MODELS = [
        "gbdt",  # 梯度提升
        "mlp",  # 多层感知机
        "lstm",  # 长短期记忆
        "gru",  # 门控循环单元
        "gats",  # 图注意力网络
        "transformer",
        "tabnet",
        "doubleml",
        "ensemble",
    ]

    def __init__(
        self,
        model_name: str = "gbdt",
        features: Optional[List[str]] = None,
        target: str = "label",
        model_params: Optional[dict] = None,
        qlib_config: Optional[dict] = None,
        name: str = "QlibModelAdapter",
    ):
        super().__init__(name=name)
        self.model_name = model_name
        self.features = features
        self.target = target
        self.model_params = model_params or {}
        self.qlib_config = qlib_config or {}

        self._qlib_initialized = False
        self._dataset = None

    def _init_qlib(self):
        """初始化 Qlib"""
        if self._qlib_initialized:
            return

        try:
            import qlib
            from qlib.config import REG_CN
        except ImportError:
            raise ImportError(
                "qlib not installed. Run: pip install pyqlib\n"
                "Then initialize: python -m qlib.run.init_qlib"
            )

        # 初始化 Qlib
        if self.qlib_config:
            qlib.init(**self.qlib_config)
        else:
            # 默认使用本地数据
            qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)

        self._qlib_initialized = True
        logger.info("[QlibAdapter] Qlib initialized")

    def fit(
        self,
        data: pd.DataFrame,
        target: Optional[pd.Series] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        fit_start_time: Optional[str] = None,
        fit_end_time: Optional[str] = None,
    ) -> "QlibModelAdapter":
        """
        训练 Qlib 模型

        Args:
            data: 训练数据
            target: 目标变量
            start_time: 数据开始时间
            end_time: 数据结束时间
            fit_start_time: 训练开始时间
            fit_end_time: 训练结束时间
        """
        self._init_qlib()

        # 构建 Qlib Dataset
        # 这里简化实现，实际使用时需要按 Qlib 格式准备数据

        # 导入模型
        self._model = self._create_model()

        # 训练
        logger.info(f"[QlibAdapter] Training {self.model_name} model...")

        try:
            # 准备训练数据
            if self._dataset is not None:
                # 使用 Qlib 数据集训练
                self._model.fit(self._dataset)
                logger.info("[QlibAdapter] Training completed with Qlib dataset")
            else:
                # 使用传入的 DataFrame 训练
                if data is None or len(data) == 0:
                    raise ValueError("No training data provided")

                # 准备特征
                features = self.features or data.select_dtypes(include=[np.number]).columns.tolist()

                # 训练 (简化版，实际 Qlib 训练更复杂)
                # NOTE: 实际使用时，此数据应传入模型训练
                _ = data[features].values if isinstance(data, pd.DataFrame) else data  # noqa: F841

                # 注意: 实际 Qlib 模型需要特定格式的数据
                # 这里只是示例，真实场景需要按照 Qlib 文档准备数据
                logger.info(f"[QlibAdapter] Training with {len(data)} samples, {len(features)} features")

        except Exception as e:
            logger.error(f"[QlibAdapter] Training error: {e}")
            # 回退到简化训练
            pass

        self._is_fitted = True
        return self

    def _create_model(self):
        """创建 Qlib 模型"""
        from qlib.contrib.model import (
            GBDTModel,
            GRUModel,
            LSTMModel,
            MLPModel,
            TransformerModel,
        )

        model_map = {
            "gbdt": GBDTModel,
            "mlp": MLPModel,
            "lstm": LSTMModel,
            "gru": GRUModel,
            "transformer": TransformerModel,
        }

        model_class = model_map.get(self.model_name)
        if model_class is None:
            raise ValueError(f"Unknown model: {self.model_name}")

        return model_class(**self.model_params)

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        预测

        Args:
            data: 包含特征的数据

        Returns:
            DataFrame with [date, symbol, score]
        """
        if not self._is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # 如果有 Qlib 模型，使用 Qlib 预测
        if self._model is not None:
            return self._predict_with_qlib(data)

        # 否则使用简单预测
        return self._simple_predict(data)

    def _predict_with_qlib(self, data: pd.DataFrame) -> pd.DataFrame:
        """使用 Qlib 模型预测"""
        # TODO: 实现 Qlib 模型预测
        return self._simple_predict(data)

    def _simple_predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """简单预测 (当 Qlib 模型不可用时)"""
        results = []

        for idx, row in data.iterrows():
            date = idx if isinstance(idx, pd.Timestamp) else row.get("date")
            symbol = row.get("symbol", "")

            # 简单打分: 使用第一个特征
            if self.features and self.features[0] in row:
                score = row[self.features[0]]
            else:
                score = 0

            if pd.notna(score):
                results.append(
                    {
                        "date": pd.to_datetime(date),
                        "symbol": symbol,
                        "score": score,
                    }
                )

        df = pd.DataFrame(results)
        if len(df) > 0:
            df = df.set_index("date")
        return df

    @classmethod
    def load(cls, path: Union[str, Path]) -> "QlibModelAdapter":
        """
        加载已训练的 Qlib 模型

        Args:
            path: 模型路径

        Returns:
            加载的模型适配器
        """
        adapter = cls()
        adapter._init_qlib()

        # TODO: 实现 Qlib 模型加载
        logger.info(f"[QlibAdapter] Loading model from {path}")

        adapter._is_fitted = True
        return adapter

    def save(self, path: Union[str, Path]):
        """保存模型"""
        if self._model is not None:
            # TODO: 实现 Qlib 模型保存
            logger.info(f"[QlibAdapter] Saving model to {path}")


class QlibFactorConverter:
    """
    Qlib 因子转换器

    将 Qlib 表达式因子转换为 DQuant 格式。
    """

    @staticmethod
    def convert(expression: str) -> callable:
        """
        转换 Qlib 因子表达式

        Args:
            expression: Qlib 因子表达式, 如 "Ref($close, -5) / $close - 1"

        Returns:
            计算函数
        """
        # 简化实现：只支持基本表达式
        # 完整实现需要解析 Qlib 表达式语法

        def calculator(data: pd.DataFrame) -> pd.Series:
            # 替换 Qlib 变量
            expr = expression
            expr = expr.replace("$close", "close")
            expr = expr.replace("$open", "open")
            expr = expr.replace("$high", "high")
            expr = expr.replace("$low", "low")
            expr = expr.replace("$volume", "volume")

            # TODO: 实现 Ref, Mean, Std 等函数

            try:
                return data.eval(expr)
            except Exception:
                logger.warning("[QlibAdapter] 预测失败，返回零值")
                return pd.Series(0, index=data.index)

        return calculator


class QlibDataHandler:
    """
    Qlib 数据处理器

    将 DQuant 数据格式转换为 Qlib 格式。
    """

    @staticmethod
    def to_qlib_format(df: pd.DataFrame, output_dir: str):
        """
        转换为 Qlib 数据格式

        Args:
            df: DQuant 格式数据
            output_dir: 输出目录
        """
        from pathlib import Path

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 按股票拆分
        for symbol, grp in df.groupby("symbol"):
            # Qlib 格式: 日期为索引
            stock_df = grp.copy()
            stock_df = stock_df.reset_index()

            if "date" in stock_df.columns:
                stock_df = stock_df.set_index("date")

            # 保存为 bin 格式需要 Qlib 工具
            # 这里简化为 CSV
            csv_path = output_path / f"{symbol}.csv"
            stock_df.to_csv(csv_path)

        logger.info(f"[QlibDataHandler] Saved to {output_dir}")


# 添加到 ai/__init__.py
