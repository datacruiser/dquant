"""
Qlib 模型适配器
"""

from pathlib import Path
from typing import List, Optional, Union

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
        # 先做契约检查：没有 Qlib Dataset 时直接抛错，
        # 避免下面去 import qlib + create_model 才暴露问题。
        if self._dataset is None:
            raise NotImplementedError(
                "QlibModelAdapter.fit() 当前仅支持预先构造好的 Qlib Dataset。"
                "请先通过 Qlib DataHandler 构建 dataset 并注入 adapter._dataset，"
                "或直接使用 dquant.ai.XGBoostFactor / LGBMFactor。"
            )

        self._init_qlib()

        # 导入模型
        self._model = self._create_model()

        # 训练
        logger.info(f"[QlibAdapter] Training {self.model_name} model...")

        try:
            self._model.fit(self._dataset)
            logger.info("[QlibAdapter] Training completed with Qlib dataset")
            self._is_fitted = True
        except Exception as e:
            logger.error(f"[QlibAdapter] Training error: {e}")
            raise

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
        if self._dataset is None or self._model is None:
            raise NotImplementedError(
                "QlibModelAdapter._predict_with_qlib 当前未实现。"
                "适配器仅在 self._dataset / self._model 都被正确注入时才能预测。"
            )
        return self._model.predict(self._dataset)

    def _simple_predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """已废弃：原来会用第一个特征原值作为 score，具有误导性。"""
        raise NotImplementedError(
            "QlibModelAdapter._simple_predict 已废弃。"
            "如需轻量预测请直接使用 dquant.ai.XGBoostFactor / LGBMFactor。"
        )

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

            # 安全检查：仅允许算术运算和列名引用
            import re

            safe_pattern = re.compile(r"^[0-9a-zA-Z_\s\.\+\-\*/\(\),]+$")
            if not safe_pattern.match(expr):
                logger.warning(f"[QlibAdapter] 表达式包含不安全字符，拒绝执行: {expr}")
                return pd.Series(0, index=data.index)

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
