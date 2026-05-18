"""
交易审计日志

JSONL 格式持久化每一笔交易事件，用于事后追踪和合规审计。
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from dquant.broker.base import Order, OrderResult
from dquant.logger import get_logger

logger = get_logger(__name__)


class TradeJournal:
    """
    JSONL 格式交易审计日志

    每日一个文件，追加写入，每笔交易一行 JSON。

    Usage:
        journal = TradeJournal("./trade_journal")

        journal.record("ORDER_PLACED", order, result, strategy_name="MyStrategy")
        journal.record("ORDER_CANCELLED", order, strategy_name="MyStrategy")
    """

    def __init__(self, journal_dir: str = "./trade_journal"):
        self.journal_dir = Path(journal_dir)
        self.journal_dir.mkdir(parents=True, exist_ok=True)
        # 写入失败缓冲队列：磁盘满时暂存，避免崩掉交易循环
        self._write_failures: list = []

    def record(
        self,
        event_type: str,
        order: Order,
        result: Optional[OrderResult] = None,
        strategy_name: str = "",
        signal_info: Optional[dict] = None,
    ):
        """
        记录交易事件

        Args:
            event_type: 事件类型 (ORDER_PLACED, ORDER_FILLED, ORDER_REJECTED, ORDER_CANCELLED)
            order: 订单对象
            result: 订单结果 (可选)
            strategy_name: 策略名称
            signal_info: 信号来源信息
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "order_id": getattr(order, "order_id", "") or "",
            "symbol": order.symbol,
            "side": order.side,
            "quantity": order.quantity,
            "price": order.price,
            "order_type": getattr(order, "order_type", "MARKET"),
            "strategy": strategy_name,
            "signal": signal_info or {},
        }

        if result is not None:
            record.update(
                {
                    "filled_quantity": result.filled_quantity,
                    "filled_price": result.filled_price,
                    "commission": result.commission,
                    "status": result.status,
                }
            )

        date_str = datetime.now().strftime("%Y-%m-%d")
        filepath = self.journal_dir / f"{date_str}.jsonl"

        try:
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            # 写入成功后尝试刷出之前缓冲的失败记录
            self._flush_failures()
        except Exception as e:
            self._write_failures.append(record)
            logger.error(f"写入交易日志失败 (已缓冲 {len(self._write_failures)} 条): {e}")
            if len(self._write_failures) > 1000:
                # 审计日志不可丢失，超限仍向上抛出
                raise RuntimeError(
                    f"审计日志缓冲超过 1000 条 ({len(self._write_failures)})，"
                    f"请检查磁盘空间。最后错误: {e}"
                ) from e

    def _flush_failures(self):
        """尝试写入之前失败的缓冲记录"""
        if not self._write_failures:
            return
        remaining = []
        for record in self._write_failures:
            date_str = record.get("timestamp", datetime.now().isoformat())[:10]
            filepath = self.journal_dir / f"{date_str}.jsonl"
            try:
                with open(filepath, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            except Exception:
                remaining.append(record)
        if remaining:
            logger.warning(f"审计日志仍有 {len(remaining)} 条缓冲未写出")
        self._write_failures = remaining

    def read_day(self, date_str: str) -> list:
        """读取某天的所有交易记录"""
        filepath = self.journal_dir / f"{date_str}.jsonl"
        if not filepath.exists():
            return []

        records = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return records

    def get_today_summary(self) -> dict:
        """获取今日交易摘要"""
        date_str = datetime.now().strftime("%Y-%m-%d")
        records = self.read_day(date_str)

        placed = [r for r in records if r.get("event_type") == "ORDER_PLACED"]
        filled = [r for r in placed if r.get("status") == "FILLED"]
        rejected = [r for r in placed if r.get("status") == "REJECTED"]

        return {
            "date": date_str,
            "total_placed": len(placed),
            "total_filled": len(filled),
            "total_rejected": len(rejected),
            "symbols": list({r["symbol"] for r in placed}),
            "total_commission": sum(r.get("commission", 0) for r in filled),
        }
