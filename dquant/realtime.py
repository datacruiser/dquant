"""
实时数据模块

支持 WebSocket 推送、实时行情订阅等。
"""

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional

from dquant.constants import DEFAULT_STAMP_DUTY
from dquant.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RealtimeQuote:
    """实时行情"""

    symbol: str
    price: float
    open: float
    high: float
    low: float
    volume: int
    turnover: float
    timestamp: datetime

    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "price": self.price,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "volume": self.volume,
            "turnover": self.turnover,
            "timestamp": self.timestamp.isoformat(),
        }


class RealtimeDataSource(ABC):
    """实时数据源基类"""

    @abstractmethod
    async def subscribe(self, symbols: List[str]):
        """订阅股票"""
        pass

    @abstractmethod
    async def unsubscribe(self, symbols: List[str]):
        """取消订阅"""
        pass

    @abstractmethod
    async def get_quote(self, symbol: str) -> Optional[RealtimeQuote]:
        """获取行情"""
        pass

    @abstractmethod
    def on_quote(self, callback: Callable[[RealtimeQuote], None]):
        """注册行情回调"""
        pass


class MockRealtimeSource(RealtimeDataSource):
    """
    模拟实时数据源

    用于测试和演示。
    """

    def __init__(self):
        self.subscribed = set()
        self.callbacks = []
        self.quotes = {}
        self._running = False

    async def subscribe(self, symbols: List[str]):
        """订阅"""
        for symbol in symbols:
            self.subscribed.add(symbol)
            # 生成初始行情
            self.quotes[symbol] = self._generate_quote(symbol)

    async def unsubscribe(self, symbols: List[str]):
        """取消订阅"""
        for symbol in symbols:
            self.subscribed.discard(symbol)
            if symbol in self.quotes:
                del self.quotes[symbol]

    async def get_quote(self, symbol: str) -> Optional[RealtimeQuote]:
        """获取行情"""
        return self.quotes.get(symbol)

    def on_quote(self, callback: Callable[[RealtimeQuote], None]):
        """注册回调"""
        self.callbacks.append(callback)

    def _generate_quote(self, symbol: str) -> RealtimeQuote:
        """生成模拟行情"""
        import random

        base_price = 10 + random.random() * 90
        price = base_price * (1 + random.random() * 0.02)

        return RealtimeQuote(
            symbol=symbol,
            price=price,
            open=base_price,
            high=price * 1.01,
            low=price * 0.99,
            volume=int(random.random() * 1000000),
            turnover=price * random.random() * 1000000,
            timestamp=datetime.now(),
        )

    async def start_streaming(self):
        """开始推送"""
        import random

        self._running = True

        while self._running:
            # 更新所有订阅的行情
            for symbol in self.subscribed:
                # 随机更新价格
                if symbol in self.quotes:
                    old_quote = self.quotes[symbol]
                    new_price = old_quote.price * (1 + random.random() * 0.002 - DEFAULT_STAMP_DUTY)

                    quote = RealtimeQuote(
                        symbol=symbol,
                        price=new_price,
                        open=old_quote.open,
                        high=max(old_quote.high, new_price),
                        low=min(old_quote.low, new_price),
                        volume=old_quote.volume + int(random.random() * 1000),
                        turnover=old_quote.turnover + new_price * random.random() * 100,
                        timestamp=datetime.now(),
                    )

                    self.quotes[symbol] = quote

                    # 触发回调
                    for callback in self.callbacks:
                        try:
                            callback(quote)
                        except Exception as e:
                            logger.error(f"回调错误: {e}")

            await asyncio.sleep(1)  # 每秒更新

    def stop_streaming(self):
        """停止推送"""
        self._running = False


class RealtimeManager:
    """
    实时数据管理器

    管理多个实时数据源。
    """

    def __init__(self):
        self.sources: Dict[str, RealtimeDataSource] = {}
        self.callbacks: List[Callable] = []

    def register_source(self, name: str, source: RealtimeDataSource):
        """注册数据源"""
        self.sources[name] = source
        source.on_quote(self._on_quote)

    def get_source(self, name: str) -> Optional[RealtimeDataSource]:
        """获取数据源"""
        return self.sources.get(name)

    async def subscribe(self, symbols: List[str], source: str = "default"):
        """订阅"""
        if source in self.sources:
            await self.sources[source].subscribe(symbols)

    async def unsubscribe(self, symbols: List[str], source: str = "default"):
        """取消订阅"""
        if source in self.sources:
            await self.sources[source].unsubscribe(symbols)

    def on_quote(self, callback: Callable[[RealtimeQuote], None]):
        """注册行情回调"""
        self.callbacks.append(callback)

    def _on_quote(self, quote: RealtimeQuote):
        """内部回调"""
        for callback in self.callbacks:
            try:
                callback(quote)
            except Exception as e:
                logger.error(f"回调错误: {e}")


# WebSocket 服务器
class RealtimeServer:
    """
    实时数据 WebSocket 服务器

    向客户端推送实时行情。
    """

    def __init__(self, manager: RealtimeManager, port: int = 8765):
        self.manager = manager
        self.port = port
        self.clients = set()

    async def handle_client(self, websocket, path):
        """处理客户端连接"""
        self.clients.add(websocket)

        try:
            # 发送欢迎消息
            await websocket.send(
                json.dumps(
                    {
                        "type": "connected",
                        "message": "Welcome to DQuant Realtime Server",
                    }
                )
            )

            # 接收客户端消息
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._handle_message(websocket, data)
                except Exception as e:
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "error",
                                "message": str(e),
                            }
                        )
                    )

        finally:
            self.clients.discard(websocket)

    async def _handle_message(self, websocket, data: Dict):
        """处理客户端消息"""
        msg_type = data.get("type")

        if msg_type == "subscribe":
            # 订阅股票
            symbols = data.get("symbols", [])
            await self.manager.subscribe(symbols)

            await websocket.send(
                json.dumps(
                    {
                        "type": "subscribed",
                        "symbols": symbols,
                    }
                )
            )

        elif msg_type == "unsubscribe":
            # 取消订阅
            symbols = data.get("symbols", [])
            await self.manager.unsubscribe(symbols)

            await websocket.send(
                json.dumps(
                    {
                        "type": "unsubscribed",
                        "symbols": symbols,
                    }
                )
            )

        else:
            await websocket.send(
                json.dumps(
                    {
                        "type": "error",
                        "message": f"Unknown message type: {msg_type}",
                    }
                )
            )

    def broadcast_quote(self, quote: RealtimeQuote):
        """广播行情"""
        message = json.dumps(
            {
                "type": "quote",
                "data": quote.to_dict(),
            }
        )

        # 向所有客户端发送
        for client in self.clients:
            try:
                asyncio.create_task(client.send(message))
            except Exception as e:
                logger.warning(f"Failed to send message to client: {e}")

    async def start(self):
        """启动服务器"""
        try:
            import websockets
        except ImportError:
            logger.warning("需要安装 websockets: pip install websockets")
            return

        # 注册行情回调
        self.manager.on_quote(self.broadcast_quote)

        async with websockets.serve(self.handle_client, "localhost", self.port):
            logger.info(f"WebSocket 服务器启动: ws://localhost:{self.port}")
            await asyncio.Future()  # 永久运行


# WebSocket 客户端
class RealtimeClient:
    """
    实时数据 WebSocket 客户端

    订阅并接收实时行情。
    """

    def __init__(self, url: str = "ws://localhost:8765"):
        self.url = url
        self.websocket = None
        self.callbacks = []

    async def connect(self):
        """连接服务器"""
        try:
            import websockets
        except ImportError:
            logger.warning("需要安装 websockets: pip install websockets")
            return

        self.websocket = await websockets.connect(self.url)

    async def subscribe(self, symbols: List[str]):
        """订阅股票"""
        if not self.websocket:
            await self.connect()

        await self.websocket.send(
            json.dumps(
                {
                    "type": "subscribe",
                    "symbols": symbols,
                }
            )
        )

    async def unsubscribe(self, symbols: List[str]):
        """取消订阅"""
        if self.websocket:
            await self.websocket.send(
                json.dumps(
                    {
                        "type": "unsubscribe",
                        "symbols": symbols,
                    }
                )
            )

    def on_quote(self, callback: Callable[[RealtimeQuote], None]):
        """注册行情回调"""
        self.callbacks.append(callback)

    async def listen(self):
        """监听消息"""
        if not self.websocket:
            await self.connect()

        async for message in self.websocket:
            try:
                data = json.loads(message)

                if data.get("type") == "quote":
                    quote_data = data.get("data", {})
                    quote = RealtimeQuote(
                        symbol=quote_data["symbol"],
                        price=quote_data["price"],
                        open=quote_data["open"],
                        high=quote_data["high"],
                        low=quote_data["low"],
                        volume=quote_data["volume"],
                        turnover=quote_data["turnover"],
                        timestamp=datetime.fromisoformat(quote_data["timestamp"]),
                    )

                    for callback in self.callbacks:
                        try:
                            callback(quote)
                        except Exception as e:
                            logger.error(f"回调错误: {e}")

            except Exception as e:
                logger.error(f"消息处理错误: {e}")

    async def close(self):
        """关闭连接"""
        if self.websocket:
            await self.websocket.close()


# 便捷函数
def create_mock_realtime_manager() -> RealtimeManager:
    """创建模拟实时数据管理器"""
    manager = RealtimeManager()
    manager.register_source("default", MockRealtimeSource())
    return manager
