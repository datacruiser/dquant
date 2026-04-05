"""
通达信本地数据加载器

直接读取通达信本地数据文件，无需网络请求。
"""

import struct
from pathlib import Path
from typing import List, Optional

import pandas as pd

from dquant.data.base import DataSource


class TDXLoader(DataSource):
    """
    通达信本地数据加载器

    直接读取通达信安装目录下的 .day/.lc5/.lc1 文件。

    Usage:
        loader = TDXLoader(
            tdx_path='C:/通达信/vipdoc',
            market='sz',  # sz, sh
            symbols=['000001', '000002'],
        )
        df = loader.load()
    """

    # 通达信文件结构
    # vipdoc/sh/lday/  - 上海日线
    # vipdoc/sz/lday/  - 深圳日线
    # vipdoc/sh/minline/ - 上海分钟线

    MARKET_MAP = {
        "sh": "sh",
        "sz": "sz",
        "bj": "bj",
    }

    def __init__(
        self,
        tdx_path: str,
        market: str = "sz",
        symbols: Optional[List[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        freq: str = "day",  # day, lc5(5分钟), lc1(1分钟)
        include_factors: bool = True,
    ):
        super().__init__(symbols=symbols, start=start, end=end)
        self.tdx_path = Path(tdx_path)
        self.market = market.lower()
        self.freq = freq
        self.include_factors = include_factors

    def _get_data_dir_and_ext(self):
        """获取数据目录和文件扩展名"""
        freq_map = {
            "day": ("lday", ".day"),
            "lc5": ("minline", ".lc5"),
            "lc1": ("minline", ".lc1"),
        }

        if self.freq not in freq_map:
            raise ValueError(f"Unknown freq: {self.freq}")

        subdir, ext = freq_map[self.freq]
        return self.tdx_path / self.market / subdir, ext

    def _get_symbol_list(self, data_dir, file_ext):
        """获取股票列表"""
        if self.symbols is None:
            files = list(data_dir.glob(f"*{file_ext}"))
            return [f.stem for f in files]
        return self.symbols

    def _load_single_symbol(self, data_dir, symbol, file_ext):
        """加载单个股票数据"""
        file_path = data_dir / f"{symbol}{file_ext}"
        if file_path.exists():
            return self._read_tdx_file(file_path, symbol)
        return None

    def load(self) -> pd.DataFrame:
        """加载数据"""
        # 获取数据目录和扩展名
        data_dir, file_ext = self._get_data_dir_and_ext()

        if not data_dir.exists():
            raise FileNotFoundError(f"TDX data directory not found: {data_dir}")

        # 获取股票列表
        symbol_list = self._get_symbol_list(data_dir, file_ext)

        all_data = []
        failed = []

        for symbol in symbol_list:
            try:
                df = self._load_single_symbol(data_dir, symbol, file_ext)
                if df is not None and len(df) > 0:
                    all_data.append(df)
            except Exception as e:
                failed.append((symbol, str(e)))

        if failed:
            print(f"  [TDX] 加载失败: {len(failed)} 只")

        if not all_data:
            raise ValueError("No data loaded")

        result = pd.concat(all_data, axis=0, ignore_index=False)

        # 过滤日期
        if self.start:
            result = result[result.index >= pd.to_datetime(self.start)]
        if self.end:
            result = result[result.index <= pd.to_datetime(self.end)]

        # 计算因子
        if self.include_factors:
            result = self._calculate_factors(result)

        self.validate(result)

        return result.sort_index()

    def _read_tdx_file(self, file_path: Path, symbol: str) -> Optional[pd.DataFrame]:
        """读取通达信数据文件"""
        try:
            with open(file_path, "rb") as f:
                data = f.read()

            # 通达信日线文件格式: 每条记录32字节
            # 4字节日期, 4字节开盘价, 4字节最高价, 4字节最低价, 4字节收盘价, 4字节成交额, 4字节成交量, 4字节保留

            record_size = 32
            num_records = len(data) // record_size

            records = []
            for i in range(num_records):
                offset = i * record_size
                record = data[offset : offset + record_size]

                # 解析 (通达信价格需要除以100)
                date_int, open_p, high_p, low_p, close_p, amount, volume, _ = (
                    struct.unpack("IIIIIIII", record)
                )

                # 日期转换
                year = date_int // 10000
                month = (date_int % 10000) // 100
                day = date_int % 100

                try:
                    date = pd.Timestamp(year=year, month=month, day=day)
                except Exception:
                    continue

                records.append(
                    {
                        "date": date,
                        "open": open_p / 100.0,
                        "high": high_p / 100.0,
                        "low": low_p / 100.0,
                        "close": close_p / 100.0,
                        "amount": amount / 10000.0,  # 万元
                        "volume": volume,
                    }
                )

            df = pd.DataFrame(records)
            df["symbol"] = symbol + (".SH" if self.market == "sh" else ".SZ")
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")

            return df

        except Exception:
            return None

    def _calculate_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算技术因子"""
        results = []

        for symbol, group in df.groupby("symbol"):
            group = group.sort_index()

            group["momentum_5"] = group["close"].pct_change(5)
            group["momentum_10"] = group["close"].pct_change(10)
            group["momentum_20"] = group["close"].pct_change(20)

            returns = group["close"].pct_change()
            group["volatility_20"] = returns.rolling(20).std()

            group["ma_5"] = group["close"].rolling(5).mean()
            group["ma_10"] = group["close"].rolling(10).mean()
            group["ma_20"] = group["close"].rolling(20).mean()

            results.append(group)

        return pd.concat(results)


class TDXBlockLoader:
    """
    通达信板块数据加载器
    """

    def __init__(self, tdx_path: str):
        self.tdx_path = Path(tdx_path)

    def get_blocks(self) -> dict:
        """获取所有板块及其成分股"""
        block_file = self.tdx_path / "T0002" / "hq_cache" / "block.dat"

        if not block_file.exists():
            return {}

        # 解析板块文件 (格式较复杂，这里简化处理)
        blocks = {}

        # 备用方案：读取板块目录
        block_dir = self.tdx_path / "T0002" / "block"
        if block_dir.exists():
            for blk_file in block_dir.glob("*.blk"):
                with open(blk_file, "r", encoding="gbk") as f:
                    content = f.read()
                    # 解析成分股
                    stocks = []
                    for line in content.split("\n"):
                        line = line.strip()
                        if line and not line.startswith("["):
                            stocks.append(line)

                    blocks[blk_file.stem] = stocks

        return blocks

    def get_concept_blocks(self) -> dict:
        """获取概念板块"""
        return self.get_blocks()
