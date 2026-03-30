"""
数据源管理器

统一管理多个数据源，支持缓存、增量更新等。
"""

from typing import Optional, List, Dict, Any, Union, Type
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import json

from dquant.data.base import DataSource
from dquant.logger import get_logger

logger = get_logger(__name__)


class DataSourceRegistry:
    """
    数据源注册表

    管理所有可用的数据源类型。
    """

    _sources: Dict[str, Type[DataSource]] = {}

    @classmethod
    def register(cls, name: str, source_class: Type[DataSource]):
        """注册数据源"""
        if '_sources' not in cls.__dict__:
            cls._sources = dict(cls._sources)
        cls._sources[name] = source_class

    @classmethod
    def get(cls, name: str) -> Optional[Type[DataSource]]:
        """获取数据源类"""
        return cls._sources.get(name)

    @classmethod
    def list_sources(cls) -> List[str]:
        """列出所有数据源"""
        return list(cls._sources.keys())

    @classmethod
    def create(cls, name: str, **kwargs) -> DataSource:
        """创建数据源实例"""
        source_class = cls.get(name)
        if source_class is None:
            raise ValueError(f"Unknown data source: {name}")
        return source_class(**kwargs)


# 注册内置数据源
def _register_builtin_sources():
    """注册内置数据源"""
    try:
        from dquant.data.csv_loader import CSVLoader
        DataSourceRegistry.register('csv', CSVLoader)
    except ImportError as e:
        logger.debug(f"CSVLoader not available: {e}")

    try:
        from dquant.data.akshare_loader import AKShareLoader
        DataSourceRegistry.register('akshare', AKShareLoader)
    except ImportError as e:
        logger.debug(f"AKShareLoader not available: {e}")

    try:
        from dquant.data.tushare_loader import TushareLoader
        DataSourceRegistry.register('tushare', TushareLoader)
    except ImportError as e:
        logger.debug(f"TushareLoader not available: {e}")

    try:
        from dquant.data.yahoo_loader import YahooLoader
        DataSourceRegistry.register('yahoo', YahooLoader)
    except ImportError as e:
        logger.debug(f"YahooLoader not available: {e}")

    try:
        from dquant.data.jqdata_loader import JQDataLoader
        DataSourceRegistry.register('jqdata', JQDataLoader)
    except ImportError as e:
        logger.debug(f"JQDataLoader not available: {e}")

    try:
        from dquant.data.ricequant_loader import RiceQuantLoader
        DataSourceRegistry.register('ricequant', RiceQuantLoader)
    except ImportError as e:
        logger.debug(f"RiceQuantLoader not available: {e}")

    try:
        from dquant.data.tdx_loader import TDXLoader
        DataSourceRegistry.register('tdx', TDXLoader)
    except ImportError as e:
        logger.debug(f"TDXLoader not available: {e}")

    try:
        from dquant.data.database_loader import DatabaseLoader, MongoLoader
        DataSourceRegistry.register('sql', DatabaseLoader)
        DataSourceRegistry.register('mongodb', MongoLoader)
    except ImportError as e:
        logger.debug(f"DatabaseLoader not available: {e}")


_register_builtin_sources()


class DataManager:
    """
    数据管理器

    统一管理数据加载、缓存、更新等。

    Usage:
        # 创建管理器
        dm = DataManager(cache_dir='./cache')

        # 加载数据
        df = dm.load(
            source='akshare',
            symbols='hs300',
            start='2022-01-01',
        )

        # 增量更新
        dm.update(
            source='akshare',
            symbols='hs300',
        )

        # 批量加载
        dfs = dm.load_batch([
            {'source': 'akshare', 'symbols': 'hs300'},
            {'source': 'tushare', 'symbols': 'zz500'},
        ])
    """

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        cache_expire: int = 24,  # 缓存过期时间(小时)
        default_source: str = 'akshare',
    ):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cache_expire = cache_expire
        self.default_source = default_source

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load(
        self,
        source: Optional[str] = None,
        symbols: Union[str, List[str]] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        use_cache: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """
        加载数据

        Args:
            source: 数据源名称
            symbols: 股票代码
            start: 开始日期
            end: 结束日期
            use_cache: 是否使用缓存
            **kwargs: 其他参数

        Returns:
            DataFrame
        """
        source = source or self.default_source

        # 检查缓存
        cache_key = self._get_cache_key(source, symbols, start, end, kwargs)
        if use_cache and self.cache_dir:
            cached = self._load_cache(cache_key)
            if cached is not None:
                logger.debug(f"Loaded from cache: {cache_key}")
                return cached

        # 创建数据源并加载
        loader = DataSourceRegistry.create(source, symbols=symbols, start=start, end=end, **kwargs)
        df = loader.load()

        # 保存缓存
        if self.cache_dir:
            self._save_cache(cache_key, df)
            logger.debug(f"Saved to cache: {cache_key}")

        return df

    def update(
        self,
        source: Optional[str] = None,
        symbols: Union[str, List[str]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        增量更新数据

        Args:
            source: 数据源名称
            symbols: 股票代码
            **kwargs: 其他参数

        Returns:
            更新后的完整数据
        """
        source = source or self.default_source

        # 获取缓存中的最新日期
        cache_key = self._get_cache_key(source, symbols, None, None, kwargs)
        cached = self._load_cache(cache_key) if self.cache_dir else None

        if cached is not None and len(cached) > 0:
            last_date = cached.index.max()
            start = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
        else:
            start = None

        # 加载新数据
        new_data = self.load(
            source=source,
            symbols=symbols,
            start=start,
            use_cache=False,
            **kwargs
        )

        # 合并
        if cached is not None and len(new_data) > 0:
            # 去重
            combined = pd.concat([cached, new_data], axis=0)
            combined = combined[~combined.index.duplicated(keep='last')]
            return combined.sort_index()

        return new_data if len(new_data) > 0 else cached

    def load_batch(
        self,
        configs: List[Dict[str, Any]],
        merge: bool = False,
    ) -> Union[List[pd.DataFrame], pd.DataFrame]:
        """
        批量加载数据

        Args:
            configs: 配置列表
            merge: 是否合并为一个 DataFrame

        Returns:
            DataFrame 列表或合并后的 DataFrame
        """
        dfs = []
        errors = []

        for i, config in enumerate(configs):
            try:
                df = self.load(**config)
                dfs.append(df)
                logger.info(f"Loaded {i+1}/{len(configs)}")
            except Exception as e:
                logger.error(f"[DataManager] Failed to load config {i+1}: {config} — {e}")
                errors.append({'index': i, 'config': config, 'error': str(e)})
                dfs.append(pd.DataFrame())

        if errors:
            logger.warning(f"[DataManager] {len(errors)}/{len(configs)} configs failed to load")

        if merge:
            return pd.concat(dfs, axis=0)

        return dfs

    def _get_cache_key(self, source: str, symbols, start, end, kwargs) -> str:
        """生成缓存键"""
        if isinstance(symbols, list):
            symbols = ','.join(symbols[:5]) + f'...{len(symbols)}'

        key_parts = [source, str(symbols), str(start), str(end)]
        key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])

        return '_'.join(key_parts).replace('/', '_').replace(' ', '_')

    def _load_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """加载缓存"""
        cache_file = self.cache_dir / f"{cache_key}.parquet"
        meta_file = self.cache_dir / f"{cache_key}.meta.json"

        if not cache_file.exists():
            return None

        # 检查过期
        if meta_file.exists():
            with open(meta_file, 'r') as f:
                meta = json.load(f)

            cached_time = datetime.fromisoformat(meta['timestamp'])
            if datetime.now() - cached_time > timedelta(hours=self.cache_expire):
                return None

        # 加载数据
        try:
            return pd.read_parquet(cache_file)
        except Exception:
            return None

    def _save_cache(self, cache_key: str, df: pd.DataFrame):
        """保存缓存"""
        cache_file = self.cache_dir / f"{cache_key}.parquet"
        meta_file = self.cache_dir / f"{cache_key}.meta.json"

        # 保存数据
        df.to_parquet(cache_file, index=True)

        # 保存元数据
        meta = {
            'timestamp': datetime.now().isoformat(),
            'rows': len(df),
            'columns': list(df.columns),
        }
        with open(meta_file, 'w') as f:
            json.dump(meta, f)

    def clear_cache(self, older_than: Optional[int] = None):
        """
        清理缓存

        Args:
            older_than: 清理多少小时前的缓存（必须显式指定）
        """
        if not self.cache_dir:
            return

        if older_than is None:
            # 安全起见，不指定 older_than 时仅清理已过期的缓存
            for cache_file in self.cache_dir.glob('*.parquet'):
                meta_file = cache_file.with_suffix('.meta.json')
                if meta_file.exists():
                    with open(meta_file, 'r') as f:
                        meta = json.load(f)
                    cached_time = datetime.fromisoformat(meta['timestamp'])
                    if datetime.now() - cached_time > timedelta(hours=self.cache_expire):
                        cache_file.unlink()
                        meta_file.unlink()
            logger.info("Expired cache cleared")
            return

        for cache_file in self.cache_dir.glob('*.parquet'):
            meta_file = cache_file.with_suffix('.meta.json')
            if meta_file.exists():
                with open(meta_file, 'r') as f:
                    meta = json.load(f)
                cached_time = datetime.fromisoformat(meta['timestamp'])
                if datetime.now() - cached_time > timedelta(hours=older_than):
                    cache_file.unlink()
                    meta_file.unlink()

        logger.info(f"Cache older than {older_than}h cleared")


def load_data(
    source: str = 'akshare',
    symbols: Union[str, List[str]] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    快捷加载函数

    Args:
        source: 数据源名称
        symbols: 股票代码
        start: 开始日期
        end: 结束日期
        **kwargs: 其他参数

    Returns:
        DataFrame
    """
    dm = DataManager()
    return dm.load(source=source, symbols=symbols, start=start, end=end, **kwargs)
