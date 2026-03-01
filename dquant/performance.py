"""
性能优化模块

提供 numba JIT 加速、并行计算等功能。
"""

from dquant.logger import get_logger

logger = get_logger(__name__)

from typing import Optional, List, Callable, Any
import pandas as pd
import numpy as np
from functools import wraps
import time
from dquant.constants import DEFAULT_COMMISSION, DEFAULT_SLIPPAGE, DEFAULT_STAMP_DUTY, DEFAULT_INITIAL_CASH, MIN_SHARES, DEFAULT_WINDOW

# 尝试导入 numba
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # 如果 numba 不可用，提供空装饰器
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

# 尝试导入多进程
try:
    from multiprocessing import Pool, cpu_count
    MULTIPROCESSING_AVAILABLE = True
except ImportError:
    MULTIPROCESSING_AVAILABLE = False


def timing(func):
    """计时装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.debug(f"[{func.__name__}] 耗时: {elapsed:.4f}s")
        return result
    return wrapper


class NumbaAccelerator:
    """
    Numba 加速器
    
    使用 JIT 编译加速数值计算。
    """
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def rolling_mean_numba(arr: np.ndarray, window: int) -> np.ndarray:
        """并行滚动均值"""
        n = len(arr)
        result = np.empty(n)
        result[:window-1] = np.nan
        
        for i in prange(window-1, n):
            result[i] = np.mean(arr[i-window+1:i+1])
        
        return result
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def rolling_std_numba(arr: np.ndarray, window: int) -> np.ndarray:
        """并行滚动标准差"""
        n = len(arr)
        result = np.empty(n)
        result[:window-1] = np.nan
        
        for i in prange(window-1, n):
            result[i] = np.std(arr[i-window+1:i+1])
        
        return result
    
    @staticmethod
    @jit(nopython=True)
    def cumsum_numba(arr: np.ndarray) -> np.ndarray:
        """累积和"""
        result = np.empty_like(arr)
        result[0] = arr[0]
        
        for i in range(1, len(arr)):
            result[i] = result[i-1] + arr[i]
        
        return result
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def pct_change_numba(arr: np.ndarray) -> np.ndarray:
        """百分比变化"""
        n = len(arr)
        result = np.empty(n)
        result[0] = np.nan
        
        for i in prange(1, n):
            if arr[i-1] != 0:
                result[i] = (arr[i] - arr[i-1]) / arr[i-1]
            else:
                result[i] = np.nan
        
        return result
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def correlation_matrix_numba(data: np.ndarray) -> np.ndarray:
        """计算相关系数矩阵"""
        n_cols = data.shape[1]
        corr = np.eye(n_cols)
        
        for i in prange(n_cols):
            for j in range(i+1, n_cols):
                c = np.corrcoef(data[:, i], data[:, j])[0, 1]
                corr[i, j] = c
                corr[j, i] = c
        
        return corr
    
    @staticmethod
    def is_available() -> bool:
        """检查 numba 是否可用"""
        return NUMBA_AVAILABLE


class ParallelProcessor:
    """
    并行处理器
    
    使用多进程并行处理数据。
    """
    
    def __init__(self, n_workers: Optional[int] = None):
        """
        Args:
            n_workers: 工作进程数，默认使用 CPU 核心数
        """
        self.n_workers = n_workers or (cpu_count() if MULTIPROCESSING_AVAILABLE else 1)
    
    def map(
        self,
        func: Callable,
        items: List[Any],
        chunksize: int = 1,
    ) -> List[Any]:
        """
        并行映射
        
        Args:
            func: 处理函数
            items: 数据列表
            chunksize: 每个进程处理的块大小
        
        Returns:
            处理结果列表
        """
        if not MULTIPROCESSING_AVAILABLE or self.n_workers == 1:
            return [func(item) for item in items]
        
        with Pool(self.n_workers) as pool:
            results = pool.map(func, items, chunksize=chunksize)
        
        return results
    
    def starmap(
        self,
        func: Callable,
        items: List[tuple],
        chunksize: int = 1,
    ) -> List[Any]:
        """
        并行星型映射
        
        Args:
            func: 处理函数
            items: 参数元组列表
        """
        if not MULTIPROCESSING_AVAILABLE or self.n_workers == 1:
            return [func(*item) for item in items]
        
        with Pool(self.n_workers) as pool:
            results = pool.starmap(func, items, chunksize=chunksize)
        
        return results
    
    def apply_to_dataframe(
        self,
        df: pd.DataFrame,
        func: Callable,
        groupby: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        并行应用到 DataFrame
        
        Args:
            df: DataFrame
            func: 处理函数
            groupby: 分组列名
        """
        if groupby:
            groups = [group for _, group in df.groupby(groupby)]
            results = self.map(func, groups)
            return pd.concat(results)
        else:
            # 按行数分割
            chunk_size = max(1, len(df) // self.n_workers)
            chunks = [
                df.iloc[i:i+chunk_size]
                for i in range(0, len(df), chunk_size)
            ]
            results = self.map(func, chunks)
            return pd.concat(results)


class VectorizedOperations:
    """
    向量化操作
    
    提供高性能的向量化计算函数。
    """
    
    @staticmethod
    def rolling_apply(
        arr: np.ndarray,
        window: int,
        func: Callable,
    ) -> np.ndarray:
        """
        滚动应用函数（向量化）
        
        比 pandas rolling 更快
        """
        from numpy.lib.stride_tricks import sliding_window_view
        
        if len(arr) < window:
            return np.full(len(arr), np.nan)
        
        # 创建滑动窗口视图
        windows = sliding_window_view(arr, window)
        
        # 向量化应用函数
        result = np.array([func(w) for w in windows])
        
        # 填充前面的 NaN
        result = np.concatenate([np.full(window-1, np.nan), result])
        
        return result
    
    @staticmethod
    def expanding_apply(
        arr: np.ndarray,
        func: Callable,
        min_periods: int = 1,
    ) -> np.ndarray:
        """
        扩展应用函数（向量化）
        """
        result = np.empty(len(arr))
        result[:min_periods-1] = np.nan
        
        for i in range(min_periods-1, len(arr)):
            result[i] = func(arr[:i+1])
        
        return result
    
    @staticmethod
    def group_apply(
        df: pd.DataFrame,
        group_col: str,
        value_col: str,
        func: Callable,
    ) -> pd.Series:
        """
        分组应用函数（优化版）
        """
        result = pd.Series(index=df.index, dtype=float)
        
        for name, group in df.groupby(group_col):
            result.loc[group.index] = func(group[value_col])
        
        return result


class CacheManager:
    """
    缓存管理器
    
    缓存计算结果以避免重复计算。
    """
    
    def __init__(self, max_size: int = MIN_SHARES):
        self.cache = {}
        self.max_size = max_size
        self.access_count = {}
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        if key in self.cache:
            self.access_count[key] += 1
            return self.cache[key]
        return None
    
    def set(self, key: str, value: Any):
        """设置缓存"""
        # LRU 淘汰
        if len(self.cache) >= self.max_size:
            # 移除访问次数最少的
            min_key = min(self.access_count, key=self.access_count.get)
            del self.cache[min_key]
            del self.access_count[min_key]
        
        self.cache[key] = value
        self.access_count[key] = 0
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.access_count.clear()
    
    def memoize(self, func: Callable) -> Callable:
        """记忆化装饰器"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            key = f"{func.__name__}:{args}:{kwargs}"
            
            # 尝试从缓存获取
            cached = self.get(key)
            if cached is not None:
                return cached
            
            # 计算并缓存
            result = func(*args, **kwargs)
            self.set(key, result)
            
            return result
        
        return wrapper


class PerformanceMonitor:
    """
    性能监控器
    
    监控函数执行时间和内存使用。
    """
    
    def __init__(self):
        self.stats = {}
    
    def record(self, name: str, elapsed: float, memory: Optional[float] = None):
        """记录性能数据"""
        if name not in self.stats:
            self.stats[name] = {
                'calls': 0,
                'total_time': 0.0,
                'avg_time': 0.0,
                'max_time': 0.0,
                'min_time': float('inf'),
            }
        
        self.stats[name]['calls'] += 1
        self.stats[name]['total_time'] += elapsed
        self.stats[name]['avg_time'] = (
            self.stats[name]['total_time'] / self.stats[name]['calls']
        )
        self.stats[name]['max_time'] = max(self.stats[name]['max_time'], elapsed)
        self.stats[name]['min_time'] = min(self.stats[name]['min_time'], elapsed)
        
        if memory is not None:
            if 'max_memory' not in self.stats[name]:
                self.stats[name]['max_memory'] = memory
            else:
                self.stats[name]['max_memory'] = max(
                    self.stats[name]['max_memory'], memory
                )
    
    def monitor(self, func: Callable) -> Callable:
        """监控装饰器"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            
            try:
                import tracemalloc
                tracemalloc.start()
            except Exception:
                logger.warning(f"Operation failed: {e}")
            
            result = func(*args, **kwargs)
            
            elapsed = time.time() - start
            
            try:
                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                memory = peak / 1024 / 1024  # MB
            except Exception:
                memory = None
            
            self.record(func.__name__, elapsed, memory)
            
            return result
        
        return wrapper
    
    def get_report(self) -> str:
        """生成性能报告"""
        report = []
        report.append("="*60)
        report.append("性能报告")
        report.append("="*60)
        
        for name, stats in self.stats.items():
            report.append(f"\n{name}:")
            report.append(f"  调用次数: {stats['calls']}")
            report.append(f"  平均时间: {stats['avg_time']:.4f}s")
            report.append(f"  最大时间: {stats['max_time']:.4f}s")
            report.append(f"  最小时间: {stats['min_time']:.4f}s")
            report.append(f"  总时间:   {stats['total_time']:.4f}s")
            
            if 'max_memory' in stats:
                report.append(f"  峰值内存: {stats['max_memory']:.2f} MB")
        
        report.append("\n" + "="*60)
        
        return "\n".join(report)


# 全局实例
accelerator = NumbaAccelerator()
parallel_processor = ParallelProcessor()
cache_manager = CacheManager()
performance_monitor = PerformanceMonitor()
