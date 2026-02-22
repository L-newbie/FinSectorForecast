# -*- coding: utf-8 -*-
"""
内存缓存系统

特性：
1. LRU (最近最少使用) 淘汰策略
2. TTL (生存时间) 过期支持
3. 缓存容量限制
4. 线程安全
5. 完整的API接口
6. 缓存统计和监控

作者: AI Assistant
版本: 2.0.0
"""

import threading
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Any, Optional, Dict, Tuple, Callable
from functools import wraps
import copy
import hashlib
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('MemoryCache')


class CacheEntry:
    """缓存条目封装类"""
    
    def __init__(self, key: str, value: Any, ttl: Optional[int] = None):
        self.key = key
        self.value = value
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        self.access_count = 0
        self.ttl = ttl  # TTL in seconds, None means no expiration
    
    def is_expired(self) -> bool:
        """检查缓存是否过期"""
        if self.ttl is None:
            return False
        age = (datetime.now() - self.created_at).total_seconds()
        return age > self.ttl
    
    def access(self) -> Any:
        """访问缓存并返回 value"""
        self.last_accessed = datetime.now()
        self.access_count += 1
        return self.value
    
    def update(self, value: Any) -> None:
        """更新缓存值"""
        self.value = value
        self.created_at = datetime.now()
    
    def get_age_seconds(self) -> float:
        """获取缓存创建以来的秒数"""
        return (datetime.now() - self.created_at).total_seconds()
    
    def get_idle_seconds(self) -> float:
        """获取缓存未被访问的秒数"""
        return (datetime.now() - self.last_accessed).total_seconds()


class LRUCache:
    """
    LRU (最近最少使用) 缓存实现
    
    特性：
    - 固定容量限制
    - LRU 淘汰策略
    - TTL 支持
    - 线程安全
    """
    
    def __init__(
        self, 
        max_size: int = 1000, 
        default_ttl: Optional[int] = None,
        on_evict: Optional[Callable[[str, Any], None]] = None
    ):
        """
        初始化 LRU 缓存
        
        Args:
            max_size: 最大缓存条目数量
            default_ttl: 默认TTL（秒），None表示不过期
            on_evict: 淘汰回调函数，签名: (key, value) -> None
        """
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_size = max_size
        self._default_ttl = default_ttl
        self._on_evict = on_evict
        self._lock = threading.RLock()
        
        # 统计信息
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'sets': 0,
            'deletes': 0
        }
        
        logger.info(f"LRUCache 初始化: max_size={max_size}, default_ttl={default_ttl}s")
    
    def get(self, key: str) -> Tuple[bool, Any]:
        """
        获取缓存
        
        Args:
            key: 缓存键
        
        Returns:
            (是否命中, 缓存值或None)
        """
        with self._lock:
            if key not in self._cache:
                self._stats['misses'] += 1
                return False, None
            
            entry = self._cache[key]
            
            # 检查是否过期
            if entry.is_expired():
                self._remove_entry(key)
                self._stats['misses'] += 1
                logger.debug(f"缓存过期: {key}")
                return False, None
            
            # 移动到末尾（最新使用）
            self._cache.move_to_end(key)
            
            # 返回值的深拷贝以避免外部修改
            value = entry.access()
            self._stats['hits'] += 1
            
            return True, copy.deepcopy(value)
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        设置缓存
        
        Args:
            key: 缓存键
            value: 缓存值
            ttl: TTL（秒），None使用默认值
        """
        with self._lock:
            # 如果键已存在，先移除（触发淘汰回调）
            if key in self._cache:
                old_entry = self._cache[key]
                self._remove_entry(key)
            
            # 如果缓存已满，执行 LRU 淘汰
            while len(self._cache) >= self._max_size:
                self._evict_lru()
            
            # 创建新条目
            actual_ttl = ttl if ttl is not None else self._default_ttl
            entry = CacheEntry(key, copy.deepcopy(value), actual_ttl)
            self._cache[key] = entry
            
            self._stats['sets'] += 1
            logger.debug(f"缓存设置: {key}, ttl={actual_ttl}s")
    
    def delete(self, key: str) -> bool:
        """
        删除缓存
        
        Args:
            key: 缓存键
        
        Returns:
            是否成功删除
        """
        with self._lock:
            if key in self._cache:
                self._remove_entry(key)
                self._stats['deletes'] += 1
                return True
            return False
    
    def clear(self) -> int:
        """
        清空所有缓存
        
        Returns:
            清空的缓存数量
        """
        with self._lock:
            count = len(self._cache)
            
            # 触发所有淘汰回调
            for key, entry in self._cache.items():
                if self._on_evict:
                    try:
                        self._on_evict(key, entry.value)
                    except Exception as e:
                        logger.error(f"淘汰回调执行失败: {e}")
            
            self._cache.clear()
            logger.info(f"缓存已清空: {count} 条目")
            return count
    
    def has(self, key: str) -> bool:
        """
        检查键是否存在且未过期
        
        Args:
            key: 缓存键
        
        Returns:
            是否存在且有效
        """
        with self._lock:
            if key not in self._cache:
                return False
            entry = self._cache[key]
            return not entry.is_expired()
    
    def _remove_entry(self, key: str) -> None:
        """移除缓存条目"""
        if key in self._cache:
            entry = self._cache.pop(key)
            if self._on_evict:
                try:
                    self._on_evict(key, entry.value)
                except Exception as e:
                    logger.error(f"淘汰回调执行失败: {e}")
            self._stats['evictions'] += 1
    
    def _evict_lru(self) -> None:
        """淘汰最近最少使用的条目"""
        if self._cache:
            # OrderedDict 的第一个元素是最久未使用的
            key = next(iter(self._cache))
            self._remove_entry(key)
            logger.debug(f"LRU 淘汰: {key}")
    
    def cleanup_expired(self) -> int:
        """
        清理所有过期缓存
        
        Returns:
            清理的过期缓存数量
        """
        with self._lock:
            expired_keys = []
            for key, entry in self._cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._remove_entry(key)
            
            if expired_keys:
                logger.info(f"清理过期缓存: {len(expired_keys)} 条目")
            
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            统计信息字典
        """
        with self._lock:
            total = self._stats['hits'] + self._stats['misses']
            hit_rate = self._stats['hits'] / total if total > 0 else 0.0
            
            return {
                'size': len(self._cache),
                'max_size': self._max_size,
                'hits': self._stats['hits'],
                'misses': self._stats['misses'],
                'hit_rate': round(hit_rate, 4),
                'evictions': self._stats['evictions'],
                'sets': self._stats['sets'],
                'deletes': self._stats['deletes']
            }
    
    def get_all_keys(self) -> list:
        """获取所有缓存键"""
        with self._lock:
            return list(self._cache.keys())
    
    def get_entries_info(self) -> list:
        """获取所有缓存条目的详细信息"""
        with self._lock:
            return [
                {
                    'key': key,
                    'age_seconds': entry.get_age_seconds(),
                    'idle_seconds': entry.get_idle_seconds(),
                    'access_count': entry.access_count,
                    'ttl': entry.ttl,
                    'expired': entry.is_expired()
                }
                for key, entry in self._cache.items()
            ]


class MemoryCache:
    """
    内存缓存管理器 - 完整的缓存系统
    
    特性：
    - 多缓存实例支持
    - 命名空间隔离
    - 默认TTL配置
    - 容量限制
    - 自动过期清理
    - 统计监控
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(
        self,
        default_ttl: int = 3600,
        max_size: int = 1000,
        cleanup_interval: int = 300,
        _force_init: bool = False
    ):
        """
        初始化内存缓存管理器
        
        Args:
            default_ttl: 默认TTL（秒），默认1小时
            max_size: 每个命名空间的最大缓存数量
            cleanup_interval: 过期清理间隔（秒），默认5分钟
            _force_init: 强制重新初始化（内部使用）
        """
        # 如果已初始化且非强制重新初始化，则跳过
        if self._initialized and not _force_init:
            return
        
        self._initialized = True
        self._caches: Dict[str, LRUCache] = {}
        self._config: Dict[str, Dict] = {}
        self._manager_lock = threading.RLock()
        
        # 默认配置
        self._default_ttl = default_ttl
        self._default_max_size = max_size
        self._cleanup_interval = cleanup_interval
        
        # 启动后台清理任务
        self._cleanup_thread: Optional[threading.Thread] = None
        self._stop_cleanup = threading.Event()
        
        logger.info(
            f"MemoryCache 初始化: default_ttl={default_ttl}s, "
            f"max_size={max_size}, cleanup_interval={cleanup_interval}s"
        )
    
    def reconfigure(
        self,
        default_ttl: int = None,
        max_size: int = None,
        cleanup_interval: int = None
    ):
        """
        重新配置缓存管理器
        
        Args:
            default_ttl: 默认TTL
            max_size: 默认最大容量
            cleanup_interval: 清理间隔
        """
        if default_ttl is not None:
            self._default_ttl = default_ttl
        if max_size is not None:
            self._default_max_size = max_size
        if cleanup_interval is not None:
            self._cleanup_interval = cleanup_interval
        
        logger.info(
            f"MemoryCache 重新配置: default_ttl={self._default_ttl}s, "
            f"max_size={self._default_max_size}, cleanup_interval={self._cleanup_interval}s"
        )
    
    def create_namespace(
        self, 
        name: str, 
        ttl: Optional[int] = None,
        max_size: Optional[int] = None
    ) -> bool:
        """
        创建命名空间
        
        Args:
            name: 命名空间名称
            ttl: TTL（秒），None使用默认值
            max_size: 最大容量，None使用默认值
        
        Returns:
            是否创建成功
        """
        with self._manager_lock:
            if name in self._caches:
                return False
            
            actual_ttl = ttl if ttl is not None else self._default_ttl
            actual_max_size = max_size if max_size is not None else self._default_max_size
            
            self._caches[name] = LRUCache(
                max_size=actual_max_size,
                default_ttl=actual_ttl
            )
            
            self._config[name] = {
                'ttl': actual_ttl,
                'max_size': actual_max_size
            }
            
            logger.info(f"创建缓存命名空间: {name}, ttl={actual_ttl}s, max_size={actual_max_size}")
            return True
    
    def delete_namespace(self, name: str) -> bool:
        """
        删除命名空间
        
        Args:
            name: 命名空间名称
        
        Returns:
            是否删除成功
        """
        with self._manager_lock:
            if name not in self._caches:
                return False
            
            self._caches[name].clear()
            del self._caches[name]
            del self._config[name]
            
            logger.info(f"删除缓存命名空间: {name}")
            return True
    
    def get(
        self, 
        namespace: str, 
        key: str, 
        default: Any = None
    ) -> Any:
        """
        获取缓存
        
        Args:
            namespace: 命名空间
            key: 缓存键
            default: 默认值
        
        Returns:
            缓存值或默认值
        """
        with self._manager_lock:
            cache = self._caches.get(namespace)
            if cache is None:
                logger.warning(f"命名空间不存在: {namespace}")
                return default
            
            hit, value = cache.get(key)
            return value if hit else default
    
    def set(
        self, 
        namespace: str, 
        key: str, 
        value: Any, 
        ttl: Optional[int] = None
    ) -> bool:
        """
        设置缓存
        
        Args:
            namespace: 命名空间
            key: 缓存键
            value: 缓存值
            ttl: TTL（秒），None使用命名空间默认值
        
        Returns:
            是否设置成功
        """
        with self._manager_lock:
            # 如果命名空间不存在，自动创建
            if namespace not in self._caches:
                self.create_namespace(namespace)
            
            try:
                self._caches[namespace].set(key, value, ttl)
                return True
            except Exception as e:
                logger.error(f"设置缓存失败: {e}")
                return False
    
    def delete(self, namespace: str, key: str) -> bool:
        """
        删除缓存
        
        Args:
            namespace: 命名空间
            key: 缓存键
        
        Returns:
            是否删除成功
        """
        with self._manager_lock:
            cache = self._caches.get(namespace)
            if cache is None:
                return False
            return cache.delete(key)
    
    def has(self, namespace: str, key: str) -> bool:
        """
        检查缓存是否存在
        
        Args:
            namespace: 命名空间
            key: 缓存键
        
        Returns:
            是否存在
        """
        with self._manager_lock:
            cache = self._caches.get(namespace)
            if cache is None:
                return False
            return cache.has(key)
    
    def clear(self, namespace: Optional[str] = None) -> int:
        """
        清空缓存
        
        Args:
            namespace: 命名空间，None表示清空所有
        
        Returns:
            清空的缓存数量
        """
        with self._manager_lock:
            if namespace is None:
                total = 0
                for cache in self._caches.values():
                    total += cache.clear()
                logger.info("所有缓存已清空")
                return total
            else:
                cache = self._caches.get(namespace)
                if cache:
                    count = cache.clear()
                    logger.info(f"命名空间 {namespace} 缓存已清空: {count} 条目")
                    return count
                return 0
    
    def cleanup_expired(self, namespace: Optional[str] = None) -> int:
        """
        清理过期缓存
        
        Args:
            namespace: 命名空间，None表示清理所有
        
        Returns:
            清理的缓存数量
        """
        with self._manager_lock:
            if namespace is None:
                total = 0
                for cache in self._caches.values():
                    total += cache.cleanup_expired()
                return total
            else:
                cache = self._caches.get(namespace)
                if cache:
                    return cache.cleanup_expired()
                return 0
    
    def get_stats(self, namespace: Optional[str] = None) -> Dict:
        """
        获取统计信息
        
        Args:
            namespace: 命名空间，None表示获取所有
        
        Returns:
            统计信息
        """
        with self._manager_lock:
            if namespace is None:
                return {
                    name: cache.get_stats()
                    for name, cache in self._caches.items()
                }
            else:
                cache = self._caches.get(namespace)
                if cache:
                    return cache.get_stats()
                return {}
    
    def get_namespaces(self) -> list:
        """获取所有命名空间"""
        with self._manager_lock:
            return list(self._caches.keys())
    
    def get_cache_info(self, namespace: str) -> Dict:
        """获取命名空间详细信息"""
        with self._manager_lock:
            if namespace not in self._caches:
                return {}
            
            cache = self._caches[namespace]
            config = self._config.get(namespace, {})
            
            return {
                'namespace': namespace,
                'config': config,
                'stats': cache.get_stats(),
                'entries': cache.get_entries_info()
            }
    
    def start_auto_cleanup(self) -> None:
        """启动自动过期清理任务"""
        if self._cleanup_thread is not None and self._cleanup_thread.is_alive():
            return
        
        self._stop_cleanup.clear()
        
        def cleanup_task():
            while not self._stop_cleanup.wait(self._cleanup_interval):
                try:
                    count = self.cleanup_expired()
                    if count > 0:
                        logger.info(f"自动清理过期缓存: {count} 条目")
                except Exception as e:
                    logger.error(f"自动清理任务失败: {e}")
        
        self._cleanup_thread = threading.Thread(target=cleanup_task, daemon=True)
        self._cleanup_thread.start()
        logger.info("自动清理任务已启动")
    
    def stop_auto_cleanup(self) -> None:
        """停止自动过期清理任务"""
        if self._cleanup_thread is not None:
            self._stop_cleanup.set()
            self._cleanup_thread.join(timeout=5)
            logger.info("自动清理任务已停止")


# 全局缓存管理器实例
memory_cache = MemoryCache()


# ============ 便捷函数 ============

def cache_get(namespace: str, key: str, default: Any = None) -> Any:
    """获取缓存的便捷函数"""
    return memory_cache.get(namespace, key, default)


def cache_set(namespace: str, key: str, value: Any, ttl: Optional[int] = None) -> bool:
    """设置缓存的便捷函数"""
    return memory_cache.set(namespace, key, value, ttl)


def cache_delete(namespace: str, key: str) -> bool:
    """删除缓存的便捷函数"""
    return memory_cache.delete(namespace, key)


def cache_clear(namespace: Optional[str] = None) -> int:
    """清空缓存的便捷函数"""
    return memory_cache.clear(namespace)


def cache_has(namespace: str, key: str) -> bool:
    """检查缓存是否存在"""
    return memory_cache.has(namespace, key)


# ============ 装饰器 ============

def cached(
    namespace: str = 'default',
    key_func: Optional[Callable[..., str]] = None,
    ttl: Optional[int] = None
):
    """
    缓存装饰器
    
    Args:
        namespace: 命名空间
        key_func: 生成缓存键的函数，默认为函数名+参数哈希
        ttl: TTL（秒）
    
    使用示例:
        @cached('my_namespace', ttl=3600)
        def expensive_function(arg1, arg2):
            # 耗时计算
            return result
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # 使用函数名和参数生成键
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                key_str = ":".join(key_parts)
                cache_key = hashlib.md5(key_str.encode()).hexdigest()
            
            # 尝试从缓存获取
            cached_value = memory_cache.get(namespace, cache_key)
            if cached_value is not None:
                logger.debug(f"缓存命中: {namespace}:{cache_key}")
                return cached_value
            
            # 执行函数
            result = func(*args, **kwargs)
            
            # 存入缓存
            memory_cache.set(namespace, cache_key, result, ttl)
            logger.debug(f"缓存存储: {namespace}:{cache_key}")
            
            return result
        return wrapper
    return decorator
