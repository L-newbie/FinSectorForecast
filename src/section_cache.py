# -*- coding: utf-8 -*-
"""
板块级内存缓存系统

功能：
1. 为每个内容板块建立独立的缓存机制
2. 确保同一板块在同一时间仅存在一个缓存版本
3. 数据更新时，仅针对对应板块的缓存文件进行刷新操作
4. 删除旧版本缓存并生成新版本缓存
5. 纯内存缓存，不生成文件

板块定义：
- index: 主页/仪表盘
- predict: 预测页面
- training: 训练页面
- analysis: 分析页面
"""

import threading
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Callable
from functools import wraps


class SectionCache:
    """板块级缓存类 - 管理单个板块的缓存"""
    
    def __init__(self, section_name: str, ttl_seconds: int = 3600):
        """
        初始化板块缓存
        
        Args:
            section_name: 板块名称
            ttl_seconds: 缓存有效期（秒），默认1小时
        """
        self.section_name = section_name
        self.ttl_seconds = ttl_seconds
        self._cache_data: Optional[Dict[str, Any]] = None
        self._cache_time: Optional[datetime] = None
        self._lock = threading.RLock()
    
    def is_valid(self) -> bool:
        """检查缓存是否有效"""
        if self._cache_data is None or self._cache_time is None:
            return False
        
        age = datetime.now() - self._cache_time
        return age.total_seconds() < self.ttl_seconds
    
    def get(self) -> Optional[Dict[str, Any]]:
        """获取缓存数据"""
        with self._lock:
            if self.is_valid():
                return self._cache_data
            return None
    
    def set(self, data: Dict[str, Any]) -> None:
        """
        设置缓存数据（自动删除旧版本）
        
        Args:
            data: 要缓存的数据
        """
        with self._lock:
            # 删除旧版本缓存
            self._cache_data = None
            self._cache_time = None
            
            # 生成新版本缓存
            self._cache_data = data
            self._cache_time = datetime.now()
    
    def invalidate(self) -> None:
        """使缓存失效（删除旧版本）"""
        with self._lock:
            self._cache_data = None
            self._cache_time = None
    
    def refresh(self, data: Dict[str, Any]) -> None:
        """
        刷新缓存（先删除旧版本，再生成新版本）
        
        Args:
            data: 新的缓存数据
        """
        with self._lock:
            # 删除旧版本
            self._cache_data = None
            self._cache_time = None
            
            # 生成新版本
            self._cache_data = data
            self._cache_time = datetime.now()
    
    def get_cache_info(self) -> Dict[str, Any]:
        """获取缓存信息"""
        with self._lock:
            if self._cache_data is None:
                return {
                    'section': self.section_name,
                    'exists': False,
                    'valid': False,
                    'age_seconds': None
                }
            
            age = 0
            if self._cache_time:
                age = (datetime.now() - self._cache_time).total_seconds()
            
            return {
                'section': self.section_name,
                'exists': True,
                'valid': self.is_valid(),
                'age_seconds': age,
                'ttl_seconds': self.ttl_seconds
            }


class SectionCacheManager:
    """板块缓存管理器 - 管理所有板块的缓存"""
    
    _instance = None
    _lock = threading.Lock()
    
    # 默认TTL配置（秒）
    DEFAULT_TTL = {
        'index': 3600,       # 主页缓存1小时
        'predict': 3600,     # 预测页面缓存1小时
        'training': 3600,    # 训练页面缓存1小时
        'analysis': 3600,    # 分析页面缓存1小时
    }
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self._section_caches: Dict[str, SectionCache] = {}
        self._manager_lock = threading.RLock()
        
        # 初始化所有板块的缓存
        for section_name, ttl in self.DEFAULT_TTL.items():
            self._section_caches[section_name] = SectionCache(section_name, ttl)
    
    def get_cache(self, section_name: str) -> Optional[SectionCache]:
        """
        获取指定板块的缓存对象
        
        Args:
            section_name: 板块名称
        
        Returns:
            板块缓存对象，如果不存在返回None
        """
        with self._manager_lock:
            return self._section_caches.get(section_name)
    
    def get(self, section_name: str) -> Optional[Dict[str, Any]]:
        """
        获取板块缓存数据
        
        Args:
            section_name: 板块名称
        
        Returns:
            缓存数据，如果不存在或过期返回None
        """
        cache = self.get_cache(section_name)
        if cache:
            return cache.get()
        return None
    
    def set(self, section_name: str, data: Dict[str, Any], ttl_seconds: Optional[int] = None) -> bool:
        """
        设置板块缓存数据
        
        Args:
            section_name: 板块名称
            data: 要缓存的数据
            ttl_seconds: 可选的TTL覆盖值
        
        Returns:
            是否设置成功
        """
        with self._manager_lock:
            # 如果板块不存在，创建它
            if section_name not in self._section_caches:
                actual_ttl = ttl_seconds if ttl_seconds else 3600
                self._section_caches[section_name] = SectionCache(section_name, actual_ttl)
            
            cache = self._section_caches[section_name]
            
            # 如果传入了TTL覆盖值，更新缓存的TTL
            if ttl_seconds and ttl_seconds != cache.ttl_seconds:
                cache.ttl_seconds = ttl_seconds
            
            cache.set(data)
            return True
    
    def invalidate(self, section_name: str) -> bool:
        """
        使指定板块的缓存失效（删除旧版本）
        
        Args:
            section_name: 板块名称
        
        Returns:
            是否成功
        """
        cache = self.get_cache(section_name)
        if cache:
            cache.invalidate()
            return True
        return False
    
    def refresh(self, section_name: str, data: Dict[str, Any]) -> bool:
        """
        刷新板块缓存（删除旧版本并生成新版本）
        
        Args:
            section_name: 板块名称
            data: 新的缓存数据
        
        Returns:
            是否刷新成功
        """
        cache = self.get_cache(section_name)
        if cache:
            cache.refresh(data)
            return True
        return False
    
    def invalidate_all(self) -> None:
        """使所有板块的缓存失效"""
        with self._manager_lock:
            for cache in self._section_caches.values():
                cache.invalidate()
    
    def get_section_names(self) -> list:
        """获取所有已注册的板块名称"""
        with self._manager_lock:
            return list(self._section_caches.keys())
    
    def get_all_cache_info(self) -> Dict[str, Dict[str, Any]]:
        """获取所有板块的缓存信息"""
        with self._manager_lock:
            return {
                name: cache.get_cache_info()
                for name, cache in self._section_caches.items()
            }
    
    def register_section(self, section_name: str, ttl_seconds: int = 3600) -> bool:
        """
        注册新的板块
        
        Args:
            section_name: 板块名称
            ttl_seconds: 缓存有效期（秒）
        
        Returns:
            是否注册成功
        """
        with self._manager_lock:
            if section_name in self._section_caches:
                return False
            
            self._section_caches[section_name] = SectionCache(section_name, ttl_seconds)
            return True
    
    def unregister_section(self, section_name: str) -> bool:
        """
        注销板块（同时删除其缓存）
        
        Args:
            section_name: 板块名称
        
        Returns:
            是否注销成功
        """
        with self._manager_lock:
            if section_name not in self._section_caches:
                return False
            
            cache = self._section_caches[section_name]
            cache.invalidate()
            del self._section_caches[section_name]
            return True


# 全局板块缓存管理器实例
section_cache_manager = SectionCacheManager()


def cached_section(section_name: str, ttl_seconds: int = 3600):
    """
    板块缓存装饰器
    
    Args:
        section_name: 板块名称
        ttl_seconds: 缓存有效期（秒）
    
    使用示例:
        @cached_section('index')
        def get_index_data():
            # expensive computation
            return data
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 尝试从缓存获取
            cached_data = section_cache_manager.get(section_name)
            if cached_data is not None:
                return cached_data
            
            # 执行函数获取数据
            result = func(*args, **kwargs)
            
            # 存入缓存
            section_cache_manager.set(section_name, result, ttl_seconds)
            
            return result
        return wrapper
    return decorator


def invalidate_section_cache(section_name: str) -> bool:
    """
    使指定板块缓存失效的便捷函数
    
    Args:
        section_name: 板块名称
    
    Returns:
        是否成功
    """
    return section_cache_manager.invalidate(section_name)


def refresh_section_cache(section_name: str, data: Dict[str, Any]) -> bool:
    """
    刷新指定板块缓存的便捷函数
    
    Args:
        section_name: 板块名称
        data: 新的缓存数据
    
    Returns:
        是否成功
    """
    return section_cache_manager.refresh(section_name, data)
