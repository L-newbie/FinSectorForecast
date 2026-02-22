# -*- coding: utf-8 -*-
"""
缓存管理模块

功能：
1. 预测结果缓存（内存）
2. 训练结果缓存（内存）
3. 分析结果缓存（内存）
4. 缓存过期管理
5. LRU淘汰策略
6. 容量限制
7. 线程安全

版本：2.0.0 - 纯内存缓存，无文件操作
"""

import hashlib
import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import threading
import logging

# 检测是否是 Flask reloader 子进程
_FLASK_RELOADER = os.environ.get('WERKZEUG_RUN_MAIN') == 'true'

from src.memory_cache import memory_cache

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('CacheManager')


class CacheManager:
    """缓存管理器 - 纯内存缓存，无文件操作"""
    
    _instance = None
    _lock = threading.Lock()
    
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
        
        # 检测是否是 Flask reloader 子进程
        # 如果是子进程，可能需要跳过某些初始化
        is_reloader = os.environ.get('WERKZEUG_RUN_MAIN') == 'true'
        
        self._initialized = True
        
        # 缓存过期时间配置（秒）
        self.cache_expiry = {
            'sectors': 3600,            # 板块列表缓存1小时
            'predict_all': 3600 * 6,      # 全板块预测缓存6小时
            'predict_multi': 3600 * 4,    # 多板块预测缓存4小时
            'analysis': 3600 * 2,         # 深度分析缓存2小时
            'train': 3600 * 12,           # 训练结果缓存12小时
        }
        
        # 缓存容量配置
        self.cache_max_size = {
            'sectors': 10,               # 板块列表缓存容量
            'predict_all': 50,
            'predict_multi': 100,
            'analysis': 100,
            'train': 30,
        }
        
        # 使用全局单例缓存
        self._memory_cache = memory_cache
        
        # 重新配置全局缓存（只在非 reloader 模式下执行）
        if not is_reloader:
            self._memory_cache.reconfigure(
                default_ttl=3600,
                max_size=500,
                cleanup_interval=300
            )
        
        # 初始化各个命名空间
        for cache_type in self.cache_expiry.keys():
            ttl = self.cache_expiry[cache_type]
            max_size = self.cache_max_size.get(cache_type, 100)
            self._memory_cache.create_namespace(
                name=cache_type,
                ttl=ttl,
                max_size=max_size
            )
        
        # 启动自动清理任务（由 memory_cache 内部检查是否已启动）
        self._memory_cache.start_auto_cleanup()
        
        logger.info("CacheManager 初始化完成 - 纯内存缓存")
    
    def _get_cache_key(self, cache_type: str, identifier: str = '') -> str:
        """生成缓存键"""
        key = f"{cache_type}_{identifier}" if identifier else cache_type
        return hashlib.md5(key.encode()).hexdigest()
    
    def get(self, cache_type: str, identifier: str = '') -> Optional[Dict]:
        """
        获取缓存数据
        
        Args:
            cache_type: 缓存类型 (predict_all, predict_multi, analysis, train)
            identifier: 标识符（如板块名称列表）
        
        Returns:
            缓存数据或None
        """
        cache_key = self._get_cache_key(cache_type, identifier)
        
        # 从内存缓存获取
        cached_data = self._memory_cache.get(cache_type, cache_key)
        
        if cached_data is not None:
            logger.debug(f"缓存命中: {cache_type}:{identifier}")
            return cached_data.get('data')
        
        logger.debug(f"缓存未命中: {cache_type}:{identifier}")
        return None
    
    def set(self, cache_type: str, data: Dict, identifier: str = '') -> bool:
        """
        设置缓存数据
        
        Args:
            cache_type: 缓存类型
            data: 要缓存的数据
            identifier: 标识符
        
        Returns:
            是否成功
        """
        cache_key = self._get_cache_key(cache_type, identifier)
        
        cache_data = {
            'cache_type': cache_type,
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        
        # 获取该缓存类型的TTL
        ttl = self.cache_expiry.get(cache_type)
        
        # 保存到内存缓存
        result = self._memory_cache.set(cache_type, cache_key, cache_data, ttl)
        
        if result:
            logger.debug(f"缓存设置成功: {cache_type}:{identifier}")
        else:
            logger.warning(f"缓存设置失败: {cache_type}:{identifier}")
        
        return result
    
    def clear(self, cache_type: str = None, identifier: str = '') -> bool:
        """
        清除缓存
        
        Args:
            cache_type: 缓存类型，None表示清除所有
            identifier: 标识符
        
        Returns:
            是否成功
        """
        try:
            if cache_type is None:
                # 清除所有缓存
                self._memory_cache.clear()
                logger.info("所有缓存已清除")
            else:
                # 清除特定缓存
                if identifier:
                    cache_key = self._get_cache_key(cache_type, identifier)
                    self._memory_cache.delete(cache_type, cache_key)
                else:
                    self._memory_cache.clear(cache_type)
                logger.info(f"缓存已清除: {cache_type}")
            
            return True
        except Exception as e:
            logger.error(f"清除缓存失败: {e}")
            return False
    
    def get_cache_info(self) -> Dict:
        """获取缓存信息"""
        try:
            all_stats = self._memory_cache.get_stats()
            
            info = {
                'memory_cache_count': 0,
                'file_cache_count': 0,  # 保持向后兼容
                'cache_size_mb': 0,     # 内存缓存无法精确计算大小
                'namespaces': {}
            }
            
            total_entries = 0
            for ns, stats in all_stats.items():
                info['namespaces'][ns] = {
                    'size': stats.get('size', 0),
                    'max_size': stats.get('max_size', 0),
                    'hits': stats.get('hits', 0),
                    'misses': stats.get('misses', 0),
                    'hit_rate': stats.get('hit_rate', 0)
                }
                total_entries += stats.get('size', 0)
            
            info['memory_cache_count'] = total_entries
            
        except Exception as e:
            logger.error(f"获取缓存信息失败: {e}")
            info = {
                'memory_cache_count': 0,
                'file_cache_count': 0,
                'cache_size_mb': 0,
                'error': str(e)
            }
        
        return info
    
    def cleanup_expired_cache(self, max_age_hours: int = 8) -> Dict:
        """
        清理过期的缓存
        
        Args:
            max_age_hours: 最大缓存时间（小时），默认8小时（此参数已弃用，TTL由各缓存类型独立控制）
        
        Returns:
            清理结果信息
        """
        result = {
            'success': True,
            'deleted_count': 0,
            'deleted_size_mb': 0,
            'errors': [],
            'log': []
        }
        
        try:
            # 清理所有命名空间的过期缓存
            total_deleted = self._memory_cache.cleanup_expired()
            
            result['deleted_count'] = total_deleted
            result['log'].append(f"清理完成: 删除 {total_deleted} 个过期缓存条目")
            result['log'].append("说明: 缓存过期由各缓存类型的TTL独立控制")
            
            logger.info(f"缓存清理完成: 删除 {total_deleted} 个条目")
            
        except Exception as e:
            result['success'] = False
            result['errors'].append(f"清理过程发生错误: {str(e)}")
            result['log'].append(f"[严重错误] {str(e)}")
            logger.error(f"缓存清理失败: {e}")
        
        return result
    
    def get_cache_timestamp(self, cache_type: str, identifier: str = '') -> Optional[str]:
        """
        获取缓存的时间戳
        
        Args:
            cache_type: 缓存类型
            identifier: 标识符
        
        Returns:
            ISO格式的时间戳字符串，如果缓存不存在则返回None
        """
        cache_key = self._get_cache_key(cache_type, identifier)
        cached_data = self._memory_cache.get(cache_type, cache_key)
        
        if cached_data is not None and isinstance(cached_data, dict):
            return cached_data.get('timestamp')
        
        return None
    
    def is_cache_expired(self, cache_type: str, identifier: str = '') -> bool:
        """
        检查缓存是否已过期
        
        Args:
            cache_type: 缓存类型
            identifier: 标识符
        
        Returns:
            True表示已过期或不存在，False表示有效
        """
        cache_key = self._get_cache_key(cache_type, identifier)
        cached_data = self._memory_cache.get(cache_type, cache_key)
        
        if cached_data is None:
            return True
        
        # 检查是否过期由内存缓存自动处理
        # 如果get返回None，说明已过期
        return False
    
    def get_stats(self, cache_type: str = None) -> Dict:
        """
        获取缓存统计信息
        
        Args:
            cache_type: 缓存类型，None表示获取所有
        
        Returns:
            统计信息
        """
        return self._memory_cache.get_stats(cache_type)


def start_cache_cleanup_scheduler(interval_hours: int = 1, max_cache_age_hours: int = 8):
    """
    启动缓存清理定时任务
    
    注意：新的内存缓存系统使用自动后台清理，此函数保留用于向后兼容。
    
    Args:
        interval_hours: 清理间隔（小时），默认1小时
        max_cache_age_hours: 缓存最大保留时间（小时），默认8小时
    """
    logger.info(
        f"缓存清理定时任务已配置: 间隔={interval_hours}h, "
        f"过期时间={max_cache_age_hours}h (由TTL控制)"
    )
    
    # 新的缓存系统使用自动清理，不需要额外启动线程
    cache_mgr = CacheManager()
    
    # 返回一个空操作线程以保持向后兼容
    def noop_task():
        while True:
            import time
            time.sleep(interval_hours * 3600)
    
    cleanup_thread = threading.Thread(target=noop_task, daemon=True)
    cleanup_thread.start()
    
    return cleanup_thread


# 全局缓存管理器实例
cache_manager = CacheManager()
