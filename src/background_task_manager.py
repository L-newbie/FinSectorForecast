# -*- coding: utf-8 -*-
"""
后台任务管理器

功能：
1. 在应用启动时自动开始训练和预测任务
2. 维护任务状态和进度
3. 提供任务状态查询接口
"""

import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
import copy


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"          # 等待中
    RUNNING = "running"          # 运行中
    COMPLETED = "completed"      # 已完成
    FAILED = "failed"            # 失败
    CANCELLED = "cancelled"      # 已取消


class BackgroundTask:
    """后台任务类"""
    
    def __init__(self, task_id: str, task_type: str, name: str, 
                 description: str = "", config: Dict = None):
        self.task_id = task_id
        self.task_type = task_type  # 'train', 'predict', 'train_all', 'predict_all'
        self.name = name
        self.description = description
        self.config = config or {}
        
        self.status = TaskStatus.PENDING
        self.progress = 0  # 0-100
        self.message = "等待中..."
        self.result = None
        self.error = None
        
        self.created_at = datetime.now()
        self.started_at = None
        self.completed_at = None
        
        # 任务相关数据
        self.sectors = []  # 涉及的板块列表
        self.total_sectors = 0
        self.completed_sectors = 0
        
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'task_id': self.task_id,
            'task_type': self.task_type,
            'name': self.name,
            'description': self.description,
            'status': self.status.value,
            'progress': self.progress,
            'message': self.message,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'sectors': self.sectors,
            'total_sectors': self.total_sectors,
            'completed_sectors': self.completed_sectors,
            'result': self.result,
            'error': self.error
        }


class BackgroundTaskManager:
    """后台任务管理器"""
    
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
        
        self._initialized = True
        self.tasks: Dict[str, BackgroundTask] = {}
        self.task_lock = threading.Lock()
        
        # 当前正在运行的任务
        self.current_task: Optional[BackgroundTask] = None
        
        # 最新的预测结果缓存
        self.latest_predictions: Optional[Dict] = None
        self.latest_predictions_time: Optional[datetime] = None
        
        # 配置
        self.auto_train_on_startup = True
        self.auto_predict_on_startup = True
        self.predict_refresh_interval = 3600  # 预测刷新间隔（秒），默认1小时
    
    def create_task(self, task_type: str, name: str, 
                   description: str = "", config: Dict = None,
                   sectors: List[str] = None) -> BackgroundTask:
        """创建新任务"""
        task_id = f"{task_type}_{int(time.time() * 1000)}"
        task = BackgroundTask(task_id, task_type, name, description, config)
        task.sectors = sectors or []
        task.total_sectors = len(sectors) if sectors else 0
        
        with self.task_lock:
            self.tasks[task_id] = task
        
        return task
    
    def start_task(self, task: BackgroundTask):
        """启动任务"""
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        self.current_task = task
        
        # 移除过期任务（保留最近10个）
        self._cleanup_old_tasks()
    
    def update_progress(self, task_id: str, progress: int, message: str = None,
                       completed_sectors: int = None):
        """更新任务进度"""
        with self.task_lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.progress = min(100, max(0, progress))
                if message:
                    task.message = message
                if completed_sectors is not None:
                    task.completed_sectors = completed_sectors
    
    def complete_task(self, task_id: str, result: Any = None):
        """完成任务"""
        with self.task_lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.status = TaskStatus.COMPLETED
                task.progress = 100
                task.message = "任务完成"
                task.completed_at = datetime.now()
                task.result = result
                
                if self.current_task and self.current_task.task_id == task_id:
                    self.current_task = None
                
                # 如果是预测任务，更新最新预测缓存
                if task.task_type in ['predict', 'predict_all']:
                    if result:
                        self.latest_predictions = result
                        self.latest_predictions_time = datetime.now()
    
    def fail_task(self, task_id: str, error: str):
        """任务失败"""
        with self.task_lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                task.status = TaskStatus.FAILED
                task.error = error
                task.message = f"任务失败: {error}"
                task.completed_at = datetime.now()
                
                if self.current_task and self.current_task.task_id == task_id:
                    self.current_task = None
    
    def get_task(self, task_id: str) -> Optional[BackgroundTask]:
        """获取任务"""
        with self.task_lock:
            return self.tasks.get(task_id)
    
    def get_all_tasks(self) -> List[BackgroundTask]:
        """获取所有任务"""
        with self.task_lock:
            return list(self.tasks.values())
    
    def get_current_task(self) -> Optional[BackgroundTask]:
        """获取当前正在运行的任务"""
        return self.current_task
    
    def get_latest_predictions(self) -> Optional[Dict]:
        """获取最新预测结果"""
        return self.latest_predictions
    
    def _cleanup_old_tasks(self):
        """清理过期任务（保留最近10个）"""
        if len(self.tasks) > 10:
            # 按创建时间排序，保留最新的10个
            sorted_tasks = sorted(self.tasks.values(), 
                                 key=lambda t: t.created_at, 
                                 reverse=True)
            for task in sorted_tasks[10:]:
                del self.tasks[task.task_id]
    
    def get_status_summary(self) -> Dict:
        """获取状态摘要"""
        with self.task_lock:
            tasks_list = list(self.tasks.values())
            
            # 统计各状态数量
            status_counts = {
                'pending': 0,
                'running': 0,
                'completed': 0,
                'failed': 0,
                'cancelled': 0
            }
            for task in tasks_list:
                status_counts[task.status.value] += 1
            
            # 获取当前运行的任务
            current = self.current_task.to_dict() if self.current_task else None
            
            # 获取最近完成的任务
            completed_tasks = [t for t in tasks_list if t.status == TaskStatus.COMPLETED]
            recent_completed = sorted(completed_tasks, 
                                     key=lambda t: t.completed_at, 
                                     reverse=True)[:3]
            recent_completed = [t.to_dict() for t in recent_completed]
            
            return {
                'total_tasks': len(tasks_list),
                'status_counts': status_counts,
                'current_task': current,
                'recent_completed': recent_completed,
                'latest_predictions': self.latest_predictions is not None,
                'latest_predictions_time': self.latest_predictions_time.isoformat() 
                                          if self.latest_predictions_time else None
            }


# 全局实例
task_manager = BackgroundTaskManager()


def start_background_training(config: Dict, sectors: List[str] = None):
    """在后台线程中启动训练任务"""
    def run_training():
        task = None  # 初始化task变量
        sectors = None # 初始化sectors变量
        try:
            from src.data_fetcher import DataFetcher
            from src.predictor import SectorPredictor, MultiSectorPredictor
            
            # 创建任务
            task = task_manager.create_task(
                task_type='train_all',
                name='全板块模型训练',
                description='启动时自动训练所有板块模型',
                config=config,
                sectors=sectors
            )
            task_manager.start_task(task)
            
            # 获取板块列表
            if not sectors:
                fetcher = DataFetcher(config)
                sectors = fetcher.get_sectors_list()
                task.sectors = sectors
                task.total_sectors = len(sectors)
            
            total = len(sectors)
            success_count = 0
            results = {}
            
            for i, sector in enumerate(sectors):
                progress = int(((i + 1) / total) * 100)
                task_manager.update_progress(
                    task.task_id,
                    progress,
                    f"正在训练板块 {i+1}/{total}: {sector}...",
                    completed_sectors=i+1
                )
                
                try:
                    predictor = SectorPredictor(sector, config)
                    result = predictor.train()
                    results[sector] = result
                    success_count += 1
                except Exception as e:
                    results[sector] = {'error': str(e)}
            
            task_manager.complete_task(task.task_id, {
                'success_count': success_count,
                'total': total,
                'results': results
            })
            
            print(f">>> 后台训练任务完成: 成功 {success_count}/{total}")
            
            # 训练完成后自动开始预测
            start_background_prediction(config, sectors)
            
        except Exception as e:
            print(f">>> 后台训练任务异常: {str(e)}")
            if task:
                task_manager.fail_task(task.task_id, str(e))
    
    # 启动后台线程
    thread = threading.Thread(target=run_training, daemon=True)
    thread.start()
    print(">>> 后台训练任务已启动")


def start_background_prediction(config: Dict, sectors: List[str] = None):
    """在后台线程中启动预测任务"""
    def run_prediction():
        task = None  # 初始化task变量
        sectors = None  # 初始化sectors变量
        try:
            from src.data_fetcher import DataFetcher
            from src.predictor import SectorPredictor, MultiSectorPredictor
            
            # 创建任务
            task = task_manager.create_task(
                task_type='predict_all',
                name='全板块预测',
                description='启动时自动预测所有板块',
                config=config,
                sectors=sectors
            )
            task_manager.start_task(task)
            
            # 获取板块列表
            if not sectors:
                fetcher = DataFetcher(config)
                sectors = fetcher.get_sectors_list()
                task.sectors = sectors
                task.total_sectors = len(sectors)
            
            total = len(sectors)
            predictions = []
            success_count = 0
            error_count = 0
            
            for i, sector in enumerate(sectors):
                progress = int(((i + 1) / total) * 100)
                task_manager.update_progress(
                    task.task_id,
                    progress,
                    f"正在预测板块 {i+1}/{total}: {sector}...",
                    completed_sectors=i+1
                )
                
                try:
                    predictor = SectorPredictor(sector, config)
                    prediction = predictor.predict()
                    features = predictor.get_historical_features()
                    prediction['features'] = features
                    predictions.append(prediction)
                    success_count += 1
                except Exception as e:
                    error_count += 1
                    predictions.append({
                        'sector_name': sector,
                        'error': str(e),
                        'predicted_return': 0,
                        'probability': 0.5,
                        'signal': '观望'
                    })
            
            # 排序
            predictions.sort(key=lambda x: x.get('predicted_return', 0), reverse=True)
            
            # 获取Top机会
            valid_opportunities = [p for p in predictions 
                                  if 'error' not in p and p.get('probability', 0) > 0]
            opportunities = sorted(valid_opportunities, 
                                  key=lambda x: x.get('probability', 0), 
                                  reverse=True)[:20]
            
            result = {
                'success': True,
                'predictions': predictions,
                'opportunities': opportunities,
                'total_sectors': total,
                'success_count': success_count,
                'error_count': error_count
            }
            
            task_manager.complete_task(task.task_id, result)
            
            print(f">>> 后台预测任务完成: 成功 {success_count}, 失败 {error_count}")
            
        except Exception as e:
            print(f">>> 后台预测任务异常: {str(e)}")
            if task:
                task_manager.fail_task(task.task_id, str(e))
    
    # 启动后台线程
    thread = threading.Thread(target=run_prediction, daemon=True)
    thread.start()
    print(">>> 后台预测任务已启动")


def start_periodic_prediction(config: Dict, interval_seconds: int = 3600):
    """启动定时预测任务"""
    def run_periodic():
        while True:
            time.sleep(interval_seconds)
            try:
                print(">>> 执行定时预测任务...")
                start_background_prediction(config)
            except Exception as e:
                print(f">>> 定时预测任务异常: {str(e)}")
    
    thread = threading.Thread(target=run_periodic, daemon=True)
    thread.start()
    print(f">>> 定时预测任务已启动，间隔: {interval_seconds}秒")


def init_background_tasks(config: Dict):
    return
    # """初始化后台任务（在Flask应用启动时调用）"""
    # print(">>> 初始化后台任务...")
    
    # # 启动后台训练
    # if task_manager.auto_train_on_startup:
    #     start_background_training(config)
    
    # # 启动定时预测
    # if task_manager.auto_predict_on_startup:
    #     start_periodic_prediction(config, task_manager.predict_refresh_interval)
    
    # print(">>> 后台任务初始化完成")
