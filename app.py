# -*- coding: utf-8 -*-
"""
板块次日涨跌预测系统 - Web应用入口
"""

import sys
import os
import json
from flask import Flask, render_template, request, jsonify, redirect, url_for
from datetime import datetime

# 添加当前目录到路径
sys.path.insert(0, '.')

from src.data_fetcher import DataFetcher
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer
from src.predictor import SectorPredictor, MultiSectorPredictor
from src.cache_manager import cache_manager, start_cache_cleanup_scheduler
from src.section_cache import section_cache_manager
from src.memory_cache import memory_cache
from src.background_task_manager import task_manager, init_background_tasks

# 加载配置
import yaml
def load_config(config_path="config/config.yaml"):
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except:
        return {}

def merge_config(base_config, custom_params):
    """合并自定义参数到基础配置"""
    import copy
    result = copy.deepcopy(base_config)
    
    # 合并数据参数
    if 'data' in custom_params:
        if 'data' not in result:
            result['data'] = {}
        result['data'].update(custom_params['data'])
    
    # 合并模型参数
    if 'model' in custom_params:
        if 'model' not in result:
            result['model'] = {}
        if 'classifier' in custom_params['model']:
            if 'classifier' not in result['model']:
                result['model']['classifier'] = {}
            if 'params' in custom_params['model']['classifier']:
                if 'params' not in result['model']['classifier']:
                    result['model']['classifier']['params'] = {}
                result['model']['classifier']['params'].update(custom_params['model']['classifier']['params'])
        if 'regressor' in custom_params['model']:
            if 'regressor' not in result['model']:
                result['model']['regressor'] = {}
            if 'params' in custom_params['model']['regressor']:
                if 'params' not in result['model']['regressor']:
                    result['model']['regressor']['params'] = {}
                result['model']['regressor']['params'].update(custom_params['model']['regressor']['params'])
        if 'training' in custom_params['model']:
            if 'training' not in result['model']:
                result['model']['training'] = {}
            result['model']['training'].update(custom_params['model']['training'])
    
    # 合并预测参数
    if 'predict' in custom_params:
        if 'predict' not in result:
            result['predict'] = {}
        result['predict'].update(custom_params['predict'])
    
    return result

config = load_config()

# 创建Flask应用
app = Flask(__name__)
app.secret_key = 'sector_predictor_2026'

# 全局预测器缓存
predictors_cache = {}
multi_predictor = None


# 全局板块列表缓存
_sectors_cache = None
_sectors_cache_time = None
SECTORS_CACHE_TTL = 3600  # 板块列表缓存1小时

# 板块列表缓存键名
SECTORS_CACHE_KEY = 'sectors_list'

# 后台刷新任务状态
_sectors_refresh_in_progress = False

def get_cached_sectors():
    """获取缓存的板块列表 - 优先从cache_manager读取"""
    global _sectors_cache, _sectors_cache_time, _sectors_refresh_in_progress
    now = datetime.now()
    
    # 优先从 cache_manager 读取
    cached_data = cache_manager.get('sectors')
    if cached_data is not None:
        return cached_data
    
    # 如果cache_manager没有，从本地缓存读取
    if _sectors_cache is not None and _sectors_cache_time is not None:
        if (now - _sectors_cache_time).total_seconds() < SECTORS_CACHE_TTL:
            # 同步到 cache_manager
            cache_manager.set('sectors', _sectors_cache)
            return _sectors_cache
    
    # 如果正在刷新中，等待刷新完成后再从缓存读取
    if _sectors_refresh_in_progress:
        import time
        max_wait = 10  # 最多等待10秒
        wait_time = 0
        while _sectors_refresh_in_progress and wait_time < max_wait:
            time.sleep(0.5)
            wait_time += 0.5
        # 刷新完成后从缓存读取
        cached_data = cache_manager.get('sectors')
        if cached_data is not None:
            return cached_data
    
    # 缓存过期或不存在，重新获取
    fetcher = DataFetcher(config)
    sectors = fetcher.get_sectors_list()
    
    # 更新本地缓存
    _sectors_cache = sectors
    _sectors_cache_time = now
    
    # 同步到 cache_manager
    cache_manager.set('sectors', sectors)
    
    return sectors


def refresh_sectors_cache_background():
    """后台刷新板块列表缓存 - 在后台线程中运行"""
    global _sectors_refresh_in_progress
    
    if _sectors_refresh_in_progress:
        return {
            'success': False,
            'message': '刷新任务正在进行中'
        }
    
    _sectors_refresh_in_progress = True
    
    try:
        print(">>> 后台任务: 开始刷新板块列表缓存...")
        fetcher = DataFetcher(config)
        new_sectors = fetcher.get_sectors_list()
        
        # 更新缓存
        global _sectors_cache, _sectors_cache_time
        _sectors_cache = new_sectors
        _sectors_cache_time = datetime.now()
        
        # 同步到 cache_manager
        cache_manager.set('sectors', new_sectors)
        
        print(f">>> 后台任务: 板块列表刷新完成，共 {len(new_sectors)} 个板块")
        
        return {
            'success': True,
            'message': f'成功刷新板块列表，共 {len(new_sectors)} 个板块',
            'count': len(new_sectors)
        }
    except Exception as e:
        print(f">>> 后台任务: 板块列表刷新失败: {str(e)}")
        return {
            'success': False,
            'message': f'刷新失败: {str(e)}'
        }
    finally:
        _sectors_refresh_in_progress = False


def start_sectors_refresh_scheduler(interval_minutes: int = 30):
    """启动板块列表定时刷新任务"""
    import threading
    import time
    
    def refresh_loop():
        while True:
            time.sleep(interval_minutes * 60)
            try:
                refresh_sectors_cache_background()
            except Exception as e:
                print(f">>> 定时刷新任务异常: {str(e)}")
    
    thread = threading.Thread(target=refresh_loop, daemon=True)
    thread.start()
    print(f">>> 板块列表定时刷新任务已启动，间隔: {interval_minutes} 分钟")
    return thread


def init_sectors_cache_on_startup():
    """启动时初始化板块列表缓存"""
    print(">>> 启动时初始化板块列表缓存...")
    try:
        sectors = refresh_sectors_cache_background()
        if sectors.get('success'):
            print(f">>> 板块列表缓存初始化完成: {sectors.get('count', 0)} 个板块")
        else:
            print(f">>> 板块列表缓存初始化失败: {sectors.get('message')}")
    except Exception as e:
        print(f">>> 板块列表缓存初始化异常: {str(e)}")


@app.route('/')
def index():
    """主页 - 仪表盘"""
    # 页面先快速渲染，板块列表通过前端异步加载
    return render_template('index.html', 
                           title='FinSectorForecast')


@app.route('/predict')
def predict_page():
    """预测页面"""
    # 页面先快速渲染，板块列表通过前端异步加载
    return render_template('predict.html', 
                           title='预测分析')


@app.route('/training')
def training_page():
    """训练页面"""
    # 页面先快速渲染，板块列表通过前端异步加载
    return render_template('training.html', 
                           title='模型训练')


@app.route('/analysis')
def analysis_page():
    """分析页面"""
    # 页面先快速渲染，板块列表通过前端异步加载
    return render_template('analysis.html', 
                           title='深度分析')


# ==================== API接口 ====================

@app.route('/api/tasks/status', methods=['GET'])
def get_tasks_status():
    """获取后台任务状态"""
    try:
        summary = task_manager.get_status_summary()
        
        # 获取最新预测结果
        latest_predictions = task_manager.get_latest_predictions()
        
        return jsonify({
            'success': True,
            'summary': summary,
            'has_predictions': latest_predictions is not None,
            'predictions': latest_predictions
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/tasks/current', methods=['GET'])
def get_current_task():
    """获取当前正在运行的任务"""
    try:
        current = task_manager.get_current_task()
        if current:
            return jsonify({
                'success': True,
                'task': current.to_dict()
            })
        else:
            return jsonify({
                'success': True,
                'task': None,
                'message': '当前没有正在运行的任务'
            })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/tasks', methods=['GET'])
def get_all_tasks():
    """获取所有任务列表"""
    try:
        tasks = task_manager.get_all_tasks()
        tasks_dict = [t.to_dict() for t in tasks]
        
        return jsonify({
            'success': True,
            'tasks': tasks_dict,
            'count': len(tasks_dict)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/sectors', methods=['GET'])
def get_sectors():
    """获取板块列表 - 优先从缓存读取"""
    force_refresh = request.args.get('force_refresh', 'false').lower() == 'true'
    
    if force_refresh:
        # 强制刷新
        result = refresh_sectors_cache_background()
        if result.get('success'):
            # 获取最新数据
            sectors = get_cached_sectors()
            return jsonify({
                'success': True, 
                'data': sectors,
                'refreshed': True,
                'message': result.get('message')
            })
        else:
            return jsonify({
                'success': False,
                'error': result.get('message')
            })
    
    # 优先从缓存读取
    sectors = get_cached_sectors()
    
    # 检查缓存状态
    cache_info = cache_manager.get_cache_timestamp('sectors')
    cache_status = 'cached' if cache_info else 'no_cache'
    
    return jsonify({
        'success': True, 
        'data': sectors,
        'cache_status': cache_status,
        'count': len(sectors)
    })


@app.route('/api/config', methods=['GET'])
def get_config():
    """获取当前配置信息"""
    try:
        # 返回前端需要的配置信息
        config_data = {
            'data': {
                'history_days': config.get('data', {}).get('history_days', 365),
                'feature_window': config.get('data', {}).get('feature_window', 60)
            },
            'model': {
                'classifier': {
                    'name': config.get('model', {}).get('classifier', {}).get('name', 'lightgbm'),
                    'params': config.get('model', {}).get('classifier', {}).get('params', {
                        'n_estimators': 200,
                        'max_depth': 8,
                        'learning_rate': 0.05,
                        'num_leaves': 64,
                        'min_child_samples': 10
                    })
                },
                'regressor': {
                    'name': config.get('model', {}).get('regressor', {}).get('name', 'lightgbm'),
                    'params': config.get('model', {}).get('regressor', {}).get('params', {
                        'n_estimators': 200,
                        'max_depth': 8,
                        'learning_rate': 0.05,
                        'num_leaves': 64,
                        'min_child_samples': 10
                    })
                },
                'training': config.get('model', {}).get('training', {
                    'test_size': 0.2,
                    'validation_split': 0.2,
                    'early_stopping_rounds': 15
                })
            },
            'predict': {
                'probability_threshold': config.get('predict', {}).get('probability_threshold', 0.8)
            }
        }
        
        return jsonify({
            'success': True,
            'config': config_data
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/predict/single', methods=['POST'])
def predict_single():
    """单板块预测"""
    data = request.get_json()
    sector_name = data.get('sector', '半导体')
    date = data.get('date')
    
    try:
        # 优先使用缓存的predictor
        cache_key = f'predictor_{sector_name}'
        if cache_key in predictors_cache:
            cached = predictors_cache[cache_key]
            predictor = cached['predictor']
            print(f"使用缓存的模型预测: {sector_name}")
        else:
            predictor = SectorPredictor(sector_name, config)
            print(f"使用新模型预测: {sector_name}")
        
        prediction = predictor.predict(date)
        
        # 获取当前特征
        features = predictor.get_historical_features()
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'features': features
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/predict/multi', methods=['POST'])
def predict_multi():
    """多板块预测"""
    data = request.get_json()
    sectors = data.get('sectors', [])
    force_refresh = data.get('force_refresh', False)
    
    if not sectors:
        return jsonify({'success': False, 'error': '请选择至少一个板块'})
    
    # 生成缓存标识
    sectors_sorted = sorted(sectors)
    cache_identifier = '_'.join(sectors_sorted[:5])  # 使用前5个板块作为标识
    
    # 尝试从缓存获取
    if not force_refresh:
        cached_data = cache_manager.get('predict_multi', cache_identifier)
        if cached_data:
            cached_data['from_cache'] = True
            return jsonify(cached_data)
    
    try:
        predictions = []
        for sector_name in sectors:
            try:
                # 优先使用缓存的predictor
                cache_key = f'predictor_{sector_name}'
                if cache_key in predictors_cache:
                    cached = predictors_cache[cache_key]
                    predictor = cached['predictor']
                else:
                    predictor = SectorPredictor(sector_name, config)
                
                prediction = predictor.predict()
                features = predictor.get_historical_features()
                prediction['features'] = features
                predictions.append(prediction)
            except Exception as e:
                predictions.append({
                    'sector_name': sector_name,
                    'error': str(e)
                })
        
        # 按预测涨幅排序
        predictions.sort(key=lambda x: x.get('predicted_return', 0), reverse=True)
        
        result = {
            'success': True,
            'predictions': predictions,
            'from_cache': False
        }
        
        # 保存到缓存
        cache_manager.set('predict_multi', result, cache_identifier)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/predict/all', methods=['POST'])
def predict_all():
    """全板块预测排名"""
    data = request.get_json() or {}
    force_refresh = data.get('force_refresh', False)
    
    # 首先尝试从后台任务管理器获取最新预测结果
    latest_predictions = task_manager.get_latest_predictions()
    if latest_predictions and not force_refresh:
        latest_predictions['from_cache'] = True
        latest_predictions['source'] = 'background_task'
        return jsonify(latest_predictions)
    
    # 尝试从缓存获取
    if not force_refresh:
        cached_data = cache_manager.get('predict_all')
        if cached_data:
            # 获取缓存的剩余时间
            cache_info = memory_cache.get_cache_info('predict_all')
            entries = cache_info.get('entries', [])
            
            remaining_seconds = 0
            cache_timestamp = ''
            
            if entries and len(entries) > 0:
                # 获取第一个条目的信息（predict_all 只有一个条目）
                entry = entries[0]
                ttl = entry.get('ttl', 0)
                age = entry.get('age_seconds', 0)
                remaining_seconds = max(0, ttl - age) if ttl > 0 else 0
            
            if remaining_seconds > 0:
                hours = int(remaining_seconds // 3600)
                mins = int((remaining_seconds % 3600) // 60)
                if hours > 0:
                    cache_expiry = f'{hours}小时{mins}分钟'
                else:
                    cache_expiry = f'{mins}分钟'
            else:
                cache_expiry = '已过期或无缓存'
            
            cached_data['from_cache'] = True
            cached_data['cache_expiry'] = cache_expiry
            return jsonify(cached_data)
    
    try:
        # 获取所有可用板块（使用缓存，而非直接调用get_sectors_list）
        all_sectors = get_cached_sectors()
        
        print(f"\n开始预测 {len(all_sectors)} 个板块...")
        
        # 使用与/api/predict/multi相同的预测方法
        predictions = []
        success_count = 0
        error_count = 0
        
        for i, sector_name in enumerate(all_sectors):
            try:
                # 优先使用缓存的predictor
                cache_key = f'predictor_{sector_name}'
                if cache_key in predictors_cache:
                    cached = predictors_cache[cache_key]
                    predictor = cached['predictor']
                else:
                    predictor = SectorPredictor(sector_name, config)
                
                prediction = predictor.predict()
                features = predictor.get_historical_features()
                prediction['features'] = features
                # 添加板块ID（使用索引）
                prediction['sector_id'] = i + 1
                predictions.append(prediction)
                success_count += 1
            except Exception as e:
                error_count += 1
                print(f"  ✗ {sector_name} 预测失败: {str(e)}")
                predictions.append({
                    'sector_name': sector_name,
                    'sector_id': i + 1,
                    'error': str(e),
                    'predicted_return': 0,
                    'probability': 0.5,
                    'signal': '观望'
                })
        
        print(f"预测完成: 成功 {success_count}, 失败 {error_count}")
        
        # 排序时只考虑有 predicted_return 字段的预测（与原来行为一致）
        predictions_sorted = sorted(predictions, key=lambda x: x.get('predicted_return', 0), reverse=True)
        
        # 获取Top机会（只包含成功预测且有概率的）
        valid_opportunities = [p for p in predictions if 'error' not in p and p.get('probability', 0) > 0]
        opportunities = sorted(valid_opportunities, key=lambda x: x.get('probability', 0), reverse=True)[:20]
        
        result = {
            'success': True,
            'predictions': predictions_sorted,
            'opportunities': opportunities,
            'total_sectors': len(all_sectors),
            'success_count': success_count,
            'error_count': error_count,
            'from_cache': False,
            'cache_expiry': '6小时',
            'source': 'realtime'
        }
        
        # 保存到缓存
        cache_manager.set('predict_all', result)
        
        return jsonify(result)
    except Exception as e:
        print(f"predict_all 错误: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/sector/detail', methods=['GET'])
def get_sector_detail():
    """获取板块详情数据"""
    sector_name = request.args.get('name', '')
    sector_id = request.args.get('id', '')
    
    if not sector_name and not sector_id:
        return jsonify({'success': False, 'error': '请提供板块名称或ID'})
    
    try:
        # 使用缓存的predictor或创建新的
        cache_key = f'predictor_{sector_name}'
        if cache_key in predictors_cache:
            cached = predictors_cache[cache_key]
            predictor = cached['predictor']
        else:
            predictor = SectorPredictor(sector_name, config)
        
        # 获取预测结果
        prediction = predictor.predict()
        
        # 获取历史特征数据
        features = predictor.get_historical_features()
        
        # 获取原始数据用于图表
        if predictor.raw_data is not None:
            raw_data = predictor.raw_data.copy()
            # 取最近60天数据
            raw_data = raw_data.tail(60)
            price_history = []
            for idx, row in raw_data.iterrows():
                price_history.append({
                    'date': row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date']),
                    'price': float(row['close']) if 'close' in row else 0,
                    'volume': float(row['volume']) if 'volume' in row else 0,
                    'change_pct': float(row['change_pct']) if 'change_pct' in row else 0
                })
        else:
            price_history = []
        
        # 构建技术指标数据
        indicators = []
        if features:
            # MA指标
            indicators.append({
                'name': 'MA5',
                'value': f"{features.get('ma_5', 0):.2f}",
                'trend': 'positive' if features.get('ma_5', 0) > features.get('ma_10', 0) else 'negative',
                'trendText': '向上' if features.get('ma_5', 0) > features.get('ma_10', 0) else '向下',
                'signal': 'buy' if features.get('ma_5', 0) > features.get('ma_10', 0) else 'sell',
                'signalText': '买入' if features.get('ma_5', 0) > features.get('ma_10', 0) else '卖出'
            })
            indicators.append({
                'name': 'MA10',
                'value': f"{features.get('ma_10', 0):.2f}",
                'trend': 'positive' if features.get('ma_10', 0) > features.get('ma_20', 0) else 'negative',
                'trendText': '向上' if features.get('ma_10', 0) > features.get('ma_20', 0) else '向下',
                'signal': 'buy' if features.get('ma_10', 0) > features.get('ma_20', 0) else 'sell',
                'signalText': '买入' if features.get('ma_10', 0) > features.get('ma_20', 0) else '卖出'
            })
            indicators.append({
                'name': 'MA20',
                'value': f"{features.get('ma_20', 0):.2f}",
                'trend': 'neutral',
                'trendText': '持平',
                'signal': 'hold',
                'signalText': '观望'
            })
            
            # RSI指标
            rsi = features.get('rsi_14', 50)
            rsi_signal = 'sell' if rsi > 70 else ('buy' if rsi < 30 else 'hold')
            rsi_signal_text = '超买' if rsi > 70 else ('超卖' if rsi < 30 else '正常')
            indicators.append({
                'name': 'RSI(14)',
                'value': f"{rsi:.2f}",
                'trend': 'positive' if rsi < 30 else ('negative' if rsi > 70 else 'neutral'),
                'trendText': rsi_signal_text,
                'signal': rsi_signal,
                'signalText': '买入' if rsi_signal == 'buy' else ('卖出' if rsi_signal == 'sell' else '观望')
            })
            
            # MACD指标
            macd = features.get('macd', 0)
            macd_signal_val = features.get('macd_signal', 0)
            macd_histogram = features.get('macd_histogram', 0)
            indicators.append({
                'name': 'MACD',
                'value': f"{macd:.4f}",
                'trend': 'positive' if macd > macd_signal_val else 'negative',
                'trendText': '金叉' if macd > macd_signal_val else '死叉',
                'signal': 'buy' if macd > macd_signal_val else 'sell',
                'signalText': '买入' if macd > macd_signal_val else '卖出'
            })
            
            # 布林带
            bb_upper = features.get('bb_upper', 0)
            bb_lower = features.get('bb_lower', 0)
            current_price = features.get('close', 0) or (price_history[-1]['price'] if price_history else 0)
            bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) != 0 else 0.5
            indicators.append({
                'name': '布林带位置',
                'value': f"{bb_position * 100:.1f}%",
                'trend': 'negative' if bb_position > 0.8 else ('positive' if bb_position < 0.2 else 'neutral'),
                'trendText': '上轨附近' if bb_position > 0.8 else ('下轨附近' if bb_position < 0.2 else '中轨区域'),
                'signal': 'sell' if bb_position > 0.8 else ('buy' if bb_position < 0.2 else 'hold'),
                'signalText': '卖出' if bb_position > 0.8 else ('买入' if bb_position < 0.2 else '观望')
            })
            
            # 成交量比率
            vol_ratio = features.get('volume_ratio', 1)
            indicators.append({
                'name': '量比',
                'value': f"{vol_ratio:.2f}",
                'trend': 'positive' if vol_ratio > 1.5 else ('negative' if vol_ratio < 0.5 else 'neutral'),
                'trendText': '放量' if vol_ratio > 1.5 else ('缩量' if vol_ratio < 0.5 else '正常'),
                'signal': 'buy' if vol_ratio > 1.5 else 'hold',
                'signalText': '关注' if vol_ratio > 1.5 else '观望'
            })
        
        # 模拟成分股数据（实际项目中应从数据源获取）
        components = [
            {'name': '个股A', 'code': '000001', 'price': '15.68', 'change': '+2.35%'},
            {'name': '个股B', 'code': '000002', 'price': '23.45', 'change': '+1.28%'},
            {'name': '个股C', 'code': '000003', 'price': '8.92', 'change': '-0.56%'},
            {'name': '个股D', 'code': '000004', 'price': '45.60', 'change': '+3.12%'},
            {'name': '个股E', 'code': '000005', 'price': '12.30', 'change': '-1.05%'},
        ]
        
        # 计算历史表现（基于真实价格数据）
        history = {}
        if price_history and len(price_history) >= 5:
            # 5日涨跌幅
            start_price_5d = price_history[0]['price']
            end_price = price_history[-1]['price']
            change_5d = ((end_price - start_price_5d) / start_price_5d * 100) if start_price_5d > 0 else 0
            history['5d'] = f"{change_5d:+.2f}%"
        else:
            history['5d'] = '--'
            
        if price_history and len(price_history) >= 10:
            start_price_10d = price_history[-10]['price']
            end_price = price_history[-1]['price']
            change_10d = ((end_price - start_price_10d) / start_price_10d * 100) if start_price_10d > 0 else 0
            history['10d'] = f"{change_10d:+.2f}%"
        else:
            history['10d'] = '--'
            
        if price_history and len(price_history) >= 30:
            start_price_30d = price_history[-30]['price']
            end_price = price_history[-1]['price']
            change_30d = ((end_price - start_price_30d) / start_price_30d * 100) if start_price_30d > 0 else 0
            history['30d'] = f"{change_30d:+.2f}%"
        else:
            history['30d'] = '--'
            
        # 年初至今（基于全年数据）
        if price_history and len(price_history) >= 60:
            start_price_year = price_history[0]['price']
            end_price = price_history[-1]['price']
            change_year = ((end_price - start_price_year) / start_price_year * 100) if start_price_year > 0 else 0
            history['year'] = f"{change_year:+.2f}%"
        else:
            history['year'] = '--'
        
        # 计算风险等级
        volatility = features.get('volatility_20', 0) if features else 0
        if volatility < 0.02:
            risk_level = '低'
            risk_desc = '波动较小，风险较低'
        elif volatility < 0.04:
            risk_level = '中'
            risk_desc = '波动适中，注意风险控制'
        else:
            risk_level = '高'
            risk_desc = '波动较大，谨慎操作'
        
        # 构建返回数据
        result = {
            'success': True,
            'data': {
                'name': sector_name,
                'id': sector_id,
                'code': sector_name,
                'currentPrice': f"{features.get('close', 0):.2f}" if features else '--',
                'priceChange': f"+{prediction.get('predicted_return', 0):.2f}%" if prediction.get('predicted_return', 0) >= 0 else f"{prediction.get('predicted_return', 0):.2f}%",
                'changePercent': f"+{prediction.get('predicted_return', 0):.2f}%" if prediction.get('predicted_return', 0) >= 0 else f"{prediction.get('predicted_return', 0):.2f}%",
                'volume': f"{features.get('volume', 0) / 10000:.0f}万" if features and features.get('volume', 0) > 0 else '--',
                'turnover': f"{features.get('turnover', 0) / 100000000:.2f}亿" if features and features.get('turnover', 0) > 0 else '--',
                'prediction': f"+{prediction.get('predicted_return', 0):.2f}%" if prediction.get('predicted_return', 0) >= 0 else f"{prediction.get('predicted_return', 0):.2f}%",
                'accuracy': f"{prediction.get('probability', 0) * 100:.1f}",
                'riskLevel': risk_level,
                'riskDesc': risk_desc,
                'signal': prediction.get('signal', '观望'),
                'confidence': prediction.get('confidence', '中等'),
                'recommendation': prediction.get('recommendation', '建议观望'),
                'signal_analysis': prediction.get('signal_analysis', {}),
                'indicators': indicators,
                'components': components,
                'history': history,
                'priceHistory': price_history,
                'prediction_date': prediction.get('prediction_date', ''),
                'base_date': prediction.get('base_date', '')
            }
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f"获取板块详情失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})


@app.route('/sector_detail')
def sector_detail_page():
    """板块详情页面"""
    return render_template('sector_detail.html')


@app.route('/api/train/single', methods=['POST'])
def train_single():
    """训练单个板块模型"""
    data = request.get_json()
    sector_name = data.get('sector', '半导体')
    custom_params = data.get('params', {})
    
    try:
        # 合并自定义参数和默认配置
        train_config = config.copy()
        if custom_params:
            train_config = merge_config(train_config, custom_params)
        
        predictor = SectorPredictor(sector_name, train_config)
        result = predictor.train()
        
        return jsonify({
            'success': True,
            'result': result,
            'message': f'{sector_name} 模型训练完成'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/train/multi', methods=['POST'])
def train_multi():
    """训练多个板块模型"""
    data = request.get_json() or {}
    sectors = data.get('sectors', [])
    custom_params = data.get('params', {})
    
    if not sectors:
        return jsonify({'success': False, 'error': '请选择至少一个板块'})
    
    try:
        # 合并自定义参数和默认配置
        train_config = config.copy()
        if custom_params:
            train_config = merge_config(train_config, custom_params)
        
        results = {}
        success_count = 0
        
        print(f"\n{'='*50}")
        print(f"开始训练 {len(sectors)} 个板块模型")
        print(f"训练参数: {custom_params}")
        print(f"{'='*50}")
        
        # 清除相关缓存，确保使用新训练的模型
        for sector in sectors:
            cache_key = f'predictor_{sector}'
            if cache_key in predictors_cache:
                del predictors_cache[cache_key]
        
        for i, sector in enumerate(sectors):
            try:
                progress = int(((i + 1) / len(sectors)) * 100)
                print(f"[进度 {progress}%] 正在训练板块 {i+1}/{len(sectors)}: {sector}...")
                
                predictor = SectorPredictor(sector, train_config)
                result = predictor.train()
                
                # 缓存训练好的predictor
                cache_key = f'predictor_{sector}'
                predictors_cache[cache_key] = {
                    'predictor': predictor,
                    'config': train_config,
                    'trained_at': datetime.now().isoformat()
                }
                
                results[sector] = result
                success_count += 1
                
                print(f"  ✓ {sector} 训练完成")
            except Exception as e:
                results[sector] = {'error': str(e)}
                print(f"  ✗ {sector} 训练失败: {str(e)}")
        
        # 清除预测缓存，确保下次预测使用新模型
        cache_manager.clear('predict_all')
        print("已清除预测缓存，下次预测将使用新训练的模型")
        
        print(f"{'='*50}")
        print(f"训练完成！成功 {success_count}/{len(sectors)} 个板块")
        print(f"{'='*50}\n")
        
        return jsonify({
            'success': True,
            'results': results,
            'success_count': success_count,
            'message': f'成功训练 {success_count}/{len(sectors)} 个板块模型',
            'cached': True
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/train/all', methods=['POST'])
def train_all():
    """训练所有板块模型"""
    try:
        global multi_predictor
        
        data = request.get_json() or {}
        custom_params = data.get('params', {})
        
        # 合并自定义参数和默认配置
        train_config = config.copy()
        if custom_params:
            train_config = merge_config(train_config, custom_params)
        
        # 获取所有可用板块（使用缓存）
        sectors = get_cached_sectors()
        
        multi_predictor = MultiSectorPredictor(train_config)
        for sector in sectors:
            multi_predictor.add_sector(sector)
        
        results = multi_predictor.train_all()
        
        return jsonify({
            'success': True,
            'results': results,
            'message': f'成功训练 {len(sectors)} 个板块模型'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/statistics/charts', methods=['GET'])
def get_statistics_charts():
    """获取统计图表数据"""
    chart_type = request.args.get('type', 'overview')
    time_range = request.args.get('range', '7d')
    
    try:
        # 尝试从缓存获取预测数据
        cached_predictions = cache_manager.get('predict_all')
        
        if chart_type == 'overview':
            # 获取所有板块的涨跌幅数据
            data = []
            
            if cached_predictions and cached_predictions.get('predictions'):
                # 使用缓存的预测数据
                for pred in cached_predictions['predictions']:
                    if 'error' not in pred:
                        data.append({
                            'name': pred.get('sector_name', ''),
                            'value': pred.get('predicted_return', 0),
                            'probability': pred.get('probability', 0.5)
                        })
            else:
                # 如果没有缓存，返回提示信息
                # 前端应该先调用 /api/predict/all 接口来生成数据
                return jsonify({
                    'success': False,
                    'error': '数据未初始化，请先调用预测接口生成数据',
                    'need_init': True,
                    'type': 'bar',
                    'data': []
                })
            
            return jsonify({
                'success': True,
                'type': 'bar',
                'data': data
            })
        
        elif chart_type == 'signal_distribution':
            # 获取交易信号分布数据
            signals = {'买入': 0, '卖出': 0, '观望': 0, '强烈买入': 0, '强烈卖出': 0}
            
            if cached_predictions and cached_predictions.get('predictions'):
                # 使用缓存的预测数据
                for pred in cached_predictions['predictions']:
                    if 'error' not in pred:
                        signal = pred.get('signal', '观望')
                        if signal in signals:
                            signals[signal] += 1
                        else:
                            signals['观望'] += 1
            else:
                # 如果没有缓存，返回默认数据（不进行实时预测）
                # 保留默认的观望信号
                pass
            
            data = [{'name': k, 'value': v} for k, v in signals.items() if v > 0]
            
            return jsonify({
                'success': True,
                'type': 'doughnut',
                'data': data
            })
        
        elif chart_type == 'trend':
            # 获取趋势数据（模拟）
            data = []
            dates = ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05']
            for date in dates:
                data.append({
                    'date': date,
                    'close': 100 + (dates.index(date) * 2)
                })
            
            return jsonify({
                'success': True,
                'type': 'line',
                'data': data
            })
        
        else:
            return jsonify({
                'success': False,
                'error': '不支持的图表类型'
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/api/analysis/full', methods=['POST'])
def full_analysis():
    """完整分析"""
    data = request.get_json()
    sector_name = data.get('sector', '半导体')
    force_refresh = data.get('force_refresh', False)
    
    # 尝试从缓存获取
    if not force_refresh:
        cached_data = cache_manager.get('analysis', sector_name)
        if cached_data:
            cached_data['from_cache'] = True
            return jsonify(cached_data)
    
    try:
        predictor = SectorPredictor(sector_name, config)
        
        # 准备数据
        predictor.prepare_data()
        
        # 训练
        train_result = predictor.train()
        
        # 预测
        prediction = predictor.predict()
        
        # 特征
        features = predictor.get_historical_features()
        
        result = {
            'success': True,
            'prediction': prediction,
            'features': features,
            'train_result': train_result,
            'from_cache': False
        }
        
        # 保存到缓存
        cache_manager.set('analysis', result, sector_name)
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/market/overview', methods=['GET'])
def market_overview():
    """市场概览数据"""
    try:
        fetcher = DataFetcher(config)
        
        # 使用缓存的板块列表
        sectors = get_cached_sectors()
        market_data = []
        
        for sector in sectors:
            try:
                # 使用 get_sector_fund_flow 代替不存在的 get_sector_data
                data = fetcher.get_sector_fund_flow(sector, days=5)
                if data is not None and len(data) > 0:
                    latest = data.iloc[-1]
                    market_data.append({
                        'sector': sector,
                        'close': float(latest.get('close', 0)),
                        'change_pct': float(latest.get('change_pct', 0)),
                        'net_inflow': float(latest.get('net_inflow', 0))
                    })
            except Exception as e:
                print(f"获取板块 {sector} 数据失败: {e}")
                pass
        
        return jsonify({
            'success': True,
            'data': market_data
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ==================== 缓存管理API ====================

@app.route('/api/cache/status', methods=['GET'])
def get_cache_status():
    """
    获取缓存状态信息
    
    返回当前缓存的数据是否过期、过期时间等信息
    用于前端判断是否需要显示刷新提示
    
    刷新机制说明：
    - 缓存数据过期时自动触发重新训练和预测
    - 用户主动点击刷新按钮时触发
    - 用户首次打开页面时触发
    - 常规页面刷新和导航不触发重新训练
    """
    try:
        # 获取预测缓存状态
        predict_all_timestamp = cache_manager.get_cache_timestamp('predict_all')
        is_predict_expired = cache_manager.is_cache_expired('predict_all')
        
        # 计算剩余有效时间
        remaining_time = None
        if predict_all_timestamp and not is_predict_expired:
            from datetime import datetime
            try:
                cache_time = datetime.fromisoformat(predict_all_timestamp)
                ttl = cache_manager.cache_expiry.get('predict_all', 21600)
                elapsed = (datetime.now() - cache_time).total_seconds()
                remaining_time = max(0, ttl - elapsed)
            except:
                pass
        
        # 获取统计信息
        stats = cache_manager.get_stats()
        
        # 缓存过期时间配置（秒）
        cache_expiry = {
            'predict_all': {
                'ttl': cache_manager.cache_expiry.get('predict_all', 21600),
                'ttl_display': f"{cache_manager.cache_expiry.get('predict_all', 21600) // 3600}小时",
                'description': '全板块预测结果缓存'
            },
            'predict_multi': {
                'ttl': cache_manager.cache_expiry.get('predict_multi', 14400),
                'ttl_display': f"{cache_manager.cache_expiry.get('predict_multi', 14400) // 3600}小时",
                'description': '多板块预测结果缓存'
            },
            'analysis': {
                'ttl': cache_manager.cache_expiry.get('analysis', 7200),
                'ttl_display': f"{cache_manager.cache_expiry.get('analysis', 7200) // 3600}小时",
                'description': '深度分析结果缓存'
            }
        }
        
        return jsonify({
            'success': True,
            'has_cache': predict_all_timestamp is not None,
            'is_expired': is_predict_expired,
            'remaining_time_seconds': remaining_time,
            'cache_timestamp': predict_all_timestamp,
            'cache_expiry': cache_expiry,
            'stats': stats,
            'refresh_conditions': {
                'description': '数据刷新触发条件',
                'conditions': [
                    '缓存数据过期时（自动）',
                    '用户主动点击刷新按钮时',
                    '用户首次打开页面时'
                ],
                'note': '常规页面刷新和导航不会触发重新训练，将使用现有缓存数据'
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/cache/info', methods=['GET'])
def get_cache_info():
    """获取缓存信息"""
    try:
        info = cache_manager.get_cache_info()
        return jsonify({
            'success': True,
            'info': info
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/cache/clear', methods=['POST'])
def clear_cache():
    """清除缓存"""
    try:
        data = request.get_json() or {}
        cache_type = data.get('cache_type')
        identifier = data.get('identifier', '')
        
        # 清除指定类型的缓存
        if cache_type:
            cache_manager.clear(cache_type, identifier)
        else:
            # 清除所有缓存
            cache_manager.clear('predict_all', '')
            cache_manager.clear('predict_multi', '')
            cache_manager.clear('predict_single', '')
            memory_cache.clear()
        
        return jsonify({
            'success': True,
            'message': '缓存已清除'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ==================== 板块缓存API ====================

@app.route('/api/section-cache/info', methods=['GET'])
def get_section_cache_info():
    """获取板块缓存信息"""
    try:
        info = section_cache_manager.get_all_cache_info()
        return jsonify({
            'success': True,
            'info': info
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/section-cache/get/<section_name>', methods=['GET'])
def get_section_cache(section_name):
    """获取指定板块的缓存数据"""
    try:
        cached_data = section_cache_manager.get(section_name)
        if cached_data:
            return jsonify({
                'success': True,
                'data': cached_data,
                'from_cache': True
            })
        return jsonify({
            'success': True,
            'data': None,
            'from_cache': False,
            'message': f'板块 "{section_name}" 缓存不存在或已过期'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/section-cache/set/<section_name>', methods=['POST'])
def set_section_cache(section_name):
    """设置指定板块的缓存数据"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': '请提供缓存数据'})
        
        ttl_seconds = request.args.get('ttl', type=int)
        
        success = section_cache_manager.set(section_name, data, ttl_seconds)
        
        return jsonify({
            'success': success,
            'message': f'板块 "{section_name}" 缓存已设置' if success else '设置失败'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/section-cache/invalidate/<section_name>', methods=['POST'])
def invalidate_section_cache(section_name):
    """使指定板块的缓存失效（删除旧版本）"""
    try:
        success = section_cache_manager.invalidate(section_name)
        
        return jsonify({
            'success': success,
            'message': f'板块 "{section_name}" 缓存已失效' if success else f'板块 "{section_name}" 不存在'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/section-cache/refresh/<section_name>', methods=['POST'])
def refresh_section_cache(section_name):
    """刷新指定板块缓存（删除旧版本并生成新版本）"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': '请提供缓存数据'})
        
        success = section_cache_manager.refresh(section_name, data)
        
        return jsonify({
            'success': success,
            'message': f'板块 "{section_name}" 缓存已刷新' if success else '刷新失败'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/section-cache/invalidate-all', methods=['POST'])
def invalidate_all_section_cache():
    """使所有板块缓存失效"""
    try:
        section_cache_manager.invalidate_all()
        
        return jsonify({
            'success': True,
            'message': '所有板块缓存已失效'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


# ==================== 板块列表缓存管理API ====================

@app.route('/api/sectors/refresh', methods=['POST'])
def refresh_sectors():
    """手动刷新板块列表缓存"""
    data = request.get_json() or {}
    background = data.get('background', False)
    
    if background:
        # 后台异步刷新
        import threading
        thread = threading.Thread(target=refresh_sectors_cache_background)
        thread.start()
        return jsonify({
            'success': True,
            'message': '后台刷新任务已启动'
        })
    else:
        # 同步刷新
        result = refresh_sectors_cache_background()
        return jsonify(result)


@app.route('/api/sectors/cache/status', methods=['GET'])
def get_sectors_cache_status():
    """获取板块列表缓存状态"""
    # 从 cache_manager 获取
    cached_data = cache_manager.get('sectors')
    timestamp = cache_manager.get_cache_timestamp('sectors')
    
    # 获取本地缓存信息
    cache_time = _sectors_cache_time.isoformat() if _sectors_cache_time else None
    
    return jsonify({
        'success': True,
        'has_cache': cached_data is not None,
        'timestamp': timestamp,
        'local_cache_time': cache_time,
        'cache_count': len(cached_data) if cached_data else 0,
        'refresh_in_progress': _sectors_refresh_in_progress
    })


# ==================== 错误处理 ====================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': '页面不存在'}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': '服务器内部错误'}), 500


# ==================== 启动应用 ====================

def delayed_init():
    """延迟初始化 - 在应用启动后执行"""
    import threading
    import time
    import os
    
    # 检测是否是 Flask debug 模式
    # WERKZEUG_RUN_MAIN 存在说明是 debug 模式
    is_debug_mode = 'WERKZEUG_RUN_MAIN' in os.environ
    is_reloader = os.environ.get('WERKZEUG_RUN_MAIN') == 'true'
    
    if is_debug_mode:
        # debug 模式下，只在子进程中执行初始化
        if not is_reloader:
            print(">>> 检测到 debug 模式主进程，跳过初始化（等待子进程）")
            return
        print(">>> 检测到 debug 模式子进程，执行初始化")
    else:
        # 非 debug 模式下，在唯一进程中执行初始化
        print(">>> 非 debug 模式，执行初始化")
    
    def init_task():
        # 延迟2秒后执行，确保Flask完全启动
        time.sleep(2)
        try:
            # # 启动时初始化板块列表缓存
            # print(">>> 延迟任务: 初始化板块列表缓存...")
            # init_sectors_cache_on_startup()
            
            # 启动板块列表定时刷新任务（每30分钟刷新一次）
            start_sectors_refresh_scheduler(interval_minutes=30)
            print(">>> 延迟任务: 板块列表缓存系统初始化完成")
            
            # # ====== 新增: 启动后台训练和预测任务 ======
            # print(">>> 延迟任务: 启动后台训练和预测任务...")
            # init_background_tasks(config)
            # print(">>> 延迟任务: 后台任务系统初始化完成")
            # # ====== 新增结束 ======
            
        except Exception as e:
            print(f">>> 延迟初始化异常: {str(e)}")
    
    thread = threading.Thread(target=init_task, daemon=True)
    thread.start()


def open_browser(url):
    """延迟打开浏览器"""
    import threading
    import time
    import webbrowser
    
    def _open():
        time.sleep(1.5)  # 等待服务器启动
        webbrowser.open(url)
    
    thread = threading.Thread(target=_open, daemon=True)
    thread.start()


if __name__ == '__main__':
    import os
    
    print("=" * 60)
    print("板块次日涨跌预测系统 - Web界面")
    print("=" * 60)
    
    # 检测是否是 Flask debug 模式
    is_debug_mode = 'WERKZEUG_RUN_MAIN' in os.environ
    is_reloader = os.environ.get('WERKZEUG_RUN_MAIN') == 'true'
    print(f">>> 运行模式: {'debug子进程' if is_reloader else ('debug主进程' if is_debug_mode else '正常模式')}")
    
    # 启动缓存清理定时任务（每小时清理一次，删除超过6小时的缓存）
    # 只在子进程或非 debug 模式下启动
    if is_reloader or not is_debug_mode:
        print("启动缓存清理定时任务...")
        start_cache_cleanup_scheduler(interval_hours=1, max_cache_age_hours=6)
    
    # 延迟初始化板块列表缓存（避免启动时阻塞导致热重载检测）
    print("调度延迟初始化任务...")
    delayed_init()
    
    # 自动打开浏览器（只在主进程或非 debug 模式下打开）
    server_url = "http://127.0.0.1:5000"
    if not is_reloader:
        print("自动打开浏览器...")
        open_browser(server_url)
    
    print("启动服务: " + server_url)
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)
