# -*- coding: utf-8 -*-
"""
预测模块

功能：
1. 整合数据获取、特征工程、模型训练
2. 提供统一的预测接口
3. 输出预测结果和建议
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from datetime import datetime
import warnings

from .data_fetcher import DataFetcher
from .feature_engineering import FeatureEngineer
from .model_training import ModelTrainer

warnings.filterwarnings('ignore')


class SectorPredictor:
    """板块预测器"""

    def __init__(self, sector_name: str, config: Optional[Dict] = None):
        """
        初始化板块预测器

        Args:
            sector_name: 板块名称
            config: 配置字典
        """
        self.sector_name = sector_name
        self.config = config or {}

        # 初始化各模块
        self.data_fetcher = DataFetcher(self.config)
        self.feature_engineer = FeatureEngineer(self.config)
        self.model_trainer = ModelTrainer(self.config)

        # 数据
        self.raw_data = None
        self.feature_data = None
        self.training_results = None

        # 配置 - 直接从config中读取，避免重复读取
        self.history_days = self.config.get('data', {}).get('history_days', 365)
        self.feature_window = self.config.get('data', {}).get('feature_window', 60)
        self.probability_threshold = self.config.get('predict', {}).get('probability_threshold', 0.8)
        
        # 交易信号判定配置 - 从config中读取，使用默认值作为后备
        signal_config = self.config.get('signal', {})
        
        # 置信度判定配置
        confidence_config = signal_config.get('confidence', {})
        self.confidence_very_high_offset = confidence_config.get('very_high_offset', 0.1)  # 非常高置信度的偏移量
        self.confidence_medium_threshold = confidence_config.get('medium_threshold', 0.55)  # 中等置信度阈值
        self.confidence_low_threshold = confidence_config.get('low_threshold', 0.45)  # 低置信度阈值
        
        # 交易信号判定配置
        trading_config = signal_config.get('trading', {})
        self.strong_signal_offset = trading_config.get('strong_signal_offset', 0.1)  # 强烈信号的偏移量
        self.predicted_return_threshold = trading_config.get('predicted_return_threshold', 0)  # 预测涨跌幅阈值
        
        # 投资建议判定配置
        recommendation_config = signal_config.get('recommendation', {})
        self.recommend_high_prob_threshold = recommendation_config.get('high_prob_threshold', 0.7)  # 高概率阈值
        self.recommend_medium_prob_threshold = recommendation_config.get('medium_prob_threshold', 0.55)  # 中等概率阈值
        self.recommend_low_prob_threshold = recommendation_config.get('low_prob_threshold', 0.4)  # 低概率阈值
        self.recommend_large_return_threshold = recommendation_config.get('large_return_threshold', 1.0)  # 大涨跌幅阈值
        
        # 信号分析配置
        analysis_config = signal_config.get('analysis', {})
        self.analysis_high_prob_threshold = analysis_config.get('high_prob_threshold', 0.6)  # 信号分析中的高概率阈值
        self.analysis_low_prob_threshold = analysis_config.get('low_prob_threshold', 0.4)  # 信号分析中的低概率阈值
        self.analysis_large_volatility_threshold = analysis_config.get('large_volatility_threshold', 1.0)  # 大波动阈值
        
        # 技术指标配置
        technical_config = signal_config.get('technical', {})
        self.rsi_overbought = technical_config.get('rsi_overbought', 70)  # RSI超买阈值
        self.rsi_oversold = technical_config.get('rsi_oversold', 30)  # RSI超卖阈值

    def prepare_data(self, days: int = 365) -> bool:
        """
        准备数据

        Args:
            days: 数据天数

        Returns:
            bool: 是否成功
        """
        # print(f"正在获取板块 [{self.sector_name}] 的数据...")

        # 获取数据
        self.raw_data = self.data_fetcher.get_sector_historical_data(self.sector_name, days)

        if self.raw_data.empty:
            # print(f"错误: 无法获取板块 {self.sector_name} 的数据")
            return False

        # print(f"获取到 {len(self.raw_data)} 条数据记录")

        # 创建特征
        # print("正在创建特征...")
        self.feature_data = self.feature_engineer.create_features(self.raw_data)
        self.feature_data = self.feature_engineer.create_target(self.feature_data)

        # print(f"特征数量: {len(self.feature_data.columns)}")

        return True

    def train(self, reload_data: bool = True) -> Dict:
        """
        训练模型

        Args:
            reload_data: 是否重新加载数据

        Returns:
            Dict: 训练结果
        """
        if reload_data or self.feature_data is None:
            if not self.prepare_data():
                return {'error': '数据准备失败'}

        # 准备训练数据
        print(f"\n>>> 开始训练板块: {self.sector_name}")
        print(f"    历史数据: {self.history_days}天, 特征窗口: {self.feature_window}天")
        # print("正在准备训练数据...")
        feature_cols, X, y_up, y_return = self.feature_engineer.prepare_training_dataset(self.feature_data)

        if len(X) < 50:
            return {'error': '训练数据不足'}

        # print(f"训练样本数: {len(X)}")
        # print(f"特征数量: {len(feature_cols)}")
        # print(f"正样本比例: {y_up.mean():.2%}")

        # print("\n开始训练模型...")
        self.training_results = self.model_trainer.train_all(
            X, y_up, y_return, feature_cols
        )

        # 打印报告
        # print("\n" + self.model_trainer.get_training_report(self.training_results))

        # 打印特征重要性
        # importance = self.model_trainer.get_feature_importance(10)
        # if not importance.empty:
        #     print("\n【Top 10 重要特征】")
        #     for i, row in importance.iterrows():
        #         print(f"  {row['feature']}: {row['importance']:.4f}")

        return self.training_results

    def predict(self, date: Optional[str] = None) -> Dict:
        """
        预测

        Args:
            date: 预测日期 (可选)

        Returns:
            Dict: 预测结果
        """
        # 检查模型是否已训练
        if self.model_trainer.classifier is None:
            # 尝试训练模型
            train_result = self.train()
            # 检查训练是否成功
            if train_result and isinstance(train_result, dict) and 'error' in train_result:
                raise ValueError(f"模型训练失败: {train_result.get('error', '未知错误')}")

        # 获取最新数据
        if self.feature_data is None:
            self.prepare_data()

        # 获取最后一条数据作为预测对象
        latest_data = self.feature_data.iloc[-1:].copy()

        # 准备特征
        feature_cols, X, _, _ = self.feature_engineer.prepare_training_dataset(
            self.feature_data, drop_na=False
        )

        # 获取最新特征
        X_latest = X.iloc[-1:].fillna(0)

        # 预测
        prediction = self.model_trainer.predict(X_latest)
        
        # 检查预测结果是否有效
        if 'error' in prediction:
            raise ValueError(f"预测失败: {prediction.get('error', '未知错误')}")

        # 计算预测目标日期（下一个交易日）
        from datetime import timedelta
        prediction_date = self._get_next_trading_day()

        # 获取特征用于分析
        features = self.get_historical_features()

        # 解析结果
        result = {
            'sector_name': self.sector_name,
            'date': prediction_date,
            'prediction_date': prediction_date,
            'base_date': datetime.now().strftime('%Y-%m-%d'),
            'probability': float(prediction['probability'][0]),
            'predicted_return': float(prediction['predicted_return'][0]),
            'prediction_up': bool(prediction['prediction_up'][0]),
            'confidence': self._get_confidence(prediction['probability'][0]),
            'signal': self._get_signal(
                prediction['probability'][0],
                prediction['predicted_return'][0]
            ),
            'recommendation': self._get_recommendation(
                prediction['probability'][0],
                prediction['predicted_return'][0]
            ),
            'signal_analysis': self._generate_signal_analysis(
                prediction['probability'][0],
                prediction['predicted_return'][0],
                features
            )
        }

        return result

    def _get_next_trading_day(self) -> str:
        """获取下一个交易日（跳过周末和节假日）"""
        from datetime import timedelta
        
        # 从网络获取今年的节假日（使用 akshare 库）
        holidays = self._get_china_stock_holidays()
        
        today = datetime.now()
        next_day = today + timedelta(days=1)
        
        # 跳过周末和节假日
        max_attempts = 15  # 最多尝试15天
        for _ in range(max_attempts):
            # 跳过周末
            if next_day.weekday() == 5:  # 周六
                next_day = next_day + timedelta(days=2)
                continue  # 重新检查新日期
            elif next_day.weekday() == 6:  # 周日
                next_day = next_day + timedelta(days=1)
                continue  # 重新检查新日期
            
            # 跳过节假日
            date_str = next_day.strftime('%Y-%m-%d')
            if date_str in holidays:
                next_day = next_day + timedelta(days=1)
                continue  # 重新检查新日期
            
            # 找到交易日
            break
        
        return next_day.strftime('%Y-%m-%d')
    
    def _get_china_stock_holidays(self) -> List[str]:
        """获取中国股市节假日列表（使用固定节假日列表，避免 akshare 依赖）"""
        try:
            # 直接返回默认节假日列表，避免 akshare 依赖
            current_year = datetime.now().year
            # 2025年节假日列表
            if current_year == 2025:
                return [
                    '2025-01-01', '2025-01-28', '2025-01-29', '2025-01-30', '2025-01-31',
                    '2025-02-01', '2025-02-02', '2025-02-03', '2025-02-04',
                    '2025-04-04', '2025-04-05', '2025-04-06',
                    '2025-05-01', '2025-05-02', '2025-05-03', '2025-05-04', '2025-05-05',
                    '2025-05-31', '2025-06-01', '2025-06-02',
                    '2025-10-01', '2025-10-02', '2025-10-03', '2025-10-04', '2025-10-05',
                    '2025-10-06', '2025-10-07', '2025-10-08',
                ]
            elif current_year == 2026:
                return [
                    '2026-01-01', '2026-01-27', '2026-01-28', '2026-01-29', '2026-01-30',
                    '2026-01-31', '2026-02-01', '2026-02-02',
                    '2026-04-04', '2026-04-05', '2026-04-06',
                    '2026-05-01', '2026-05-02', '2026-05-03', '2026-05-04', '2026-05-05',
                    '2026-06-07', '2026-06-08', '2026-06-09',
                    '2026-10-01', '2026-10-02', '2026-10-03', '2026-10-04', '2026-10-05',
                    '2026-10-06', '2026-10-07',
                ]
            else:
                # 其他年份默认节假日（2025年）
                return [
                    f'{current_year}-01-01',
                    f'{current_year}-04-04', f'{current_year}-04-05', f'{current_year}-04-06',
                    f'{current_year}-05-01', f'{current_year}-05-02', f'{current_year}-05-03', f'{current_year}-05-04', f'{current_year}-05-05',
                    f'{current_year}-10-01', f'{current_year}-10-02', f'{current_year}-10-03', f'{current_year}-10-04', f'{current_year}-10-05',
                    f'{current_year}-10-06', f'{current_year}-10-07',
                ]
        
        except Exception as e:
            # print(f"获取节假日数据失败: {e}，使用默认节假日")
            # 如果获取失败，使用默认节假日列表（2025年）
            return [
                '2025-01-01', '2025-01-28', '2025-01-29', '2025-01-30', '2025-01-31',
                '2025-02-01', '2025-02-02', '2025-02-03', '2025-02-04',
                '2025-04-04', '2025-04-05', '2025-04-06',
                '2025-05-01', '2025-05-02', '2025-05-03', '2025-05-04', '2025-05-05',
                '2025-05-31', '2025-06-01', '2025-06-02',
                '2025-10-01', '2025-10-02', '2025-10-03', '2025-10-04', '2025-10-05',
                '2025-10-06', '2025-10-07', '2025-10-08',
            ]

    def _get_confidence(self, probability: float) -> str:
        """获取预测置信度 - 使用配置中的阈值"""
        threshold = self.probability_threshold
        # 使用配置中的阈值判定置信度等级
        if probability > threshold + self.confidence_very_high_offset:
            return "非常高"
        elif probability > threshold:
            return "高"
        elif probability > self.confidence_medium_threshold:
            return "中等"
        elif probability > self.confidence_low_threshold:
            return "低"
        else:
            return "非常低"

    def _get_signal(self, probability: float, predicted_return: float) -> str:
        """获取交易信号 - 使用配置中的阈值"""
        threshold = self.probability_threshold
        # 使用配置中的阈值判定交易信号
        if probability > threshold + self.strong_signal_offset and predicted_return > self.predicted_return_threshold:
            return "强烈买入"
        elif probability > threshold and predicted_return > self.predicted_return_threshold:
            return "买入"
        elif probability < 1 - threshold - self.strong_signal_offset and predicted_return < self.predicted_return_threshold:
            return "强烈卖出"
        elif probability < 1 - threshold and predicted_return < self.predicted_return_threshold:
            return "卖出"
        else:
            return "观望"

    def _get_recommendation(self, probability: float, predicted_return: float) -> str:
        """获取投资建议 - 使用配置中的阈值"""
        # 使用配置中的阈值判定投资建议
        if probability > self.recommend_high_prob_threshold:
            if predicted_return > self.recommend_large_return_threshold:
                return f"板块{self.sector_name}预计上涨{predicted_return:.2f}%，建议重点关注"
            elif predicted_return > 0:
                return f"板块{self.sector_name}预计小幅上涨{predicted_return:.2f}%，可适当关注"
            else:
                return f"板块{self.sector_name}上涨概率较高但幅度有限，建议谨慎"
        elif probability > self.recommend_medium_prob_threshold:
            return f"板块{self.sector_name}有一定的上涨机会，建议保持关注"
        elif probability < self.recommend_low_prob_threshold:
            return f"板块{self.sector_name}下跌风险较大，建议回避"
        else:
            return f"板块{self.sector_name}方向不明，建议观望"

    def _generate_signal_analysis(self, probability: float, predicted_return: float, features: Dict) -> str:
        """生成详细的交易信号分析说明 - 使用配置中的阈值"""
        analysis_parts = []
        
        # 1. 整体信号解读
        signal = self._get_signal(probability, predicted_return)
        if signal == "强烈买入":
            analysis_parts.append(f"<div class='alert alert-success'><strong>📈 强烈买入信号</strong>：模型预测该板块次日有较大概率和幅度上涨，建议积极布局。</div>")
        elif signal == "买入":
            analysis_parts.append(f"<div class='alert alert-success'><strong>📈 买入信号</strong>：模型预测该板块次日上涨概率较高，可考虑适当配置。</div>")
        elif signal == "强烈卖出":
            analysis_parts.append(f"<div class='alert alert-danger'><strong>📉 强烈卖出信号</strong>：模型预测该板块次日下跌风险较大，建议规避风险。</div>")
        elif signal == "卖出":
            analysis_parts.append(f"<div class='alert alert-danger'><strong>📉 卖出信号</strong>：模型预测该板块次日下跌概率较高，建议减仓或观望。</div>")
        else:
            analysis_parts.append(f"<div class='alert alert-warning'><strong>⏸️ 观望信号</strong>：模型预测该板块方向不明确，建议暂时观望等待机会。</div>")
        
        # 2. 概率分析 - 使用配置中的阈值
        prob_level = "高" if probability > self.analysis_high_prob_threshold else "中等" if probability > self.analysis_low_prob_threshold else "低"
        analysis_parts.append(f"<h6 class='text-primary mt-3'>📊 概率分析</h6>")
        analysis_parts.append(f"<p>上涨概率为 <strong>{probability*100:.1f}%</strong>，置信度{prob_level}。")
        if probability > self.analysis_high_prob_threshold:
            analysis_parts.append(f"该概率超过{int(self.analysis_high_prob_threshold*100)}%阈值，表明模型对上涨趋势有较强信心。</p>")
        elif probability < self.analysis_low_prob_threshold:
            analysis_parts.append(f"该概率低于{int(self.analysis_low_prob_threshold*100)}%，表明模型对下跌趋势有较强信心。</p>")
        else:
            analysis_parts.append(f"该概率处于中间区域，市场方向存在不确定性。</p>")
        
        # 3. 涨跌幅分析 - 使用配置中的阈值
        analysis_parts.append(f"<h6 class='text-success mt-3'>📈 涨跌幅分析</h6>")
        analysis_parts.append(f"<p>预测涨跌幅为 <strong class='{'text-up' if predicted_return >= 0 else 'text-down'}'>{predicted_return:+.2f}%</strong>。")
        if abs(predicted_return) > self.analysis_large_volatility_threshold:
            analysis_parts.append(f"预期波动幅度较大，{'上涨空间可观' if predicted_return > 0 else '下跌风险显著'}。</p>")
        else:
            analysis_parts.append(f"预期波动幅度较小，市场可能处于震荡状态。</p>")
        
        # 4. 技术指标分析 - 使用配置中的阈值
        if features:
            analysis_parts.append(f"<h6 class='text-warning mt-3' style='color: #ffd43b !important;'>🔧 技术指标分析</h6>")
            
            # RSI分析 - 使用配置中的超买超卖阈值
            rsi = features.get('rsi_14', 50)
            if rsi > self.rsi_overbought:
                analysis_parts.append(f"<p>• <strong>RSI(14)={rsi:.1f}</strong>：处于超买区域（>{self.rsi_overbought}），短期可能面临回调压力。</p>")
            elif rsi < self.rsi_oversold:
                analysis_parts.append(f"<p>• <strong>RSI(14)={rsi:.1f}</strong>：处于超卖区域（<{self.rsi_oversold}），可能存在反弹机会。</p>")
            else:
                analysis_parts.append(f"<p>• <strong>RSI(14)={rsi:.1f}</strong>：处于正常区间（{self.rsi_oversold}-{self.rsi_overbought}），无明显超买超卖信号。</p>")
            
            # MACD分析
            macd = features.get('macd', 0)
            macd_signal = features.get('macd_signal', 0)
            if macd > macd_signal:
                analysis_parts.append(f"<p>• <strong>MACD</strong>：MACD线位于信号线上方，呈多头排列，短期趋势向好。</p>")
            else:
                analysis_parts.append(f"<p>• <strong>MACD</strong>：MACD线位于信号线下方，呈空头排列，短期趋势偏弱。</p>")
            
            # 资金流向分析
            net_inflow = features.get('net_inflow', 0)
            if net_inflow > 0:
                analysis_parts.append(f"<p>• <strong>资金流向</strong>：主力资金净流入{net_inflow/10000:.2f}万，资金面支撑上涨。</p>")
            elif net_inflow < 0:
                analysis_parts.append(f"<p>• <strong>资金流向</strong>：主力资金净流出{abs(net_inflow)/10000:.2f}万，资金面存在压力。</p>")
            else:
                analysis_parts.append(f"<p>• <strong>资金流向</strong>：主力资金净流入0.00万，资金面平衡。</p>")
            
            # 均线分析
            return_5d = features.get('return_5d', 0)
            return_20d = features.get('return_20d', 0)
            analysis_parts.append(f"<p>• <strong>均线趋势</strong>：5日涨跌{return_5d:+.2f}%，20日涨跌{return_20d:+.2f}%。")
            if return_5d > 0 and return_20d > 0:
                analysis_parts.append(f"短期和中期趋势均向上，走势健康。</p>")
            elif return_5d < 0 and return_20d < 0:
                analysis_parts.append(f"短期和中期趋势均向下，需谨慎对待。</p>")
            else:
                analysis_parts.append(f"短期和中期趋势不一致，市场处于震荡整理阶段。</p>")
        
        # 5. 风险提示
        analysis_parts.append(f"<h6 class='text-danger mt-3'>⚠️ 风险提示</h6>")
        analysis_parts.append(f"<p>本预测基于历史数据和技术指标，仅供参考，不构成投资建议。股市有风险，投资需谨慎。建议结合基本面分析和市场情绪综合判断。</p>")
        
        return ''.join(analysis_parts)

    def batch_predict(self, n_predictions: int = 5) -> List[Dict]:
        """
        批量预测（用于回测）

        Args:
            n_predictions: 预测数量

        Returns:
            List[Dict]: 预测结果列表
        """
        results = []

        if self.feature_data is None:
            self.prepare_data()

        # 准备特征
        feature_cols, X, y_up, y_return = self.feature_engineer.prepare_training_dataset(
            self.feature_data, drop_na=False
        )

        # 最后n_predictions个样本
        X_predict = X.tail(n_predictions).fillna(0)

        # 预测
        predictions = self.model_trainer.predict(X_predict)

        # 实际值
        y_up_actual = y_up.tail(n_predictions)
        y_return_actual = y_return.tail(n_predictions)

        for i in range(len(predictions['probability'])):
            result = {
                'probability': float(predictions['probability'][i]),
                'predicted_return': float(predictions['predicted_return'][i]),
                'actual_return': float(y_return_actual.iloc[i]) if i < len(y_return_actual) else None,
                'actual_up': bool(y_up_actual.iloc[i]) if i < len(y_up_actual) else None,
                'correct': predictions['prediction_up'][i] == y_up_actual.iloc[i] if i < len(y_up_actual) else None
            }
            results.append(result)

        return results

    def get_historical_features(self, date: Optional[str] = None) -> Dict:
        """
        获取历史特征快照

        Args:
            date: 日期

        Returns:
            Dict: 特征快照
        """
        if self.feature_data is None:
            self.prepare_data()

        # 获取最后一条
        latest = self.feature_data.iloc[-1]

        def _to_native(value):
            """将numpy类型转换为Python原生类型"""
            import numpy as np
            if isinstance(value, (np.integer, np.int32, np.int64)):
                return int(value)
            elif isinstance(value, (np.floating, np.float32, np.float64)):
                return float(value)
            elif isinstance(value, np.bool_):
                return bool(value)
            return value

        # 关键特征
        key_features = {
            'date': str(latest.get('date', '')),
            'close': _to_native(latest.get('close', 0)),
            'change_pct': _to_native(latest.get('change_pct', 0)),
            'net_inflow': _to_native(latest.get('net_inflow', 0)),
            'volume': _to_native(latest.get('volume', 0)),
            'turnover': _to_native(latest.get('turnover', 0)),
            'rsi_14': _to_native(latest.get('rsi_14', 0)),
            'macd': _to_native(latest.get('macd', 0)),
            'macd_signal': _to_native(latest.get('macd_signal', 0)),
            'macd_histogram': _to_native(latest.get('macd_histogram', 0)),
            'bb_position': _to_native(latest.get('bb_position', 0)),
            'volume_ratio_5d': _to_native(latest.get('volume_ratio_5d', 0)),
            'return_5d': _to_native(latest.get('return_5d', 0)),
            'return_10d': _to_native(latest.get('return_10d', 0)),
            'return_20d': _to_native(latest.get('return_20d', 0)),
            'net_inflow_5d': _to_native(latest.get('net_inflow_5d', 0)),
            'net_inflow_10d': _to_native(latest.get('net_inflow_10d', 0)),
            # 移动平均线 (注意字段名是 ma5, ma10, ma20)
            'ma_5': _to_native(latest.get('ma5', 0)),
            'ma_10': _to_native(latest.get('ma10', 0)),
            'ma_20': _to_native(latest.get('ma20', 0)),
            # 波动率 (注意字段名是 volatility_20d)
            'volatility_20': _to_native(latest.get('volatility_20d', 0)),
        }

        return key_features


class MultiSectorPredictor:
    """多板块预测器"""

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化多板块预测器

        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.predictors = {}

    def add_sector(self, sector_name: str):
        """添加板块"""
        self.predictors[sector_name] = SectorPredictor(sector_name, self.config)

    def train_all(self) -> Dict:
        """训练所有板块模型"""
        results = {}
        total_sectors = len(self.predictors)
        current_index = 0

        for sector_name, predictor in self.predictors.items():
            current_index += 1
            print(f"\n{'='*50}")
            print(f"训练板块 [{current_index}/{total_sectors}]: {sector_name}")
            print('='*50)
            try:
                result = predictor.train()
                results[sector_name] = result
            except Exception as e:
                print(f"训练失败: {e}")
                results[sector_name] = {'error': str(e)}

        return results

    def predict_all(self) -> List[Dict]:
        """预测所有板块"""
        results = []
        total_sectors = len(self.predictors)
        current_index = 0

        for sector_name, predictor in self.predictors.items():
            current_index += 1
            try:
                print(f"预测板块 [{current_index}/{total_sectors}]: {sector_name}")
                result = predictor.predict()
                results.append(result)
            except Exception as e:
                print(f"预测失败 {sector_name}: {e}")

        # 按上涨概率排序
        results.sort(key=lambda x: x.get('probability', 0), reverse=True)

        return results

    def get_top_opportunities(self, n: int = 5) -> List[Dict]:
        """
        获取最佳投资机会

        Args:
            n: 返回数量

        Returns:
            List[Dict]: 最佳机会列表
        """
        predictions = self.predict_all()

        # 筛选上涨概率较高的
        opportunities = [
            p for p in predictions
            if p.get('probability', 0) > 0.55
        ]

        return opportunities[:n]


# 测试代码
if __name__ == "__main__":
    # 创建预测器
    print("创建板块预测器...")
    predictor = SectorPredictor("半导体")

    # 训练模型
    print("\n训练模型...")
    results = predictor.train()

    # 预测
    print("\n预测结果:")
    prediction = predictor.predict()
    print(f"  板块: {prediction['sector_name']}")
    print(f"  上涨概率: {prediction['probability']:.2%}")
    print(f"  预测涨幅: {prediction['predicted_return']:.2f}%")
    print(f"  信号: {prediction['signal']}")
    print(f"  建议: {prediction['recommendation']}")

    # 获取当前特征
    print("\n当前市场特征:")
    features = predictor.get_historical_features()
    for k, v in features.items():
        print(f"  {k}: {v}")
