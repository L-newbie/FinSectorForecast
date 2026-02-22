# -*- coding: utf-8 -*-
"""
板块次日涨跌预测系统 - 主程序入口

功能：
1. 命令行接口
2. 模型训练
3. 预测执行
4. 批量分析
"""

import sys
import argparse
import yaml
from typing import Optional
from datetime import datetime

# 添加当前目录到路径
sys.path.insert(0, '.')

from src.data_fetcher import DataFetcher
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer
from src.predictor import SectorPredictor, MultiSectorPredictor


def load_config(config_path: str = "config/config.yaml") -> dict:
    """
    加载配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        dict: 配置字典
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"配置文件 {config_path} 未找到，使用默认配置")
        return {}


def train_single_sector(sector_name: str, config: dict) -> None:
    """
    训练单个板块模型

    Args:
        sector_name: 板块名称
        config: 配置字典
    """
    print(f"\n{'='*60}")
    print(f"训练板块: {sector_name}")
    print('='*60)

    # 创建预测器
    predictor = SectorPredictor(sector_name, config)

    # 训练
    results = predictor.train()

    print("\n训练完成!")


def predict_single_sector(sector_name: str, config: Optional[dict] = None, date: Optional[str] = None) -> dict:
    """
    预测单个板块

    Args:
        sector_name: 板块名称
        config: 配置字典
        date: 预测日期
    """
    print(f"\n{'='*60}")
    print(f"预测板块: {sector_name}")
    print('='*60)

    # 创建预测器
    predictor = SectorPredictor(sector_name, config)

    # 预测
    prediction = predictor.predict(date)

    # 输出结果
    print("\n" + "="*40)
    print("预测结果")
    print("="*40)
    print(f"板块名称:     {prediction['sector_name']}")
    print(f"预测日期:     {prediction['date']}")
    print(f"上涨概率:     {prediction['probability']:.2%}")
    print(f"预测涨幅:     {prediction['predicted_return']:.2f}%")
    print(f"置信度:       {prediction['confidence']}")
    print(f"交易信号:     {prediction['signal']}")
    print("-"*40)
    print(f"投资建议:")
    print(f"  {prediction['recommendation']}")
    print("="*40)


def train_all_sectors(config: dict) -> None:
    """
    训练所有板块模型

    Args:
        config: 配置字典
    """
    sectors = config.get('data', {}).get('sectors', [])

    if not sectors:
        # 获取默认板块列表
        fetcher = DataFetcher(config)
        sectors = fetcher.get_sectors_list()

    print(f"\n{'='*60}")
    print(f"训练所有板块模型")
    print(f"板块数量: {len(sectors)}")
    print('='*60)

    # 创建多板块预测器
    multi_predictor = MultiSectorPredictor(config)

    for sector in sectors:
        multi_predictor.add_sector(sector)

    # 训练所有
    results = multi_predictor.train_all()

    print("\n所有板块训练完成!")


def predict_all_sectors(config: dict, top_n: int = 10) -> None:
    """
    预测所有板块并排序

    Args:
        config: 配置字典
        top_n: 返回前N个
    """
    sectors = config.get('data', {}).get('sectors', [])

    if not sectors:
        fetcher = DataFetcher(config)
        sectors = fetcher.get_sectors_list()

    print(f"\n{'='*60}")
    print(f"预测所有板块")
    print('='*60)

    # 创建多板块预测器
    multi_predictor = MultiSectorPredictor(config)

    for sector in sectors:
        multi_predictor.add_sector(sector)

    # 训练所有
    print("训练模型...")
    multi_predictor.train_all()

    # 预测所有
    print("\n预测所有板块...")
    predictions = multi_predictor.predict_all()

    # 排序并输出
    print("\n" + "="*60)
    print("板块上涨概率排名")
    print("="*60)
    print(f"{'排名':<4} {'板块名称':<15} {'上涨概率':<10} {'预测涨幅':<10} {'信号':<10}")
    print("-"*60)

    for i, pred in enumerate(predictions[:top_n], 1):
        print(f"{i:<4} {pred['sector_name']:<15} {pred['probability']:.2%}      "
              f"{pred['predicted_return']:>6.2f}%    {pred['signal']:<10}")

    # 最佳机会
    opportunities = multi_predictor.get_top_opportunities(5)
    if opportunities:
        print("\n" + "="*60)
        print("最佳投资机会 (Top 5)")
        print("="*60)
        for i, opp in enumerate(opportunities, 1):
            print(f"{i}. {opp['sector_name']}: 上涨概率 {opp['probability']:.2%}, "
                  f"预测涨幅 {opp['predicted_return']:.2f}%")
            print(f"   建议: {opp['recommendation']}")


def analyze_sector(sector_name: str, config: dict) -> None:
    """
    完整分析单个板块

    Args:
        sector_name: 板块名称
        config: 配置字典
    """
    print(f"\n{'='*60}")
    print(f"全面分析板块: {sector_name}")
    print('='*60)

    # 创建预测器
    predictor = SectorPredictor(sector_name, config)

    # 准备数据
    predictor.prepare_data()

    # 训练模型
    print("\n[1/3] 训练模型...")
    predictor.train()

    # 预测
    print("\n[2/3] 预测...")
    prediction = predictor.predict()

    # 特征快照
    print("\n[3/3] 当前市场特征...")
    features = predictor.get_historical_features()

    # 输出
    print("\n" + "="*60)
    print("分析结果汇总")
    print("="*60)
    print(f"\n【预测结果】")
    print(f"  上涨概率:   {prediction['probability']:.2%}")
    print(f"  预测涨幅:   {prediction['predicted_return']:.2f}%")
    print(f"  交易信号:   {prediction['signal']}")
    print(f"  置信度:     {prediction['confidence']}")
    print(f"\n【投资建议】")
    print(f"  {prediction['recommendation']}")
    print(f"\n【当前市场特征】")
    print(f"  收盘价:     {features.get('close', 0):.2f}")
    print(f"  当日涨跌:   {features.get('change_pct', 0):.2f}%")
    print(f"  5日涨跌:    {features.get('return_5d', 0):.2f}%")
    print(f"  10日涨跌:   {features.get('return_10d', 0):.2f}%")
    print(f"  资金流向:   {features.get('net_inflow', 0):,.0f}")
    print(f"  RSI(14):    {features.get('rsi_14', 0):.2f}")
    print(f"  MACD:       {features.get('macd', 0):.4f}")
    print(f"  布林带位置: {features.get('bb_position', 0):.2f}")
    print("="*60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='板块次日涨跌预测系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 训练单个板块
  python main.py --mode train --sector 半导体

  # 预测单个板块
  python main.py --mode predict --sector 半导体

  # 训练所有板块
  python main.py --mode train-all

  # 预测所有板块并排名
  python main.py --mode predict-all --top 10

  # 完整分析
  python main.py --mode analyze --sector 半导体
        """
    )

    parser.add_argument('--mode', type=str, default='predict',
                        choices=['train', 'predict', 'train-all', 'predict-all', 'analyze'],
                        help='运行模式')
    parser.add_argument('--sector', type=str, default='半导体',
                        help='板块名称')
    parser.add_argument('--date', type=str, default=None,
                        help='预测日期 (YYYY-MM-DD)')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='配置文件路径')
    parser.add_argument('--top', type=int, default=10,
                        help='返回结果数量')

    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)

    # 执行
    if args.mode == 'train':
        train_single_sector(args.sector, config)
    elif args.mode == 'predict':
        predict_single_sector(args.sector, config, args.date)
    elif args.mode == 'train-all':
        train_all_sectors(config)
    elif args.mode == 'predict-all':
        predict_all_sectors(config, args.top)
    elif args.mode == 'analyze':
        analyze_sector(args.sector, config)


if __name__ == "__main__":
    main()
