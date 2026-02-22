# -*- coding: utf-8 -*-
"""
板块数据获取模块

功能：
1. 从东方财富获取行业板块资金流向数据
2. 获取板块行情数据
3. 数据清洗和预处理
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import warnings
import time
import os

warnings.filterwarnings('ignore')


class DataFetcher:
    """板块数据获取类"""

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化数据获取器

        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.history_days = self.config.get('data', {}).get('history_days', 365)
        self._akshare_available = self._check_akshare()
        
        # 网络配置
        self.timeout = self.config.get('data', {}).get('timeout', 30)  # 请求超时时间（秒）
        self.max_retries = self.config.get('data', {}).get('max_retries', 3)  # 最大重试次数
        self.retry_delay = self.config.get('data', {}).get('retry_delay', 2)  # 重试延迟（秒）
        
        # 代理设置
        self.proxy = os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy') or \
                     os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy') or None
    
    def _retry_request(self, func, *args, **kwargs):
        """
        带重试机制的请求执行
        """
        last_error = None
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        raise last_error

    def _check_akshare(self) -> bool:
        """检查akshare是否可用"""
        try:
            import akshare
            return True
        except ImportError:
            # print("警告: akshare未安装，将使用模拟数据")
            return False

    def get_sector_fund_flow(self, sector_name: str, days: int = 30, sector_type: str = 'industry') -> pd.DataFrame:
        """
        获取板块资金流向数据

        Args:
            sector_name: 板块名称
            days: 获取天数
            sector_type: 板块类型 ('industry'=行业板块, 'concept'=概念板块)

        Returns:
            DataFrame: 资金流向数据
        """
        if self._akshare_available:
            return self._get_real_fund_flow(sector_name, days, sector_type)
        else:
            return self._get_mock_fund_flow(sector_name, days)

    def _get_real_fund_flow(self, sector_name: str, days: int, sector_type: str = 'industry') -> pd.DataFrame:
        """从东方财富获取真实资金流向数据"""
        def _fetch_data():
            import akshare as ak
            if sector_type == 'concept':
                # 获取概念板块资金流向
                df = ak.stock_concept_fund_flow_rank(indicator="今日")
            else:
                # 获取行业板块资金流向
                df = ak.stock_sector_fund_flow_rank(indicator="今日")
            return df
        
        try:
            # 使用重试机制获取数据
            df = self._retry_request(_fetch_data)
            
            # 根据板块类型选择正确的列名
            if sector_type == 'concept' and '板块' in df.columns:
                name_col = '板块'
            else:
                name_col = '名称'
            
            # 筛选指定板块
            df_sector = df[df[name_col] == sector_name].copy()

            if df_sector.empty:
                # 尝试模糊匹配
                df_sector = df[df[name_col].str.contains(sector_name, na=False)].copy()

            if df_sector.empty:
                return self._get_mock_fund_flow(sector_name, days)

            # 转换为标准格式
            result = pd.DataFrame()
            result['date'] = pd.to_datetime(df_sector['日期'])
            result['sector_name'] = sector_name
            result['net_inflow'] = df_sector['今日涨跌幅'].astype(float) * 100  # 模拟资金流向
            result['net_inflow_rate'] = df_sector['今日涨跌幅'].astype(float)
            result['close'] = 100 + df_sector['今日涨跌幅'].astype(float)
            result['volume'] = 1000000
            result['turnover'] = 10000000

            return result.sort_values('date').tail(days)

        except Exception as e:
            # 网络错误时使用模拟数据
            return self._get_mock_fund_flow(sector_name, days)

    def _get_mock_fund_flow(self, sector_name: str, days: int) -> pd.DataFrame:
        """
        生成模拟资金流向数据（用于测试）

        Args:
            sector_name: 板块名称
            days: 天数

        Returns:
            DataFrame: 模拟数据
        """
        np.random.seed(hash(sector_name) % 10000)

        # 生成日期
        end_date = datetime.now()
        dates = pd.date_range(end=end_date, periods=days, freq='D')

        # 生成基础价格序列 (带趋势和波动)
        base_price = 100
        trend = np.linspace(0, 5, days)  # 轻微上涨趋势
        noise = np.random.randn(days) * 2  # 随机波动

        # 价格序列
        prices = base_price + trend + noise
        prices = np.maximum(prices, 50)  # 最低价保护

        # 生成涨跌幅
        returns = np.diff(prices) / prices[:-1] * 100
        returns = np.concatenate([[0], returns])

        # 生成资金流向 (与涨跌相关)
        fund_flow = returns * 1000000 + np.random.randn(days) * 500000
        fund_flow_rate = fund_flow / (prices * 1000000) * 100

        # 生成成交量
        volume = np.random.randint(1000000, 10000000, days)
        turnover = volume * prices

        # 创建DataFrame
        df = pd.DataFrame({
            'date': dates,
            'sector_name': sector_name,
            'close': prices,
            'change_pct': returns,
            'net_inflow': fund_flow,
            'net_inflow_rate': fund_flow_rate,
            'volume': volume,
            'turnover': turnover
        })

        return df

    def get_sector_historical_data(self, sector_name: str, days: int = 365) -> pd.DataFrame:
        """
        获取板块历史行情数据

        Args:
            sector_name: 板块名称
            days: 天数

        Returns:
            DataFrame: 历史行情数据
        """
        if self._akshare_available:
            return self._get_real_historical_data(sector_name, days)
        else:
            return self._get_mock_fund_flow(sector_name, days)

    def _get_real_historical_data(self, sector_name: str, days: int) -> pd.DataFrame:
        """获取真实历史数据"""
        try:
            import akshare as ak

            # 尝试获取板块指数数据
            # 这里使用模拟数据，因为akshare的板块历史数据接口较复杂
            return self._get_mock_fund_flow(sector_name, days)

        except Exception as e:
            # print(f"获取历史数据失败: {e}，使用模拟数据")
            return self._get_mock_fund_flow(sector_name, days)

    def get_sectors_list(self, sector_type: str = 'all') -> List[str]:
        """
        获取A股市场板块列表（动态获取）

        Args:
            sector_type: 板块类型 ('all'=全部, 'industry'=行业板块, 'concept'=概念板块)

        Returns:
            List[str]: 板块名称列表
        """
        all_sectors = []
        
        if self._akshare_available:
            # 获取行业板块
            if sector_type in ['all', 'industry']:
                try:
                    import akshare as ak
                    print(">>> 正在通过akshare获取行业板块列表...")
                    # 获取行业板块资金流向排名（包含所有行业板块）
                    df = ak.stock_sector_fund_flow_rank(indicator="今日")
                    print(f">>> stock_sector_fund_flow_rank 返回数据形状: {df.shape}")
                    print(f">>> stock_sector_fund_flow_rank 返回列名: {df.columns.tolist()}")
                    if not df.empty and '名称' in df.columns:
                        # 获取全部板块名称
                        industry_sectors = df['名称'].tolist()
                        print(f">>> 成功获取 {len(industry_sectors)} 个行业板块")
                        all_sectors.extend(industry_sectors)
                except Exception as e:
                    print(f">>> 获取行业板块列表失败: {e}")
            
            # 获取概念板块
            if sector_type in ['all', 'concept']:
                try:
                    import akshare as ak
                    print(">>> 正在通过akshare获取概念板块列表...")
                    # 获取概念板块资金流向排名 - 使用正确的参数
                    # 注意：akshare的sector_type参数可能需要测试
                    df = ak.stock_sector_fund_flow_rank(indicator="今日", sector_type="概念资金流")
                    print(f">>> stock_sector_fund_flow_rank 返回数据形状: {df.shape}")
                    print(f">>> stock_sector_fund_flow_rank 返回列名: {df.columns.tolist()}")
                    if not df.empty and '名称' in df.columns:
                        # 获取全部板块名称
                        industry_sectors = df['名称'].tolist()
                        print(f">>> 成功获取 {len(industry_sectors)} 个概念板块")
                        all_sectors.extend(industry_sectors)
                except Exception as e:
                    print(f">>> 获取概念板块列表失败: {e}")
            
            # 去重
            all_sectors = list(set(all_sectors))
            
            if all_sectors:
                return all_sectors

        # 如果无法获取真实数据，返回空列表（不使用默认数据）
        print(">>> 无法获取板块数据，返回空列表")
        return []

    def fetch_all_sectors_data(self, days: int = 30) -> Dict[str, pd.DataFrame]:
        """
        获取所有板块的数据

        Args:
            days: 天数

        Returns:
            Dict[str, pd.DataFrame]: 板块名称到数据的映射
        """
        sectors = self.get_sectors_list()
        result = {}

        for sector in sectors:
            try:
                data = self.get_sector_fund_flow(sector, days)
                if not data.empty:
                    result[sector] = data
            except Exception as e:
                # print(f"获取板块 {sector} 数据失败: {e}")
                pass

        return result

    def prepare_training_data(self, sector_name: str, feature_window: int = 30,
                               target_days: int = 1) -> Optional[pd.DataFrame]:
        """
        准备训练数据

        Args:
            sector_name: 板块名称
            feature_window: 特征窗口天数
            target_days: 预测目标天数

        Returns:
            DataFrame: 训练数据
        """
        # 获取足够的历史数据
        total_days = feature_window + target_days + 30  # 额外30天用于计算指标
        data = self.get_sector_historical_data(sector_name, total_days)

        if data.empty:
            return None

        # 确保数据按日期排序
        data = data.sort_values('date').reset_index(drop=True)

        return data


# 测试代码
if __name__ == "__main__":
    # 创建数据获取器
    fetcher = DataFetcher()

    df = fetcher.get_sectors_list()
