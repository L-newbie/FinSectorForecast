# -*- coding: utf-8 -*-
"""
特征工程模块

功能：
1. 价格特征计算
2. 资金特征计算
3. 动量指标计算
4. 成交量特征计算
5. 生成30天状态序列特征
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict
import warnings

warnings.filterwarnings('ignore')


class FeatureEngineer:
    """特征工程类"""

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化特征工程师

        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.feature_window = self.config.get('data', {}).get('feature_window', 60)

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建所有特征

        Args:
            df: 原始数据DataFrame

        Returns:
            DataFrame: 包含所有特征的DataFrame
        """
        if df.empty:
            return df

        df = df.copy()

        # 计算各类特征
        df = self._create_price_features(df)
        df = self._create_fund_features(df)
        df = self._create_momentum_features(df)
        df = self._create_volume_features(df)
        df = self._create_sequence_features(df)

        return df

    def _create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建价格特征"""
        # 确保有必要的列
        if 'close' not in df.columns:
            return df

        # 获取收盘价
        close = df['close'].values

        # 计算涨跌幅
        if 'change_pct' not in df.columns:
            df['change_pct'] = pd.Series(close).pct_change() * 100

        # 5日涨跌幅
        df['return_5d'] = df['change_pct'].rolling(window=5).sum()

        # 10日涨跌幅
        df['return_10d'] = df['change_pct'].rolling(window=10).sum()

        # 20日涨跌幅
        df['return_20d'] = df['change_pct'].rolling(window=20).sum()

        # 20日均线
        df['ma20'] = df['close'].rolling(window=20).mean()

        # 收盘价相对20日均线位置 (0-100)
        df['ma20_position'] = (df['close'] - df['ma20']) / df['ma20'] * 100

        # 5日均线
        df['ma5'] = df['close'].rolling(window=5).mean()

        # 5日均线与20日均线金叉死叉 (1=金叉, -1=死叉, 0=无)
        ma5_above_ma20 = (df['ma5'] > df['ma20']).astype(int)
        ma5_above_ma20_prev = ma5_above_ma20.shift(1)
        df['ma5_cross_ma20'] = (ma5_above_ma20 - ma5_above_ma20_prev).fillna(0)

        # 10日均线
        df['ma10'] = df['close'].rolling(window=10).mean()

        # 10日均线与20日均线关系
        df['ma10_above_ma20'] = (df['ma10'] > df['ma20']).astype(int)

        # 价格波动率
        df['volatility_5d'] = df['change_pct'].rolling(window=5).std()
        df['volatility_10d'] = df['change_pct'].rolling(window=10).std()
        df['volatility_20d'] = df['change_pct'].rolling(window=20).std()

        # 最高价/最低价相对位置
        if 'high' in df.columns and 'low' in df.columns:
            df['high_low_ratio'] = (df['high'] - df['low']) / df['close'] * 100

        return df

    def _create_fund_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建资金特征"""
        # 资金净流入
        if 'net_inflow' not in df.columns:
            # 从涨跌幅模拟资金流向
            df['net_inflow'] = df['change_pct'] * 1000000

        # 资金净流入率
        if 'net_inflow_rate' not in df.columns and 'turnover' in df.columns:
            df['net_inflow_rate'] = df['net_inflow'] / df['turnover'] * 100

        # 5日累计资金流向
        df['net_inflow_5d'] = df['net_inflow'].rolling(window=5).sum()

        # 10日累计资金流向
        df['net_inflow_10d'] = df['net_inflow'].rolling(window=10).sum()

        # 20日累计资金流向
        df['net_inflow_20d'] = df['net_inflow'].rolling(window=20).sum()

        # 资金净流入均值
        df['net_inflow_mean_5d'] = df['net_inflow'].rolling(window=5).mean()
        df['net_inflow_mean_10d'] = df['net_inflow'].rolling(window=10).mean()
        df['net_inflow_mean_20d'] = df['net_inflow'].rolling(window=20).mean()

        # 资金净流入标准差
        df['net_inflow_std_5d'] = df['net_inflow'].rolling(window=5).std()
        df['net_inflow_std_10d'] = df['net_inflow'].rolling(window=10).std()

        # 资金净流入Z-Score
        df['net_inflow_zscore'] = (df['net_inflow'] - df['net_inflow_mean_20d']) / df['net_inflow_std_10d']

        # 资金流向趋势 (5日 vs 10日)
        df['fund_trend'] = df['net_inflow_5d'] - df['net_inflow_10d']

        return df

    def _create_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建动量特征"""
        close = df['close'].values

        # RSI指标
        delta = pd.Series(close).diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))

        # 6日RSI
        gain_6 = (delta.where(delta > 0, 0)).rolling(window=6).mean()
        loss_6 = (-delta.where(delta < 0, 0)).rolling(window=6).mean()
        rs_6 = gain_6 / loss_6
        df['rsi_6'] = 100 - (100 / (1 + rs_6))

        # 12日RSI
        gain_12 = (delta.where(delta > 0, 0)).rolling(window=12).mean()
        loss_12 = (-delta.where(delta < 0, 0)).rolling(window=12).mean()
        rs_12 = gain_12 / loss_12
        df['rsi_12'] = 100 - (100 / (1 + rs_12))

        # MACD
        ema12 = pd.Series(close).ewm(span=12, adjust=False).mean()
        ema26 = pd.Series(close).ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # MACD金叉死叉
        macd_above_signal = (df['macd'] > df['macd_signal']).astype(int)
        macd_above_signal_prev = macd_above_signal.shift(1)
        df['macd_cross'] = (macd_above_signal - macd_above_signal_prev).fillna(0)

        # 布林带
        ma20 = df['close'].rolling(window=20).mean()
        std20 = df['close'].rolling(window=20).std()
        df['bb_upper'] = ma20 + 2 * std20
        df['bb_lower'] = ma20 - 2 * std20
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']) * 100
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / ma20 * 100

        # KDJ指标
        low_n = df['close'].rolling(window=9).min()
        high_n = df['close'].rolling(window=9).max()
        k = 100 * (df['close'] - low_n) / (high_n - low_n)
        df['kdj_k'] = k.rolling(window=3).mean()
        df['kdj_d'] = df['kdj_k'].rolling(window=3).mean()
        df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']

        # KDJ金叉死叉
        k_above_d = (df['kdj_k'] > df['kdj_d']).astype(int)
        k_above_d_prev = k_above_d.shift(1)
        df['kdj_cross'] = (k_above_d - k_above_d_prev).fillna(0)

        return df

    def _create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建成交量特征"""
        # 成交量
        if 'volume' not in df.columns:
            df['volume'] = 1000000

        # 成交量变化率
        df['volume_change_rate'] = df['volume'].pct_change() * 100

        # 5日均量
        df['volume_ma5'] = df['volume'].rolling(window=5).mean()

        # 10日均量
        df['volume_ma10'] = df['volume'].rolling(window=10).mean()

        # 20日均量
        df['volume_ma20'] = df['volume'].rolling(window=20).mean()

        # 成交量相对均量
        df['volume_ratio_5d'] = df['volume'] / df['volume_ma5']
        df['volume_ratio_10d'] = df['volume'] / df['volume_ma10']
        df['volume_ratio_20d'] = df['volume'] / df['volume_ma20']

        # 量价配合指标 (涨跌时成交量是否配合)
        # 上涨时放量=1, 下跌时缩量=1
        price_up = (df['change_pct'] > 0).astype(int)
        volume_up = (df['volume'] > df['volume_ma5']).astype(int)
        df['price_volume_match'] = (price_up == volume_up).astype(int)

        # 能量潮 (OBV)
        obv = [0]
        for i in range(1, len(df)):
            if df['change_pct'].iloc[i] > 0:
                obv.append(obv[-1] + df['volume'].iloc[i])
            elif df['change_pct'].iloc[i] < 0:
                obv.append(obv[-1] - df['volume'].iloc[i])
            else:
                obv.append(obv[-1])
        df['obv'] = obv

        # OBV变化率
        df['obv_change'] = pd.Series(obv).pct_change() * 100

        return df

    def _create_sequence_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建30天状态序列特征"""
        window = self.feature_window

        # 创建状态序列 (涨跌状态: 1=涨, -1=跌, 0=平)
        df['status'] = np.sign(df['change_pct'])

        # 创建30天状态序列
        for i in range(1, window + 1):
            df[f'status_lag_{i}'] = df['status'].shift(i)

        # 创建30天资金流向序列
        for i in range(1, window + 1):
            df[f'fund_lag_{i}'] = df['net_inflow'].shift(i)

        # 创建30天涨跌幅序列
        for i in range(1, window + 1):
            df[f'return_lag_{i}'] = df['change_pct'].shift(i)

        # 创建30天成交量序列
        for i in range(1, window + 1):
            df[f'volume_lag_{i}'] = df['volume'].shift(i)

        # 过去30天的累计涨幅
        df['cumulative_return_30d'] = df['close'].pct_change(periods=30) * 100

        # 过去30天的平均涨幅
        df['mean_return_30d'] = df['change_pct'].rolling(window=30).mean()

        # 过去30天上涨天数
        df['up_days_30d'] = (df['status'] > 0).rolling(window=30).sum()

        # 过去30天下跌天数
        df['down_days_30d'] = (df['status'] < 0).rolling(window=30).sum()

        # 过去30天资金净流入总和
        df['total_fund_30d'] = df['net_inflow'].rolling(window=30).sum()

        # 状态转移矩阵特征
        # 连续上涨天数
        df['consecutive_up'] = self._count_consecutive(df['status'], 1)
        # 连续下跌天数
        df['consecutive_down'] = self._count_consecutive(df['status'], -1)

        return df

    def _count_consecutive(self, series: pd.Series, value: int) -> pd.Series:
        """计算连续相同值的次数"""
        result = pd.Series(0, index=series.index)
        count = 0

        for i in range(len(series)):
            if series.iloc[i] == value:
                count += 1
            else:
                count = 0
            result.iloc[i] = count

        return result

    def create_target(self, df: pd.DataFrame, target_days: int = 1) -> pd.DataFrame:
        """
        创建预测目标

        Args:
            df: 特征数据
            target_days: 预测目标天数

        Returns:
            DataFrame: 包含目标的DataFrame
        """
        df = df.copy()

        # 第二天是否继续上涨 (二分类目标)
        df['target_up'] = (df['change_pct'].shift(-target_days) > 0).astype(int)

        # 第二天涨幅 (回归目标)
        df['target_return'] = df['change_pct'].shift(-target_days)

        # 连续上涨预测 (n天后是否累计上涨)
        for n in [2, 3, 5]:
            df[f'target_up_{n}d'] = (df['change_pct'].shift(-1) +
                                      df['change_pct'].shift(-2) +
                                      df['change_pct'].shift(-n)) > 0
            df[f'target_up_{n}d'] = df[f'target_up_{n}d'].astype(int)

        return df

    def prepare_training_dataset(self, df: pd.DataFrame,
                                  drop_na: bool = True) -> tuple:
        """
        准备训练数据集

        Args:
            df: 完整数据
            drop_na: 是否删除NaN

        Returns:
            tuple: (特征列名列表, X, y_up, y_return)
        """
        # 排除非特征列
        exclude_cols = ['date', 'sector_name', 'target_up', 'target_return',
                       'target_up_2d', 'target_up_3d', 'target_up_5d',
                       'status', 'open', 'high', 'low', 'pre_close']

        # 选择数值列作为特征
        feature_cols = [col for col in df.columns
                       if col not in exclude_cols and df[col].dtype in ['float64', 'int64', 'int32']]

        # 删除NaN
        if drop_na:
            df_clean = df[feature_cols + ['target_up', 'target_return']].dropna()
        else:
            df_clean = df[feature_cols + ['target_up', 'target_return']]

        X = df_clean[feature_cols]
        y_up = df_clean['target_up']
        y_return = df_clean['target_return']

        return feature_cols, X, y_up, y_return

    def get_feature_importance(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """
        获取特征重要性排名

        Args:
            df: 数据
            feature_cols: 特征列名

        Returns:
            DataFrame: 特征重要性
        """
        # 计算每个特征与目标的相关性
        correlations = []

        for col in feature_cols:
            if col in df.columns and 'target_up' in df.columns:
                corr = df[col].corr(df['target_up'])
                correlations.append({
                    'feature': col,
                    'correlation': corr
                })

        return pd.DataFrame(correlations).sort_values('correlation', ascending=False)


# 测试代码
if __name__ == "__main__":
    # 导入数据获取器
    from data_fetcher import DataFetcher

    # 创建数据
    fetcher = DataFetcher()
    df = fetcher.get_sector_fund_flow("半导体", 100)

    # 创建特征
    engineer = FeatureEngineer()
    df_features = engineer.create_features(df)
    df_features = engineer.create_target(df_features)

    print("特征列数:", len(df_features.columns))
    print("\n特征列表:")
    print(df_features.columns.tolist()[:30])

    # 准备训练数据
    feature_cols, X, y_up, y_return = engineer.prepare_training_dataset(df_features)
    print(f"\n训练数据形状: X={X.shape}, y_up={y_up.shape}")
    print(f"上涨样本比例: {y_up.mean():.2%}")
