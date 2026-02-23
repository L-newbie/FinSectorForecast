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
        
        # 从配置中读取技术指标参数，使用默认值作为后备
        ti_config = self.config.get('technical_indicators', {})
        
        # RSI参数
        rsi_config = ti_config.get('rsi', {})
        self.rsi_periods = rsi_config.get('periods', [6, 12, 14])  # RSI周期列表
        
        # MACD参数
        macd_config = ti_config.get('macd', {})
        self.macd_fast_period = macd_config.get('fast_period', 12)  # 快线EMA周期
        self.macd_slow_period = macd_config.get('slow_period', 26)  # 慢线EMA周期
        self.macd_signal_period = macd_config.get('signal_period', 9)  # 信号线EMA周期
        
        # 布林带参数
        bollinger_config = ti_config.get('bollinger', {})
        self.bollinger_period = bollinger_config.get('period', 20)  # 布林带中轨周期
        self.bollinger_std_dev = bollinger_config.get('std_dev', 2)  # 标准差倍数
        
        # KDJ参数
        kdj_config = ti_config.get('kdj', {})
        self.kdj_n_period = kdj_config.get('n_period', 9)  # KDJ的N周期
        self.kdj_m1_period = kdj_config.get('m1_period', 3)  # K值的M1平滑周期
        self.kdj_m2_period = kdj_config.get('m2_period', 3)  # D值的M2平滑周期
        
        # 均线参数
        ma_config = ti_config.get('moving_average', {})
        self.ma_short_period = ma_config.get('short_period', 5)  # 短期均线周期
        self.ma_medium_period = ma_config.get('medium_period', 10)  # 中期均线周期
        self.ma_long_period = ma_config.get('long_period', 20)  # 长期均线周期
        
        # 滚动窗口参数
        rolling_config = ti_config.get('rolling_windows', {})
        self.rolling_short = rolling_config.get('short', 5)  # 短期滚动窗口
        self.rolling_medium = rolling_config.get('medium', 10)  # 中期滚动窗口
        self.rolling_long = rolling_config.get('long', 20)  # 长期滚动窗口
        self.rolling_extended = rolling_config.get('extended', 30)  # 扩展滚动窗口

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
        """创建价格特征 - 使用配置中的参数"""
        # 确保有必要的列
        if 'close' not in df.columns:
            return df

        # 获取收盘价
        close = df['close'].values

        # 计算涨跌幅
        if 'change_pct' not in df.columns:
            df['change_pct'] = pd.Series(close).pct_change() * 100

        # 使用配置中的滚动窗口计算涨跌幅
        df[f'return_{self.rolling_short}d'] = df['change_pct'].rolling(window=self.rolling_short).sum()
        df[f'return_{self.rolling_medium}d'] = df['change_pct'].rolling(window=self.rolling_medium).sum()
        df[f'return_{self.rolling_long}d'] = df['change_pct'].rolling(window=self.rolling_long).sum()

        # 使用配置中的均线周期
        df[f'ma{self.ma_long_period}'] = df['close'].rolling(window=self.ma_long_period).mean()

        # 收盘价相对长期均线位置 (0-100)
        df['ma_long_position'] = (df['close'] - df[f'ma{self.ma_long_period}']) / df[f'ma{self.ma_long_period}'] * 100

        # 短期均线
        df[f'ma{self.ma_short_period}'] = df['close'].rolling(window=self.ma_short_period).mean()

        # 短期均线与长期均线金叉死叉 (1=金叉, -1=死叉, 0=无)
        ma_short_above_long = (df[f'ma{self.ma_short_period}'] > df[f'ma{self.ma_long_period}']).astype(int)
        ma_short_above_long_prev = ma_short_above_long.shift(1)
        df['ma_short_cross_long'] = (ma_short_above_long - ma_short_above_long_prev).fillna(0)

        # 中期均线
        df[f'ma{self.ma_medium_period}'] = df['close'].rolling(window=self.ma_medium_period).mean()

        # 中期均线与长期均线关系
        df['ma_medium_above_long'] = (df[f'ma{self.ma_medium_period}'] > df[f'ma{self.ma_long_period}']).astype(int)

        # 价格波动率 - 使用配置中的滚动窗口
        df[f'volatility_{self.rolling_short}d'] = df['change_pct'].rolling(window=self.rolling_short).std()
        df[f'volatility_{self.rolling_medium}d'] = df['change_pct'].rolling(window=self.rolling_medium).std()
        df[f'volatility_{self.rolling_long}d'] = df['change_pct'].rolling(window=self.rolling_long).std()

        # 最高价/最低价相对位置
        if 'high' in df.columns and 'low' in df.columns:
            df['high_low_ratio'] = (df['high'] - df['low']) / df['close'] * 100

        return df

    def _create_fund_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建资金特征 - 使用配置中的参数"""
        # 资金净流入
        if 'net_inflow' not in df.columns:
            # 从涨跌幅模拟资金流向
            df['net_inflow'] = df['change_pct'] * 1000000

        # 资金净流入率
        if 'net_inflow_rate' not in df.columns and 'turnover' in df.columns:
            df['net_inflow_rate'] = df['net_inflow'] / df['turnover'] * 100

        # 使用配置中的滚动窗口计算累计资金流向
        df[f'net_inflow_{self.rolling_short}d'] = df['net_inflow'].rolling(window=self.rolling_short).sum()
        df[f'net_inflow_{self.rolling_medium}d'] = df['net_inflow'].rolling(window=self.rolling_medium).sum()
        df[f'net_inflow_{self.rolling_long}d'] = df['net_inflow'].rolling(window=self.rolling_long).sum()

        # 资金净流入均值 - 使用配置中的滚动窗口
        df[f'net_inflow_mean_{self.rolling_short}d'] = df['net_inflow'].rolling(window=self.rolling_short).mean()
        df[f'net_inflow_mean_{self.rolling_medium}d'] = df['net_inflow'].rolling(window=self.rolling_medium).mean()
        df[f'net_inflow_mean_{self.rolling_long}d'] = df['net_inflow'].rolling(window=self.rolling_long).mean()

        # 资金净流入标准差 - 使用配置中的滚动窗口
        df[f'net_inflow_std_{self.rolling_short}d'] = df['net_inflow'].rolling(window=self.rolling_short).std()
        df[f'net_inflow_std_{self.rolling_medium}d'] = df['net_inflow'].rolling(window=self.rolling_medium).std()

        # 资金净流入Z-Score
        df['net_inflow_zscore'] = (df['net_inflow'] - df[f'net_inflow_mean_{self.rolling_long}d']) / df[f'net_inflow_std_{self.rolling_medium}d']

        # 资金流向趋势 (短期 vs 中期)
        df['fund_trend'] = df[f'net_inflow_{self.rolling_short}d'] - df[f'net_inflow_{self.rolling_medium}d']

        return df

    def _create_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建动量特征 - 使用配置中的参数"""
        close = df['close'].values

        # RSI指标 - 使用配置中的周期列表
        delta = pd.Series(close).diff()
        for period in self.rsi_periods:
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        # MACD - 使用配置中的参数
        ema_fast = pd.Series(close).ewm(span=self.macd_fast_period, adjust=False).mean()
        ema_slow = pd.Series(close).ewm(span=self.macd_slow_period, adjust=False).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=self.macd_signal_period, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # MACD金叉死叉
        macd_above_signal = (df['macd'] > df['macd_signal']).astype(int)
        macd_above_signal_prev = macd_above_signal.shift(1)
        df['macd_cross'] = (macd_above_signal - macd_above_signal_prev).fillna(0)

        # 布林带 - 使用配置中的参数
        ma_period = df['close'].rolling(window=self.bollinger_period).mean()
        std_period = df['close'].rolling(window=self.bollinger_period).std()
        df['bb_upper'] = ma_period + self.bollinger_std_dev * std_period
        df['bb_lower'] = ma_period - self.bollinger_std_dev * std_period
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']) * 100
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / ma_period * 100

        # KDJ指标 - 使用配置中的参数
        low_n = df['close'].rolling(window=self.kdj_n_period).min()
        high_n = df['close'].rolling(window=self.kdj_n_period).max()
        k = 100 * (df['close'] - low_n) / (high_n - low_n)
        df['kdj_k'] = k.rolling(window=self.kdj_m1_period).mean()
        df['kdj_d'] = df['kdj_k'].rolling(window=self.kdj_m2_period).mean()
        df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']

        # KDJ金叉死叉
        k_above_d = (df['kdj_k'] > df['kdj_d']).astype(int)
        k_above_d_prev = k_above_d.shift(1)
        df['kdj_cross'] = (k_above_d - k_above_d_prev).fillna(0)

        return df

    def _create_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """创建成交量特征 - 使用配置中的参数"""
        # 成交量
        if 'volume' not in df.columns:
            df['volume'] = 1000000

        # 成交量变化率
        df['volume_change_rate'] = df['volume'].pct_change() * 100

        # 使用配置中的均线周期计算均量
        df[f'volume_ma{self.ma_short_period}'] = df['volume'].rolling(window=self.ma_short_period).mean()
        df[f'volume_ma{self.ma_medium_period}'] = df['volume'].rolling(window=self.ma_medium_period).mean()
        df[f'volume_ma{self.ma_long_period}'] = df['volume'].rolling(window=self.ma_long_period).mean()

        # 成交量相对均量 - 使用配置中的周期
        df[f'volume_ratio_{self.ma_short_period}d'] = df['volume'] / df[f'volume_ma{self.ma_short_period}']
        df[f'volume_ratio_{self.ma_medium_period}d'] = df['volume'] / df[f'volume_ma{self.ma_medium_period}']
        df[f'volume_ratio_{self.ma_long_period}d'] = df['volume'] / df[f'volume_ma{self.ma_long_period}']

        # 量价配合指标 (涨跌时成交量是否配合)
        # 上涨时放量=1, 下跌时缩量=1
        price_up = (df['change_pct'] > 0).astype(int)
        volume_up = (df['volume'] > df[f'volume_ma{self.ma_short_period}']).astype(int)
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
        """创建状态序列特征 - 使用配置中的参数"""
        window = self.feature_window

        # 创建状态序列 (涨跌状态: 1=涨, -1=跌, 0=平)
        df['status'] = np.sign(df['change_pct'])

        # 创建状态序列（使用配置中的feature_window）
        for i in range(1, window + 1):
            df[f'status_lag_{i}'] = df['status'].shift(i)

        # 创建资金流向序列
        for i in range(1, window + 1):
            df[f'fund_lag_{i}'] = df['net_inflow'].shift(i)

        # 创建涨跌幅序列
        for i in range(1, window + 1):
            df[f'return_lag_{i}'] = df['change_pct'].shift(i)

        # 创建成交量序列
        for i in range(1, window + 1):
            df[f'volume_lag_{i}'] = df['volume'].shift(i)

        # 使用配置中的扩展滚动窗口计算累计涨幅
        df[f'cumulative_return_{self.rolling_extended}d'] = df['close'].pct_change(periods=self.rolling_extended) * 100

        # 使用配置中的扩展滚动窗口计算平均涨幅
        df[f'mean_return_{self.rolling_extended}d'] = df['change_pct'].rolling(window=self.rolling_extended).mean()

        # 使用配置中的扩展滚动窗口计算上涨天数
        df[f'up_days_{self.rolling_extended}d'] = (df['status'] > 0).rolling(window=self.rolling_extended).sum()

        # 使用配置中的扩展滚动窗口计算下跌天数
        df[f'down_days_{self.rolling_extended}d'] = (df['status'] < 0).rolling(window=self.rolling_extended).sum()

        # 使用配置中的扩展滚动窗口计算资金净流入总和
        df[f'total_fund_{self.rolling_extended}d'] = df['net_inflow'].rolling(window=self.rolling_extended).sum()

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
