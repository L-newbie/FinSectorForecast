# -*- coding: utf-8 -*-
"""
模型训练模块

功能：
1. LightGBM分类模型训练 (预测上涨概率)
2. LightGBM回归模型训练 (预测涨幅)
3. 模型评估与调参
4. 模型保存与加载
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')

# 尝试导入机器学习库
try:
    from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                  f1_score, roc_auc_score, mean_squared_error,
                                  mean_absolute_error, r2_score)
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("警告: scikit-learn未安装")

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
    print("警告: lightgbm未安装")

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("警告: xgboost未安装")


class ModelTrainer:
    """模型训练类"""

    def __init__(self, config: Optional[Dict] = None):
        """
        初始化模型训练器

        Args:
            config: 配置字典
        """
        self.config = config or {}
        self.model_config = self.config.get('model', {})

        # 模型参数
        self.classifier_params = self.model_config.get('classifier', {}).get('params', {})
        self.regressor_params = self.model_config.get('regressor', {}).get('params', {})

        # 训练参数
        training_config = self.model_config.get('training', {})
        self.test_size = training_config.get('test_size', 0.2)
        self.early_stopping_rounds = training_config.get('early_stopping_rounds', 10)

        # 模型
        self.classifier = None
        self.regressor = None

        # 特征重要性
        self.feature_importance = None

    def _get_classifier(self) -> object:
        """获取分类模型"""
        if LGBM_AVAILABLE:
            # 添加silent参数抑制警告
            params = {**self.classifier_params, 'silent': True, 'verbose': -1}
            return lgb.LGBMClassifier(**params)
        elif XGB_AVAILABLE:
            return xgb.XGBClassifier(**self.classifier_params)
        elif SKLEARN_AVAILABLE:
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42
            )
        else:
            raise ImportError("没有可用的机器学习库")

    def _get_regressor(self) -> object:
        """获取回归模型"""
        if LGBM_AVAILABLE:
            # 添加silent参数抑制警告
            params = {**self.regressor_params, 'silent': True, 'verbose': -1}
            return lgb.LGBMRegressor(**params)
        elif XGB_AVAILABLE:
            return xgb.XGBRegressor(**self.regressor_params)
        elif SKLEARN_AVAILABLE:
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=6,
                random_state=42
            )
        else:
            raise ImportError("没有可用的机器学习库")

    def train_classifier(self, X: pd.DataFrame, y: pd.Series,
                         feature_names: Optional[List[str]] = None) -> Dict:
        """
        训练分类模型 (预测上涨概率)

        Args:
            X: 特征数据
            y: 目标变量 (0或1)
            feature_names: 特征名称

        Returns:
            Dict: 训练结果
        """
        if len(X) == 0:
            return {'error': '数据为空'}

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42, shuffle=False
        )

        # 创建模型
        self.classifier = self._get_classifier()

        # 训练
        if LGBM_AVAILABLE:
            self.classifier.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)]
            )
        else:
            self.classifier.fit(X_train, y_train)

        # 预测
        y_pred = self.classifier.predict(X_test)
        y_pred_proba = self.classifier.predict_proba(X_test)[:, 1]

        # 评估
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0.5
        }

        # 特征重要性
        if hasattr(self.classifier, 'feature_importances_'):
            importance = self.classifier.feature_importances_
            if feature_names is None:
                feature_names = X.columns.tolist()
            self.feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)

        return {
            'metrics': metrics,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'n_features': X.shape[1]
        }

    def train_regressor(self, X: pd.DataFrame, y: pd.Series,
                        feature_names: Optional[List[str]] = None) -> Dict:
        """
        训练回归模型 (预测涨幅)

        Args:
            X: 特征数据
            y: 目标变量 (涨幅)
            feature_names: 特征名称

        Returns:
            Dict: 训练结果
        """
        if len(X) == 0:
            return {'error': '数据为空'}

        # 去除NaN和Inf
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        y = y.replace([np.inf, -np.inf], np.nan).fillna(0)

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42, shuffle=False
        )

        # 创建模型
        self.regressor = self._get_regressor()

        # 训练
        if LGBM_AVAILABLE:
            self.regressor.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)]
            )
        else:
            self.regressor.fit(X_train, y_train)

        # 预测
        y_pred = self.regressor.predict(X_test)

        # 评估
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }

        return {
            'metrics': metrics,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'n_features': X.shape[1]
        }

    def train_all(self, X: pd.DataFrame, y_up: pd.Series, y_return: pd.Series,
                  feature_names: Optional[List[str]] = None) -> Dict:
        """
        训练所有模型

        Args:
            X: 特征数据
            y_up: 上涨目标 (二分类)
            y_return: 涨幅目标 (回归)
            feature_names: 特征名称

        Returns:
            Dict: 训练结果
        """
        results = {}

        # 训练分类模型
        print(f"    [-] 训练分类模型 (预测上涨概率)...")
        results['classifier'] = self.train_classifier(X, y_up, feature_names)
        print(f"    [√] 分类模型完成 - 准确率: {results['classifier']['metrics']['accuracy']:.2%}, AUC: {results['classifier']['metrics']['auc']:.4f}")

        # 训练回归模型
        print(f"    [-] 训练回归模型 (预测涨幅)...")
        results['regressor'] = self.train_regressor(X, y_return, feature_names)
        print(f"    [√] 回归模型完成 - RMSE: {results['regressor']['metrics']['rmse']:.4f}, MAE: {results['regressor']['metrics']['mae']:.4f}")

        return results

    def predict(self, X: pd.DataFrame) -> Dict:
        """
        使用训练好的模型进行预测

        Args:
            X: 特征数据

        Returns:
            Dict: 预测结果
        """
        if self.classifier is None or self.regressor is None:
            return {'error': '模型未训练'}

        # 处理输入数据
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

        # 确保列顺序一致
        if hasattr(self.classifier, 'feature_names_in_'):
            X = X[self.classifier.feature_names_in_]

        # 预测
        probability = self.classifier.predict_proba(X)[:, 1]
        predicted_return = self.regressor.predict(X)

        return {
            'probability': probability,
            'predicted_return': predicted_return,
            'prediction_up': (probability > 0.5).astype(int)
        }

    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict:
        """
        交叉验证

        Args:
            X: 特征数据
            y: 目标变量
            cv: 折数

        Returns:
            Dict: 交叉验证结果
        """
        # 使用时间序列分割
        tscv = TimeSeriesSplit(n_splits=cv)

        # 训练分类模型
        classifier = self._get_classifier()
        scores = cross_val_score(classifier, X, y, cv=tscv, scoring='roc_auc')

        return {
            'cv_scores': scores.tolist(),
            'mean_score': scores.mean(),
            'std_score': scores.std()
        }

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        获取特征重要性

        Args:
            top_n: 返回前N个重要特征

        Returns:
            DataFrame: 特征重要性
        """
        if self.feature_importance is None:
            return pd.DataFrame()

        return self.feature_importance.head(top_n)

    def save_models(self, path: str):
        """保存模型"""
        import pickle

        models = {
            'classifier': self.classifier,
            'regressor': self.regressor,
            'feature_importance': self.feature_importance
        }

        with open(path, 'wb') as f:
            pickle.dump(models, f)

        print(f"模型已保存到: {path}")

    def load_models(self, path: str):
        """加载模型"""
        import pickle

        with open(path, 'rb') as f:
            models = pickle.load(f)

        self.classifier = models['classifier']
        self.regressor = models['regressor']
        self.feature_importance = models.get('feature_importance')

        print(f"模型已从: {path} 加载")

    def get_training_report(self, results: Dict) -> str:
        """
        生成训练报告

        Args:
            results: 训练结果

        Returns:
            str: 报告文本
        """
        report = []
        report.append("=" * 50)
        report.append("板块涨跌预测模型 - 训练报告")
        report.append("=" * 50)

        # 分类模型结果
        if 'classifier' in results:
            report.append("\n【分类模型 - 上涨概率预测】")
            metrics = results['classifier']['metrics']
            report.append(f"  准确率 (Accuracy):  {metrics['accuracy']:.4f}")
            report.append(f"  精确率 (Precision): {metrics['precision']:.4f}")
            report.append(f"  召回率 (Recall):    {metrics['recall']:.4f}")
            report.append(f"  F1分数:             {metrics['f1']:.4f}")
            report.append(f"  AUC:                {metrics['auc']:.4f}")
            report.append(f"  训练样本数:         {results['classifier']['train_size']}")
            report.append(f"  测试样本数:         {results['classifier']['test_size']}")

        # 回归模型结果
        if 'regressor' in results:
            report.append("\n【回归模型 - 涨幅预测】")
            metrics = results['regressor']['metrics']
            report.append(f"  MSE:                {metrics['mse']:.4f}")
            report.append(f"  RMSE:               {metrics['rmse']:.4f}")
            report.append(f"  MAE:                {metrics['mae']:.4f}")
            report.append(f"  R²:                 {metrics['r2']:.4f}")

        report.append("\n" + "=" * 50)

        return "\n".join(report)


# 测试代码
if __name__ == "__main__":
    # 创建模拟数据
    np.random.seed(42)
    n_samples = 1000
    n_features = 50

    X = pd.DataFrame(np.random.randn(n_samples, n_features))
    y_up = pd.Series(np.random.randint(0, 2, n_samples))
    y_return = pd.Series(np.random.randn(n_samples) * 2)

    # 训练模型
    trainer = ModelTrainer()
    results = trainer.train_all(X, y_up, y_return)

    # 打印报告
    print(trainer.get_training_report(results))

    # 预测
    X_new = pd.DataFrame(np.random.randn(5, n_features))
    prediction = trainer.predict(X_new)
    print("\n预测结果:")
    print(f"  上涨概率: {prediction['probability'][:5]}")
    print(f"  预测涨幅: {prediction['predicted_return'][:5]}")
