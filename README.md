# 📈 A股金融板块预测系统

基于历史30天的资金流向、涨跌幅等技术指标，预测板块第二天继续上涨的概率及涨幅。


[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)](https://flask.palletsprojects.com/)
[![LightGBM](https://img.shields.io/badge/LightGBM-optimized-orange.svg)](https://lightgbm.readthedocs.io/)

## ✨ 功能特性

| 功能 | 说明 |
|------|------|
| 📊 数据获取 | 从东方财富获取板块资金流向数据 |
| 🔧 特征工程 | 30+ 技术指标特征 |
| 🤖 机器学习 | LightGBM/XGBoost 分类+回归模型 |
| 💾 高性能缓存 | LRU淘汰、TTL过期、线程安全 |
| 🎨 Web界面 | 预测、分析、训练三大模块 |

## 🚀 快速开始

### 安装

```bash
pip install -r requirements.txt
```

### 运行

```bash
# Web应用
python app.py

# 命令行预测
python main.py --mode predict --sector 半导体
```

### Python API

```python
from src.predictor import SectorPredictor

predictor = SectorPredictor("半导体")
result = predictor.predict()

print(f"上涨概率: {result['probability']:.2%}")
print(f"预测涨幅: {result['predicted_return']:.2f}%")
```

## 📁 项目结构

```
FinSectorForecast/
├── config/
│   └── config.yaml           # 配置文件
├── src/
│   ├── data_fetcher.py       # 数据获取
│   ├── feature_engineering.py # 特征工程
│   ├── model_training.py     # 模型训练
│   ├── predictor.py          # 预测模块
│   ├── memory_cache.py       # 内存缓存
│   ├── cache_manager.py      # 缓存管理
│   └── section_cache.py      # 页面缓存
├── templates/                 # Web模板
├── app.py                    # Web入口
├── main.py                   # CLI入口
└── requirements.txt
```

## 🗂️ 核心模块

- **[`src/predictor.py`](src/predictor.py)** - 预测核心逻辑
- **[`src/model_training.py`](src/model_training.py)** - 模型训练
- **[`src/feature_engineering.py`](src/feature_engineering.py)** - 特征提取
- **[`src/memory_cache.py`](src/memory_cache.py)** - 缓存系统

## ⚙️ 配置

修改 `config/config.yaml`:

```yaml
data:
  history_days: 365
  feature_window: 30
  sectors: ["半导体", "新能源", "医药"]

model:
  classifier:
    name: lightgbm
    params:
      n_estimators: 100
      max_depth: 6

predict:
  probability_threshold: 0.6
```

## 📤 提交到 GitHub

### 初始化 Git 仓库（首次）

```bash
# 初始化本地仓库
git init

# 添加所有文件
git add .

# 提交代码
git commit -m "Initial commit"

# 添加远程仓库（替换为你的仓库地址）
git remote add origin https://github.com/你的用户名/FinSectorForecast.git

# 推送到 GitHub
git push -u origin main
```

### 分支管理

```bash
# 创建新分支
git checkout -b feature/你的功能名

# 切换分支
git checkout main

# 合并分支
git merge feature/你的功能名
```

## ⚠️ 风险提示

本系统仅供学习研究使用，不构成投资建议。股市有风险，投资需谨慎。

---

## 📦 依赖

- flask >= 2.3.0
- pandas >= 1.5.0
- numpy >= 1.23.0
- scikit-learn >= 1.2.0
- lightgbm >= 3.3.0
- matplotlib >= 3.6.0
- pyyaml >= 6.0

---

*MIT License*
