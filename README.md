# 📈 板块次日涨跌预测系统

基于历史30天的资金流向、涨跌幅等技术指标，预测板块第二天继续上涨的概率及涨幅。
在线预览: https://L-newbie.github.io/FinSectorForecast/

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

### 后续提交

```bash
# 添加修改的文件
git add .

# 提交修改
git commit -m "描述你的修改"

# 推送到 GitHub
git push
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

---

## 🌐 部署到 GitHub Pages

### 方法一：使用 docs 目录

1. 在项目根目录创建 `docs` 文件夹
2. 将静态文件放入 `docs` 目录
3. 在 GitHub 仓库设置中启用 GitHub Pages，Source 选择 `docs` 文件夹

```bash
# 示例：创建docs目录并添加静态文件
mkdir docs
# 将你的HTML/CSS/JS文件复制到docs目录
git add docs/
git commit -m "Add static files for GitHub Pages"
git push
```

### 方法二：使用 gh-pages 分支

```bash
# 安装 gh-pages 工具（可选）
npm install -g gh-pages

# 或者使用 Python
pip install ghp-import

# 部署 docs 目录到 gh-pages 分支
ghp-import -p docs/

# 或者使用 git subtree
git subtree push --prefix docs origin gh-pages
```

### 方法三：使用 GitHub Actions 自动部署

创建 `.github/workflows/deploy.yml`：

```yaml
name: Deploy to GitHub Pages

on:
  push:
    branches:
      - main

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.8'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Build static files
        run: |
          # 这里添加生成静态文件的命令
          # 例如：python build.py
          mkdir -p docs
      
      - name: Deploy to GitHub Pages
        uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: ./docs
```

### GitHub Pages 访问

部署完成后，通过以下地址访问：

```
https://你的用户名.github.io/FinSectorForecast/
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
