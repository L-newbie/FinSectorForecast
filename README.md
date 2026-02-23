# 📈 FinSectorForecast

**A股金融板块智能预测系统**

基于机器学习的A股板块次日涨跌概率预测系统

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3+-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![LightGBM](https://img.shields.io/badge/LightGBM-3.3+-F7931E?style=flat-square&logo=lightgbm&logoColor=white)](https://lightgbm.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

[功能特性](#-功能特性) • [快速开始](#-快速开始) • [使用指南](#-使用指南) • [API文档](#-api文档) • [配置说明](#️-配置说明)

---

## 📖 项目简介

FinSectorForecast 是一个基于历史资金流向、涨跌幅等技术指标的A股板块预测系统。系统采用 LightGBM/XGBoost 机器学习模型，通过分类模型预测板块次日涨跌概率，通过回归模型预测预期涨幅，为投资决策提供数据支持。

### 核心能力

- 🎯 **双模型预测**：分类模型预测涨跌概率 + 回归模型预测涨幅空间
- 📊 **30+ 技术指标**：涵盖价格、资金、动量、成交量等多维度特征
- 🔄 **实时数据获取**：通过 akshare 接口获取东方财富板块数据
- 💾 **多级缓存系统**：LRU淘汰 + TTL过期 + 线程安全的高性能缓存
- 🖥️ **双模式运行**：Web可视化界面 + 命令行批量处理
- ⚡ **后台任务管理**：异步任务执行、定时刷新、智能调度
- 📱 **响应式Web界面**：现代SPA架构，流畅的用户体验

---

## ✨ 功能特性

### 📊 数据获取模块

| 功能 | 说明 |
|:-----|:-----|
| 板块资金流向 | 从东方财富获取行业/概念板块资金流向数据 |
| 历史行情数据 | 获取板块历史行情，支持自定义天数 |
| 数据清洗 | 自动处理缺失值、异常值，确保数据质量 |
| 板块列表管理 | 智能缓存板块列表，支持后台定时刷新 |

### 🔧 特征工程模块

| 特征类型 | 包含指标 |
|:---------|:---------|
| 价格特征 | 涨跌幅、均线系统、价格波动率、布林带位置 |
| 资金特征 | 主力净流入、超大单/大单/中单/小单流向、资金净流入比率 |
| 动量指标 | RSI相对强弱指数(6/12/14日)、MACD指标、KDJ指标 |
| 成交量特征 | 成交量均线、量价关系、换手率 |
| 序列特征 | 30天状态序列、趋势编码、动量变化序列 |

### 🤖 机器学习模块

- **分类模型**：预测次日上涨概率（0-100%）
- **回归模型**：预测次日预期涨幅
- **支持算法**：LightGBM、XGBoost、RandomForest
- **模型评估**：准确率、精确率、召回率、F1-Score、AUC

### 💾 缓存系统

- **LRU淘汰策略**：自动淘汰最近最少使用的缓存
- **TTL过期机制**：支持不同类型数据设置独立过期时间
- **线程安全**：多线程环境下安全运行
- **缓存监控**：实时统计命中率、容量使用情况
- **多级缓存架构**：
  - 内存缓存 (MemoryCache)
  - 缓存管理器 (CacheManager)
  - 页面片段缓存 (SectionCache)

### 🎨 Web界面

| 页面 | 功能 |
|:-----|:-----|
| 仪表盘 | 全板块预测排名、市场概览、最佳投资机会 |
| 预测分析 | 单板块深度预测、特征详情、历史回测 |
| 模型训练 | 自定义参数训练、模型评估、特征重要性分析 |
| 深度分析 | 多维度数据可视化、技术指标图表 |

---

## 🚀 快速开始

### 环境要求

- Python 3.8+
- pip 包管理器

### 安装步骤

```bash
# 1. 克隆项目
git clone https://github.com/L-newbie/FinSectorForecast.git
cd FinSectorForecast

# 2. 创建虚拟环境（推荐）
python -m venv venv

# Windows 激活
venv\Scripts\activate

# Linux/Mac 激活
source venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 安装数据源库（获取真实数据必需）
pip install akshare
```

### 启动方式

#### 方式一：Web应用（推荐）

```bash
python app.py
```

访问 http://127.0.0.1:5000 即可使用Web界面。

#### 方式二：命令行模式

```bash
# 预测单个板块
python main.py --mode predict --sector 半导体

# 训练单个板块模型
python main.py --mode train --sector 半导体

# 预测所有板块并排名
python main.py --mode predict-all --top 10

# 完整分析单个板块
python main.py --mode analyze --sector 半导体
```

---

## 📚 使用指南

### Python API 调用

```python
from src.predictor import SectorPredictor, MultiSectorPredictor

# 单板块预测
predictor = SectorPredictor("半导体")
result = predictor.predict()

print(f"板块名称: {result['sector_name']}")
print(f"上涨概率: {result['probability']:.2%}")
print(f"预测涨幅: {result['predicted_return']:.2f}%")
print(f"交易信号: {result['signal']}")
print(f"投资建议: {result['recommendation']}")

# 多板块预测
multi_predictor = MultiSectorPredictor()
multi_predictor.add_sector("半导体")
multi_predictor.add_sector("新能源")
multi_predictor.add_sector("医药")

predictions = multi_predictor.predict_all()
opportunities = multi_predictor.get_top_opportunities(5)
```

### 预测结果说明

| 字段 | 说明 |
|:-----|:-----|
| `probability` | 次日上涨概率（0-1） |
| `predicted_return` | 预测次日涨跌幅（%） |
| `confidence` | 预测置信度（高/中/低） |
| `signal` | 交易信号（强烈买入/买入/持有/卖出） |
| `recommendation` | 详细投资建议 |

### 交易信号判定规则

| 信号 | 条件 |
|:-----|:-----|
| 🔥 强烈买入 | 上涨概率 ≥ 80% 且 预测涨幅 > 2% |
| ✅ 买入 | 上涨概率 ≥ 60% 且 预测涨幅 > 0.5% |
| ⏸️ 持有 | 其他情况 |
| ❌ 卖出 | 上涨概率 < 40% 且 预测涨幅 < -0.5% |

---

## 🔌 API文档

### REST API 接口

#### 获取板块列表
```http
GET /api/sectors
```

**响应示例：**
```json
{
  "success": true,
  "data": ["半导体", "新能源", "医药", "..."],
  "count": 100,
  "cache_status": "cached"
}
```

#### 单板块预测
```http
POST /api/predict/single
Content-Type: application/json

{
  "sector": "半导体",
  "date": "2024-01-15"
}
```

#### 多板块预测
```http
POST /api/predict/multi
Content-Type: application/json

{
  "sectors": ["半导体", "新能源", "医药"],
  "force_refresh": false
}
```

#### 全板块预测排名
```http
POST /api/predict/all
Content-Type: application/json

{
  "force_refresh": false
}
```

#### 获取配置信息
```http
GET /api/config
```

#### 后台任务状态
```http
GET /api/tasks/status
```

---

## ⚙️ 配置说明

配置文件位于 [`config/config.yaml`](config/config.yaml)，支持以下配置项：

### 数据配置

```yaml
data:
  source: akshare           # 数据源
  history_days: 365         # 历史数据天数
  feature_window: 30        # 特征窗口天数
  timeout: 30               # 请求超时（秒）
  max_retries: 3            # 最大重试次数
  retry_delay: 2            # 重试延迟（秒）
```

### 模型配置

```yaml
model:
  # 分类模型（预测上涨概率）
  classifier:
    name: lightgbm
    params:
      n_estimators: 100
      max_depth: 6
      learning_rate: 0.1
      num_leaves: 31
      min_child_samples: 20

  # 回归模型（预测涨幅）
  regressor:
    name: lightgbm
    params:
      n_estimators: 100
      max_depth: 6
      learning_rate: 0.1
      num_leaves: 31
      min_child_samples: 20

  # 训练配置
  training:
    test_size: 0.2              # 测试集比例
    validation_split: 0.2       # 验证集比例
    early_stopping_rounds: 10   # 早停轮数
```

### 缓存配置

```yaml
cache:
  expiry:
    sectors: 3600       # 板块列表缓存1小时
    predict_all: 21600  # 全板块预测缓存6小时
    predict_multi: 14400  # 多板块预测缓存4小时
  max_size:
    sectors: 10
    predict_all: 50
  global:
    default_ttl: 3600
    max_entries: 500
```

---

## 📁 项目结构

```
FinSectorForecast/
├── 📂 config/
│   ├── __init__.py
│   └── config.yaml              # 配置文件
├── 📂 src/
│   ├── __init__.py
│   ├── data_fetcher.py          # 数据获取模块
│   ├── feature_engineering.py   # 特征工程模块
│   ├── model_training.py        # 模型训练模块
│   ├── predictor.py             # 预测核心模块
│   ├── memory_cache.py          # LRU内存缓存
│   ├── cache_manager.py         # 缓存管理器
│   ├── section_cache.py         # 页面片段缓存
│   └── background_task_manager.py # 后台任务管理
├── 📂 templates/
│   ├── base.html                # 基础模板
│   ├── index.html               # 仪表盘页面
│   ├── predict.html             # 预测页面
│   ├── training.html            # 训练页面
│   └── analysis.html            # 分析页面
├── 📄 app.py                    # Web应用入口
├── 📄 main.py                   # CLI入口
├── 📄 requirements.txt          # 依赖列表
├── 📄 .gitignore
└── 📄 README.md
```

### 核心模块说明

| 模块 | 功能描述 |
|:-----|:---------|
| [`data_fetcher.py`](src/data_fetcher.py) | 从东方财富获取板块资金流向和行情数据 |
| [`feature_engineering.py`](src/feature_engineering.py) | 计算30+技术指标特征，生成训练数据 |
| [`model_training.py`](src/model_training.py) | LightGBM/XGBoost模型训练与评估 |
| [`predictor.py`](src/predictor.py) | 整合数据、特征、模型的预测核心 |
| [`memory_cache.py`](src/memory_cache.py) | 高性能LRU缓存实现 |
| [`cache_manager.py`](src/cache_manager.py) | 统一缓存管理 |
| [`section_cache.py`](src/section_cache.py) | 页面片段级缓存 |
| [`background_task_manager.py`](src/background_task_manager.py) | 后台异步任务管理 |

---

## 📦 依赖说明

| 依赖 | 版本 | 用途 |
|:-----|:-----|:-----|
| flask | ≥2.3.0 | Web框架 |
| pandas | ≥1.5.0 | 数据处理 |
| numpy | ≥1.23.0 | 数值计算 |
| scikit-learn | ≥1.2.0 | 机器学习工具 |
| lightgbm | ≥3.3.0 | 梯度提升模型 |
| matplotlib | ≥3.6.0 | 数据可视化 |
| seaborn | ≥0.12.0 | 统计可视化 |
| pyyaml | ≥6.0 | 配置文件解析 |
| tqdm | ≥4.65.0 | 进度条显示 |
| akshare | ≥1.12.0 | 金融数据获取（可选） |

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

### 开发流程

```bash
# 1. Fork 项目
# 2. 创建特性分支
git checkout -b feature/your-feature-name

# 3. 提交更改
git commit -m "Add: your feature description"

# 4. 推送到分支
git push origin feature/your-feature-name

# 5. 创建 Pull Request
```

### 代码规范

- 遵循 PEP 8 编码规范
- 添加必要的文档字符串
- 编写单元测试

---

## ⚠️ 风险提示

<div align="center">

**本系统仅供学习研究使用，不构成任何投资建议。**

股市有风险，投资需谨慎。任何投资决策都应基于个人独立判断，
并充分考虑自身风险承受能力。过往表现不代表未来收益。

</div>

---

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源协议。

---

## 📮 联系方式

如有问题或建议，欢迎通过以下方式联系：

- 🐛 Issues: [GitHub Issues](https://github.com/L-newbie/FinSectorForecast/issues)

---

<div align="center">

**⭐ 如果这个项目对你有帮助，请给一个 Star ⭐**

Made with ❤️ by FinSectorForecast Team

</div>
