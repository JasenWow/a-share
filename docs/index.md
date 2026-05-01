# Big-A

Big-A 是一个基于 [Qlib](https://github.com/microsoft/qlib) 的 A 股量化交易研究框架，支持多模型训练、预测、回测和评估的完整工作流。

## 📖 详细文档

如果你是量化交易新手，建议按以下顺序阅读：

### 基础概念
- **[量化交易基础概念](concepts/quant-basics.md)** — 什么是量化交易、A股、因子、回测、信号？
- **[Qlib 框架介绍](concepts/qlib-intro.md)** — Qlib 是什么？核心概念、数据格式、表达式引擎

### 项目架构
- **[项目架构总览](architecture/overview.md)** — 模块关系、数据流、配置系统详解

### 模块详解
- **[数据管理模块](modules/data.md)** — 数据下载、更新、验证流程
- **[因子工程模块](modules/factors.md)** — Alpha158、自定义因子、算子
- **[模型模块](modules/models.md)** — LightGBM 与 Kronos 模型原理与使用
- **[Hedge Fund 多智能体模块](modules/hedge-fund.md)** — LLM 多智能体选股系统（13 个分析智能体）
- **[回测引擎](modules/backtest.md)** — 回测原理、TopkDropout 策略、A股交易规则
- **[实盘交易策略](modules/real-trading.md)** — 小资金实盘策略：止损、周频调仓、仓位上限
- **[模拟交易引擎](modules/simulation.md)** — 模拟交易执行、订单管理、持仓跟踪
- **[Broker 券商接口](modules/broker.md)** — 券商接口抽象层，支持模拟和实盘券商
- **[调度工作流](modules/scheduler.md)** — Prefect 自动化流水线、每日任务调度
- **[评估模块](modules/evaluation.md)** — IC、Rank IC、ICIR、Sharpe 等指标详解

### 使用指南
- **[滚动回测指南](guides/rolling-backtest.md)** — Walk-forward 滚动回测方法
- **[端到端流程指南](guides/e2e.md)** — 9 步完整流水线详解
- **[自选股分析指南](guides/watchlist.md)** — 多模型打分、趋势分析、持仓模拟、AI 定性分析

### 参考
- **[术语表](reference/glossary.md)** — 所有量化术语的简单解释

---

## 核心能力

- **数据管理**：自动从 [chenditc/investment_data](https://github.com/chenditc/investment_data) 下载和更新 Qlib 格式的 A 股数据
- **多模型支持**：LightGBM（传统因子模型）和 Kronos（基于 Transformer 的时序模型）
- **Alpha158 因子**：内置 Qlib Alpha158 特征工程流水线
- **Hedge Fund 多智能体**：13 个 LLM 智能体并行分析，模拟大师投资风格
- **自选股分析**：多模型打分、趋势追踪、持仓模拟、AI 定性分析
- **回测引擎**：基于 Qlib 的回测引擎，支持 TopkDropout、RealTradingStrategy（小资金实盘）等策略
- **模拟交易引擎**：完整的模拟交易系统，支持止损、熔断、调仓
- **模型评估**：IC、Rank IC、ICIR 等指标计算和可视化对比
- **实验追踪**：集成 MLflow 进行实验参数和指标记录
- **端到端流水线**：一键运行从数据验证到分析报告的完整流程

## 项目结构

```
big-a/
├── configs/                  # 配置文件
│   ├── base.yaml             # 基础配置（Qlib 初始化、市场、基准）
│   ├── data/                 # 数据处理配置
│   │   ├── handler_alpha158.yaml
│   │   └── handler_custom.yaml
│   ├── model/                # 模型配置
│   │   ├── lightgbm.yaml
│   │   └── kronos.yaml
│   ├── backtest/             # 回测配置
│   │   ├── topk_csi300.yaml
│   │   ├── real_trading.yaml
│   │   └── rolling_csi300.yaml
│   └── simulation/           # 模拟交易配置
├── src/big_a/                # 核心源码
│   ├── config.py             # 配置加载与合并
│   ├── qlib_config.py        # Qlib 初始化
│   ├── experiment.py         # MLflow 实验管理
│   ├── data/                 # 数据更新与验证
│   ├── factors/              # 因子计算（Alpha158、自定义算子）
│   ├── models/               # 模型实现
│   │   ├── lightgbm_model.py
│   │   ├── kronos.py
│   │   ├── kronos_model/
│   │   └── hedge_fund/       # LLM 多智能体系统
│   ├── backtest/             # 回测引擎与分析
│   │   ├── engine.py
│   │   ├── analysis.py
│   │   ├── evaluation.py
│   │   ├── metrics.py
│   │   ├── plots.py
│   │   ├── reporting.py
│   │   └── rolling.py
│   ├── report/               # 报告生成与格式化
│   │   ├── formatter.py
│   │   └── scorer.py          # 自选股评分器
│   ├── strategy/             # 交易策略
│   │   └── real_trading.py    # 小资金实盘策略
│   ├── simulation/           # 模拟交易引擎
│   │   ├── engine.py
│   │   ├── types.py
│   │   ├── config.py
│   │   └── storage.py
│   ├── broker/               # 券商接口
│   │   ├── base.py
│   │   └── in_memory.py
│   ├── scheduler/            # Prefect 工作流调度
│   │   └── flows.py
│   └── workflow/             # 工作流编排
├── scripts/                  # CLI 入口脚本
│   ├── run.py                # 统一 CLI（train / predict / backtest / evaluate）
│   ├── e2e.py                # 端到端流水线
│   ├── watchlist_report.py   # 自选股分析报告
│   ├── train_lightgbm.py     # 单独训练 LightGBM
│   ├── predict_lightgbm.py   # 单独预测 LightGBM
│   ├── predict_kronos.py     # 单独预测 Kronos
│   ├── backtest.py           # 单独回测
│   ├── roll_backtest.py      # 滚动回测
│   ├── real_trading_backtest.py  # 实盘策略回测
│   ├── evaluate.py           # 模型评估
│   ├── analyze.py            # 分析报告
│   ├── update_data.py        # 数据更新
│   └── validate_data.py      # 数据验证
├── tests/                    # 测试
├── data/                     # Qlib 数据目录（需下载）
├── output/                   # 输出目录
└── mlruns/                   # MLflow 实验追踪
```

## 环境要求

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/)（推荐的包管理器）

## 安装

```bash
cd big-a
uv sync
```

## 快速开始

### 1. 下载数据

首次使用需要下载 Qlib 格式的 A 股数据：

```bash
# 查看当前数据状态（dry-run）
uv run python scripts/update_data.py --dry-run

# 下载数据
uv run python scripts/update_data.py update
```

数据默认存储在 `data/qlib_data/cn_data/` 目录下。

### 2. 训练模型

```bash
# 使用默认配置训练 LightGBM
uv run python scripts/run.py train

# 指定配置文件
uv run python scripts/run.py train \
  --model-config configs/model/lightgbm.yaml \
  --data-config configs/data/handler_alpha158.yaml \
  --output output/lightgbm_model.pkl
```

### 3. 生成预测

```bash
# LightGBM 预测
uv run python scripts/run.py predict \
  --model lightgbm \
  --model-path output/lightgbm_model.pkl \
  --output output/predictions.parquet

# Kronos 预测
uv run python scripts/run.py predict \
  --model kronos \
  --kronos-config configs/model/kronos.yaml \
  --output output/kronos_predictions.csv
```

### 4. 回测

```bash
uv run python scripts/run.py backtest \
  --signal-file output/predictions.parquet \
  --config configs/backtest/topk_csi300.yaml \
  --output output/backtest_report.parquet
```

### 5. 模型评估

```bash
uv run python scripts/run.py evaluate \
  --kronos output/kronos_predictions.csv \
  --lightgbm output/predictions.parquet \
  --actual output/actual_returns.csv \
  --output-dir output/evaluation
```

### 6. 端到端流水线（推荐）

一键运行完整的 LightGBM + Kronos 对比流程：

```bash
uv run python scripts/e2e.py

# 跳过训练，使用已有模型
uv run python scripts/e2e.py --skip-train

# 跳过 Kronos 步骤，仅运行 LightGBM
uv run python scripts/e2e.py --skip-kronos
```

端到端流程包含 9 个步骤：

1. 数据验证
2. Qlib 初始化
3. 训练 LightGBM
4. 生成预测
5. 运行回测
6. 生成分析报告
7. Kronos 滚动信号生成
8. Kronos 回测
9. 模型对比

## 配置说明

所有配置文件位于 `configs/` 目录，采用 YAML 格式，支持多层叠加合并。

### 基础配置 (`configs/base.yaml`)

```yaml
qlib_init:
  provider_uri: "data/qlib_data/cn_data"  # Qlib 数据路径
  region: cn                                # 市场（中国 A 股）

market: csi300          # 股票池（沪深 300）
benchmark: SH000300     # 基准指数

data_handler:
  start_time: "2010-01-01"
  end_time: "2024-12-31"
  instruments: csi300
```

### 数据集配置 (`configs/data/`)

- `handler_alpha158.yaml` — 使用 Qlib Alpha158 特征，数据分段为：
  - 训练集：2010-01-01 ~ 2018-12-31
  - 验证集：2019-01-01 ~ 2021-12-31
  - 测试集：2022-01-01 ~ 2024-12-31

### 模型配置 (`configs/model/`)

- `lightgbm.yaml` — LightGBM 模型参数（损失函数、学习率、树深度等）
- `kronos.yaml` — Kronos 模型参数（HuggingFace 模型 ID、回看窗口、预测长度等）

### 回测配置 (`configs/backtest/`)

- `topk_csi300.yaml` — TopkDropout 策略，选股 topk=50，每次调仓 n_drop=5
- `rolling_csi300.yaml` — 滚动回测配置（训练 5 年 → 验证 1 年 → 测试 1 年，步长 1 年）

回测默认参数：
- 初始资金：1 亿元
- 涨跌停阈值：9.5%
- 买入成本：0.05%
- 卖出成本：0.15%
- 最低手续费：5 元

## 常用操作

### 检查数据状态

```bash
cd big-a
uv run python -c "
import sys; sys.path.insert(0, 'src')
from big_a.data.updater import get_last_update_date
print(f'Last update: {get_last_update_date()}')
"
```

### 更新数据

```bash
uv run python scripts/update_data.py update
```

### 验证数据完整性

```bash
uv run python scripts/validate_data.py
```
