# 调度模块详解

## 概述

调度模块是 Big-A 的自动化中枢，负责协调日常的数据更新和模拟交易流程。

为什么要用调度系统？因为量化策略需要定期执行，比如每天收盘后更新数据、生成信号、执行交易。如果全靠手动操作，既费时又容易出错。调度系统就像一个尽职的助理，按照预定的时间表自动完成这些工作。

Big-A 使用 Prefect 作为调度框架。Prefect 是一个现代的工作流编排工具，专门为数据管道设计，相比传统的 cron 任务，它有更清晰的任务依赖管理、更强大的重试机制、更友好的可视化界面。

Prefect 的核心概念：
- **Flow（流）**：一个完整的工作流程，相当于一篇文章
- **Task（任务）**：流中的单个步骤，相当于文章中的一个段落
- **Deployment（部署）**：把流发布到 Prefect 服务器，设置执行时间表

## 核心文件

调度相关的代码分布在以下位置：

```
src/big_a/scheduler/flows.py     # Prefect flow 和 task 定义
scripts/run_simulation.py        # 模拟交易的命令行工具
src/big_a/data/sector.py        # 板块分类数据管理
src/big_a/data/rotation.py      # 板块轮动信号计算
```

## Daily Pipeline

`daily_pipeline` 是每日数据更新的主流程，它串联了市场数据更新和板块数据更新两个任务。

```python
@flow
def daily_pipeline() -> None:
    """每日数据更新管道，按顺序执行市场和板块更新。"""
    try:
        update_market_data()
    except Exception as e:
        logger.error(f"Market data update failed: {e}")

    try:
        update_sector_data()
    except Exception as e:
        logger.error(f"Sector data update failed: {e}")
```

这个流程的设计思路是"尽力而为"。每个任务独立执行，某个任务失败不会阻止其他任务继续。这就像一列火车，如果一节车厢出了问题，不应该让整列火车都停下来。

流程执行顺序是串行的：先完成市场数据更新，再进行板块数据更新。这样可以确保后续任务使用最新的数据。

## 可用任务

### update_market_data

这个任务负责增量更新市场数据。它调用 `big_a.data.updater` 模块的 `update_incremental()` 函数，从 GitHub 数据仓库下载最新的日频数据。

```python
@task
def update_market_data() -> None:
    """增量更新市场数据。"""
    from big_a.data.updater import update_incremental

    update_incremental()
```

执行这个任务时，会自动检测本地数据的最后更新日期，只下载从那天到今天的新数据。这避免了大量重复下载，节省时间和带宽。

### update_sector_data

这个任务刷新申万一级行业分类数据。它调用 `big_a.data.sector` 模块的 `refresh_sector_data()` 函数，从 AKShare 获取最新的板块分类信息。

```python
@task
def update_sector_data() -> None:
    """刷新板块分类数据。"""
    from big_a.data.sector import refresh_sector_data

    refresh_sector_data()
```

板块分类数据会缓存到 `data/sector_data/sw_classification.parquet` 文件。任务会优先从缓存读取，只有强制刷新或者缓存失效时才重新获取。

### run_simulation_daily

这个任务执行一天的模拟交易。它加载最新的组合快照，用模拟信号运行一天的交易，然后保存结果。

```python
@task
def run_simulation_daily() -> dict:
    """执行单日模拟交易。"""
    # 加载配置和存储
    config = load_simulation_config("configs/simulation/default.yaml")
    storage = SimulationStorage(...)

    # 恢复上一天的状态
    latest = storage.load_latest_snapshot()
    broker = InMemoryBroker(...)

    if latest:
        broker._cash = latest.cash
        for code, pos in latest.positions.items():
            broker._positions[code] = pos

    # 运行模拟
    engine = SimulationEngine(config=config, broker=broker)
    engine.initialize()

    # 生成模拟信号（目前是占位符）
    mock_signals = [
        StockSignal(stock_code="600519.SH", score=0.8, signal=SignalStrength.BUY, source=SignalSource.fused),
        StockSignal(stock_code="000858.SZ", score=0.6, signal=SignalStrength.BUY, source=SignalSource.fused),
    ]

    portfolio = engine.run_daily(trading_date=today, signals=mock_signals, prices={...})

    # 保存状态
    storage.save_snapshot(portfolio, today)

    return {"total_value": portfolio.total_value, "cash": portfolio.cash}
```

注意这里的信号是模拟的占位符。未来会用真实的模型信号替换它们。

## 板块轮动信号

除了基本的数据更新，调度模块还提供了板块轮动信号功能。这个功能在 `src/big_a/data/rotation.py` 中实现。

板块轮动是动量策略的一种。核心逻辑是：过去一段时间涨得好的板块，未来可能继续涨。`rank_sectors()` 函数计算每个申万一级行业的动量得分，然后排序。

```python
from big_a.data.rotation import rank_sectors, get_top_sectors

# 获取所有板块的动量排名
ranked = rank_sectors(lookback_days=20)
# 返回: [("银行", 3.5), ("食品饮料", 2.8), ("电子", -1.2), ...]

# 获取top-k板块
top_banks = get_top_sectors(top_k=5, lookback_days=20)
# 返回: ["银行", "食品饮料", "医药生物", "房地产", "汽车"]
```

计算动量的方法是：取每个板块内所有股票的等权重指数，计算 N 日前的价格和当前价格的涨跌幅。

## 模拟交易 CLI

模拟交易的入口脚本是 `scripts/run_simulation.py`，它提供了几个命令：

### 初始化账户

```bash
python scripts/run_simulation.py init
```

这会创建必要的目录结构，初始化一个空仓组合。默认初始资金是 100 万。

### 查看状态

```bash
python scripts/run_simulation.py status
```

显示当前账户的现金、持仓、市值和盈亏情况。如果持仓不为空，会以表格形式展示每只股票的详细信息。

输出示例：
```
Simulation Account: default
Initial Capital: 1,000,000.00
Current Cash: 850,000.00
Total Portfolio Value: 980,000.00
Total P&L: -20,000.00 (-2.00%)

Positions:
Stock        Qty   Avg Price      Current       P&L   P&L%
--------------------------------------------------------------
600519.SH      100      1600.00      1550.00   -5000.00   -3.12%
```

### 运行模拟

```bash
python scripts/run_simulation.py run
```

执行一天的交易模拟。默认使用模拟信号，随机选择几只股票生成买卖信号，然后模拟实际成交。

如果想预览信号而不实际交易：
```bash
python scripts/run_simulation.py run --dry-run
```

这会显示会生成哪些信号，但不会执行任何操作。

连续运行模式（持续监控新数据）：
```bash
python scripts/run_simulation.py run --continuous
```

### 查看交易历史

```bash
python scripts/run_simulation.py history
```

显示所有历史成交记录，按时间倒序排列。

## Prefect 集成

### 本地运行

直接运行 flow：
```python
from big_a.scheduler.flows import daily_pipeline

# 执行一次
daily_pipeline()
```

### 部署到 Prefect Cloud

```python
from big_a.scheduler.flows import daily_pipeline

# 发布到 Prefect
daily_pipeline.serve(
    cron="0 19 * * 1-5",  # 工作日晚上7点执行
    name="daily-pipeline"
)
```

这会启动一个持久化的服务，按照 cron 表达式的时间表自动执行。

### 用 Deployment 方式部署

```python
from prefect import flow
from big_a.scheduler.flows import daily_pipeline

# 另一种部署方式
daily_pipeline.deploy(
    name="daily-pipeline",
    cron="0 19 * * 1-5"
)
```

 Prefect 的优势在于可以到 Prefect Cloud 的 dashboard 上查看每个任务的执行状态、日志、耗时。如果某个任务失败了，可以重试单个任务而不需要重新运行整个流程。

## 数据存储

调度模块依赖模拟存储系统来保存状态：

- **快照（Snapshot）**：保存每天收盘后的组合状态，包括现金、持仓、市值
- **交易记录（Trades）**：保存每天的实际成交明细
- **决策记录（Decisions）**：保存每天的信号和决策理由

这些数据默认存在 `data/simulation/` 目录下：
```
data/simulation/
├── snapshots/       # 每日组合快照
├── trades/         # 成交记录
├── decisions/      # 决策记录
└── storage.json     # 存储元数据
```

## 与数据更新模块的关系

调度模块和数据管理模块（data.md）紧密配合：

1. **update_market_data 任务** 调用 `updater.update_incremental()`，这正是数据管理模块的核心功能
2. **update_sector_data 任务** 调用 `sector.refresh_sector_data()`，管理申万行业分类缓存
3. **板块轮动信号** 依赖板块分类数据，用来做行业动量分析

可以这样理解：数据管理模块提供原材料，调度模块负责按照时间表采购这些原材料。

## 常见问题

### 任务失败了怎么办？

Prefect 会在 dashboard 中显示失败状态。可以点击失败的任务查看错误日志，然后手动重试。

如果某个任务需要重试：
```python
# 在 Prefect dashboard 上点击 retry，或者用代码：
from big_a.scheduler.flows import update_market_data
update_market_data.with_options(retries=3, retry_delay_seconds=60)()
```

### 可以同时运行多个 pipeline 吗？

可以。Prefect 支持并发执行多个 flow。但要注意数据一致性，避免多个 flow 同时写入同一个文件。

### 交易信号从哪里来？

目前的实现使用的是占位符信号（mock signals）。未来会接入 Kronos 和 LightGBM 模型，生成真实的预测信号。

### 如何查看调度任务的执行历史？

在 Prefect Cloud dashboard 上可以查看所有历史执行记录，包括开始时间、结束时间、耗时、状态。

---

## 总结

调度模块是 Big-A 的自动化核心，它解决了"什么时候做什么事"的问题。

主要能力：
1. **daily_pipeline**：串联市场数据和板块数据的每日更新流程
2. **run_simulation_daily**：执行每日模拟交易，保存组合快照
3. ** Prefect 集成**：支持定时执行、失败重试、执行监控
4. **板块轮动**：提供行业动量信号，辅助选股

调度模块把人工操作变成了自动化流程，让策略可以持续、稳定地运行。