# Qlib 框架介绍

Qlib 是微软开源的 AI 量化投资平台，是 Big-A 项目的核心依赖。本指南将帮你理解 Qlib 的核心概念和使用方法。

---

## 1. 什么是 Qlib？

### 1.1 Qlib 是什么

**Qlib** 是微软亚洲研究院（MSRA）开源的面向 AI 的量化投资平台。

**官方定义**：Qlib 旨在通过 AI 方案，为量化投资提供一站式的解决方案，让研究者、从业者更方便地进行 AI 量化研究。

**生活类比**：
- Qlib 就像是一个"量化交易的瑞士军刀"
- 里面包含了你需要的大部分工具：
  - 数据处理工具（像刀）
  - 因子计算工具（像剪刀）
  - 模型训练工具（像锯子）
  - 回测工具（像开瓶器）
- 你不需要自己造这些工具，直接用就行

### 1.2 Qlib 的特点

| 特点 | 说明 |
|------|------|
| **开源免费** | GitHub 上开源，任何人都可以使用 |
| **AI 原生** | 专为机器学习/深度学习设计 |
| **完整工具链** | 从数据处理到回测，一应俱全 |
| **高性能** | 用 Python 和 C++ 混合编写，速度快 |
| **可扩展** | 可以自定义因子、模型、策略 |

### 1.3 Qlib 的官方网站

- **GitHub 仓库**：https://github.com/microsoft/qlib
- **官方文档**：https://qlib.readthedocs.io/
- **论文**：Qlib: An AI-oriented Quantitative Investment Platform (KDD 2020)

### 1.4 Qlib 提供了什么

Qlib 提供了量化交易的"完整工具链"：

```
原始数据 → 数据处理 → 因子计算 → 模型训练 → 策略回测 → 绩效分析
   ↓           ↓           ↓           ↓           ↓           ↓
 Provider   DataHandler  Expression  LGBModel   Strategy   Portfolio
```

**类比**：就像汽车制造厂的生产线
- 原材料（原始数据）→ 零件加工（数据处理）→ 组装（因子计算）→ 质检（模型训练）→ 测试驾驶（回测）→ 交付（绩效分析）

---

## 2. Qlib 的核心概念

### 2.1 Provider（数据提供者）

#### 2.1.1 什么是 Provider

**Provider** 是 Qlib 的数据提供者，负责读取和存储股票数据。

**关键特点**：
- Qlib 不使用 CSV 等文本格式，而是使用**二进制格式**（.bin 文件）
- 二进制格式读取速度非常快（比 CSV 快 10-100 倍）
- 数据按时间序列存储，方便查询

**生活类比**：
- CSV 格式就像"手写的日记"，每行一个记录，读取时要一行行解析
- 二进制格式就像"数据库"，直接定位到位置，瞬间读取

#### 2.1.2 为什么用二进制格式

**CSV 格式的问题**：
```csv
date,close,volume
2020-01-01,10.5,1000000
2020-01-02,10.8,1200000
...
```
- 每次读取都要解析字符串（"10.5" → 10.5）
- 文件大，占空间
- 读取慢

**二进制格式**：
- 直接存储数值（10.5 直接存储为浮点数）
- 文件小，省空间
- 读取快

**性能对比**（估算）：
- 读取 300 只股票 × 5 年数据：
  - CSV：约 10 秒
  - 二进制：约 0.1 秒
- **快了 100 倍！**

#### 2.1.3 在代码中使用 Provider

```python
import qlib

# 初始化 Qlib，指定数据目录
qlib.init(
    provider_uri="data/qlib_data/cn_data",  # 数据目录
    region="cn"  # 中国市场
)

# 数据目录结构：
# data/qlib_data/cn_data/
# ├── calendars/day.txt          # 交易日历
# ├── instruments/all.txt         # 股票列表
# └── features/                   # 特征数据
#     ├── SH600000/
#     │   ├── close.bday.bin     # 收盘价
#     │   ├── volume.bday.bin    # 成交量
#     │   └── ...
#     ├── SH600001/
#     └── ...
```

---

### 2.2 Expression Engine（表达式引擎）

#### 2.2.1 什么是表达式引擎

**Expression Engine** 是 Qlib 的因子计算引擎，用类似数学公式的语法来计算因子。

**核心优势**：
- 不需要写循环代码
- 用简单的表达式就能计算复杂因子
- 自动处理时间序列操作
- 性能优化（底层用 C++ 实现）

#### 2.2.2 基本语法

**常用操作符**：

| 操作符 | 含义 | 例子 | 说明 |
|--------|------|------|------|
| `$close` | 收盘价 | `$close` | 当天收盘价 |
| `$open` | 开盘价 | `$open` | 当天开盘价 |
| `$high` | 最高价 | `$high` | 当天最高价 |
| `$low` | 最低价 | `$low` | 当天最低价 |
| `$volume` | 成交量 | `$volume` | 当天成交量 |
| `+` | 加法 | `$close + $open` | 收盘价 + 开盘价 |
| `-` | 减法 | `$close - $open` | 收盘价 - 开盘价 |
| `*` | 乘法 | `$close * $volume` | 收盘价 × 成交量（成交额） |
| `/` | 除法 | `$close / $open` | 收盘价 / 开盘价 |

**常用函数**：

| 函数 | 含义 | 例子 | 说明 |
|------|------|------|------|
| `Mean(x, n)` | n 日均值 | `Mean($close, 20)` | 20 日收盘价均值 |
| `StdDev(x, n)` | n 日标准差 | `StdDev($close, 20)` | 20 日收盘价波动率 |
| `Max(x, n)` | n 日最大值 | `Max($high, 20)` | 20 日最高价 |
| `Min(x, n)` | n 日最小值 | `Min($low, 20)` | 20 日最低价 |
| `Sum(x, n)` | n 日求和 | `Sum($volume, 5)` | 5 日成交量总和 |
| `Ref(x, n)` | 引用 n 天前的值 | `Ref($close, 5)` | 5 天前的收盘价 |
| `Delay(x, n)` | 延迟 n 天 | `Delay($close, 5)` | 5 天前的收盘价（同 Ref） |
| `Return(x, n)` | n 日收益率 | `Return($close, 5)` | 5 日收益率 |
| `Rank(x)` | 横截面排名 | `Rank($close)` | 当天收盘价在所有股票中的排名 |

#### 2.2.3 实战例子

**例子 1：计算 20 日收益率**

```python
# 表达式
expr = "Return($close, 20)"

# 含义：(今天收盘价 - 20天前收盘价) / 20天前收盘价

# 对应的 Python 代码（如果不使用表达式引擎）
def calculate_return(close_prices, n=20):
    if len(close_prices) < n + 1:
        return None
    today = close_prices[-1]
    n_days_ago = close_prices[-n-1]
    return (today - n_days_ago) / n_days_ago
```

**例子 2：计算 20 日均线**

```python
# 表达式
expr = "Mean($close, 20)"

# 含义：过去 20 天收盘价的平均值

# 对应的 Python 代码
def calculate_ma(close_prices, n=20):
    if len(close_prices) < n:
        return None
    return sum(close_prices[-n:]) / n
```

**例子 3：计算动量因子（20 日收益率排名）**

```python
# 表达式
expr = "Rank(Return($close, 20))"

# 含义：
# 1. 先计算每只股票的 20 日收益率
# 2. 然后在所有股票中排名
# 3. 返回排名（0-1 之间的分数）

# 假设有 3 只股票：
# 股票A：20 日收益率 10%
# 股票B：20 日收益率 5%
# 股票C：20 日收益率 15%
# 排名：C(15%) > A(10%) > B(5%)
# Rank 值：C=1.0, A=0.5, B=0.0
```

**例子 4：计算波动率因子**

```python
# 表达式
expr = "StdDev($close, 20) / Mean($close, 20)"

# 含义：
# 20 日标准差 / 20 日均值
# 数值越大，波动越大

# 类比：
# 学生 A 的成绩：80, 82, 78, 81, 79（波动小）
# 学生 B 的成绩：60, 90, 50, 100, 70（波动大）
# 学生 B 的波动率更高
```

#### 2.2.4 在代码中使用表达式引擎

```python
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

# 定义一个简单的表达式
expressions = [
    "$close",  # 收盘价
    "$volume",  # 成交量
    "Mean($close, 20)",  # 20 日均线
    "Return($close, 5)",  # 5 日收益率
]

# 加载数据
data_handler = DataHandlerLP(
    instruments="csi300",  # 沪深 300 指数成分股
    start_time="2020-01-01",
    end_time="2024-12-31",
    infer_processors=[
        {"class": "FilterCol", "module_path": "qlib.contrib.data.processor", "kwargs": {"fields_group": "feature", "col_list": expressions}}
    ]
)

# 获取数据
df = data_handler.fetch()
print(df.head())
```

---

### 2.3 DataHandler（数据处理器）

#### 2.3.1 什么是 DataHandler

**DataHandler** 是 Qlib 的数据处理器，负责：
- 加载原始数据
- 计算因子（使用表达式引擎）
- 处理缺失值
- 标准化/归一化数据
- 划分训练/验证/测试集

**生活类比**：
- DataHandler 就像"厨师"
- 原材料（原始数据）→ 清洗、切菜、调味（处理）→ 成品（模型可用的数据）

#### 2.3.2 Alpha158 Handler

**Alpha158 Handler** 是 Qlib 内置的数据处理器，包含了 158 个常用的量化因子。

**特点**：
- 开箱即用，不需要自己定义因子
- 涵盖了常用的技术指标
- 自动处理数据标准化

**代码示例**：

```python
from qlib.contrib.data.handler import Alpha158

# 创建 Alpha158 Handler
data_handler = Alpha158(
    instruments="csi300",  # 沪深 300
    start_time="2020-01-01",
    end_time="2024-12-31",
    fit_start_time="2020-01-01",  # 训练集开始时间
    fit_end_time="2022-12-31",    # 训练集结束时间
)

# 获取数据
df = data_handler.fetch()
print(df.shape)  # (样本数, 158个因子)
```

#### 2.3.3 fit_start_time 和 fit_end_time

**重要概念**：防止数据泄露（Data Leakage）

**问题**：如果用全部数据（包括未来数据）来标准化，就是"作弊"。

**例子**：
- 你在 2020 年 1 月做预测
- 如果用 2021-2024 年的数据来计算均值和标准差
- 就相当于"看到了未来"

**正确做法**：
- fit_start_time 和 fit_end_time 定义了"训练集"的时间范围
- 只用训练集的数据来学习标准化参数
- 验证集和测试集使用相同的参数

**代码示例**：

```python
data_handler = Alpha158(
    instruments="csi300",
    start_time="2020-01-01",    # 数据开始时间
    end_time="2024-12-31",      # 数据结束时间
    fit_start_time="2020-01-01",  # 只用这部分数据学习标准化参数
    fit_end_time="2022-12-31",    # （防止数据泄露）
)

# 时间划分：
# 2020-01-01 ~ 2022-12-31：训练集（用于学习模型）
# 2023-01-01 ~ 2023-12-31：验证集（用于调参）
# 2024-01-01 ~ 2024-12-31：测试集（用于评估）
```

**生活类比**：
- 就像考试复习
- 你在考试前（2022 年 12 月）只能看之前的教材
- 不能看考后的答案（2023-2024 年）
- 否则就是"作弊"

#### 2.3.4 自定义 DataHandler

你可以自定义 DataHandler 来添加自己的因子：

```python
from qlib.data.dataset.handler import DataHandlerLP

# 自定义表达式
custom_expressions = [
    "$close",
    "$volume",
    "Mean($close, 20)",
    "StdDev($close, 20)",
    "Return($close, 5)",
    "$close / Mean($close, 20) - 1",  # 相对 20 日均线的位置
]

data_handler = DataHandlerLP(
    instruments="csi300",
    start_time="2020-01-01",
    end_time="2024-12-31",
    fit_start_time="2020-01-01",
    fit_end_time="2022-12-31",
    infer_processors=[
        {
            "class": "FilterCol",
            "module_path": "qlib.contrib.data.processor",
            "kwargs": {
                "fields_group": "feature",
                "col_list": custom_expressions
            }
        }
    ]
)
```

---

### 2.4 DatasetH（数据集）

#### 2.4.1 什么是 DatasetH

**DatasetH** 是 Qlib 的数据集类，负责：
- 管理 DataHandler
- 划分训练/验证/测试集
- 提供数据访问接口

**生活类比**：
- DatasetH 就像"图书馆管理员"
- 管理所有书籍（数据）
- 按类别分类（训练集、验证集、测试集）
- 帮你找需要的书

#### 2.4.2 segments（数据集划分）

**segments** 定义了数据集的划分：

```python
from qlib.data.dataset import DatasetH
from qlib.contrib.data.handler import Alpha158

dataset = DatasetH(
    handler={
        "class": "Alpha158",
        "module_path": "qlib.contrib.data.handler",
    },
    segments={
        "train": ("2020-01-01", "2022-12-31"),  # 训练集
        "valid": ("2023-01-01", "2023-12-31"),  # 验证集
        "test": ("2024-01-01", "2024-12-31"),   # 测试集
    }
)
```

**时间划分的作用**：
- **训练集**：用于训练模型（学习因子与收益的关系）
- **验证集**：用于调整模型参数（选择最好的模型）
- **测试集**：用于评估最终性能（不参与训练和调参）

**生活类比**：
- 就像准备考试
- 练习题（训练集）：学习知识
- 模拟考试（验证集）：检查掌握程度，调整学习方法
- 真实考试（测试集）：最终评估

---

### 2.5 D 对象（数据访问接口）

#### 2.5.1 什么是 D 对象

**D 对象**是 Qlib 的数据访问接口，提供了类似数据库查询的功能。

**常用方法**：

| 方法 | 功能 | 例子 |
|------|------|------|
| `D.features()` | 查询特征数据 | 查询收盘价、成交量 |
| `D.instruments()` | 查询股票列表 | 查询沪深 300 成分股 |
| `D.calendar()` | 查询交易日历 | 查询 2024 年的所有交易日 |
| `D.list_data()` | 列出可用的数据 | 查看有哪些数据可用 |

#### 2.5.2 D.features() - 查询特征数据

```python
from qlib.data import D

# 查询收盘价
df = D.features(
    instruments=["SH600000"],  # 股票代码
    fields=["$close"],         # 字段
    start_time="2024-01-01",
    end_time="2024-01-31"
)
print(df.head())

# 输出：
# instrument    close
# datetime
# 2024-01-01  SH600000  10.5
# 2024-01-02  SH600000  10.8
# ...
```

**查询多个字段**：

```python
df = D.features(
    instruments=["SH600000", "SH600001"],
    fields=["$close", "$volume", "Mean($close, 20)"],
    start_time="2024-01-01",
    end_time="2024-01-31"
)
print(df.head())

# 输出：
# instrument    close   volume  Mean($close, 20)
# datetime
# 2024-01-01  SH600000  10.5   1000000  10.3
# 2024-01-02  SH600000  10.8   1200000  10.4
# ...
```

#### 2.5.3 D.instruments() - 查询股票列表

```python
# 查询沪深 300 成分股
instruments = D.instruments(market="csi300")
print(len(instruments))  # 300
print(instruments[:10])  # 前 10 只股票

# 输出：
# ['SH600000', 'SH600001', 'SH600004', ..., 'SZ300750']
```

#### 2.5.4 D.calendar() - 查询交易日历

```python
# 查询 2024 年的交易日
trading_days = D.calendar(start_time="2024-01-01", end_time="2024-12-31")
print(len(trading_days))  # 约 244 天（扣除周末和节假日）
print(trading_days[:10])  # 前 10 个交易日

# 输出：
# ['2024-01-02', '2024-01-03', '2024-01-04', ..., '2024-12-31']
```

**注意**：2024-01-01 是元旦，不是交易日，所以从 01-02 开始。

#### 2.5.5 在 Big-A 项目中使用 D 对象

```python
import qlib
from big_a.qlib_config import init_qlib

# 初始化 Qlib
init_qlib()

# 查询某只股票的数据
from qlib.data import D

df = D.features(
    instruments=["SH600519"],  # 贵州茅台
    fields=["$close", "$volume"],
    start_time="2024-01-01",
    end_time="2024-12-31"
)

print(df.tail())
```

---

### 2.6 LGBModel（LightGBM 模型）

#### 2.6.1 什么是 LGBModel

**LGBModel** 是 Qlib 内置的 LightGBM 模型封装。

**LightGBM** 是微软开源的梯度提升决策树（GBDT）框架，特点：
- 训练速度快
- 内存占用少
- 准确率高
- 支持分类和回归

**生活类比**：
- LightGBM 就像一个"超级学霸"
- 他看了大量历史数据（因子 → 收益）
- 学会了其中的规律
- 给他新的数据（今天的因子），他能预测明天的收益

#### 2.6.2 在 Qlib 中使用 LGBModel

```python
from qlib.contrib.model.gbdt import LGBModel
from qlib.data.dataset import DatasetH

# 创建数据集
dataset = DatasetH(
    handler={
        "class": "Alpha158",
        "module_path": "qlib.contrib.data.handler",
    },
    segments={
        "train": ("2020-01-01", "2022-12-31"),
        "valid": ("2023-01-01", "2023-12-31"),
        "test": ("2024-01-01", "2024-12-31"),
    }
)

# 创建模型
model = LGBModel(
    loss="mse",  # 均方误差（回归任务）
    colsample_bytree=0.8,
    learning_rate=0.05,
    max_depth=5,
    n_estimators=200,
    subsample=0.8,
    seed=42,
)

# 训练模型
model.fit(dataset)

# 预测
pred = model.predict(dataset)
print(pred)
```

#### 2.6.3 模型参数说明

| 参数 | 含义 | 常用值 | 说明 |
|------|------|--------|------|
| `loss` | 损失函数 | `mse` | 回归任务用 mse，分类任务用 logloss |
| `learning_rate` | 学习率 | 0.01-0.1 | 越小越稳健，但训练慢 |
| `max_depth` | 树的最大深度 | 3-8 | 越大越复杂，容易过拟合 |
| `n_estimators` | 树的数量 | 100-1000 | 越多越强，但训练慢 |
| `subsample` | 样本采样率 | 0.6-1.0 | 防止过拟合 |
| `colsample_bytree` | 特征采样率 | 0.6-1.0 | 防止过拟合 |

**参数调优建议**：
- 从简单参数开始：`max_depth=5, learning_rate=0.05, n_estimators=200`
- 用验证集调整参数
- 不要调得太细（容易过拟合）

---

### 2.7 TopkDropoutStrategy（选股策略）

#### 2.7.1 什么是 TopkDropoutStrategy

**TopkDropoutStrategy** 是 Qlib 内置的选股策略。

**策略逻辑**：
1. 获取所有股票的预测分数（信号）
2. 选出分数最高的 K 只股票
3. 按等权重或市值权重买入
4. 持有 N 天后卖出
5. 重复步骤 1-4

**生活类比**：
- 就像选"班长候选人"
1. 给每个学生打分（信号）
2. 选出分数最高的 10 个（Top 10）
3. 这 10 个就是候选人
4. 下个月重新评选

#### 2.7.2 在 Qlib 中使用 TopkDropoutStrategy

```python
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy

# 创建策略
strategy = TopkDropoutStrategy(
    topk=50,           # 选出信号最高的 50 只股票
    n_drop=5,          # 每次交易时随机丢弃 5 只（增加多样性）
    risk_degree=0.95,  # 风险控制（仓位控制）
)

# 执行策略
portfolio = strategy.generate_predict_from_model(
    model=model,
    dataset=dataset,
)
```

#### 2.7.3 策略参数说明

| 参数 | 含义 | 常用值 | 说明 |
|------|------|--------|------|
| `topk` | 选股数量 | 30-100 | 越少越集中，越多越分散 |
| `n_drop` | 随机丢弃数量 | 0-10 | 防止过拟合，增加多样性 |
| `risk_degree` | 风险度 | 0.8-1.0 | 控制仓位大小 |

**策略示例**：
- `topk=50, n_drop=5`：从 50 只中随机选 45 只
- `topk=30, n_drop=0`：固定选 30 只

---

### 2.8 backtest_daily()（日频回测）

#### 2.8.1 什么是 backtest_daily

**backtest_daily** 是 Qlib 内置的日频回测函数。

**功能**：
- 模拟真实交易
- 计算收益、风险指标
- 生成回测报告

#### 2.8.2 在 Qlib 中使用 backtest_daily

```python
from qlib.backtest import backtest
from qlib.contrib.evaluate import risk_analysis

# 执行回测
portfolio = backtest(
    strategy=strategy,
    executor=executor,  # 交易执行器
    start_time="2024-01-01",
    end_time="2024-12-31",
)

# 分析回测结果
report = risk_analysis(
    portfolio,
    return_type="pandas"
)

print(report)
```

**回测报告包含的指标**：

| 指标 | 含义 | 说明 |
|------|------|------|
| `return` | 总收益率 | (最终资金 - 初始资金) / 初始资金 |
| `annualized_return` | 年化收益率 | 转换为年度收益率 |
| `max_drawdown` | 最大回撤 | 最大的亏损幅度 |
| `sharpe_ratio` | 夏普比率 | 收益风险比（越高越好） |
| `information_ratio` | 信息比率 | 超额收益/跟踪误差 |

---

## 3. Qlib 数据格式

### 3.1 目录结构

Qlib 使用二进制格式存储数据，目录结构如下：

```
data/qlib_data/cn_data/
├── calendars/
│   └── day.txt                    # 交易日历（每行一个日期）
├── instruments/
│   └── all.txt                    # 股票列表（每行一个代码）
└── features/
    ├── SH600000/                  # 浦发银行
    │   ├── close.bday.bin         # 收盘价（按交易日）
    │   ├── open.bday.bin          # 开盘价
    │   ├── high.bday.bin          # 最高价
    │   ├── low.bday.bin           # 最低价
    │   ├── volume.bday.bin        # 成交量
    │   ├── amount.bday.bin        # 成交额
    │   └── vwap.bday.bin          # 成交量加权平均价
    ├── SH600001/                  # 邯郸钢铁
    │   ├── close.bday.bin
    │   └── ...
    ├── SZ000001/                  # 平安银行
    │   ├── close.bday.bin
    │   └── ...
    └── ...
```

### 3.2 文件说明

#### 3.2.1 calendars/day.txt

**内容**：交易日历，每行一个日期。

```
2020-01-02
2020-01-03
2020-01-06
...
2024-12-31
```

**作用**：定义哪些是交易日（排除周末和节假日）。

**读取代码**：

```python
from qlib.data import D

trading_days = D.calendar()
print(len(trading_days))  # 约 1200 天（5 年）
```

#### 3.2.2 instruments/all.txt

**内容**：股票列表，每行一个股票代码。

```
SH600000
SH600001
SH600004
...
SZ300750
```

**作用**：定义有哪些股票可用。

**读取代码**：

```python
from qlib.data import D

instruments = D.instruments()
print(len(instruments))  # 约 5000 只
```

#### 3.2.3 features/*.bday.bin

**内容**：股票特征数据，二进制格式。

**文件名规则**：
- `close.bday.bin`：收盘价（bday = business day，交易日）
- `open.bday.bin`：开盘价
- `volume.bday.bin`：成交量

**数据结构**：
- 每个文件是一个时间序列
- 按交易日历的顺序存储
- 用 32 位浮点数存储

**无法直接读取**：需要通过 Qlib 的 API 读取。

### 3.3 本项目数据来源

**Big-A 项目**使用的数据来自：

- **GitHub 仓库**：chenditc/investment_data
- **数据频率**：日频（daily）
- **数据时间**：约 2005 年至今
- **股票范围**：A 股全市场

**下载数据**：

```bash
# 项目根目录
cd /Volumes/data/documents/codes/stock-big-a

# 下载数据（如果还没下载）
python scripts/download_qlib_data.py
```

**数据位置**：

```
data/
└── qlib_data/
    └── cn_data/          # 中国市场数据
        ├── calendars/
        ├── instruments/
        └── features/
```

---

## 4. Qlib 初始化流程

### 4.1 基本初始化

使用 Qlib 之前，必须先初始化：

```python
import qlib

# 初始化 Qlib
qlib.init(
    provider_uri="data/qlib_data/cn_data",  # 数据目录
    region="cn",                            # 市场区域
    redis_cache_dir="data/qlib_data/redis",  # Redis 缓存目录（可选）
)
```

**参数说明**：

| 参数 | 含义 | 必填 | 说明 |
|------|------|------|------|
| `provider_uri` | 数据目录 | 是 | 存储 .bin 文件的目录 |
| `region` | 市场区域 | 是 | `cn`（中国）或 `us`（美国） |
| `redis_cache_dir` | Redis 缓存目录 | 否 | 加速数据读取 |

### 4.2 Big-A 项目的初始化封装

**Big-A 项目**将初始化封装在 `big_a.qlib_config.init_qlib()` 中：

```python
# big_a/qlib_config.py
from pathlib import Path
import qlib

def init_qlib():
    """初始化 Qlib"""
    # 项目根目录
    project_root = Path(__file__).parent.parent

    # 数据目录
    data_dir = project_root / "data" / "qlib_data" / "cn_data"

    # 初始化 Qlib
    qlib.init(
        provider_uri=str(data_dir),
        region="cn",
    )

    # 注册自定义算子（因子）
    _register_custom_ops()

    print(f"Qlib 初始化完成")
    print(f"数据目录：{data_dir}")
```

### 4.3 注册自定义算子

Qlib 允许注册自定义算子（自定义因子）：

```python
from qlib.contrib.ops import register_all_ops

# 注册所有内置算子
register_all_ops()

# 注册自定义算子
@qlib.register_op
class LimitStatus(qlib.Operator):
    """涨跌停状态"""
    def __init__(self):
        super().__init__()

    def __call__(self, close, high, low):
        # 计算涨跌停状态
        limit_up = (high - close.shift(1)) / close.shift(1) >= 0.095
        limit_down = (low - close.shift(1)) / close.shift(1) <= -0.095
        return limit_up.astype(float) - limit_down.astype(float)
```

**使用自定义算子**：

```python
# 在表达式中使用
expr = "LimitStatus($close, $high, $low)"

# 含义：
# 返回 1 表示涨停
# 返回 -1 表示跌停
# 返回 0 表示未涨跌停
```

### 4.4 在代码中调用初始化

```python
from big_a.qlib_config import init_qlib

# 初始化 Qlib
init_qlib()

# 现在可以使用 Qlib 的所有功能
from qlib.data import D
df = D.features(...)
```

---

## 5. Qlib 与本项目的关系

### 5.1 架构关系

```
┌─────────────────────────────────────────────────────────┐
│                    Big-A 项目                            │
│  (应用层：具体的量化交易策略和工具)                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ 数据加载     │  │ 因子计算     │  │ 模型训练     │  │
│  │ (基于 Qlib)  │  │ (基于 Qlib)  │  │ (基于 Qlib)  │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                         │
├─────────────────────────────────────────────────────────┤
│                    Qlib 框架                            │
│  (引擎层：提供基础功能和工具)                              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Provider     │  │ Expression   │  │ LGBModel     │  │
│  │ (数据存储)    │  │ Engine       │  │ (模型封装)    │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**类比**：
- Qlib 就像"汽车引擎"
- Big-A 就像"基于引擎制造的汽车"
- 你可以直接用引擎（Qlib），也可以直接开汽车（Big-A）

### 5.2 Big-A 项目使用 Qlib 的部分

| 功能 | Big-A 实现 | Qlib 提供 |
|------|------------|-----------|
| **数据加载** | 封装在 `big_a.qlib_config` | Provider |
| **因子计算** | Alpha158 + 自定义因子 | Expression Engine |
| **数据处理** | 自定义 DataHandler | DataHandlerLP |
| **模型训练** | LGBModel, Kronos | LGBModel |
| **策略回测** | 自定义策略 | backtest_daily |
| **绩效分析** | 自定义指标 | risk_analysis |

### 5.3 Big-A 项目额外添加的功能

Big-A 项目在 Qlib 的基础上，额外添加了：

#### 5.3.1 Kronos 深度学习模型

**Kronos** 是一个基于 Transformer 的时序预测模型。

**特点**：
- 使用 Attention 机制捕捉长期依赖
- 适合处理时间序列数据
- 比传统 GBDT 模型更强大

**代码位置**：`big_a/models/kronos.py`

#### 5.3.2 自定义因子

Big-A 添加了 Qlib 没有的因子：

- **VWAP**（成交量加权平均价）
- **VolumeRatio**（量比）
- **LimitStatus**（涨跌停状态）
- **AmihudIlliquidity**（非流动性指标）

**代码位置**：`big_a/qlib_config.py`

#### 5.3.3 完整流水线

Big-A 提供了从数据加载到回测的完整流水线：

```python
from big_a.pipeline import run_pipeline

# 运行完整流水线
results = run_pipeline(
    model_type="lgb",  # 或 "kronos"
    start_date="2024-01-01",
    end_date="2024-12-31",
)

# results 包含：
# - 预测结果
# - 回测报告
# - 绩效指标
```

#### 5.3.4 可视化工具

Big-A 提供了丰富的可视化工具：

- 收益曲线图
- 因子重要性图
- 持仓分析图
- 风险指标图

**代码位置**：`big_a/visualization/`

### 5.4 如何选择使用 Qlib 还是 Big-A

**使用 Qlib 直接开发**：
- 适合：想深入理解 Qlib 机制、需要高度自定义
- 缺点：需要写很多代码

**使用 Big-A 开发**：
- 适合：快速实现策略、初学者
- 优点：代码简洁、开箱即用

**建议**：
- 初学者：先用 Big-A，理解整体流程
- 进阶者：学习 Qlib，理解底层机制
- 高级者：基于 Qlib 开发自定义功能

---

## 6. 实战示例：用 Qlib 做一个简单策略

### 6.1 完整代码示例

```python
import qlib
from qlib.data import D
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.backtest import backtest
from qlib.contrib.evaluate import risk_analysis

# 1. 初始化 Qlib
qlib.init(
    provider_uri="data/qlib_data/cn_data",
    region="cn",
)

# 2. 准备数据
from qlib.data.dataset import DatasetH
from qlib.contrib.data.handler import Alpha158

dataset = DatasetH(
    handler={
        "class": "Alpha158",
        "module_path": "qlib.contrib.data.handler",
    },
    segments={
        "train": ("2020-01-01", "2022-12-31"),
        "valid": ("2023-01-01", "2023-12-31"),
        "test": ("2024-01-01", "2024-12-31"),
    }
)

# 3. 训练模型
model = LGBModel(
    loss="mse",
    learning_rate=0.05,
    max_depth=5,
    n_estimators=200,
)

model.fit(dataset)

# 4. 创建策略
strategy = TopkDropoutStrategy(
    topk=50,
    n_drop=5,
)

# 5. 执行回测
portfolio = backtest(
    strategy=strategy,
    executor=executor,
    start_time="2024-01-01",
    end_time="2024-12-31",
)

# 6. 分析结果
report = risk_analysis(portfolio)
print(report)
```

### 6.2 代码解释

**步骤 1：初始化 Qlib**
- 加载数据
- 注册算子

**步骤 2：准备数据**
- 创建数据集
- 划分训练/验证/测试集

**步骤 3：训练模型**
- 使用 LightGBM
- 用训练集训练，用验证集调参

**步骤 4：创建策略**
- 选出信号最高的 50 只股票
- 随机丢弃 5 只（防止过拟合）

**步骤 5：执行回测**
- 模拟 2024 年的交易
- 计算收益

**步骤 6：分析结果**
- 计算各种风险指标
- 评估策略表现

---

## 7. 总结

### 7.1 Qlib 核心概念回顾

| 概念 | 作用 | 类比 |
|------|------|------|
| Provider | 数据存储和读取 | 数据库 |
| Expression Engine | 因子计算 | 公式计算器 |
| DataHandler | 数据处理 | 厨师 |
| DatasetH | 数据集管理 | 图书馆管理员 |
| D 对象 | 数据访问接口 | SQL 查询 |
| LGBModel | 模型训练 | 超级学霸 |
| TopkDropoutStrategy | 选股策略 | 班长评选 |
| backtest_daily | 回测 | 模拟考试 |

### 7.2 Big-A 与 Qlib 的关系

- **Qlib** = 引擎（提供基础功能）
- **Big-A** = 汽车（基于引擎构建的应用）

Big-A 在 Qlib 的基础上添加了：
- Kronos 深度学习模型
- 自定义因子
- 完整流水线
- 可视化工具

### 7.3 下一步学习

1. **运行 Big-A 项目**：
   ```bash
   cd /Volumes/data/documents/codes/stock-big-a
   python examples/simple_strategy.py
   ```

2. **阅读 Qlib 官方文档**：
   - https://qlib.readthedocs.io/

3. **尝试自定义因子**：
   - 修改 `big_a/qlib_config.py`
   - 添加自己的因子

4. **尝试自定义模型**：
   - 参考 `big_a/models/kronos.py`
   - 实现自己的模型

5. **深入理解回测**：
   - 学习 `qlib.backtest` 模块
   - 理解交易成本和滑点

### 7.4 学习资源

- **Qlib GitHub**：https://github.com/microsoft/qlib
- **Qlib 文档**：https://qlib.readthedocs.io/
- **Qlib 论文**：Qlib: An AI-oriented Quantitative Investment Platform
- **Big-A 项目文档**：`docs/` 目录

---

恭喜你完成了 Qlib 框架的学习！现在你应该能够：
- 理解 Qlib 的核心概念
- 使用 Qlib 加载数据、计算因子
- 训练模型、执行回测
- 理解 Big-A 项目如何使用 Qlib

下一步，建议你：
1. 运行 Big-A 项目的示例代码
2. 尝试修改参数，观察结果变化
3. 实现一个简单的自定义策略

祝你在量化交易的道路上越走越远！
