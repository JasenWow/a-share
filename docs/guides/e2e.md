# 端到端流程指南

## 什么是端到端流程？

想象你要做一顿完整的晚餐：
1. 去菜市场买菜
2. 洗菜、切菜
3. 调味、炒菜
4. 摆盘、上菜

如果你每次都要手动执行这些步骤，会很麻烦。如果有一个"一键晚餐"按钮，按一下就自动完成所有步骤，那就省事多了。

端到端流程（End-to-End Pipeline，简称 E2E）就是这样一个"一键完成"的流程：
- 从原始数据开始
- 训练模型
- 生成预测
- 运行回测
- 生成报告
- 对比模型

所有步骤自动化，不需要手动干预。

---

## 为什么推荐用端到端？

### 1. 避免遗漏步骤

手动操作时，容易忘掉某些步骤：
- 训练了模型，但忘了记录参数
- 做了回测，但忘了保存结果
- 生成了预测，但忘了做对比

端到端流程保证所有步骤都按顺序执行，不会遗漏。

### 2. 所有结果自动记录到 MLflow

MLflow 是一个机器学习实验跟踪工具。端到端流程会自动记录：
- 模型参数
- 训练数据版本
- 评估指标（IC、Sharpe 等）
- 模型文件
- 预测结果

这样你可以：
- 随时查看历史实验
- 对比不同实验的效果
- 复现最好的结果

### 3. 一键对比两个模型

端到端流程默认会同时运行 LightGBM 和 Kronos 两个模型，并自动对比：
- IC 对比
- Rank IC 对比
- ICIR 对比

不需要手动写对比代码，结果自动生成。

---

## 9 个步骤详解

### 步骤 1：数据验证

**做了什么**：检查数据目录是否存在

```python
# 确保数据目录存在
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"数据目录不存在: {data_dir}")
```

**为什么需要**：
- 避免后面运行到一半才发现数据缺失
- 提前报错，节省时间

**类比**：做饭前先检查食材够不够，不要切到一半发现没肉了。

---

### 步骤 2：Qlib 初始化

**做了什么**：初始化 Qlib 量化框架

```python
import qlib
qlib.init(provider_uri=data_dir, region="cn")
```

**为什么需要**：
- Qlib 是微软开源的量化投资平台
- 提供数据加载、因子计算、回测等基础功能
- 所有模块都依赖 Qlib 的数据格式

**类比**：打开厨房的燃气和水电，这是做饭的基础设施。

---

### 步骤 3：训练 LightGBM（或加载已有模型）

**做了什么**：
- 如果 `--skip-train` 不指定，则训练新的 LightGBM 模型
- 如果 `--skip-train` 指定，则加载已有的 `lightgbm_model.pkl`

```python
if not skip_train:
    model = LGBModel()
    model.train(train_data)
    model.save('lightgbm_model.pkl')
else:
    model = LGBModel.load('lightgbm_model.pkl')
```

**为什么需要**：
- 模型是预测的核心
- 训练可能需要 10-30 分钟，可以跳过节省时间

**类比**：如果是第一次做这道菜，需要按照菜谱一步步做。如果已经做过很多次，可以直接用之前的经验。

---

### 步骤 4：在测试集上生成预测

**做了什么**：
- 用训练好的模型在测试集上生成预测
- 保存预测结果到 `lightgbm_predictions.parquet`

```python
predictions = model.predict(test_data)
predictions.to_parquet('lightgbm_predictions.parquet')
```

**为什么需要**：
- 预测结果是回测的输入
- 保存下来可以重复使用

**类比**：菜做好了，先尝一口味道，确认没问题再正式上菜。

---

### 步骤 5：运行回测（使用 TopkDropout）

**做了什么**：
- 基于预测结果构建投资组合
- 使用 TopkDropout 策略（选择预测分数最高的股票）
- 模拟交易，计算收益和风险
- 生成回测报告

```python
from big_a.backtest.strategy import TopkDropout

strategy = TopkDropout(top_k=50, drop_num=5)
backtest_report = strategy.run(predictions, test_data)
backtest_report.to_parquet('backtest_report.parquet')
```

**为什么需要**：
- 验证预测在交易中的实际效果
- 评估收益、回撤、换手率等

**类比**：正式开始吃这顿饭，看看味道如何，有没有吃坏肚子。

---

### 步骤 6：生成分析报告

**做了什么**：生成多种可视化图表和统计报告

```python
from big_a.analysis import generate_report

generate_report(backtest_report, output_dir='analysis/')
```

生成的文件：
- `analysis/summary.txt`：文字摘要（净值、收益、最大回撤等）
- `analysis/nav.png`：净值曲线图
- `analysis/drawdown.png`：回撤曲线图
- `analysis/monthly_returns.png`：月度收益热力图

**为什么需要**：
- 图表比数字更直观
- 快速了解策略表现

**类比**：吃完饭后，服务员给你一张账单和一份评价表，让你知道这顿饭花多少钱、味道如何。

---

### 步骤 7：Kronos 滚动信号生成

**做了什么**：
- 对测试集的每个日期、每只股票
- 调用 Kronos 模型生成信号
- 保存到 `kronos_predictions.parquet`

```python
from big_a.models.kronos import KronosModel

kronos = KronosModel()
kronos_predictions = kronos.predict(test_data)
kronos_predictions.to_parquet('kronos_predictions.parquet')
```

**为什么需要**：
- Kronos 是大语言模型，预测方式和 LightGBM 不同
- 需要单独生成预测结果

**类比**：除了主菜，再做一个配菜，看看哪个更好吃。

---

### 步骤 8：Kronos 回测

**做了什么**：
- 用同样的 TopkDropout 策略
- 基于 Kronos 的预测进行回测
- 生成 `kronos_backtest_report.parquet`

```python
kronos_backtest = strategy.run(kronos_predictions, test_data)
kronos_backtest.to_parquet('kronos_backtest_report.parquet')
```

**为什么需要**：
- 公平对比两个模型（使用相同的回测策略）
- 评估哪个模型更好

**类比**：用同样的标准和评分，对比主菜和配菜哪个更好吃。

---

### 步骤 9：模型对比

**做了什么**：
- 对比 LightGBM 和 Kronos 的 IC、Rank IC、ICIR
- 生成对比表格

```python
from big_a.evaluation.metrics import compare_models

comparison = compare_models(
    {
        'lightgbm': lightgbm_predictions,
        'kronos': kronos_predictions,
    },
    test_data['returns'],
)
print(comparison)
```

**为什么需要**：
- 量化地判断哪个模型更好
- 决定后续使用哪个模型

**类比**：吃完饭后，总结一下主菜和配菜哪个更值得做，下次就多做那个。

---

## 输出文件说明

### 模型文件

| 文件 | 说明 | 用途 |
|------|------|------|
| `lightgbm_model.pkl` | LightGBM 模型参数 | 后续加载使用，跳过训练 |

### 预测文件

| 文件 | 说明 | 用途 |
|------|------|------|
| `lightgbm_predictions.parquet` | LightGBM 的预测结果 | 回测、分析、对比 |
| `kronos_predictions.parquet` | Kronos 的预测结果 | 回测、分析、对比 |

### 回测文件

| 文件 | 说明 | 用途 |
|------|------|------|
| `backtest_report.parquet` | LightGBM 回测报告 | 生成分析报告 |
| `kronos_backtest_report.parquet` | Kronos 回测报告 | 生成分析报告 |
| `positions.pkl` | 持仓详情 | 分析具体持仓情况 |

### 分析报告

| 文件/目录 | 说明 | 内容 |
|-----------|------|------|
| `analysis/summary.txt` | 文字摘要 | 净值、收益、回撤等关键指标 |
| `analysis/nav.png` | 净值曲线 | 投资组合净值随时间的变化 |
| `analysis/drawdown.png` | 回撤曲线 | 从高点到低点的跌幅 |
| `analysis/monthly_returns.png` | 月度收益 | 每个月的收益热力图 |
| `kronos_analysis/` | Kronos 分析 | 和 LightGBM 相同的分析文件 |

---

## 常用选项

### 完整运行

```bash
uv run python scripts/e2e.py
```

这会执行所有 9 个步骤，包括：
- 训练 LightGBM 模型
- 生成两个模型的预测
- 运行回测
- 生成分析报告
- 对比模型

预计时间：
- 训练：10-30 分钟
- 预测：5-10 分钟
- 回测：2-5 分钟
- 总计：约 20-50 分钟

---

### 跳过训练（用已有模型）

```bash
uv run python scripts/e2e.py --skip-train
```

这会：
- 加载已有的 `lightgbm_model.pkl`
- 跳过步骤 3（训练）
- 其他步骤不变

**适用场景**：
- 已经训练过模型，只是想重新回测
- 调整了回测参数，但不需要重新训练
- 节省时间，快速验证想法

---

### 只跑 LightGBM

```bash
uv run python scripts/e2e.py --skip-kronos
```

这会：
- 跳过步骤 7-9（Kronos 相关）
- 只执行 LightGBM 的完整流程

**适用场景**：
- 只关心 LightGBM 的表现
- Kronos API 不可用或超时
- 快速验证想法

---

### 完整运行并指定数据目录

```bash
uv run python scripts/e2e.py --data-dir /path/to/data
```

默认数据目录是 `data/`，如果数据在其他位置，需要指定。

---

### 只生成预测，不做回测

如果你想只生成预测，不做回测，可以修改代码或单独运行预测脚本：

```bash
uv run python scripts/predict.py --model lightgbm
```

---

## 如何阅读结果

### 1. 先看 summary.txt

打开 `analysis/summary.txt`，你会看到类似这样的内容：

```
===== 策略回测摘要 =====

时间范围: 2020-01-01 至 2024-12-31
初始资金: 1000000
最终净值: 1500000

收益指标:
- 总收益: 50.00%
- 年化收益: 10.68%
- 基准收益: 20.00%
- 超额收益: 30.00%

风险指标:
- 年化波动率: 15.00%
- 最大回撤: -18.50%
- Sharpe 比率: 1.25
- Information Ratio: 0.89

交易指标:
- 换手率: 50.00%
- 交易次数: 250
- 平均持仓天数: 10

评估指标:
- IC: 0.045
- Rank IC: 0.062
- ICIR: 0.89
```

**关注重点**：
- **年化收益 > 基准收益**：说明跑赢了市场
- **Sharpe > 1.0**：收益匹配风险
- **最大回撤 < 20%**：亏得不会太惨
- **IC > 0.03**：预测方向正确

---

### 2. 看净值曲线（nav.png）

净值曲线告诉你：
- 策略是否持续赚钱
- 什么时候表现好，什么时候表现差
- 和基准（比如沪深 300）相比如何

好的净值曲线应该：
- 整体向上
- 波动相对平稳
- 没有突然的大幅下跌

---

### 3. 看回撤曲线（drawdown.png）

回撤曲线告诉你：
- 历史上最惨的时候亏了多少
- 回撤持续了多久
- 是否经常出现大回撤

好的回撤曲线应该：
- 最大回撤不超过 20%
- 回撤恢复时间不要太长（比如不超过 6 个月）
- 不要频繁出现深度回撤

---

### 4. 看月度收益（monthly_returns.png）

月度收益热力图告诉你：
- 哪些月份表现好，哪些月份表现差
- 是否有季节性规律
- 是否连续亏损

好的月度收益应该：
- 红色（正收益）多于绿色（负收益）
- 不要有大片的深绿色（大亏损）
- 亏损后能快速反弹

---

### 5. 看模型对比

最后的模型对比表格会告诉你哪个模型更好：

| 模型 | IC | Rank IC | ICIR |
|------|----|---------|------|
| LightGBM | 0.045 | 0.062 | 0.89 |
| Kronos | 0.038 | 0.055 | 0.75 |

从这个表格可以看出：
- LightGBM 在所有指标上都优于 Kronos
- 如果只能选一个，选 LightGBM

但也要考虑：
- Kronos 是大语言模型，可能有独特优势
- 两个模型可以结合使用（比如取平均）

---

## 完整示例

```python
# scripts/e2e.py 的核心逻辑

import os
import argparse
from big_a.models.lgb_model import LGBModel
from big_a.models.kronos import KronosModel
from big_a.backtest.strategy import TopkDropout
from big_a.evaluation.metrics import compare_models
from big_a.analysis import generate_report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-train', action='store_true')
    parser.add_argument('--skip-kronos', action='store_true')
    parser.add_argument('--data-dir', default='data')
    args = parser.parse_args()

    # 步骤 1: 数据验证
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"数据目录不存在: {args.data_dir}")

    # 步骤 2: Qlib 初始化
    import qlib
    qlib.init(provider_uri=args.data_dir, region="cn")

    # 步骤 3: 训练 LightGBM
    if not args.skip_train:
        lgb_model = LGBModel()
        lgb_model.train(train_data)
        lgb_model.save('lightgbm_model.pkl')
    else:
        lgb_model = LGBModel.load('lightgbm_model.pkl')

    # 步骤 4: 生成预测
    lgb_predictions = lgb_model.predict(test_data)
    lgb_predictions.to_parquet('lightgbm_predictions.parquet')

    # 步骤 5: 回测
    strategy = TopkDropout(top_k=50, drop_num=5)
    lgb_backtest = strategy.run(lgb_predictions, test_data)
    lgb_backtest.to_parquet('backtest_report.parquet')

    # 步骤 6: 生成分析报告
    generate_report(lgb_backtest, output_dir='analysis/')

    if not args.skip_kronos:
        # 步骤 7: Kronos 预测
        kronos = KronosModel()
        kronos_predictions = kronos.predict(test_data)
        kronos_predictions.to_parquet('kronos_predictions.parquet')

        # 步骤 8: Kronos 回测
        kronos_backtest = strategy.run(kronos_predictions, test_data)
        kronos_backtest.to_parquet('kronos_backtest_report.parquet')
        generate_report(kronos_backtest, output_dir='kronos_analysis/')

        # 步骤 9: 模型对比
        comparison = compare_models(
            {
                'lightgbm': lgb_predictions,
                'kronos': kronos_predictions,
            },
            test_data['returns'],
        )
        print(comparison)

if __name__ == '__main__':
    main()
```

---

## 总结

端到端流程是量化交易的"一键操作"，它能：
- 自动化所有步骤，避免遗漏
- 统一记录所有结果
- 快速对比多个模型

记住：
- 第一次运行需要训练模型，时间较长
- 后续可以跳过训练，节省时间
- 先看 summary.txt，再详细看图表
- 不仅看绝对表现，还要看和基准的对比

善用端到端流程，让你的量化研究更高效、更可靠。
