# 对冲基金多智能体系统

## 1. 概述

想象你是一位基金经理，面对一只陌生的股票。你不会只听一个人的意见对吧？你可能会：

- 听听技术分析师怎么说
- 看看估值专家的意见
- 参考一下其他投资者的看法
- 让风险管理员评估一下整体风险
- 最后，综合所有意见做出决策

**这就是对冲基金多智能体系统的工作方式。**

它不是一个单一的人工智能，而是一个"智能体团队"。每个智能体就像一位虚拟的投资专家，有自己独特的分析视角和投资风格。它们同时分析同一只股票，然后汇总意见，最终给出交易建议。

### 为什么需要多个智能体？

单一模型有局限性。就像你不会只听一个人的建议就做出重大投资决策一样，单一 AI 模型也可能存在偏见或盲点。

多智能体系统的好处：

1. **多角度分析**：不同智能体从不同角度审视同一只股票，减少盲点
2. **风格互补**：有的智能体擅长价值投资，有的擅长成长投资，有的擅长技术分析
3. **风险分散**：不会因为某个智能体判断失误就全军覆没
4. **集体智慧**：就像巴菲特说的"多元思维模型"，综合不同框架的分析更可靠

### 它和 LightGBM、Kronos 有什么关系？

Big-A 项目有三种模型：

| 模型 | 类型 | 类比 |
|------|------|------|
| LightGBM | 传统机器学习 | 经验丰富的老股民 |
| Kronos | 深度学习 | 看过海量K线的超级AI |
| HedgeFund | 多智能体LLM | 一个投资团队开会讨论 |

HedgeFund 与前两者不同，它不依赖历史数据训练，而是让大语言模型（LLM）直接分析股票，给出投资建议。

## 2. 系统架构

### 2.1 整体流程

系统采用 **fan-out/fan-in** 模式，就像一场会议：

```
                    ┌─────────────────┐
                    │      start       │
                    └────────┬─────────┘
                             │
            ┌────────────────┼────────────────┐
            ▼                ▼                ▼
    ┌───────────┐    ┌───────────┐    ┌───────────┐
    │ 技术分析  │    │  估值分析  │    │  情绪分析  │   ... 13个智能体并行
    │  智能体   │    │   智能体   │    │   智能体   │
    └─────┬─────┘    └─────┬─────┘    └─────┬─────┘
          │                │                │
          └────────────────┼────────────────┘
                           ▼
                  ┌─────────────────┐
                  │    风险管理员    │
                  │  (汇总信号，    │
                  │   评估风险)      │
                  └────────┬─────────┘
                           ▼
                  ┌─────────────────┐
                  │    投资组合经理   │
                  │  (综合所有意见， │
                  │   最终决策)      │
                  └────────┬─────────┘
                           ▼
                         END
```

**三个阶段：**

1. **并行分析阶段**：13个智能体同时分析股票，各自给出信号
2. **风险评估阶段**：风险管理员汇总所有信号，评估整体风险
3. **决策阶段**：投资组合经理综合风险评估和各智能体意见，做出最终决策

### 2.2 为什么这样设计？

**并行分析**的好处：

- 节省时间：13个智能体同时工作，总耗时等于最慢的那个
- 独立判断：每个智能体不受其他智能体影响，保持独立思考
- 风格多样：同时获取价值投资、成长投资、技术分析等不同风格的建议

**风险管理员**的作用：

- 汇总分散的信号，变成统一的判断
- 识别高风险情况，降低亏损概率
- 根据信号一致性调整置信度

**投资组合经理**的作用：

- 综合所有信息，做出最终交易决策
- 给出具体的分数（用于量化回测）
- 决定是买、卖还是持有

### 2.3 状态管理

系统使用 LangGraph 的状态图来管理信息流转：

```python
HedgeFundState:
  messages: list        # 对话历史
  data: dict            # 分析数据（智能体信号、风险评估等）
  metadata: dict        # 元数据（股票代码、日期、配置等）
```

不同智能体从不同位置读取数据：

- `technicals_agent` 和 `valuation_agent` 从 `state["data"]` 读取
- 投资者智能体（巴菲特、芒格等）从 `state["metadata"]` 读取
- `risk_manager` 和 `portfolio_manager` 汇总所有信息

## 3. 智能体详解

系统包含 **13个智能体**，分为三类：

### 3.1 技术分析智能体

| 智能体 | 职责 | 分析重点 |
|--------|------|----------|
| technicals_agent | 技术分析 | 趋势、动量、均值回归、波动率 |
| valuation_agent | 估值分析 | 价格位置、成交量加权、动量指标 |
| sentiment_agent | 情绪分析 | 新闻舆情、市场情绪 |

**technicals_agent** 就像一个认真看盘的技术分析师。它计算：

- EMA（指数移动平均线）：8日、21日、55日
- RSI（相对强弱指标）：14日
- MACD（异同移动平均线）
- 布林带：20日均线加减2倍标准差
- ADX（平均方向性指数）：趋势强度
- ATR（平均真实波幅）：波动率

然后综合这些指标，判断是看涨、看跌还是中性。

**valuation_agent** 关注的是价格相对位置：

- 当前价格 vs MA5、MA20、MA60
- 价格在历史区间中的位置（百分位）
- 成交量加权平均价（VWAP）
- 短期动量变化

**sentiment_agent** 负责分析新闻舆情。它获取股票相关的新闻，然后判断整体情绪是正面、负面还是中性。

### 3.2 投资者智能体（10位大师）

这10个智能体模拟了真实世界著名的投资大师，每个都有独特的分析框架：

| 智能体 | 人物 | 投资风格 |
|--------|------|----------|
| warren_buffett_agent | 沃伦·巴菲特 | 价值投资，寻找护城河 |
| charlie_munger_agent | 查理·芒格 | 品质投资，看重商业模式 |
| peter_lynch_agent | 彼得·林奇 | 成长投资，关注公司故事 |
| ben_graham_agent | 本杰明·格雷厄姆 | 价值投资之父，看重安全边际 |
| phil_fisher_agent | 菲利普·费雪 | 成长投资，看重管理质量 |
| bill_ackman_agent | 比尔·阿克曼 | 激进投资，看重品牌护城河 |
| michael_burry_agent | 迈克尔·伯里 | 反向投资，擅长发现泡沫 |
| nassim_taleb_agent | 纳西姆·塔勒布 | 反脆弱，关注尾部风险 |
| cathie_wood_agent | 凯西·伍德 | 创新投资，看重颠覆性创新 |
| aswath_damodaran_agent | 阿斯瓦特·达莫达兰 | 估值大师，DCF分析 |

**每个投资者智能体的工作方式类似：**

1. 获取股票的基础数据（价格、成交量、涨跌幅）
2. 用自己的分析框架生成 prompt
3. 调用 LLM 获取分析结论
4. 输出信号（bullish/bearish/neutral）和置信度

例如，巴菲特智能体的 prompt 会关注：

- 公司是否有持久的竞争优势（护城河）
- 盈利能力是否稳定且有增长潜力
- 管理层是否值得信赖
- 当前价格是否低于内在价值（安全边际）
- 与沪深300相比是否被低估

而塔勒布智能体则会更关注：

- 公司能否承受极端市场情况
- 是否有"黑天鹅"风险
- 收益是否非对称（上行空间大，下行风险有限）

### 3.3 风险管理智能体

**risk_manager_agent** 是团队的"风控官"。

它的职责是：

1. **汇总信号**：收集所有分析师的信号
2. **计算风险指标**：
   - 看涨/看跌/中性信号的数量
   - 平均置信度
   - 信号一致性比例
3. **生成风险评估**：
   - `adjusted_signal`：调整后的信号
   - `confidence`：风险置信度
   - `max_position_weight`：建议的最大仓位

**风险评估规则（A股市场）：**

- 高一致性（>70%）：可以较高仓位
- 中等一致性（40%-70%）：谨慎建仓
- 低一致性（<40%）：观望为主
- 混合信号：倾向于中性

### 3.4 投资组合管理智能体

**portfolio_manager_agent** 是"基金经理"，做最终决策。

它的输入：

- risk_manager 的风险评估
- 所有分析师的信号

它的输出：

- `action`：buy / sell / hold
- `score`：量化分数 [-1, 1]
- `reasoning`：决策理由

**决策原则：**

- 风险优先：风控建议是核心考量
- 仓位控制：score 的绝对值不超过 `max_position_weight`
- 信号一致性：分析师意见与风险评估一致时，增强信心
- 动态调整：
  - bullish → 正分数（0.5 到 1.0）
  - bearish → 负分数（-0.5 到 -1.0）
  - neutral → 接近 0（-0.2 到 0.2）

## 4. 信号流转

### 4.1 完整的信号流程

```
股票代码: SH600000
日期范围: 2024-01-01 至 2024-12-31

┌─────────────────────────────────────────────────────────────┐
│ 第一阶段：并行分析                                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  technicals_agent → bullish (0.8)                            │
│  valuation_agent   → bullish (0.7)                           │
│  sentiment_agent   → bullish (0.6)                          │
│  warren_buffett    → bullish (0.85)                         │
│  charlie_munger    → bullish (0.75)                         │
│  peter_lynch       → bullish (0.9)   ← 10个投资者智能体并行  │
│  ben_graham        → neutral (0.5)                          │
│  phil_fisher       → bullish (0.8)                          │
│  bill_ackman       → bullish (0.7)                          │
│  michael_burry     → neutral (0.4)                          │
│  nassim_taleb      → bearish (0.3)                          │
│  cathie_wood       → bullish (0.85)                         │
│  aswath_damodaran  → bullish (0.75)                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 第二阶段：风险评估                                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  risk_manager 计算:                                          │
│  - 看涨: 9个, 看跌: 1个, 中性: 3个                            │
│  - 平均置信度: 0.68                                          │
│  - 一致性比例: 69%                                          │
│                                                              │
│  输出:                                                       │
│  - adjusted_signal: bullish                                  │
│  - confidence: 0.75                                         │
│  - max_position_weight: 0.6                                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 第三阶段：最终决策                                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  portfolio_manager 综合所有信息:                              │
│  - 多数分析师看涨                                            │
│  - 风险评估支持 bullish                                       │
│  - 一致性较高                                                │
│                                                              │
│  输出:                                                       │
│  - action: buy                                              │
│  - score: 0.55                                              │
│  - reasoning: "多数分析师看好，风险评估支持..."               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 输出数据格式

最终输出是一个 DataFrame，格式符合 Qlib 标准：

```python
import pandas as pd

# 假设我们分析了两只股票
signals = gen.generate_signals(
    instruments=["SH600000", "SZ000001"],
    start_date="2024-01-01",
    end_date="2024-12-31",
)

print(signals)
#                           score
# datetime    instrument
# 2024-12-31  SH600000    0.55
# 2024-12-31  SZ000001   -0.32
```

如果需要获取每个智能体的详细分析，可以设置 `return_details=True`：

```python
result = gen.generate_signals(
    instruments=["SH600000"],
    start_date="2024-01-01",
    end_date="2024-12-31",
    return_details=True,
)

# result 包含:
# - signals: DataFrame（和上面一样）
# - details: dict，每个股票的各智能体信号

print(result["details"]["SH600000"]["warren_buffett_agent"])
# {'signal': 'bullish', 'confidence': 0.85, 'reasoning': '...'}
```

## 5. 使用示例

### 5.1 基础用法

```python
from big_a.models.hedge_fund import HedgeFundSignalGenerator

# 创建信号生成器（使用默认配置）
gen = HedgeFundSignalGenerator()

# 生成信号
signals = gen.generate_signals(
    instruments=["SH600000", "SZ000001"],
    start_date="2024-01-01",
    end_date="2024-12-31",
)

print(signals)
#                           score
# datetime    instrument
# 2024-12-31  SH600000    0.55
# 2024-12-31  SZ000001   -0.32
```

### 5.2 获取详细分析

```python
result = gen.generate_signals(
    instruments=["SH600000"],
    start_date="2024-01-01",
    end_date="2024-12-31",
    return_details=True,
)

# 查看巴菲特的分析
buffett_signal = result["details"]["SH600000"]["warren_buffett_agent"]
print(f"信号: {buffett_signal['signal']}")
print(f"置信度: {buffett_signal['confidence']}")
print(f"理由: {buffett_signal['reasoning']}")
```

### 5.3 使用完整模式（13个智能体）

```python
config = {"mode": "full"}

gen = HedgeFundSignalGenerator(config=config)

signals = gen.generate_signals(
    instruments=["SH600000"],
    start_date="2024-01-01",
    end_date="2024-12-31",
)
```

### 5.4 配置 LLM

默认使用智谱 GLM（ZhipuAI），也可以使用 OpenRouter：

```python
config = {
    "mode": "full",
    "llm": {
        "provider": "openrouter",
        "api_key": "your-api-key",
        "model": "google/gemma-4-31b-it:free",
        "temperature": 0.1,
        "max_tokens": 2000,
    }
}

gen = HedgeFundSignalGenerator(config=config)
```

### 5.5 在回测中使用

由于输出格式兼容 Qlib，可以直接用于回测：

```python
from big_a.backtest import backtest

# 生成信号
signals = gen.generate_signals(
    instruments=["SH600000", "SZ000001"],
    start_date="2024-01-01",
    end_date="2024-12-31",
)

# 用于回测（需要配置 Qlib 和其他参数）
result = backtest(signals, config=backtest_config)
```

## 6. 配置说明

### 6.1 基本配置

```python
config = {
    # 运行模式
    "mode": "slim",  # "slim"（3个智能体）或 "full"（13个智能体）

    # LLM 配置
    "llm": {
        "provider": "zhipu",  # "zhipu" 或 "openrouter"
        "api_key": "your-api-key",
        "model": "glm-4-flash",
        "temperature": 0.1,
        "max_tokens": 2000,
    }
}
```

### 6.2 环境变量

使用智谱 API 时，需要设置环境变量：

```bash
export ZHIPU_API_KEY="your-api-key"
```

使用 OpenRouter 时：

```bash
export OPENROUTER_API_KEY="your-api-key"
```

### 6.3 slim 模式 vs full 模式

**slim 模式**（默认）：

- 只使用 3 个智能体：technicals_agent、valuation_agent、warren_buffett_agent
- 速度快，成本低
- 适合快速测试和开发

**full 模式**：

- 使用全部 13 个智能体
- 速度慢，成本高
- 适合最终决策或研究

```python
# slim 模式（默认）
gen = HedgeFundSignalGenerator()

# full 模式
gen = HedgeFundSignalGenerator({"mode": "full"})
```

## 7. 分数解读

### 7.1 为什么是 [-1, 1]？

这个设计是为了与 Qlib 回测系统兼容。

| 分数范围 | 含义 | 操作建议 |
|----------|------|----------|
| 0.5 到 1.0 | 强烈看涨 | 买入或持有 |
| 0.2 到 0.5 | 轻度看涨 | 谨慎买入 |
| -0.2 到 0.2 | 中性 | 观望 |
| -0.5 到 -0.2 | 轻度看跌 | 谨慎卖出 |
| -1.0 到 -0.5 | 强烈看跌 | 卖出或做空 |

### 7.2 分数的限制

分数不会超过风险管理员设定的 `max_position_weight`。

例如：

- 如果 `max_position_weight = 0.6`
- 即使所有分析师都强烈看涨
- 最终分数也不会超过 0.6

这是系统的保护机制，防止过度乐观。

### 7.3 与其他模型对比

| 模型 | 输出范围 | 数据来源 |
|------|----------|----------|
| LightGBM | [-1, 1] | Alpha158 + 自定义因子 |
| Kronos | [-1, 1] | 原始 OHLCV 数据 |
| HedgeFund | [-1, 1] | LLM 分析（无结构化数据） |

三者可以互补，用于 ensemble（集成）策略。

## 8. 总结

对冲基金多智能体系统是一个**模拟投资团队决策过程**的系统。

**核心特点：**

1. **多智能体并行**：13个智能体同时分析，取长补短
2. **大语言模型驱动**：每个智能体都是一个 LLM，可以进行复杂的推理
3. **风险控制**：独立的风险管理员，防止过度冒险
4. **Qlib 兼容**：输出格式标准化，可以直接用于回测

**适用场景：**

- 需要理解投资逻辑的场景
- 需要详细分析报告的场景
- 需要综合多角度意见的场景

**不适用场景：**

- 需要毫秒级响应的场景（LLM 调用较慢）
- 批量处理大量股票的场景（成本较高）
- 完全自动化的交易场景（建议结合其他模型使用）

**使用建议：**

1. 开发阶段用 slim 模式（省钱）
2. 最终决策用 full 模式（更全面）
3. 结合 LightGBM 和 Kronos 使用（更可靠）
4. 始终关注风险，不盲目依赖模型输出

记住，系统只是工具，真正的投资智慧来自你自己的判断。