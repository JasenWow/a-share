# 模型文档

本文档介绍 big-a 项目中支持的各种预测模型。

---

## LightGBM 模型

### 模型名称
LightGBM (Light Gradient Boosting Machine)

### 概述
LightGBM 是基于梯度提升决策树的高效机器学习模型，适用于处理结构化特征数据。在量化投资中，它常用于基于技术指标和基本面特征的股票收益率预测。

### 原理
- **梯度提升**：通过逐步拟合残差来提升模型性能
- **直方图算法**：将连续特征离散化为直方图，加速训练
- **叶子生长策略**：采用叶子-wise 策略，降低计算复杂度
- **特征并行与数据并行**：支持分布式训练

### 配置说明
配置文件位于 `configs/model/lgbm.yaml`，包含：
- 模型超参数（learning_rate, num_leaves, max_depth 等）
- 训练参数（early_stopping_rounds, n_estimators 等）
- 特征选择配置

### 使用示例
```python
from big_a.config import load_config
from big_a.models.lgbm_trainer import LGBMTrainer

# 加载配置
config = load_config("configs/model/lgbm.yaml", "configs/data/cn_stock.yaml")

# 训练模型
trainer = LGBMTrainer(config)
trainer.train()

# 预测
predictions = trainer.predict(test_data)
```

---

## Kronos 模型

### 模型名称
Kronos (时间序列 Transformer 模型)

### 概述
Kronos 是专为金融时间序列预测设计的 Transformer 模型，能够捕捉长期时间依赖关系。它使用因果注意力机制，避免信息泄露，适合股票价格和波动率预测。

### 原理
- **因果注意力机制**：确保预测时仅使用历史信息
- **位置编码**：增强模型对时间序列位置的理解
- **多头注意力**：从不同角度捕捉时间序列模式
- **前馈网络**：提取非线性特征

### 配置说明
配置文件位于 `configs/model/kronos.yaml`，包含：
- 模型配置（model_id, tokenizer_id）
- 推理参数（lookback, pred_len, max_context）
- 设备配置（device: cpu/gpu）
- 信号生成模式（signal_mode）

### 使用示例
```python
from big_a.config import load_config
from big_a.models.kronos_predictor import KronosPredictor

# 加载配置
config = load_config("configs/model/kronos.yaml", "configs/data/cn_stock.yaml")

# 初始化预测器
predictor = KronosPredictor(config)

# 生成预测信号
signals = predictor.predict(instruments=["000001.SZ", "600000.SH"])
```

---

## Hedge Fund 模型

### 模型名称
Hedge Fund (基于 LLM 的多智能体量化分析系统)

### 概述
Hedge Fund 模型是一个创新的大语言模型驱动的多智能体量化分析系统。它模拟顶级投资大师的思维模式，通过多个专业智能体并行分析市场，生成高质量的投资信号。该系统使用 ZhipuAI 的 GLM-4 作为核心推理引擎，结合 LangGraph 实现复杂的工作流编排。

### 原理
- **多智能体架构**：技术分析、估值分析、情绪分析等多个专业智能体并行工作
- **投资大师模拟**：集成巴菲特、芒格、林奇等 10 位传奇投资者的投资哲学
- **LangGraph 工作流**：使用有向图编排智能体间的协作流程
- **信号聚合**：通过加权平均或投票机制综合各智能体观点
- **风险控制**：内置仓位限制、止损阈值和置信度过滤

### 配置说明
配置文件位于 `configs/model/hedge_fund.yaml`，包含：

**LLM 配置**
- `model`: 使用的模型（默认：glm-4-flash）
- `base_url`: API 基础 URL
- `api_key_env`: API 环境变量名
- `temperature`: 温度参数（0.1，偏向确定性输出）
- `max_tokens`: 最大生成长度（2000）

**智能体配置**
- `technicals`: 技术分析智能体（enabled: true）
- `valuation`: 估值分析智能体（enabled: true）
- `sentiment`: 情绪分析智能体（enabled: true）
- `fundamentals`: 基本面分析智能体（enabled: false，因 Qlib 缺乏 A 股财务数据而暂缓）

**投资大师配置**
启用的投资大师列表：
- warren_buffett: 价值投资，寻找被低估的优质公司
- charlie_munger: 多学科思维，逆向思考
- peter_lynch: 成长投资，关注生活中熟悉的优秀公司
- ben_graham: 价值投资之父，安全边际
- phil_fisher: 成长股投资，注重管理层质量
- bill_ackman: 激进价值投资，公司治理
- michael_burry: 深度价值挖掘，市场异象
- nassim_taleb: 反脆弱，黑天鹅风险
- cathie_wood: 颠覆性创新，长期成长
- aswath_damodaran: 估值理论专家，基于数据的投资决策

**新闻配置**
- `source`: 新闻来源（zhipu_websearch，使用 ZhipuAI 的网络搜索能力）

**风险配置**
- `max_position_weight`: 单只股票最大仓位权重（0.25）
- `stop_loss_threshold`: 止损阈值（-0.08，即 -8%）
- `min_confidence`: 最小置信度（0.3）

**工作流配置**
- `signal_aggregation`: 信号聚合方式（weighted_average：加权平均）
- `parallel_agents`: 智能体并行执行（true）

### Agent 列表

| 智能体 | 角色 | 状态 |
|--------|------|------|
| Technicals | 技术分析：K线形态、技术指标、趋势判断 | 已启用 |
| Valuation | 估值分析：PE、PB、PEG 等估值指标评估 | 已启用 |
| Sentiment | 情绪分析：市场情绪、新闻舆情、投资者心理 | 已启用 |
| Fundamentals | 基本面分析：财务报表、盈利能力、成长性 | 已禁用（待数据支持） |

### 使用示例
```python
from big_a.config import load_config
from big_a.models.hedge_fund import HedgeFundSignalGenerator

# 加载配置
config = load_config("configs/model/hedge_fund.yaml", "configs/data/cn_stock.yaml")

# 初始化信号生成器
generator = HedgeFundSignalGenerator(config)

# 生成投资信号
signal = generator.generate_signal(
    instrument="000001.SZ",
    date="2025-04-26",
    context={
        "price_history": [...],
        "news": [...]
    }
)

print(f"信号: {signal.action}, 置信度: {signal.confidence:.2f}")
```

### 与其他模型对比

| 特性 | LightGBM | Kronos | Hedge Fund |
|------|----------|--------|------------|
| **模型类型** | 传统机器学习 | 深度学习（Transformer） | 大语言模型 |
| **训练数据需求** | 大量标注数据 | 大量时间序列数据 | 少量样本，依赖预训练 |
| **推理速度** | 极快 | 中等 | 较慢（API 调用） |
| **可解释性** | 中等（特征重要性） | 较低 | 高（自然语言解释） |
| **优势** | 高效、稳定、适合特征工程 | 捕捉长期依赖 | 综合分析、投资逻辑、适应性 |
| **适用场景** | 高频交易、大规模回测 | 中长期趋势预测 | 深度研究、投资决策支持 |
| **成本** | 低 | 中等 | 较高（API 费用） |
| **更新频率** | 定期重训 | 定期重训 | 实时（无训练） |

**选择建议**：
- **LightGBM**：适合高频策略、需要快速推理的场景
- **Kronos**：适合捕捉复杂时间模式、中短期趋势预测
- **Hedge Fund**：适合深度研究、需要投资逻辑解释、多维度综合分析的场景
