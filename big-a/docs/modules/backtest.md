# 回测模块文档

本文档介绍 big-a 项目中支持的各种回测策略。

---

## 1. TopkDropout 策略

### 策略名称
TopkDropout（前K名择优策略）

### 概述
TopkDropout 是基于信号选股的经典回测策略，它选择预测得分最高的 K 只股票进行投资，并定期淘汰表现最差的 N 只股票。

### 原理
- **信号驱动选股**：根据模型的预测信号（收益率）选择得分最高的 K 只股票
- **定期调仓**：按照设定的频率（如每日、每周）进行调仓
- **淘汰机制**：每次调仓时淘汰表现最差的 N 只股票
- **风险控制**：通过 topk 和 n_drop 参数控制投资组合的规模和稳定性

### 配置说明
配置文件位于 `configs/backtest/topk_dropout.yaml`，包含：
- **strategy**：策略类配置（class, module_path, kwargs）
- **signal**：预测信号源（<PRED> 表示使用预测结果）
- **topk**：选股数量（如 5，表示选择得分最高的 5 只股票）
- **n_drop**：每次淘汰的股票数量（如 1）
- **risk_degree**：风险度（0.95 表示 95% 仓位）

### 使用示例
```python
from big_a.strategy.topk_dropout import TopkDropoutStrategy
from big_a.config import load_config

# 加载配置
config = load_config("configs/backtest/topk_dropout.yaml")

# 初始化策略
strategy = TopkDropoutStrategy(
    signal=<PRED>,
    topk=5,
    n_drop=1,
    risk_degree=0.95
)

# 生成交易决策
trade_decision = strategy.generate_trade_decision()
```

---

## 2. 实盘交易策略（RealTradingStrategy）

### 策略名称
RealTradingStrategy（带风控的实盘交易策略）

### 概述
RealTradingStrategy 是在 TopkDropout 策略基础上，增加了止损、周频调仓、仓位上限等风控机制的实盘友好型策略。它专门为小资金实盘交易设计，通过严格的风险控制降低回撤，提高实盘的稳定性。

### 原理
RealTradingStrategy 的核心机制包括三个阶段：

**第一阶段：每日止损检查**
- 每个交易日都检查所有持仓股票的盈亏情况
- 计算每只股票的盈亏比例：(当前价格 - 买入价格) / 买入价格
- 如果某只股票的亏损达到止损阈值（默认 -8%），立即全部卖出
- 卖出后从持仓中移除该股票，并清除其买入价格记录

**第二阶段：定期调仓（每周）**
- 按照设定的调仓频率（默认 5 个交易日，即每周）执行调仓
- 调用父类 TopkDropoutStrategy 的逻辑生成基础买卖信号
- 得到买入和卖出订单列表

**第三阶段：仓位上限控制**
- 对于每笔买入订单，检查买入后的总仓位权重
- 如果某只股票的现有仓位 + 新买入仓位超过上限（默认 25%），则限制买入数量
- 计算允许的最大买入金额：max_weight × 总资产 - 现有持仓价值
- 调整买入数量以符合仓位上限要求

### 与 TopkDropout 的对比

| 特性 | TopkDropout | RealTradingStrategy |
|------|-------------|---------------------|
| **止损机制** | 无 | 每日检查，亏损达到 -8% 自动卖出 |
| **调仓频率** | 可配置（默认每日） | 固定为 5 个交易日（每周） |
| **仓位限制** | 无 | 单只股票最大仓位 25% |
| **适合场景** | 回测研究、策略对比 | 实盘交易、小资金运作 |
| **风险控制** | 弱 | 强（三层风控） |
| **交易成本** | 较高（频繁交易） | 较低（周频调仓） |
| **实盘友好度** | 一般 | 高 |

### 配置说明
配置文件位于 `configs/backtest/real_trading.yaml`：

```yaml
strategy:
  class: RealTradingStrategy
  module_path: big_a.strategy.real_trading
  kwargs:
    signal: <PRED>
    topk: 5
    n_drop: 1
    risk_degree: 0.95

risk_controls:
  stop_loss: -0.08      # 止损阈值：-8%
  max_weight: 0.25      # 仓位上限：25%
  rebalance_freq: 5     # 调仓频率：5 个交易日（每周）

backtest:
  start_time: "2022-01-01"
  end_time: "2024-12-31"
  account: 1000000
  benchmark: SH000300
  exchange_kwargs:
    freq: day
    limit_threshold: 0.095
    deal_price: close
    open_cost: 0.0005
    close_cost: 0.0015
    min_cost: 5
```

### 使用示例
```python
from big_a.strategy.real_trading import RealTradingStrategy
from big_a.config import load_config

# 加载配置
config = load_config("configs/backtest/real_trading.yaml")

# 初始化策略
strategy = RealTradingStrategy(
    signal=<PRED>,
    topk=5,
    n_drop=1,
    risk_degree=0.95,
    stop_loss=-0.08,
    max_weight=0.25,
    rebalance_freq=5
)

# 生成交易决策
trade_decision = strategy.generate_trade_decision()

# 处理执行结果
strategy.post_exe_step(execute_result)
```

### 什么时候使用 RealTradingStrategy？

**使用 RealTradingStrategy，如果：**
- 计划进行实盘交易，需要严格的风险控制
- 资金规模较小（约 100 万），需要控制交易成本
- 希望降低回撤，提高收益的稳定性
- 担心单只股票亏损过大，需要止损保护
- 频繁调仓的交易成本过高，希望降低交易频率

**不使用 RealTradingStrategy，如果：**
- 纯粹进行回测研究，不需要考虑实盘因素
- 资金规模很大，单只股票 25% 的上限限制太严格
- 希望测试更高频的交易策略
- 需要更灵活的调仓机制

### 注意事项

1. **止损的双重作用**：止损既能限制单次亏损，也能释放资金用于更好的投资机会
2. **周频调仓的权衡**：虽然降低了交易成本，但可能错过短期的快速行情
3. **仓位上限的影响**：25% 的上限意味着最多只能持有 4 只股票（100% ÷ 25%），需要注意分散度
4. **小资金的适用性**：该策略是为小资金设计的，如果资金规模很大，可能需要调整参数
5. **买入价格追踪**：策略会自动记录每只股票的买入价格用于计算盈亏，不要在策略外部修改持仓

### 常见问题

**Q: 为什么止损阈值设为 -8%？**
A: -8% 是一个经验值。太宽松（如 -20%）起不到风控作用，太严格（如 -3%）会导致频繁止损、错过反弹。8% 的亏损通常意味着判断失误，及时止损是合理的选择。

**Q: 为什么调仓频率是 5 个交易日而不是 1 个交易日？**
A: 每日调仓的交易成本很高，尤其是对于小资金。周频调仓（5 个交易日）可以显著降低交易成本，同时仍然能够及时响应市场变化。

**Q: 如果某只股票一直涨，超过 25% 的仓位限制怎么办？**
A: 策略不会强制卖出已经超过 25% 的持仓。仓位限制只在买入时生效，这是为了让盈利的股票能够继续上涨，而不是人为截断收益。

**Q: 止损后是否会立即重新买入？**
A: 不会。止损后，该股票会被移除持仓。只有在下一次定期调仓时，如果它的预测信号仍然很好，才可能重新被买入。这样可以避免在股票下跌趋势中反复买卖。
