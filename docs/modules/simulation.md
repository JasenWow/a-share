# 模拟引擎 (Simulation Engine)

模拟引擎是 Big-A 系统的核心组件，负责在历史数据或模拟环境中执行交易策略、模拟订单执行、管理投资组合资金流动。它不涉及真实市场接入，而是扮演"沙盘演练"的角色，让交易员在投入真金白银之前验证策略表现。

## 核心概念

### 订单 (Order)

订单代表一次买卖操作的基本单元，包含以下关键字段：

```python
from big_a.simulation.types import Order, OrderSide, OrderType, OrderStatus

order = Order(
    stock_code="000001.SZ",  # 股票代码
    side=OrderSide.BUY,      # 买入或卖出
    order_type=OrderType.MARKET,  # 市价单或限价单
    quantity=1000,           # 数量（手）
    price=15.5              # 价格
)
```

**订单方向 (OrderSide)**：
- `BUY`：买入
- `SELL`：卖出

**订单类型 (OrderType)**：
- `MARKET`：市价单，以当前市场价格立即成交
- `LIMIT`：限价单，设定最高买入价或最低卖出价

**订单状态 (OrderStatus)**：
- `PENDING`：待成交
- `FILLED`：已成交
- `CANCELLED`：已取消
- `REJECTED`：被拒绝（资金不足、仓位不足等）

订单生命周期：创建时状态为 PENDING，提交给 Broker 执行后，状态变为 FILLED、CANCELLED 或 REJECTED。

### 仓位 (Position)

仓位代表当前持有某只股票的情况，包含成本、盈亏等关键信息：

```python
from big_a.simulation.types import Position

position = Position(
    stock_code="000001.SZ",
    quantity=1000,      # 持有股数
    avg_price=15.0,     # 平均成本价
    current_price=16.5, # 当前市价
    unrealized_pnl=1500.0,  # 未实现盈亏（浮盈/浮亏）
    realized_pnl=0.0,      # 已实现盈亏（平仓后结算）
    entry_date="2024-01-15"  # 建仓日期
)
```

**仓位核心属性**：
- `market_value`：市值 = quantity × current_price
- `pnl_pct`：盈亏比例 = (current_price - avg_price) / avg_price

PnL 计算示例：买入 1000 股，成本 15.0 元，当前价 16.5 元，则浮盈 = (16.5 - 15.0) × 1000 = 1500 元，盈亏比例 = (16.5 - 15.0) / 15.0 = 10%。

### 投资组合 (Portfolio)

投资组合汇总账户整体资金状况，包括现金、持仓和总权益：

```python
from big_a.simulation.types import Portfolio

portfolio = Portfolio(
    cash=300000.0,      # 可用资金
    positions={},      # 持仓字典，key 是股票代码
    total_value=500000.0  # 总权益 = 现金 + 持仓市值
)
```

**计算公式**：total_value = cash + Σ(position.market_value)

### 股票信号 (StockSignal)

信号是模型对个股的评分和操作建议：

```python
from big_a.simulation.types import StockSignal, SignalStrength, SignalSource

signal = StockSignal(
    stock_code="000001.SZ",
    score=0.85,              # 评分，范围 [-1.0, 1.0]
    signal=SignalStrength.BUY,  # 信号强度
    source=SignalSource.kronos,  # 信号来源
    reasoning="模型预测上涨趋势明确"  # 分析理由
)
```

**信号强度 (SignalStrength)**：
- `STRONG_BUY`：强烈买入
- `BUY`：买入
- `HOLD`：持有
- `SELL`：卖出
- `STRONG_SELL`：强烈卖出

**信号来源 (SignalSource)**：
- `kronos`：Kronos Transformer 模型
- `lightgbm`：LightGBM 因子模型
- `llm`：大语言模型分析
- `fused`：多模型融合

## SimulationEngine 工作机制

SimulationEngine 是模拟引擎的核心类，负责每日交易模拟的调度和执行。

### 初始化

```python
from big_a.simulation import SimulationConfig
from big_a.broker.in_memory import InMemoryBroker
from big_a.simulation.engine import SimulationEngine

# 创建配置和经纪商
config = SimulationConfig(
    initial_capital=500000.0,  # 初始资金 50 万
    topk=5,                   # 持有最多 5 只股票
    stop_loss=-0.08,          # 止损线 -8%
    rebalance_freq=5,         # 每 5 天调仓一次
    max_total_loss=-0.20      # 熔断线 -20%
)

broker = InMemoryBroker(
    initial_cash=500000.0,
    open_cost=0.0005,         # 买入佣金 0.05%
    close_cost=0.0015         # 卖出佣金 0.15%
)

engine = SimulationEngine(config=config, broker=broker)
engine.initialize()
```

### 每日模拟流程

每日模拟 `run_daily()` 方法按以下顺序执行：

```
1. 日期计数 +1
2. 更新所有持仓价格
3. 检查止损（遍历持仓，卖出浮亏超过阈值的）
4. 检查熔断（总亏损超限则跳过调仓）
5. 调仓（每 N 天执行一次）
6. 保存投资组合快照
```

**代码流程图**：

```python
def run_daily(self, trading_date: str, signals: list[StockSignal], prices: dict[str, float]):
    self._day_count += 1
    self.broker.update_prices(prices)  # 更新持仓价格

    # 止损检查：持仓浮亏 <= stop_loss 则卖出
    for stock_code, pos in self.broker.get_all_positions().items():
        if pos.pnl_pct <= self.config.stop_loss:
            self.broker.submit_order(Order(
                stock_code=stock_code,
                side=OrderSide.SELL,
                quantity=pos.quantity,
                price=pos.current_price
            ))

    # 熔断检查：总亏损 <= max_total_loss 则跳过调仓
    portfolio = self.broker.get_portfolio()
    total_pnl_pct = (portfolio.total_value - self.config.initial_capital) / self.config.initial_capital
    if total_pnl_pct > self.config.max_total_loss:
        # 调仓日，执行再平衡
        if self._day_count == 1 or self._day_count % self.config.rebalance_freq == 1:
            self._rebalance(signals, portfolio.total_value, prices, trading_date)

    # 保存快照
    self._daily_snapshots.append(self.broker.get_portfolio())
```

### 止损机制 (Stop-Loss)

止损是风险控制的最后防线。当某只持仓的浮亏比例达到或超过 `stop_loss` 阈值时，系统会自动发出卖出指令。

假设配置 `stop_loss = -0.08`（-8%）：

```
持仓：000001.SZ，买入价 10 元，当前价 9.2 元
浮亏 = (9.2 - 10) / 10 = -8%，触发止损
系统自动卖出全部持仓
```

止损的优势：自动锁定亏损，避免情绪化扛单。止损的局限：在市场流动性差时，可能以更低价格成交。

### 熔断机制 (Circuit Breaker)

熔断是全局风险控制。当整个投资组合的总亏损达到或超过 `max_total_loss` 阈值时，系统会跳过后续调仓操作，等待市场恢复或手动干预。

假设配置 `max_total_loss = -0.20`（-20%），初始资金 50 万，当前总权益 38 万：

```
总亏损 = (380000 - 500000) / 500000 = -24%
触发熔断，跳过本次调仓
```

熔断的意义：防止策略在持续亏损中越陷越深，给予交易员冷静评估的时间。

### 调仓逻辑 (Rebalancing)

调仓是策略执行的核心环节。当达到调仓日（首日或每 N 天），系统会：

1. **排序信号**：按 score 降序排列，取 topk 只股票
2. **等权分配**：每只股票分配相同权重，总权重 100%
3. **卖出**：清仓不在 topk 中的持仓
4. **买入**：调整现有持仓至目标权重，不足则买入，超配则卖出

**调仓权重计算示例**：

```python
config.topk = 5
信号列表：[score=0.9, score=0.8, score=0.75, score=0.6, score=0.55, score=0.3]

目标持仓：["股票A", "股票B", "股票C", "股票D", "股票E"]
每只权重：1.0 / 5 = 20%

假设总权益 50 万，每只股票目标市值 = 50万 × 20% = 10 万
```

调仓频率建议：过于频繁增加交易成本，过于稀少可能错失机会。默认 5 天是较为均衡的选择。

## Broker 系统

Broker（经纪商）是订单执行和仓位管理的抽象接口，允许模拟系统与不同后端对接。

### BrokerInterface 接口

```python
from big_a.broker.base import BrokerInterface

class BrokerInterface(Protocol):
    """经纪商接口定义"""

    def submit_order(self, order: Order) -> Order:
        """提交订单并返回最终状态"""

    def cancel_order(self, order_id: str) -> Order:
        """取消待成交订单"""

    def get_position(self, stock_code: str) -> Position | None:
        """查询单只股票持仓"""

    def get_all_positions(self) -> dict[str, Position]:
        """查询所有持仓"""

    def get_balance(self) -> float:
        """查询可用资金"""

    def get_portfolio(self) -> Portfolio:
        """查询完整投资组合"""

    def update_prices(self, prices: dict[str, float]) -> None:
        """批量更新持仓价格，计算浮盈亏"""
```

接口设计的优势：SimulationEngine 不关心具体实现，可以通过 `InMemoryBroker` 进行模拟测试，也可以接入 `QMTBroker` 进行实盘交易。

### InMemoryBroker 实现

InMemoryBroker 是纯内存实现的经纪商，不连接真实交易所，适合回测和模拟。

**核心属性**：

```python
self._cash = initial_cash           # 可用资金
self._positions: dict[str, Position]  # 持仓字典
self._orders: list[Order]            # 历史订单
self._trade_records: list[TradeRecord]  # 成交记录
self._prices: dict[str, float]      # 当前价格
```

**买入处理流程**：

```
1. 检查现金是否充足（成本 + 佣金）
2. 扣除现金（成本 + 佣金）
3. 更新或创建持仓：
   - 新持仓：avg_price = 成交价
   - 已有持仓：加权平均计算新成本
4. 记录成交
```

**卖出处理流程**：

```
1. 检查持仓是否充足
2. 计算已实现盈亏：(卖出价 - 成本价) × 数量 - 佣金
3. 增加现金（成交额 - 佣金）
4. 更新持仓数量，如果清仓则删除记录
5. 记录成交
```

**佣金计算**：

```python
commission = max(成交额 × 费率, 最低佣金)
# 例如：成交额 10000 元，费率 0.0005，最低佣金 5 元
# 佣金 = max(10000 × 0.0005, 5) = max(5, 5) = 5 元
```

## 配置参数

SimulationConfig 定义了模拟引擎的所有可调参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `initial_capital` | 500000.0 | 初始资金（元） |
| `account` | "sim_001" | 账户标识 |
| `max_weight` | 0.25 | 单只股票最大权重（25%） |
| `stop_loss` | -0.08 | 止损线（-8%） |
| `rebalance_freq` | 5 | 调仓频率（每 N 天） |
| `topk` | 5 | 持仓股票数量上限 |
| `n_drop` | 1 | 每日减持数量 |
| `risk_degree` | 0.95 | 风险敞口比例 |
| `max_total_loss` | -0.20 | 熔断阈值（-20%） |
| `min_cash` | 10000.0 | 最低现金保留 |
| `open_cost` | 0.0005 | 买入费率（0.05%） |
| `close_cost` | 0.0015 | 卖出费率（0.15%） |
| `min_commission` | 5.0 | 最低佣金（元） |
| `limit_threshold` | 0.095 | 涨跌停阈值（9.5%） |
| `deal_price` | "close" | 成交价格基准 |

**YAML 配置示例**：

```yaml
simulation:
  initial_capital: 500000
  topk: 5
  rebalance_freq: 5
  stop_loss: -0.08
  max_weight: 0.25

circuit_breaker:
  max_total_loss: -0.20
  min_cash: 10000

exchange:
  open_cost: 0.0005
  close_cost: 0.0015
  min_commission: 5.0
  limit_threshold: 0.095
  deal_price: close

universe:
  base_pool: csi300
  watchlist: configs/watchlist.yaml
```

**加载配置**：

```python
from big_a.simulation.config import load_simulation_config

config = load_simulation_config("configs/simulation.yaml")
```

## 使用示例

完整的模拟回测流程：

```python
from big_a.simulation import SimulationConfig
from big_a.simulation.engine import SimulationEngine
from big_a.broker.in_memory import InMemoryBroker
from big_a.simulation.types import StockSignal, SignalStrength, SignalSource

# 1. 初始化
config = SimulationConfig(
    initial_capital=500000.0,
    topk=5,
    stop_loss=-0.08,
    rebalance_freq=5,
    max_total_loss=-0.20
)
broker = InMemoryBroker(
    initial_cash=500000.0,
    open_cost=0.0005,
    close_cost=0.0015
)
engine = SimulationEngine(config=config, broker=broker)
engine.initialize()

# 2. 每日模拟（循环历史交易日）
trading_dates = ["2024-01-02", "2024-01-03", "2024-01-04", ...]
for date in trading_dates:
    # 获取当日信号和价格
    signals = [
        StockSignal(stock_code="000001.SZ", score=0.85,
                   signal=SignalStrength.BUY, source=SignalSource.kronos),
        StockSignal(stock_code="000002.SZ", score=0.78,
                   signal=SignalStrength.BUY, source=SignalSource.kronos),
        # ...更多信号
    ]
    prices = {
        "000001.SZ": 15.5,
        "000002.SZ": 28.3,
        # ...更多价格
    }

    # 执行每日模拟
    portfolio = engine.run_daily(date, signals, prices)
    print(f"{date}: 总价值={portfolio.total_value:.2f}, 现金={portfolio.cash:.2f}")

# 3. 获取回测结果
snapshots = engine.get_daily_snapshots()
day_count = engine.get_day_count()

# 计算收益率
initial = config.initial_capital
final = snapshots[-1].total_value if snapshots else initial
total_return = (final - initial) / initial
print(f"总收益率: {total_return*100:.2f}%")
```

**输出示例**：

```
2024-01-02: 总价值=500000.00, 现金=500000.00
2024-01-03: 总价值=502500.00, 现金=350000.00
Day 3 [2024-01-04]: BUY 000001.SZ qty=10000@15.50
Day 3 [2024-01-04]: portfolio total_value=510000.00, cash=160000.00
```

## 总结

模拟引擎是 Big-A 系统的策略验证层，通过以下几个关键机制保障策略的安全执行：

**风险控制三保险**：
- 止损（stop_loss）：单只股票浮亏超限自动卖出
- 熔断（max_total_loss）：总亏损超限暂停调仓
- 最低现金（min_cash）：保留应急资金

**调仓逻辑**：等权分配 + topk 筛选，兼顾分散与聚焦

**Broker 抽象**：解耦策略层与执行层，支持模拟回测和实盘交易的无缝切换

使用模拟引擎时，建议先用小资金、保守参数（高止损、低仓位）进行试跑，确认策略有效后再逐步加大敞口。