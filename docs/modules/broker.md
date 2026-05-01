# Broker 交易券商模块

## 1. 概述

Broker 模块是 Big-A 系统中的交易执行抽象层，负责处理订单提交、持仓管理、现金结算等核心交易操作。

**为什么需要 Broker 抽象？**

量化系统需要对接不同的交易渠道：
- 回测时：用模拟券商（InMemoryBroker）模拟真实交易环境
- 实盘时：用真实券商（如 QMT）连接证券公司接口

如果代码直接写死某种券商的实现，换到其他券商时就要大幅修改。Broker 接口定义了统一的操作规范，换券商时只需替换实现类，业务逻辑代码不需要改动。

```
策略/回测引擎
       ↓
   BrokerInterface  ← 抽象层
       ↓
  ┌─────┴─────┐
  ↓           ↓
InMemoryBroker   QMTBroker（未来扩展）
```

## 2. BrokerInterface 抽象接口

`BrokerInterface` 是一个 Protocol（结构化子类型协议），定义了所有券商必须实现的方法。

**源码位置：** `src/big_a/broker/base.py`

```python
from typing import Protocol

class BrokerInterface(Protocol):
    """Abstract broker interface. InMemory implementation for simulation, QMT for real trading."""

    def submit_order(self, order: Order) -> Order:
        """提交订单，返回更新后的订单（含最终状态）"""

    def cancel_order(self, order_id: str) -> Order:
        """根据订单ID取消待处理订单"""

    def get_position(self, stock_code: str) -> Position | None:
        """获取股票持仓，不持有则返回 None"""

    def get_all_positions(self) -> dict[str, Position]:
        """获取所有持仓，以 stock_code 为键的字典"""

    def get_balance(self) -> float:
        """获取当前现金余额"""

    def get_portfolio(self) -> Portfolio:
        """获取当前投资组合状态（含所有持仓和总权益）"""

    def update_prices(self, prices: dict[str, float]) -> None:
        """更新持仓股价，重算未实现盈亏"""
```

### 关键设计说明

**submit_order**：接收一个 Order 对象，执行买卖逻辑后返回更新后的 Order。订单状态可能是 FILLED（成交）、REJECTED（拒绝）、CANCELLED（取消）。

**update_prices**：批量更新持仓股价，用于每日收盘后重算浮动盈亏。不返回任何内容，直接修改持仓对象的状态。

**get_portfolio**：返回一个 Portfolio 对象，包含现金、全部持仓、总权益等信息。

## 3. InMemoryBroker 内存模拟券商

`InMemoryBroker` 是 BrokerInterface 的默认实现，用于回测环境下的交易模拟。

**源码位置：** `src/big_a/broker/in_memory.py`

```python
class InMemoryBroker:
    def __init__(
        self,
        initial_cash: float,
        open_cost: float = 0.0005,    # 买入佣金率 0.05%
        close_cost: float = 0.0015,  # 卖出佣金率 0.15%
        min_commission: float = 5.0, # 最低佣金 5 元
        limit_threshold: float = 0.095,  # 涨跌停阈值 9.5%
    ) -> None:
        ...
```

### 交易执行逻辑

**买入（BUY）：**

1. 计算买入总金额：`quantity × price`
2. 计算佣金：`max(总金额 × open_cost, min_commission)`
3. 检查现金是否足够，现金不足则拒绝订单
4. 扣除现金和佣金，更新持仓
5. 如果股票已持仓，计算新的加权平均价格
6. 如果是新持仓，创建 Position 对象

**卖出（SELL）：**

1. 检查是否有该股票持仓，没有则拒绝订单
2. 检查持仓数量是否足够，不够则拒绝订单
3. 计算卖出所得：`quantity × price`
4. 计算佣金（含印花税）：`max(所得 × close_cost, min_commission)`
5. 计算已实现盈亏：`(卖出价 - 买入均价) × quantity - 佣金`
6. 增加现金，扣除佣金
7. 如果持仓卖完，删除该 Position 对象

### 佣金计算示例

假设买入 10000 元股票（佣金率 0.05%，最低 5 元）：
- 理论佣金 = 10000 × 0.0005 = 5 元
- 由于 5 元恰好等于最低佣金，实际收取 5 元

假设买入 1000 元股票：
- 理论佣金 = 1000 × 0.0005 = 0.5 元
- 低于最低 5 元，按 5 元收取

### 持仓跟踪

Position 对象记录每只股票的详细信息：

```python
class Position(BaseModel):
    stock_code: str      # 股票代码，如 "000001.SZ"
    quantity: int        # 持仓数量
    avg_price: float     # 买入均价
    current_price: float # 当前市价
    unrealized_pnl: float  # 未实现盈亏（浮动盈亏）
    realized_pnl: float    # 已实现盈亏（卖出后结算）
    entry_date: str        # 建仓日期

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def pnl_pct(self) -> float:
        return (self.current_price - self.avg_price) / self.avg_price
```

每次调用 `update_prices` 时，系统会更新所有持仓的 `current_price` 和 `unrealized_pnl`。

### 订单状态

```python
class OrderStatus(str, Enum):
    PENDING = "PENDING"    # 待处理
    FILLED = "FILLED"      # 成交
    CANCELLED = "CANCELLED"  # 已取消
    REJECTED = "REJECTED"    # 拒绝
```

## 4. 使用示例

### 初始化券商

```python
from big_a.broker.in_memory import InMemoryBroker

broker = InMemoryBroker(
    initial_cash=1000000.0,  # 初始资金 100 万
    open_cost=0.0005,        # 买入佣金 0.05%
    close_cost=0.0015,      # 卖出佣金 0.15%（含印花税）
    min_commission=5.0,     # 最低 5 元
)
```

### 提交订单

```python
from big_a.simulation.types import Order, OrderSide, OrderType

order = Order(
    stock_code="000001.SZ",
    side=OrderSide.BUY,
    order_type=OrderType.MARKET,
    quantity=1000,
    price=10.0,
)

# 提交订单，order 对象会被更新
result = broker.submit_order(order)

print(f"订单状态: {result.status}")
print(f"手续费: {result.commission}")
print(f"成交时间: {result.filled_at}")
```

### 查询持仓和资金

```python
# 查询单只股票持仓
pos = broker.get_position("000001.SZ")
if pos:
    print(f"数量: {pos.quantity}, 均价: {pos.avg_price}")

# 查询所有持仓
all_positions = broker.get_all_positions()
for code, position in all_positions.items():
    print(f"{code}: {position.quantity}股, 市值: {position.market_value:.2f}")

# 查询现金
cash = broker.get_balance()
print(f"现金: {cash:.2f}")

# 查询完整投资组合
portfolio = broker.get_portfolio()
print(f"总权益: {portfolio.total_value:.2f}")
```

### 更新价格和查询盈亏

```python
# 收盘后批量更新价格
prices = {
    "000001.SZ": 10.5,
    "000002.SZ": 8.8,
}
broker.update_prices(prices)

# 查看浮动盈亏
portfolio = broker.get_portfolio()
for code, pos in portfolio.positions.items():
    print(f"{code}: 浮动盈亏 {pos.unrealized_pnl:.2f}")
```

### 完整回测循环示例

```python
from big_a.broker.in_memory import InMemoryBroker
from big_a.simulation.types import Order, OrderSide

broker = InMemoryBroker(initial_cash=1000000.0)

# 模拟每日交易
for date, signal in signals.items():
    # 更新当日价格
    broker.update_prices(signal.prices)

    # 根据信号生成并执行订单
    for stock_code, score in signal.top_stocks:
        if score > threshold:
            order = Order(
                stock_code=stock_code,
                side=OrderSide.BUY,
                quantity=1000,
                price=signal.prices[stock_code],
            )
            broker.submit_order(order)

    # 查看当日收盘持仓
    portfolio = broker.get_portfolio()
    print(f"日期: {date}, 总权益: {portfolio.total_value:.2f}")
```

## 5. 设计模式

Broker 模块采用了**接口隔离模式**（Interface Segregation Principle）：

- 策略层只依赖 `BrokerInterface`，不关心具体实现
- `InMemoryBroker` 用于回测，性能高、无外部依赖
- 未来扩展 `QMTBroker` 只需实现相同接口，策略代码无需修改

这种设计的好处：

1. **回测和实盘切换方便**：改一行初始化代码即可
2. **单元测试容易**：可以用内存模拟对象隔离测试
3. **扩展新券商不影响现有代码**：只要实现接口就能接入

## 6. 数据类型参考

Broker 模块依赖 `simulation/types.py` 中定义的数据结构：

| 类型 | 说明 |
|------|------|
| `Order` | 订单对象，含订单ID、股票代码、方向、数量、价格、状态、手续费等 |
| `OrderSide` | 枚举：BUY（买）、SELL（卖） |
| `OrderStatus` | 枚举：PENDING、FILLED、CANCELLED、REJECTED |
| `Position` | 持仓对象，含股票代码、数量、均价、现价、盈亏等 |
| `Portfolio` | 组合对象，含现金、全部持仓、总权益、更新时间等 |
| `TradeRecord` | 交易记录，含订单ID、股票代码、方向、数量、成交价、手续费、时间戳 |

## 7. 总结

Broker 模块是 Big-A 系统的交易执行层核心：

- **接口抽象**：`BrokerInterface` 定义了券商的通用操作规范
- **内存模拟**：`InMemoryBroker` 提供了回测所需的交易模拟能力
- **佣金计算**：考虑了 A 股的特殊费用结构（最低佣金、印花税等）
- **持仓管理**：自动跟踪所有持仓的买入均价和浮动盈亏
- **易于扩展**：新券商只需实现相同接口即可接入

通过 Broker 抽象，回测引擎可以和具体券商实现解耦，无论是在本地模拟环境还是连接真实券商，策略代码都能保持一致。