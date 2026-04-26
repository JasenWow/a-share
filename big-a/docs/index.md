# Big-A 文档中心

欢迎来到 Big-A 量化交易平台的文档中心。本文档提供项目介绍、使用指南和参考手册。

---

## 快速开始

如果你是第一次使用 Big-A，建议按照以下顺序阅读：

1. **[项目概述](#项目概述)** — 了解 Big-A 是什么，能做什么
2. **[安装指南](#安装指南)** — 快速安装和配置环境
3. **[快速上手](#快速上手)** — 运行第一个回测示例
4. **[使用指南](#使用指南)** — 深入了解各个模块

---

## 项目概述

Big-A 是一个基于 Python 的量化交易平台，支持多种预测模型（LightGBM、Kronos、Hedge Fund）和回测策略（TopkDropout、RealTradingStrategy）。它适用于 A 股市场的研究和实盘交易。

**核心特性：**
- 多种预测模型：传统机器学习（LightGBM）、深度学习（Kronos）、大语言模型（Hedge Fund）
- 灵活的回测框架：基于 Qlib，支持自定义策略
- 实盘友好策略：RealTradingStrategy 提供止损、仓位控制等风控机制
- 完整的数据处理：支持 A 股历史数据的下载和更新
- 可扩展架构：模块化设计，易于添加新模型和策略

---

## 安装指南

### 环境要求

- Python 3.10+
- macOS 或 Linux
- 8GB+ 内存
- 推荐 GPU/MPS 加速（用于 Kronos 和 Hedge Fund 模型）

### 安装步骤

```bash
# 1. 克隆仓库
git clone https://github.com/your-org/stock-big-a.git
cd stock-big-a

# 2. 安装依赖
pip install -r requirements.txt

# 3. 下载 A 股数据
python big-a/scripts/update_data.py update --data-dir data/qlib_data/cn_data

# 4. 验证安装
python -m big_a.scripts.check_installation
```

### 配置说明

配置文件位于 `configs/` 目录，主要配置包括：
- `configs/data/` — 数据源配置
- `configs/model/` — 模型配置
- `configs/backtest/` — 回测策略配置

---

## 快速上手

### 训练一个 LightGBM 模型

```bash
python -m big_a.scripts.train_model --model lightgbm
```

### 运行回测

```bash
# 使用 TopkDropout 策略回测
python -m big_a.scripts.backtest \
    --strategy topk_dropout \
    --model lightgbm \
    --start_date 2022-01-01 \
    --end_date 2024-12-31

# 使用 RealTradingStrategy 回测
python -m big_a.scripts.backtest \
    --strategy real_trading \
    --model lightgbm \
    --start_date 2022-01-01 \
    --end_date 2024-12-31
```

### 查看结果

回测结果保存在 `outputs/backtest/` 目录，包括：
- 收益曲线图
- 绩效指标表
- 交易明细记录

---

## 使用指南

### 模型模块

- **[LightGBM 模型](modules/models.md#lightgbm-模型)** — 基于梯度提升决策树的传统机器学习模型
- **[Kronos 模型](modules/models.md#kronos-模型)** — 专为金融时间序列设计的 Transformer 模型
- **[Hedge Fund 模型](modules/models.md#hedge-fund-模型)** — 基于大语言模型的多智能体量化分析系统

### 回测策略

- **[TopkDropout 策略](modules/backtest.md#1-topkdropout-策略)** — 经典的前K名择优策略
- **[实盘交易策略](modules/backtest.md#2-实盘交易策略realtradingstrategy)** — 带风控的实盘友好型策略

### 数据模块

- **[数据获取](../data.md)** — A 股数据的下载和更新
- **[数据预处理](../data.md)** — 特征工程和数据清洗

### 评估指标

- **[绩效指标](evaluation.md)** — 收益率、夏普比率、最大回撤等
- **[风险指标](evaluation.md)** — 波动率、下行风险、VaR 等

### 因子模块

- **[Alpha158 因子](factors.md)** — 经典的 158 个技术指标因子
- **[自定义因子](factors.md)** — 如何添加和测试自定义因子

### 参考文档

- **[术语表](reference/glossary.md)** — 常用术语和概念解释
- **[配置说明](../architecture/config.md)** — 配置文件结构和参数说明
- **[API 文档](../architecture/api.md)** — 核心模块的 API 参考

---

## 进阶主题

### 模型对比

如何选择合适的模型？参见 [模型对比表](modules/models.md#与其他模型对比)。

### 策略优化

- 参数调优：使用网格搜索或贝叶斯优化
- 组合策略：多个策略的加权组合
- 风险预算：根据风险偏好调整仓位

### 实盘交易

- 从回测到实盘的注意事项
- 实盘风险控制
- 交易成本管理

---

## 常见问题

### Q: Big-A 支持哪些市场？

A: 目前主要支持 A 股市场（沪深京），未来计划扩展到港股和美股。

### Q: 如何添加自定义模型？

A: 参考 [API 文档](../architecture/api.md) 中的模型接口说明，实现预测方法并注册到模型工厂。

### Q: 回测结果和实盘结果为什么有差异？

A: 回测使用的是历史数据，没有考虑滑点、冲击成本、流动性限制等实盘因素。实盘前建议进行模拟交易验证。

### Q: 如何获取帮助？

A: 可以通过以下方式获取帮助：
- 查看本文档和示例代码
- 在 GitHub 上提 Issue
- 加入用户交流群

---

## 更新日志

### v1.2.0 (2025-04-26)
- 新增 RealTradingStrategy 策略，支持止损、周频调仓、仓位上限
- 新增 Hedge Fund 模型，基于大语言模型的多智能体分析
- 优化数据更新脚本，支持增量更新
- 完善文档，新增术语表和使用指南

### v1.1.0 (2025-03-15)
- 新增 Kronos 模型支持
- 改进 LightGBM 训练流程
- 新增 Alpha158 因子库
- 修复回测框架的若干 bug

### v1.0.0 (2025-01-01)
- 首个正式版本发布
- 支持 LightGBM 模型和 TopkDropout 策略
- 完整的回测框架
- A 股数据支持

---

## 许可证

Big-A 采用 MIT 许可证，详见 [LICENSE](../LICENSE) 文件。

---

## 贡献指南

欢迎贡献代码、文档或提出建议！详见 [CONTRIBUTING.md](../CONTRIBUTING.md)。

---

**最后更新：2025 年 4 月 26 日**
