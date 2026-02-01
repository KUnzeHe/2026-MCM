# 2026 MCM Problem B: Space Transportation Optimization

[![MCM 2026](https://img.shields.io/badge/MCM-2026-blue.svg)](https://www.comap.com/contests/mcm-icm)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 项目概述

本项目是 **2026年美国大学生数学建模竞赛 (MCM)** Problem B 的解决方案，研究主题为：**太空运输系统优化** —— 探索如何高效地将约 **1亿吨** 建材从地球运输至月球，用于建设月球基地。

### 核心问题

在给定的太空电梯系统和传统火箭运输两种方式下，如何制定最优的混合运输策略，实现 **时间** 与 **成本** 的最佳权衡？

---

## 项目结构

```
2026MCM/
├── README.md                          # 项目说明文档
└── Problem B/
    ├── draft/
    │   ├── Question 1/                # 问题1：理想工况下的运输优化
    │   │   ├── 1a&1b/                 # 单一运输系统模型
    │   │   │   ├── codes/             # 单一模式代码实现
    │   │   │   └── mdFile/            # 单一模式分析文档
    │   │   ├── 1c/                    # 混合运输系统优化
    │   │   │   ├── codes/             # 核心优化算法
    │   │   │   │   └── comprehensive_transport_model_v5.py  # 主模型V5
    │   │   │   └── mdFile/            # 优化报告与分析
    │   │   │       ├── 24year_deadline_analysis_report.md  # 2050完工可行性分析
    │   │   │       └── comprehensive_optimization_framework_v4.md
    │   │   └── others/                # 草稿与实验性分析 (Drafts)
    │   │       ├── rocket_launch_prediction_analysis.md    # 辅助分析草稿
    │   │       └── pareto_analysis.py
    │   └── Question 2/                # 问题2：非理想工况分析
    │       ├── codes/                 # 可靠性与场景模拟代码
    │       │   ├── q2_visualization.py
    │       │   └── scenario_100Mt/    # 1亿吨场景模拟结果
    │       └── mdFile/                # 不确定性分析文档
    │           └── Q2_reliability_and_cost_analysis.md
    └── Reference/
        └── reference_list.txt         # 参考资料列表
```

---

## 研究内容

### Question 1: 理想工况下的运输系统优化

#### 1a. 纯电梯系统方案 (Space Elevator Only)

建模太空电梯作为唯一运输手段的场景：

- **系统架构**：地面 → 电梯 → GEO锚点 → 转运火箭 → 月球
- **瓶颈分析**：$T_{\text{chain}} = \min(T_E, T_{R,\text{anchor}})$
- **成本模型**：$C_{1a} = F_E + c_E \cdot M_{\text{tot}}$

#### 1b. 纯火箭系统方案 (Traditional Rockets Only)

建模传统火箭作为唯一运输手段的场景：

- **系统架构**：地面发射场 → 重型火箭 → 月球
- **运力计算**：$T_R = N_{\text{sites}} \cdot L_{\max} \cdot p_B$
- **成本模型**：$C_{1b} = c_R \cdot M_{\text{tot}}$

#### 1c. 混合运输系统优化 (Hybrid System)

**核心创新**：建立了完整的多目标优化框架，包含：

1. **Logistic 基础设施增长模型**
   $$N(t) = \frac{K}{1 + \left( \frac{K - N_0}{N_0} \right) e^{-r t}}$$

2. **多目标优化求解 (Ver 5.0)** —— 生成 Pareto 前沿，分析时间与成本的权衡。

3. **2050 完工 (24年工期) 可行性专项分析**：
   - **风险评估**：Monte Carlo 模拟显示按时完工概率仅为 **24.3%**。
   - **成本预估**：NPV 约为 **\$40.50 万亿**（主要由火箭承担）。
   - **运力余量**：仅 7.8%，极易受干扰。

---

### Question 2: 非理想工况下的系统分析

引入真实世界的不确定性因素与风险模型：

| 维度 | 关键因素 | 建模方法 |
|:-----|:---------|:---------|
| **可靠性** | 故障率 $\lambda$、维修 | 引入有效利用率因子 $\beta \in [0,1]$ |
| **成本** | 运维成本、事故损失 | $C_{real} = C_{base} \cdot (1 + \alpha) + E[C_{risk}]$ |
| **场景** | 1亿吨需求波动 | Monte Carlo 敏感性测试 |

**关键结论**：
- 总成本预计上升 **30%-50%**
- **鲁棒性策略**：当火箭故障率 $\lambda_R$ 较高时，应显著增加电梯的运量分配。

---

## 核心参数配置

| 参数 | 符号 | 数值 | 说明 |
|:-----|:-----|:-----|:-----|
| 总运输质量 | $M_{\text{tot}}$ | $10^8$ 吨 | 题目给定 |
| 电梯年吞吐量 | $T_E$ | 537,000 吨/年 | 3 Harbours × 179,000 |
| 电梯单位成本 | $c_E$ | \$2.7/kg | 电力 + 转运燃料 |
| 电梯固定成本 | $F_E$ | \$100B | 基础设施建设 |
| 火箭单位成本 | $c_R$ | \$720/kg (未来) | Starship级可重复使用 |
| 火箭单次载荷 | $p_B$ | 150 吨 | Starship级 |
| 发射场承载力 | $K$ | 80 | 全球最大容量 |

---

## 环境配置与运行

### 依赖安装

```bash
pip install numpy scipy matplotlib
```

### 运行主模型

```bash
cd "Problem B/draft/Question 1/1c/codes"
python comprehensive_transport_model_v5.py
```

### 主要输出

1. **Pareto 前沿曲线** —— 时间-成本权衡可视化
2. **最优分配方案** —— 不同时间约束下的 $x^*(Y)$
3. **敏感性分析图** —— 关键参数影响评估
4. **Monte Carlo 鲁棒性分析** —— 不确定性量化

---

## 核心结论与建议

### 1. 策略建议

| 目标导向 | 完工时间 (年) | 策略特征 | 预期成本 (NPV) |
|:---------|:-------------:|:---------|:---------|
| **极速推进** | 20-25 | 激进扩建火箭发射场 (>85%载荷) | > $45T (极高) |
| **推荐平衡** | **28-32** | **混合模式最佳点 (Knee Point)** | **$15T - $25T** |
| **成本最优** | > 40 | 以电梯为主 (>90%载荷) | < $5T (最低) |

### 2. 关键决策点
- **2050 "死线" 警告**：若必须在 2050 年前完工（24年工期），项目失败风险极高。建议至少延长工期至 **28年** 以确保 >95% 的成功率。
- **电梯虽然慢，但是稳**：在非理想工况下，电梯系统由于其“连续流”特性，比“离散脉冲”式的火箭发射具有更好的抗干扰能力（前提是解决缆绳风险）。
- **风险对冲**：混合方案提供了分布式冗余，降低单点故障风险。

---

## 参考资料

1. [火箭发射成本分析](https://spaceinsider.tech/2023/08/16/how-much-does-it-cost-to-launch-a-rocket/)
2. [美国电力价格](https://zh.globalpetrolprices.com/USA/electricity_prices/)


