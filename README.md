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

```text
2026MCM/
├── README.md                          # 项目说明文档
└── Problem B/
    ├── draft/
   │   ├── Question 1/                # 问题1：理想工况下的运输优化
   │   │   ├── 1a&1b/                 # 单一运输系统模型
   │   │   │   ├── codes/             # 单一模式代码实现 + 可视化
   │   │   │   │   ├── single_mode_opt.py
   │   │   │   │   └── visualization_optimized.py
   │   │   │   └── mdFile/            # 单一模式分析文档
   │   │   ├── 1c/                    # 混合运输系统优化 + 可视化
   │   │   │   ├── codes/
   │   │   │   │   ├── comprehensive_transport_model_v5.py  # 主模型 V5
   │   │   │   │   └── visualization_1c_platinum.py         # Platinum Quartet 可视化
   │   │   │   ├── image/             # Q1c 图表输出
   │   │   │   └── mdFile/            # 优化报告与分析
   │   │   ├── others/                # 草稿与实验性分析 (Drafts)
   │   │   └── ...
   │   ├── Question 2/                # 问题2：非理想工况分析（可靠性/鲁棒性）
   │   │   ├── codes/
   │   │   ├── image/
   │   │   └── mdFile/
   │   ├── Question 3/                # 问题3：水资源可持续性与补给物流
   │   │   ├── codes/
   │   │   │   ├── water_supply_analysis.py
   │   │   │   └── q3_visualization_platinum.py
   │   │   ├── image/                 # Q3 图表输出
   │   │   └── mdFile/
   │   └── Paper/                     # 论文 LaTeX
   └── Reference/                     # 参考资料列表
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
| :----- | :--------- | :--------- |
| **可靠性** | 故障率 $\lambda$、维修 | 引入有效利用率因子 $\beta \in [0,1]$ |
| **成本** | 运维成本、事故损失 | $C_{real} = C_{base} \cdot (1 + \alpha) + E[C_{risk}]$ |
| **场景** | 1亿吨需求波动 | Monte Carlo 敏感性测试 |

**关键结论**：

- 总成本预计上升 **30%-50%**
- **鲁棒性策略**：当火箭故障率 $\lambda_R$ 较高时，应显著增加电梯的运量分配。

---

### Question 3: 水资源可持续性与补给物流 (Water Sustainability)

在 10 万人月球殖民地完全运行后，建立水资源“代谢-回收-补给”的闭环模型，并将净补给需求作为 **Q1/Q2 运输系统** 的负载输入，评估其：

- **运力占用 (Capacity Tax)**：净补给需求占太空电梯系统年运力的比例。
- **经济影响 (OPEX Focus)**：电梯 vs 火箭运水的年度成本差距（对数尺度展示）。
- **安全时间线 (Strategic Reserve Timeline)**：在最坏中断情景下所需的战略安全库存与预部署积累周期。

可视化输出采用与 Q1/Q2 统一的“Golden Trio / Platinum”风格，便于论文整体叙事一致。

---

## 核心参数配置

| 参数 | 符号 | 数值 | 说明 |
| :----- | :----- | :----- | :----- |
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
pip install numpy scipy matplotlib pandas
```

### 运行（推荐流程）

以下命令均以仓库根目录为起点。

#### Question 1

##### 1a&1b 单一模式 + 可视化

```bash
python "Problem B/draft/Question 1/1a&1b/codes/single_mode_opt.py"
python "Problem B/draft/Question 1/1a&1b/codes/visualization_optimized.py"
```

##### 1c 混合优化 + Platinum Quartet 可视化

```bash
python "Problem B/draft/Question 1/1c/codes/comprehensive_transport_model_v5.py"
python "Problem B/draft/Question 1/1c/codes/visualization_1c_platinum.py"
```

#### Question 2

```bash
python "Problem B/draft/Question 2/codes/q2-4.py"
python "Problem B/draft/Question 2/codes/q2_visualization.py"
```

#### Question 3

```bash
python "Problem B/draft/Question 3/codes/water_supply_analysis.py"
python "Problem B/draft/Question 3/codes/q3_visualization_platinum.py"
```

### 主要输出（图表）

- Q1a/1b：`Problem B/draft/Question 1/1a&1b/image/`（对比图、成本/时间尺度优化后的图）
- Q1c：`Problem B/draft/Question 1/1c/image/`（Platinum Quartet 4 张核心叙事图）
- Q2：`Problem B/draft/Question 2/image/`（Golden Trio 关键结果图）
- Q3：`Problem B/draft/Question 3/image/`（Platinum Trio：Feasibility Frontier / Cost Chasm / Reserve Timeline）

---

## 当前进度

- Question 1：建模、求解与可视化已完成（含统一风格输出）
- Question 2：鲁棒性/可靠性分析与可视化已完成
- Question 3：水资源可持续性模型、情景对比与可视化已完成

如需论文整合，可直接将各题 `image/` 中的 PNG 插入 `Problem B/draft/Paper/main.tex`。

---

## 核心结论与建议

### 1. 策略建议

| 目标导向 | 完工时间 (年) | 策略特征 | 预期成本 (NPV) |
| :--------- | :-------------: | :--------- | :--------- |
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


