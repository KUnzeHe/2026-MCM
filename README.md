# 2026 MCM Problem B: Space Transportation Optimization

[![MCM 2026](https://img.shields.io/badge/MCM-2026-blue.svg)](https://www.comap.com/contests/mcm-icm)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 项目概述

本项目是 **2026年美国大学生数学建模竞赛 (MCM)** Problem B 的完整解决方案，研究主题为：**太空运输系统优化与可持续性评估** —— 探索如何高效地将约 **1亿吨** 建材从地球运输至月球，用于建设月球殖民基地，并评估其环境与长期可持续性。

### 核心问题

在给定的太空电梯系统和传统火箭运输两种方式下：
1. 如何制定最优的混合运输策略，实现 **时间** 与 **成本** 的最佳权衡？
2. 如何在不确定性条件下保障系统的 **可靠性与鲁棒性**？
3. 10万人殖民地的 **水资源补给** 如何实现可持续？
4. 整体方案的 **环境影响** 如何？太空电梯是否真的环保？

### 论文标题
**Ascending to the Moon: A Multi-Objective Optimization Framework for Sustainable Space Logistics and Environmental Assessment**
*(Subtitle: Balancing Cost, Reliability, and Planetary Impact in the Era of Space Colonization)*


---

## 项目结构

```
2026MCM/
├── README.md                                      # 项目说明文档
└── Problem B/
    ├── draft/
    │   ├── Question 1/                            # 问题1：理想工况下的运输优化
    │   │   ├── 1a&1b/                             # 单一运输系统模型
    │   │   │   ├── codes/
    │   │   │   │   ├── 1a_cost.py                 # 纯电梯成本模型
    │   │   │   │   ├── single_mode_opt.py         # 单一模式优化
    │   │   │   │   └── visualization_optimized.py # 可视化脚本
    │   │   │   ├── image/                         # 输出图表
    │   │   │   │   ├── fig1_optimized_overview.png
    │   │   │   │   ├── fig3_optimized_cost_structure.png
    │   │   │   │   ├── fig4_optimized_breakeven.png
    │   │   │   │   └── fig7_optimized_logistic.png
    │   │   │   └── mdFile/
    │   │   │       ├── single_mode_models.md      # 单一模式模型说明
    │   │   │       └── sensitivity_analysis_ideas_q1.md
    │   │   ├── 1c/                                # 混合运输系统优化
    │   │   │   ├── codes/
    │   │   │   │   ├── comprehensive_transport_model_v5.py  # 主模型V5
    │   │   │   │   ├── q1_sensitivity_platinum.py           # 敏感性分析
    │   │   │   │   └── visualization_1c_platinum.py         # Platinum风格可视化
    │   │   │   ├── image/                         # 输出图表
    │   │   │   │   ├── Fig1_Strategic_Landscape.png
    │   │   │   │   ├── Fig2_Execution_Plan.png
    │   │   │   │   ├── Fig3_Capacity_Dynamics.png
    │   │   │   │   ├── Fig4_Tradeoff_DeepDive.png
    │   │   │   │   └── Fig5_Sensitivity_Core.png
    │   │   │   └── mdFile/
    │   │   │       ├── 24year_deadline_analysis_report.md   # 2050完工可行性分析
    │   │   │       └── comprehensive_transport_model_v3.md  # 优化框架文档
    │   │   └── others/                            # 草稿与实验性分析
    │   │
    │   ├── Question 2/                            # 问题2：非理想工况与可靠性分析
    │   │   ├── codes/
    │   │   │   ├── q2-4.py                        # Q2核心分析模型
    │   │   │   ├── q2_sensitivity_platinum.py     # 可靠性敏感性分析
    │   │   │   └── q2_visualization_final.py      # Q2可视化
    │   │   ├── image/                             # 输出图表
    │   │   │   ├── Fig1_Radar_Gap.png             # 理想vs现实对比雷达
    │   │   │   ├── Fig2_Cost_Waterfall.png        # 成本瀑布分解
    │   │   │   ├── Fig3_Carbon_Truth.png          # 碳排放真相
    │   │   │   └── Fig4_Reliability_Sensitivity.png # 可靠性敏感度
    │   │   └── mdFile/
    │   │       ├── transport_reliability_carbon_model.md
    │   │       ├── sensitivity_analysis_ideas_q2.md
    │   │       └── [场景模拟结果目录]/             # CarbonTest_* 等多场景
    │   │
    │   ├── Question 3/                            # 问题3：水资源补给与循环策略
    │   │   ├── codes/
    │   │   │   ├── water_supply_analysis.py       # 水资源补给模型
    │   │   │   ├── q3_visualization.py            # 基础可视化
    │   │   │   └── q3_visualization_platinum.py   # Platinum风格可视化
    │   │   ├── image/
    │   │   │   ├── A_feasibility_frontier_platinum.png
    │   │   │   ├── B_cost_chasm_platinum.png
    │   │   │   ├── C_reserve_timeline_platinum.png
    │   │   │   └── D_reserve_sensitivity_platinum.png
    │   │   └── mdFile/
    │   │       ├── 2026MCM_ProblemB_Q3_Analysis_Report.md
    │   │       ├── water_sustainability_model.md
    │   │       └── sensitivity_analysis_ideas_q3.md
    │   │
    │   └── Question 4/                            # 问题4：环境影响与可持续性评估
    │       ├── codes/
    │       │   ├── q4_comprehensive_analysis.py   # 全面分析主模型
    │       │   ├── q4_main.py                     # Q4主程序入口
    │       │   ├── q4_environmental_model.py      # 环境模型
    │       │   ├── q4_orbital_risk_model.py       # 轨道风险模型
    │       │   ├── q4_sensitivity_analysis.py     # 敏感性分析
    │       │   └── q4_visualization_platinum.py   # "Platinum Sextet" 绘图
    │       ├── data/                              # 分析数据输出
    │       │   ├── q4_analysis_results.json
    │       │   ├── q4_pareto_front.csv
    │       │   └── q4_sensitivity_analysis.csv
    │       ├── image/                             # Platinum Sextet 六图
    │       │   ├── Fig1_Environmental_Radar.png   # 环境影响雷达图
    │       │   ├── Fig2_Carbon_Debt_LCA.png       # 全生命周期碳债
    │       │   ├── Fig3_Kessler_Warning.png       # 轨道危机预警
    │       │   ├── Fig4_SEIS_Scorecard.png        # 最终评分卡
    │       │   ├── Fig5_Green_Transition.png      # 绿色转型敏感度
    │       │   └── Fig6_Galactic_Scaleup.png      # 星际扩展效益
    │       └── mdFile/
    │           ├── comprehensive_impact_assessment_report.md
    │           ├── environmental_impact_model_revised.md
    │           └── sensitivity_analysis_ideas_q4.md
    │
    ├── Paper/                                     # 论文主文件
    │   ├── main.tex                               # LaTeX主文件
    │   ├── main.pdf                               # 编译后的PDF
    │   ├── Paper_Outline.md                       # 论文大纲
    │   ├── Figures/                               # 论文插图汇总
    │   └── Reference/
    │
    └── Reference/                                 # 参考文献
        ├── ref.bib                                # BibTeX引用
        ├── ISEC-Study-2024-Apex-Anchor-*.pdf      # 电梯研究报告
        └── falcon-users-guide-2025-*.pdf          # SpaceX参考
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

2. **发射频率物理极限模型** —— 基于周转时间约束
   $$L_{\max} = \frac{365 \cdot \eta}{t_{\text{cycle}}}$$

3. **多目标优化求解 (Ver 5.0)** —— 生成 Pareto 前沿，分析时间与成本的权衡

4. **2050 完工 (24年工期) 可行性专项分析**：
   - **风险评估**：Monte Carlo 模拟显示按时完工概率仅为 **24.3%**
   - **成本预估**：NPV 约为 **\$40.50 万亿**（主要由火箭承担）
   - **运力余量**：仅 7.8%，极易受干扰

**可视化输出 (Platinum Five)**：
| 图表 | 内容 |
|:-----|:-----|
| Fig1_Strategic_Landscape | 战略全景图 - Pareto前沿 |
| Fig2_Execution_Plan | 执行计划时间线 |
| Fig3_Capacity_Dynamics | 运力动态演化 |
| Fig4_Tradeoff_DeepDive | 权衡深度分析 |
| Fig5_Sensitivity_Core | 核心敏感性分析 |

---

### Question 2: 非理想工况下的系统分析

引入真实世界的不确定性因素与风险模型，从"理想"走向"现实"：

#### 可靠性建模框架

| 系统 | 基础可用性 | 故障率 | 有效可用性 |
|:-----|:---------|:------|:---------|
| **电梯** | $\beta_E^{base} = 0.85$ | $\lambda_E = 0.02$ | $\beta_E^{eff} \approx 0.849$ |
| **火箭** | $\beta_R^{base} = 0.90$ | $P_f = 0.01$ | $\beta_R^{eff} \approx 0.90$ |

#### 分析维度

| 维度 | 关键因素 | 建模方法 |
|:-----|:---------|:---------|
| **可靠性** | 故障率 $\lambda$、维修时间 | 引入有效利用率因子 $\beta \in [0,1]$ |
| **成本** | 运维成本、事故损失、冗余投资 | $C_{real} = C_{base} \cdot (1 + \alpha) + E[C_{risk}]$ |
| **碳排放** | 全生命周期核算 | LCA 方法论整合 |
| **场景** | 1亿吨需求波动 | Monte Carlo 敏感性测试 |

**关键结论**：
- 总成本预计上升 **30%-50%**
- **需求放大效应**：由于运输失败，实际需运输量 $M_{eff} = \frac{M_{tot}}{1 - P_f^{combined}}$
- **鲁棒性策略**：当火箭故障率 $\lambda_R$ 较高时，应显著增加电梯的运量分配

**可视化输出 (Quadrant)**：
| 图表 | 内容 |
|:-----|:-----|
| Fig1_Radar_Gap | 理想vs现实五维雷达对比 |
| Fig2_Cost_Waterfall | 成本构成瀑布图 |
| Fig3_Carbon_Truth | 碳排放真相曲线 |
| Fig4_Reliability_Sensitivity | 可靠性参数敏感度 |

---

### Question 3: 水资源补给策略 (Water Supply Strategy)

针对 10 万人月球殖民地的生存刚需——水资源，建立供需平衡与物流模型。

#### 需求侧模型 (Water Metabolism)

年总用水需求：
$$ D_{gross} = P \cdot (w_{dom} + w_{ag} + w_{ind}) \cdot 365 $$

净补给需求（考虑循环）：
$$ M_{water} = D_{gross} \cdot (1 - \eta_{sys}) + L_{struct} $$

#### 核心发现

| 循环效率 $\eta$ | 年补给需求 | 电梯占用率 | 年成本 | 可行性 |
|:---------------|:----------|:---------|:--------|:----:|
| **70%** (故障) | 1.64 Mt | **305%** | $1.3T | ✗ 不可行 |
| **90%** (ISS标准) | 273,750 t | **51%** | $550B | △ 勉强 |
| **98%** (优化目标) | 30,660 t | **5.7%** | **$61B** | ✓ 可行 |

- **经济死局**：使用火箭运输水资源成本高达 **$1.3 万亿美元/年**，经济上完全不可行
- **循环是关键**：必须达到 **98% 循环效率**，才能使系统可持续
- **战略储备**：为应对潜在中断，建议建立 **3.3万吨** 战略水储备（约需提前累积6个月）

**可视化输出 (Quadrant)**：
| 图表 | 内容 |
|:-----|:-----|
| A_feasibility_frontier | 循环效率-运力占用可行性前沿 |
| B_cost_chasm | 火箭运水的"成本深渊" |
| C_reserve_timeline | 战略储备累积时间线 |
| D_reserve_sensitivity | 储备量敏感性分析 |

---

### Question 4: 环境影响与可持续性评估 (Environmental Impact Assessment)

建立全生命周期 (LCA) 环境影响模型与 **SEIS (Space Environmental Impact Score)** 评分体系，回答"太空电梯是否真的环保"这一核心争议。

#### SEIS 评估框架

| 维度 | 评估内容 | 关键指标 |
|:-----|:--------|:--------|
| **大气层** | 火箭对平流层臭氧/中间层的破坏 | 年排放量 (Mt CO2-eq) |
| **轨道环境** | Kessler综合症风险 | 碎片风险倍数 |
| **地面LCA** | 全生命周期碳足迹 | 建设期碳债 + 运营期排放 |
| **资源** | 材料消耗与能源效率 | 单位载荷能耗 |
| **可持续性** | 环境回收期 | Payback Years |

#### 核心情景对比

| 评估指标 | **纯电梯 (理想)** | **纯火箭 (基准)** | **混合方案 (现实)** |
| :--- | :---: | :---: | :---: |
| **建设期碳负债 (Mt CO2)** | 20.0 | 1669.7 | 1466.0 |
| **运营期年排放 (Mt/yr)** | ~0.03 | 13.7 | 1.8 (初期) |
| **凯斯勒风险倍数 (2050)** | 1.0x (Baseline) | **5.0x (Critical!)** | 1.5x (Safe) |
| **环境回收期 (Years)** | **13.6** | **∞ (永不回本)** | 997.2 |
| **SEIS 评分** | **0.12 (A+)** | **12.5 (F)** | 8.5 (D-) |

#### 关键结论

1. **纯火箭方案 = 环境自杀**：
   - 2050年即触发轨道碎片危机临界点（风险 5.0x）
   - 运营排放（13.7 Mt/yr）远超减排红利（1.5 Mt/yr），永远无法回本
   - **SEIS: F (Fail)**

2. **太空电梯 = 唯一可持续解**：
   - 虽有初期建设排放，但运营 **13.6年** 后即可实现碳中和
   - **SEIS: A+**

3. **星际扩展效应**：
   | 扩展阶段 | 殖民人口 | 年碳红利 | 回收期 |
   |:--------|:------:|:-------:|:-----:|
   | Phase 1: Moon | 100,000 | 1.5 Mt | 989年 |
   | Phase 2: +Mars | 300,000 | 4.5 Mt | 560年 |
   | Phase 3: +Venus | 350,000 | 5.25 Mt | 621年 |
   | **Phase 4: Full System** | **780,000** | **11.7 Mt** | **330年** |

**可视化输出 (Platinum Sextet)**：
| 图表 | 内容 |
|:-----|:-----|
| Fig1_Environmental_Radar | 五维环境指纹雷达图 |
| Fig2_Carbon_Debt_LCA | 全生命周期碳债盈亏曲线 |
| Fig3_Kessler_Warning | 凯斯勒危机预警时间线 |
| Fig4_SEIS_Scorecard | SEIS最终评分卡 |
| Fig5_Green_Transition | 绿色转型敏感度分析 |
| Fig6_Galactic_Scaleup | 星际扩展边际效益 |

---

## 核心参数配置

### 运输系统参数

| 参数 | 符号 | 数值 | 说明 |
|:-----|:-----|:-----|:-----|
| 总运输质量 | $M_{\text{tot}}$ | $10^8$ 吨 | 题目给定 |
| 电梯年吞吐量 | $T_E$ | 537,000 吨/年 | 3 Harbours × 179,000 |
| 电梯单位成本 | $c_E$ | \$2.7/kg | 电力 + 转运燃料 |
| 电梯固定成本 | $F_E$ | \$100B | 基础设施建设 |
| 火箭单位成本 | $c_R$ | \$720/kg (未来) | Starship级可重复使用 |
| 火箭单次载荷 | $p_B$ | 150 吨 | Starship级 |
| 发射场初始数量 | $N_0$ | 10 | 初始可用场地 |
| 发射场承载力 | $K$ | 80 | 全球最大容量 (Logistic上限) |

### 可靠性参数

| 参数 | 电梯系统 | 火箭系统 |
|:-----|:--------|:--------|
| 基础可用性 | $\beta_E^{base} = 0.85$ | $\beta_R^{base} = 0.90$ |
| 年故障率 | $\lambda_E = 0.02$ | $P_f = 0.01$ |
| 维修时间 | 14 天 | 15 天 |
| 灾难性故障概率 | 0.1% | N/A |

### 水资源参数

| 参数 | 数值 | 说明 |
|:-----|:-----|:-----|
| 殖民地人口 | 100,000 人 | 题目给定 |
| 人均日用水 | 7.5 kg/人/天 | ISS标准 (生活+农业+工业) |
| 目标循环效率 | 98% | 生存刚需边界 |
| 战略储备量 | 33,750 吨 | 6个月安全缓冲 |

---

## 环境配置与运行

### 依赖安装

```bash
pip install numpy scipy matplotlib pandas
```

### 运行 Question 1 (单一模式优化与混合系统)

```bash
# 运行 Q1a 和 Q1b 单一模式
cd "Problem B/draft/Question 1/1a&1b/codes"
python 1a_cost.py
python single_mode_opt.py
python visualization_optimized.py

# 运行 Q1c 混合运输系统优化
cd "Problem B/draft/Question 1/1c/codes"
python comprehensive_transport_model_v5.py
python q1_sensitivity_platinum.py
python visualization_1c_platinum.py
```

### 运行 Question 2 (可靠性与场景模拟)

```bash
cd "Problem B/draft/Question 2/codes"
python q2-4.py
python q2_sensitivity_platinum.py
python q2_visualization_final.py
```

### 运行 Question 3 (水资源补给与循环系统)

```bash
cd "Problem B/draft/Question 3/codes"
python water_supply_analysis.py
python q3_visualization_platinum.py
```

### 运行 Question 4 (全面环境影响评估)

```bash
cd "Problem B/draft/Question 4/codes"
python q4_comprehensive_analysis.py     # 生成定量分析数据与报告
python q4_visualization_platinum.py     # 生成 Platinum Sextet 六图
```

### 编译论文

```bash
cd "Problem B/Paper"
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### 主要输出

#### Question 1 输出
| 文件 | 说明 |
|:-----|:-----|
| `Fig1_Strategic_Landscape.png` | Pareto前沿与战略全景 |
| `Fig2_Execution_Plan.png` | 最优执行计划时间线 |
| `Fig3_Capacity_Dynamics.png` | 运力动态演化 (Logistic增长) |
| `Fig4_Tradeoff_DeepDive.png` | 时间-成本权衡深度分析 |
| `Fig5_Sensitivity_Core.png` | 核心参数敏感性分析 |

#### Question 2 输出
| 文件 | 说明 |
|:-----|:-----|
| `Fig1_Radar_Gap.png` | 理想vs现实五维差距雷达 |
| `Fig2_Cost_Waterfall.png` | 成本构成瀑布分解 |
| `Fig3_Carbon_Truth.png` | 碳排放真相对比 |
| `Fig4_Reliability_Sensitivity.png` | 可靠性参数敏感度热力图 |

#### Question 3 输出
| 文件 | 说明 |
|:-----|:-----|
| `A_feasibility_frontier_platinum.png` | 循环效率-运力占用可行性前沿 |
| `B_cost_chasm_platinum.png` | 火箭运水的"成本深渊" |
| `C_reserve_timeline_platinum.png` | 战略水储备累积时间线 |
| `D_reserve_sensitivity_platinum.png` | 储备量参数敏感性分析 |

#### Question 4 输出 (Platinum Sextet)
| 文件 | 说明 |
|:-----|:-----|
| `Fig1_Environmental_Radar.png` | 五维环境影响雷达图 |
| `Fig2_Carbon_Debt_LCA.png` | 全生命周期碳债盈亏曲线 |
| `Fig3_Kessler_Warning.png` | 凯斯勒危机预警时间线 |
| `Fig4_SEIS_Scorecard.png` | SEIS最终评分卡 |
| `Fig5_Green_Transition.png` | 绿色转型敏感度曲线 |
| `Fig6_Galactic_Scaleup.png` | 星际扩展边际效益图 |

---

## 核心结论与建议

### 0. 全项目执行摘要 (Executive Summary)

本项目通过四个递进式的问题，构建了从**理想优化 → 现实约束 → 资源循环 → 环境评估**的完整决策框架：

```
┌─────────────────────────────────────────────────────────────────────┐
│  Q1: 理想优化        Q2: 现实约束       Q3: 资源循环       Q4: 环境评估  │
│  ───────────────────────────────────────────────────────────────────│
│  Pareto前沿    →   可靠性修正    →   水循环刚需    →   SEIS评级     │
│  时间-成本权衡      故障/维修         98%循环率         A+/F评价     │
│  混合策略           Monte Carlo       战略储备          LCA分析      │
└─────────────────────────────────────────────────────────────────────┘
```

**核心结论：太空电梯不仅是成本最优方案，更是唯一符合长期可持续性要求的选择。**

---

### 1. Q1 策略建议：时间-成本权衡

| 目标导向 | 完工时间 | 策略特征 | 预期成本 | 风险度 |
|:---------|:--------:|:---------|:---------|:-----:|
| **极速推进** | 20-25年 | 激进扩建火箭场 (>85%) | > $45T | 极高 |
| **推荐平衡** | **28-32年** | **混合最优点 (Knee Point)** | **$15T-$25T** | **中** |
| **成本最优** | > 40年 | 以电梯为主 (>90%) | < $5T | 低 |

**关键结论**：
- 2050年完工 (24年工期) 的按时率仅 **24.3%**，极不现实。
- 推荐将工期延长至 **28-32年**，在时间和成本间找到最佳平衡。

---

### 2. Q2 可靠性指导：鲁棒性策略

在引入真实故障率、维修周期、事故风险后：
- **总成本上升 30-50%**
- **电梯优势凸显**：由于"连续流"特性，面对不确定性时更稳健
- **推荐策略**：当火箭故障率升高时，应主动增加电梯分配比例，形成有效的风险对冲

---

### 3. Q3 生存刚需：水资源循环的硬边界

| 循环效率 | 年补给需求 | 电梯占用率 | 年成本 | 可行性 |
|:--------|:----------|:---------|:--------|:----:|
| **90%** (ISS标准) | 273,750吨 | **51%** | $550B | 勉强 |
| **98%** (优化) | 30,660吨 | **5.7%** | **$61B** | ✓ |
| **70%** (故障) | 1.64M吨 | 305% | $1.3T | ✗ |

**核心发现**：
- **经济死局**：火箭运水的成本高达 **$1.3万亿/年**，完全不可行。
- **物理刚需**：必须达到 **98%循环效率**，才能使系统可持续。
- **战略储备**：建议积累 **33,750吨** 战略水资源，约需6个月预先投入。

---

### 4. Q4 环保终审：全生命周期环境评分

#### 对标三大方案：

| 方案 | 初始碳债 | 年运营排放 | 回本时间 | **SEIS** | 评级 |
|:-----|:------:|:-------:|:-------:|:------:|:----:|
| **纯火箭** | 1,669 Mt | +13.7 Mt/yr | ∞ | 12.5 | **F** |
| **混合建设** | 1,466 Mt | +1.8 Mt/yr | 997yr | 8.5 | **D-** |
| **纯电梯** | **20 Mt** | **≈0** | **13.6yr** | **0.12** | **A+** |

#### 关键发现：

1. **轨道危机**：纯火箭方案在2050年即逼近凯斯勒综合症临界点（风险5.0x），触发灾难级轨道碎片危机。
2. **碳债陷阱**：火箭方案的运营排放（13.7 Mt/yr）远超减排红利（1.5 Mt/yr），永远无法实现环境回本。
3. **星际救赎**：虽然月球独立项目回收期长达989年，但若扩展至火星和金星，利用规模效应，回收期可缩短至330年。

---

### 5. 最终建议：综合框架

**统一的"分阶段、多约束"决策框架**：

```
┌─────────────────────────────────────────────────────────────────────┐
│                     2026-2050 【建设期】                              │
├─────────────────────────────────────────────────────────────────────┤
│  ▸ 时间约束: 28-32年 (不要死卡2050，按时完工概率仅24.3%)               │
│  ▸ 运输模式: 13% 电梯 + 87% 火箭 (受限于电梯早期运力上限)              │
│  ▸ 环保代价: ~1450 Mt CO2 债务 (必要的基础设施投资)                   │
│  ▸ 成本范围: $15T-$25T (Knee Point最优区间)                          │
│  ▸ 运力扩建: 发射场按Logistic曲线从10→60+扩张                        │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                     2050+ 【运营期】                                  │
├─────────────────────────────────────────────────────────────────────┤
│  ▸ 运输模式: 100% 电梯 (零排放运营)                                   │
│  ▸ 水循环: 98% 闭环 (生存刚需边界)                                    │
│  ▸ 战略储备: 33,750吨水 (6个月安全缓冲)                               │
│  ▸ 环保转向: 年度负碳排放 1.5 Mt (开始偿还建设期债务)                  │
│  ▸ 环境回收: 单月球989年，全太阳系330年                               │
└─────────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────────┐
│                    2100+ 【星际扩展】                                 │
├─────────────────────────────────────────────────────────────────────┤
│  ▸ 扩展目标: 月球 → 火星 → 金星 → 全太阳系                            │
│  ▸ 殖民人口: 10万 → 78万+                                            │
│  ▸ 规模效应: 环境回收期从989年骤降至330年                             │
│  ▸ 终极愿景: "越开发，越环保" —— 永续发展                             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 参考资料与进一步阅读

### 核心技术文献
1. **太空电梯研究**：ISEC Study 2024 - Apex Anchor Transportation Node
2. **火箭发射成本**：[SpaceX Starship Economics](https://spaceinsider.tech/2023/08/16/how-much-does-it-cost-to-launch-a-rocket/)
3. **美国电力价格**：[Global Petrol Prices - USA Electricity](https://zh.globalpetrolprices.com/USA/electricity_prices/)
4. **水循环系统**：ISS ECLSS (Environmental Control and Life Support System)
5. **轨道碎片研究**：Kessler & Cour-Palais (1978), "Collision Frequency of Artificial Satellites"
6. **SpaceX参考**：Falcon Users Guide 2025

### 项目相关文档

#### Question 1 文档
- [单一模式模型说明](Problem%20B/draft/Question%201/1a%261b/mdFile/single_mode_models.md)
- [混合优化框架 V3](Problem%20B/draft/Question%201/1c/mdFile/comprehensive_transport_model_v3.md)
- [2050年截止可行性分析](Problem%20B/draft/Question%201/1c/mdFile/24year_deadline_analysis_report.md)

#### Question 2 文档
- [可靠性与碳排放模型](Problem%20B/draft/Question%202/mdFile/transport_reliability_carbon_model.md)
- [Q2敏感性分析思路](Problem%20B/draft/Question%202/mdFile/sensitivity_analysis_ideas_q2.md)

#### Question 3 文档
- [水资源可持续性模型](Problem%20B/draft/Question%203/mdFile/water_sustainability_model.md)
- [Q3完整分析报告](Problem%20B/draft/Question%203/mdFile/2026MCM_ProblemB_Q3_Analysis_Report.md)

#### Question 4 文档
- [环境影响综合评估报告](Problem%20B/draft/Question%204/mdFile/comprehensive_impact_assessment_report.md)
- [环境模型详细文档](Problem%20B/draft/Question%204/mdFile/environmental_impact_model_revised.md)

#### 论文相关
- [论文大纲](Problem%20B/Paper/Paper_Outline.md)
- [论文PDF](Problem%20B/Paper/main.pdf)

---

*Last Updated: 2026.2.2*


