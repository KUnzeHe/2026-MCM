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
    │   │   │   └── mdFile/                        # 分析文档
    │   │   ├── 1c/                                # 混合运输系统优化
    │   │   │   ├── codes/
    │   │   │   │   ├── comprehensive_transport_model_v5.py  # 主模型V5
    │   │   │   │   └── visualization_1c_platinum.py         # Platinum风格可视化
    │   │   │   ├── image/                         # 输出图表
    │   │   │   └── mdFile/
    │   │   │       ├── 24year_deadline_analysis_report.md   # 2050完工可行性分析
    │   │   │       └── comprehensive_transport_model_v3.md   # 优化框架文档
    │   │   └── others/                            # 草稿与实验性分析
    │   │
    │   ├── Question 2/                            # 问题2：非理想工况与可靠性分析
    │   │   ├── codes/
    │   │   │   ├── q2-4.py                        # Q2核心分析模型
    │   │   │   └── q2_visualization_final.py      # Q2可视化
    │   │   ├── image/                             # 输出图表
    │   │   └── mdFile/
    │   │       ├── transport_reliability_carbon_model.md
    │   │       ├── CarbonTest_100Mt/              # 场景模拟结果
    │   │       ├── CarbonTest_10Mt/
    │   │       ├── CarbonTest_1Mt/
    │   │       ├── CarbonTest_10Mt_5p000yr/
    │   │       ├── CarbonTest_1Mt_0p500yr/
    │   │       └── Custom_100Mt_24p000yr/
    │   │
    │   ├── Question 3/                            # 问题3：水资源补给与循环策略
    │   │   ├── codes/
    │   │   │   ├── water_supply_analysis.py       # 水资源补给模型
    │   │   │   └── q3_visualization_platinum.py   # Q3 Platinum风格可视化
    │   │   ├── image/
    │   │   │   ├── A_feasibility_frontier_platinum.png
    │   │   │   ├── B_cost_chasm_platinum.png
    │   │   │   └── C_reserve_timeline_platinum.png
    │   │   └── mdFile/
    │   │       ├── 2026MCM_ProblemB_Q3_Analysis_Report.md
    │   │       └── water_sustainability_model.md
    │   │
    │   ├── Question 4/                            # 问题4：环境影响与可持续性评估
    │   │   ├── codes/
    │   │   │   ├── q4_comprehensive_analysis.py   # 全面分析主模型
    │   │   │   ├── q4_main.py                     # Q4主程序入口
    │   │   │   └── q4_visualization_platinum.py   # "Platinum Sextet" 绘图程序
    │   │   ├── image/
    │   │   │   ├── Fig1_Environmental_Radar.png                    # 环境影响雷达图
    │   │   │   ├── Fig2_Carbon_Debt_LCA.png                        # 全生命周期碳债
    │   │   │   ├── Fig3_Kessler_Warning.png                        # 轨道危机预警
    │   │   │   ├── Fig4_SEIS_Scorecard.png                         # 最终评分卡
    │   │   │   ├── Fig5_Green_Transition.png                       # 绿色转型敏感度
    │   │   │   └── Fig6_Galactic_Scaleup.png                       # 星际扩展效益
    │   │   └── mdFile/
    │   │       ├── comprehensive_impact_assessment_report.md       # 综合评估报告
    │   │       └── environmental_impact_model_revised.md           # 环境模型详文档
    │   │
    │   └── Paper/                                 # 论文主文件
    │       └── main.tex
    │
    └── Reference/
        └── reference_list.txt                     # 参考资料列表
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

### Question 3: 水资源补给策略 (Water Supply Strategy)

针对 10 万人月球殖民地的生存刚需——水资源，建立供需平衡与物流模型。

**核心发现**：
- **经济死局**：使用火箭运输水资源成本高达 **$1.3 万亿美元/年**，经济上完全不可行。
- **循环是关键**：
  - **Baseline (90%循环)**：占用电梯 **51%** 运力，系统不堪重负。
  - **Optimized (98%循环)**：占用电梯 **5.7%** 运力，不仅可行且成本可控（约 $61亿/年）。
- **战略储备**：为应对潜在中断，建议建立 **3.3万吨** 战略水储备（约需提前累积6个月）。

---

### Question 4: 环境影响与可持续性评估 (Environmental Impact Assessment)

建立全生命周期 (LCA) 环境影响模型，回答“太空电梯是否真的环保”这一核心争议。

**核心模型 (SEIS 体系)**：
1. **大气层影响**：量化火箭对平流层臭氧和中间层的累积破坏。
2. **轨道危机 (Kessler Syndrome)**：模拟高频发射下的轨道碎片指数增长风险。
3. **全生命周期评估 (LCA)**：
   - **建设期 (2026-2050)**：承担基础设施建设和早期火箭运输的碳负债。
   - **运营期 (2050+)**：利用电梯的低碳特性和太空移民红利实现“环境回本”。

**关键结论**：
- **纯火箭方案 = 环境自杀**：在2050年即触发轨道碎片危机临界点，且碳债务无法偿还 (**SEIS: F**)。
- **太空电梯 = 唯一解**：虽然有初期建设排放，但能在运营 **13.6年** 后实现碳中和 (**SEIS: A+**)。
- **星际扩展效应**：随着向火星/金星扩展，环境回收期可从 **989年** (仅月球) 骤降至 **330年** (全太阳系)。

**可视化成果 (Platinum Sextet)**：
- 包含 **环境指纹雷达图**、**碳债盈亏平衡曲线**、**绿色转型敏感度分析** 等6张高精度图表。

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
python visualization_1c_platinum.py
```

### 运行 Question 2 (可靠性与场景模拟)

```bash
cd "Problem B/draft/Question 2/codes"
python q2-4.py
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
python q4_visualization_platinum.py     # 生成 6 张高精度图表
```

### 主要输出

#### Question 1 输出
1. **Pareto 前沿曲线** —— 时间-成本权衡可视化 (Q1c)
2. **最优分配方案** —— 不同时间约束下的 $x^*(Y)$ 
3. **成本对比分析** —— Q1a vs Q1b vs Q1c 横向对标

#### Question 2 输出
1. **可靠性蒙特卡洛模拟结果** —— 1亿吨场景下的风险评估
2. **不同规模场景分析** —— 10Mt、100Mt等多尺度模拟
3. **成本与可靠性权衡图** —— 敏感性分析

#### Question 3 输出
1. **可行性前沿曲线** (`A_feasibility_frontier_platinum.png`) —— 循环效率与运力占用关系
2. **成本深渊分析图** (`B_cost_chasm_platinum.png`) —— 火箭运水的经济不可行性
3. **战略储备时间线** (`C_reserve_timeline_platinum.png`) —— 水资源风险应对方案

#### Question 4 输出
1. **Fig 1: 环境影响雷达** —— 五维多指标综合对比
2. **Fig 2: 碳债盈亏平衡** —— 全生命周期LCA评估
3. **Fig 3: 凯斯勒危机预警** —— 轨道碎片风险演变
4. **Fig 4: SEIS评分卡** —— 最终环保等级评定
5. **Fig 5: 绿色转型曲线** —— 电梯比例敏感度分析
6. **Fig 6: 星际扩展效益** —— 未来开发的环保边际效应

---

## 核心结论与建议

### 0. 全项目执行摘要 (Project Executive Summary)

本项目通过四个递进式的问题，构建了从**理想优化 → 现实约束 → 资源循环 → 环境评估**的完整决策框架，最终证明：

**太空电梯不仅是成本最优方案，更是唯一符合长期可持续性要求的选择。**

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
2026-2050 [建设期]
├─ 时间约束: 28-32年 (不要死卡2050)
├─ 运输模式: 13% 电梯 + 87% 火箭 (容量限制)
├─ 环保代价: ~1450 Mt CO2 债务 (必要的infrastructure investment)
└─ 成本范围: $15T-$25T (可控范围)

2050+ [运营期]
├─ 运输模式: 100% 电梯 (零排放)
├─ 水循环: 98% 闭环 (生存刚需)
├─ 环保转向: 年度负碳排放 1.5 Mt (开始偿还债务)
└─ 未来展望: 星际扩展→330年完全回本→永续发展
```

---

## 参考资料与进一步阅读

### 核心技术文献
1. 火箭发射成本分析：[SpaceX Starship Economics](https://spaceinsider.tech/2023/08/16/how-much-does-it-cost-to-launch-a-rocket/)
2. 美国电力价格数据：[Global Petrol Prices - USA Electricity](https://zh.globalpetrolprices.com/USA/electricity_prices/)
3. 国际空间站水循环系统：ISS ECLSS (Environmental Control and Life Support System)
4. 凯斯勒综合症研究：Kessler & Cour-Palais (1978), "Collision Frequency of Artificial Satellites"

### 项目相关文档
- [Q1 优化框架详文档](Problem%20B/draft/Question%201/1c/mdFile/comprehensive_transport_model_v3.md)
- [Q1 2050年截止可行性分析](Problem%20B/draft/Question%201/1c/mdFile/24year_deadline_analysis_report.md)
- [Q2 可靠性与碳排放模型](Problem%20B/draft/Question%202/mdFile/transport_reliability_carbon_model.md)
- [Q3 水资源可持续性模型](Problem%20B/draft/Question%203/mdFile/water_sustainability_model.md)
- [Q4 环境影响评估报告](Problem%20B/draft/Question%204/mdFile/comprehensive_impact_assessment_report.md)
- [Q4 环境模型详细文档](Problem%20B/draft/Question%204/mdFile/environmental_impact_model_revised.md)


