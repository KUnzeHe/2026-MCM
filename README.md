# 2026 MCM Problem B: Space Transportation Optimization

[![MCM 2026](https://img.shields.io/badge/MCM-2026-blue.svg)](https://www.comap.com/contests/mcm-icm)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 项目概述

本项目是 **2026年美国大学生数学建模竞赛 (MCM)** Problem B 的解决方案，研究主题为：**太空运输系统优化** —— 探索如何高效地将约 **1亿吨** 建材从地球运输至月球，用于建设月球基地。

### 🎯 核心问题

在给定的太空电梯系统和传统火箭运输两种方式下，如何制定最优的混合运输策略，实现 **时间** 与 **成本** 的最佳权衡？

---

## 📁 项目结构

```
2026MCM/
├── README.md                          # 项目说明文档
└── Problem B/
    ├── draft/
    │   ├── Question 1/                # 问题1：理想工况下的运输优化
    │   │   ├── 1a&1b/                 # 单一运输系统模型
    │   │   │   ├── codes/             # 代码实现
    │   │   │   │   ├── 1a_cost.py     # 电梯系统成本计算
    │   │   │   │   └── single_mode_opt.py
    │   │   │   └── mdFile/
    │   │   │       └── single_mode_models.md
    │   │   ├── 1c/                    # 混合运输系统优化
    │   │   │   ├── codes/
    │   │   │   │   ├── comprehensive_transport_model_v4.py
    │   │   │   │   └── comprehensive_transport_model_v5.py  # 主模型
    │   │   │   └── mdFile/
    │   │   │       ├── comprehensive_optimization_framework_v4.md  # 核心理论框架
    │   │   │       └── transport_optimization_readme.md
    │   │   └── others/                # 其他辅助分析
    │   │       ├── pareto_analysis.py
    │   │       └── mixed_plan_opt.py
    │   └── Question 2/                # 问题2：非理想工况分析
    │       └── mdFile/
    │           └── Q2_reliability_and_cost_analysis.md
    └── Reference/
        └── reference_list.txt         # 参考资料列表
```

---

## 🚀 研究内容

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

#### 1c. 混合运输系统优化 (Hybrid System) ⭐

**核心创新**：建立了完整的多目标优化框架，包含：

1. **Logistic 基础设施增长模型**
   $$N(t) = \frac{K}{1 + \left( \frac{K - N_0}{N_0} \right) e^{-r t}}$$

2. **周转时间模型** —— 确定物理发射频率上限
   $$L_{\max} = \frac{365 \cdot \eta}{t_{\text{cycle}}}$$

3. **火箭方程物理增益分析** —— 量化电梯vs火箭效率差异
   $$\beta = p_A / p_B \approx 4 \sim 8$$

4. **多目标优化求解** —— 生成 Pareto 前沿
   $$\min_{x} C(x, Y_{\max}) = C_{\text{capex}} + C_{\text{opex}}$$

---

### Question 2: 非理想工况下的系统分析

引入真实世界的不确定性因素：

| 因素类型 | 电梯系统 | 火箭系统 |
|:---------|:---------|:---------|
| **可靠性** | 定期检修、缆绳晃动、故障维修 | 发射失败、故障停飞 |
| **成本修正** | $c_E^{real} = c_E \cdot (1+\alpha) + E[cost_{fault}]$ | $c_R^{real} = c_R + c_{R,maint} + c_{risk}$ |
| **运力修正** | $T_E^{real} = T_E \cdot \beta_E$ | $L_{\max}^{real} = L_{\max} \cdot (1 - P_{loss})$ |

**关键结论**：
- 总成本预计上升 **30%-50%**
- 火箭故障率高时，最优策略向太空电梯倾斜
- 混合方案提供更好的系统冗余性

---

## ⚙️ 核心参数配置

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

## 🔧 环境配置与运行

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

## 📊 核心结论

### 最优策略建议

| 目标导向 | 策略 | 分配比例 | 预期成本 |
|:---------|:-----|:---------|:---------|
| **时间优先** | 全速建设发射场，激进技术路线 | 火箭 >80% | 极高 |
| **成本优先** | 延长工期至30-40年 | 电梯 >90% | 较低 |
| **最优平衡 (推荐)** | 识别 Knee Point (~25年) | 混合优化 | 中等 |

### 关键洞察

1. **电梯优先原则**：由于 $c_E \ll c_R$，在时间允许范围内应优先使用电梯
2. **瓶颈识别**：电梯系统受限于锚点转运能力，火箭受限于发射场建设速度
3. **风险对冲**：混合方案提供了分布式冗余，降低单点故障风险

---

## 📚 参考资料

1. [火箭发射成本分析](https://spaceinsider.tech/2023/08/16/how-much-does-it-cost-to-launch-a-rocket/)
2. [美国电力价格](https://zh.globalpetrolprices.com/USA/electricity_prices/)
3. SpaceX Starship 技术参数
4. NASA 空间电梯概念研究

---

## 👥 团队成员

*[待补充]*

---

## 📄 许可证

本项目仅用于学术研究与数学建模竞赛，遵循 MIT 许可证。

---

<p align="center">
  <i>🌙 Ad Astra Per Aspera - 循此苦旅，以达天际</i>
</p>