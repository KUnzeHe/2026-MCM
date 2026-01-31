# 运输系统可靠性与碳排放综合分析模型说明
# Transport System Reliability and Carbon Emissions Analysis Model

## 1. 建模思路与演进逻辑

### 1.1 模型定位
本模型是对 **Comprehensive Transport Model V5** 的扩展，专门针对 **2026 MCM Problem B Question 2** 中关于运输系统**可靠性分析**和**环境影响评估**的核心问题。

### 1.2 问题背景
在 V5 模型中，我们假设了理想化的运行条件：
- 电梯和火箭系统 100% 可用
- 无发射失败和设备故障
- 不考虑维护停机
- 不考虑碳排放成本

然而，现实世界中的大规模航天运输系统必然面临：
1. **设备可靠性问题**：故障、维修、灾难性事件
2. **运营限制**：发射窗口、天气、维护周期
3. **环境约束**：碳排放与气候影响

### 1.3 方法论演进：从理想到现实
本模型采用**双轨对比分析**策略：

| 分析层次 | 理想条件 (Ideal) | 现实条件 (Real) |
|---------|-----------------|----------------|
| 可用性 | $\beta = 1.0$ | $\beta_{E} \approx 0.85$, $\beta_{R} \approx 0.90$ |
| 故障处理 | 忽略 | 需求放大 + 成本加成 |
| 碳排放 | 忽略 | 全生命周期核算 |
| 成本构成 | CAPEX + OPEX | CAPEX + OPEX + 碳成本 + 冗余 |

---

## 2. 可靠性建模框架

### 2.1 电梯系统可靠性模型

#### 2.1.1 有效可用性因子
电梯系统的有效可用性 $\beta_E^{eff}$ 由多个因素复合决定：

$$\beta_E^{eff} = \beta_E^{base} \cdot \beta_E^{fail} \cdot \beta_E^{cat}$$

其中：
- **基础可用性** $\beta_E^{base} = 0.85$：考虑计划内维护、气象条件等
- **故障停机因子** $\beta_E^{fail} = 1 - \frac{\lambda_E \cdot t_{repair}}{365}$
  - $\lambda_E = 0.02$：年故障率
  - $t_{repair} = 14$ 天：平均修复时间
- **灾难性事件因子** $\beta_E^{cat} = 1 - P_{cat}$
  - $P_{cat} = 0.001$：灾难性故障概率

#### 2.1.2 数值计算
$$\beta_E^{fail} = 1 - \frac{0.02 \times 14}{365} \approx 0.9992$$
$$\beta_E^{eff} = 0.85 \times 0.9992 \times 0.999 \approx 0.849$$

### 2.2 火箭系统可靠性模型

#### 2.2.1 有效可用性因子
火箭系统的有效可用性 $\beta_R^{eff}$ 考虑：

$$\beta_R^{eff} = \beta_R^{base} \cdot \beta_R^{fail}$$

其中：
- **基础可用性** $\beta_R^{base} = 1 - (\delta_{window} + \delta_{maint})$
  - $\delta_{window} = 0.05$：发射窗口限制
  - $\delta_{maint} = 0.05$：维护停机占比
- **故障停机因子** $\beta_R^{fail} = 1 - \frac{P_f \cdot T_{down}}{365}$
  - $P_f = 0.01$：发射失败概率
  - $T_{down} = 15$ 天：失败后停机时间

#### 2.2.2 数值计算
$$\beta_R^{base} = 1 - (0.05 + 0.05) = 0.90$$
$$\beta_R^{fail} = 1 - \frac{0.01 \times 15}{365} \approx 0.9996$$
$$\beta_R^{eff} = 0.90 \times 0.9996 \approx 0.90$$

### 2.3 需求放大效应 (Demand Amplification)

由于运输过程中的失败会导致货物损失，实际需要运输的质量大于目标质量：

$$M_{eff} = \frac{M_{tot}}{1 - P_f^{combined}}$$

其中综合失败率：
$$P_f^{combined} = 0.3 \cdot \lambda_E + 0.7 \cdot P_f^R$$

这反映了：
- 30% 的货物经电梯链路，受电梯故障率影响
- 70% 的货物经火箭链路，受发射失败率影响

---

## 3. 容量修正模型

### 3.1 电梯链路有效容量
在非理想条件下，电梯链路的年运力受可用性限制：

$$C_E^{eff}(Y) = \beta_E^{eff} \cdot T_E \cdot Y$$

同时，锚点转运能力也需修正：

$$C_{anchor}^{eff}(Y) = \beta_R^{eff} \cdot N_{anchor} \cdot L_{anchor} \cdot p_A \cdot Y$$

电梯链路总容量取两者最小值（瓶颈约束）：

$$C_E(Y) = \min\left( C_E^{eff}(Y), C_{anchor}^{eff}(Y) \right)$$

### 3.2 火箭链路有效容量
火箭链路容量需考虑动态增长和可用性：

$$C_R(Y) = \int_0^Y \beta_R^{eff} \cdot N(t) \cdot L_{site} \cdot p_B \, dt$$

其中 $N(t)$ 为发射场 Logistic 增长函数（继承自 V5）。

### 3.3 可行性条件
系统可行性要求总容量满足有效需求：

$$C_E(Y) + C_R(Y) \geq M_{eff}$$

---

## 4. 成本修正模型

### 4.1 运营成本加成 (OPEX Adjustment)

#### 4.1.1 电梯 OPEX 修正
$$c_E^{adj} = \frac{c_E}{\eta_{energy}} + \frac{\lambda_E \cdot C_{fix}}{T_E}$$

- $\eta_{energy} = 0.95$：能源效率因子
- $C_{fix} = \$50M$：单次故障修复成本

#### 4.1.2 火箭 OPEX 修正
$$c_R^{adj} = c_R + \frac{P_f \cdot (C_{rocket} + C_{cargo})}{p_B} + \frac{C_{maint}}{L_{site} \cdot p_B}$$

- $C_{rocket} = \$200M$：火箭损失成本
- $C_{cargo} = \$100M$：货物损失成本
- $C_{maint} = \$100M$：单站年维护成本

### 4.2 资本支出冗余 (CAPEX Redundancy)
非理想条件下，需要额外 10% 的基础设施冗余：

$$C_{capex}^{real} = 1.10 \cdot C_{capex}^{ideal}$$

### 4.3 维护成本纳入
电梯系统的年度维护成本 $C_{E,main} = \$500M$/年 需纳入 NPV 计算：

$$C_{opex,E}^{total} = C_{opex,E}^{var} + C_{E,main} \cdot \frac{1 - e^{-\rho Y}}{\rho}$$

---

## 5. 碳排放建模框架

### 5.1 碳排放源识别

本模型考虑两类碳排放：

| 排放类型 | 电梯系统 | 火箭系统 |
|---------|---------|---------|
| **运营排放** | ≈ 0（可再生能源） | 发射燃烧排放 |
| **建设排放** | 电梯建设 | 发射场建设 |

### 5.2 火箭运营碳排放

#### 5.2.1 单次发射排放
基于 Starship 级别火箭的甲烷/液氧燃烧：
$$CO_2^{launch} = 1200 \text{ tons/launch}$$

#### 5.2.2 总运营排放
$$E_{R,op} = N_{launches} \cdot CO_2^{launch}$$

其中总发射次数：
$$N_{launches} = N_{sites} \cdot L_{site}^{eff} \cdot Y$$

### 5.3 建设碳排放

#### 5.3.1 电梯建设排放
$$E_{E,con} = 1 \times 10^6 \text{ tons CO}_2$$

（考虑碳纤维/纳米管生产、空间站建设等）

#### 5.3.2 发射场建设排放
$$E_{R,con} = N_{new} \cdot 50,000 \text{ tons CO}_2 \text{/site}$$

其中 $N_{new} = N_{required} - N_0$ 为新建发射场数量。

### 5.4 碳强度指标
碳强度定义为单位载荷的碳排放：

$$I_{carbon} = \frac{E_{total}}{M_{tot}} \quad [\text{tCO}_2/\text{t payload}]$$

### 5.5 碳成本计算
采用碳价格 $P_{carbon} = \$100$/tCO_2：

$$C_{carbon} = (E_{E,total} + E_{R,total}) \cdot P_{carbon}$$

---

## 6. 综合成本模型

### 6.1 总成本构成
$$C_{total} = C_{capex} + C_{opex} + C_{carbon}$$

展开为组件形式：

$$C_{total} = \underbrace{(C_{E,capex} + C_{R,capex})}_{\text{资本支出}} + \underbrace{(C_{E,opex} + C_{R,opex})}_{\text{运营支出}} + \underbrace{(C_{E,carbon} + C_{R,carbon})}_{\text{碳成本}}$$

### 6.2 组件成本详解

| 成本项 | 理想条件 | 现实条件 |
|-------|---------|---------|
| 电梯 CAPEX | $F_E$ | $F_E$ |
| 电梯 OPEX | $c_E \cdot x \cdot NPV$ | $c_E^{adj} \cdot x \cdot NPV + C_{E,main} \cdot NPV$ |
| 电梯碳成本 | - | $E_{E,total} \cdot P_{carbon}$ |
| 火箭 CAPEX | $N_{new} \cdot C_{site}$ | $1.1 \cdot N_{new} \cdot C_{site}$ |
| 火箭 OPEX | $c_R \cdot m_R \cdot NPV$ | $c_R^{adj} \cdot m_R \cdot NPV$ |
| 火箭碳成本 | - | $E_{R,total} \cdot P_{carbon}$ |

---

## 7. 参数赋值体系

### 7.1 可靠性参数

| 参数 | 符号 | 数值 | 说明 |
|-----|------|-----|------|
| 电梯基础可用性 | $\beta_E^{base}$ | 0.85 | 计划内维护、天气等 |
| 电梯年故障率 | $\lambda_E$ | 0.02 | 约每 50 年一次 |
| 电梯修复时间 | $t_{repair}$ | 14 天 | 平均修复周期 |
| 灾难性故障概率 | $P_{cat}$ | 0.001 | 不可恢复故障 |
| 能源效率 | $\eta_{energy}$ | 0.95 | 实际/理论能耗比 |
| 发射窗口限制 | $\delta_{window}$ | 0.05 | 5% 时间不可发射 |
| 维护停机占比 | $\delta_{maint}$ | 0.05 | 5% 时间维护 |
| 发射失败率 | $P_f$ | 0.01 | 1% 失败率 |
| 失败后停机 | $T_{down}$ | 15 天 | 调查与恢复 |

### 7.2 成本参数

| 参数 | 符号 | 数值 | 说明 |
|-----|------|-----|------|
| 电梯年维护成本 | $C_{E,main}$ | $500M | 年度运维 |
| 电梯故障修复成本 | $C_{fix}$ | $50M | 单次修复 |
| 火箭损失成本 | $C_{rocket}$ | $200M | 火箭报废 |
| 货物损失成本 | $C_{cargo}$ | $100M | 平均货物价值 |
| 单站年维护成本 | $C_{R,maint}$ | $100M | 发射场运维 |
| 冗余系数 | - | 1.10 | 10% 额外投资 |

### 7.3 碳排放参数

| 参数 | 符号 | 数值 | 说明 |
|-----|------|-----|------|
| 单次发射 CO₂ | $CO_2^{launch}$ | 1,200 t | 甲烷/LOX 燃烧 |
| 碳价格 | $P_{carbon}$ | $100/t | 国际碳市场参考 |
| 电梯运营碳排 | $CO_2^{E,op}$ | 0 t/t | 假设可再生能源 |
| 电梯建设碳排 | $CO_2^{E,con}$ | 1 Mt | 全生命周期 |
| 发射场建设碳排 | $CO_2^{site}$ | 50 kt/site | 土建+设备 |

---

## 8. 分析输出与可视化

### 8.1 组件分解表
模型输出详细的组件级对比：

```
Duration     | Component       | Ideal           | Real            | Δ
-------------|-----------------|-----------------|-----------------|-------
24.0 years   | MASS (Mt)       |                 |                 |
             |   Elevator      | 12.888          | 10.960          | -1.928
             |   Rocket        | 87.112          | 92.124          | +5.012
             | CO2 (Mt)        |                 |                 |
             |   Rocket Op     | 522.65          | 575.92          | +53.27
             |   TOTAL         | 523.70          | 576.97          | +53.27
             | COSTS ($B)      |                 |                 |
             |   System TOTAL  | 1,234.5         | 1,456.8         | +222.3
```

### 8.2 关键指标
- **时间延长率**：$(Y_{real} - Y_{ideal}) / Y_{ideal} \times 100\%$
- **成本增加率**：$(C_{real} - C_{total}) / C_{ideal} \times 100\%$
- **碳排放增量**：$\Delta E = E_{real} - E_{ideal}$
- **碳强度对比**：$I_{carbon}^{real}$ vs $I_{carbon}^{ideal}$

---

## 9. 模型优势与局限

### 9.1 主要优势
1. **双轨对比**：同时输出理想与现实结果，量化"理论-实践"差距
2. **组件透明**：成本、质量、排放均按组件分解，便于敏感性分析
3. **碳足迹全核算**：涵盖运营与建设两阶段排放
4. **交互式场景**：支持任意质量和时间约束的快速分析

### 9.2 已知局限
1. **可靠性参数估计**：缺乏空间电梯的历史数据，参数基于类比估计
2. **碳排放假设**：电梯使用可再生能源为乐观假设
3. **线性碳价**：未考虑碳价随时间/排放量的非线性变化
4. **静态可靠性**：未建模可靠性随时间改进（学习曲线）

---

## 10. 与 V5 模型的关系

本模型（q2-4.py）继承了 V5 的核心框架：
- Logistic 基础设施增长模型
- NPV 贴现计算
- 贪心分配策略（电梯优先）

并扩展了以下功能：

| 功能模块 | V5 | Q2-4 |
|---------|----|----- |
| 可靠性建模 | ❌ | ✅ |
| 碳排放计算 | ❌ | ✅ |
| 理想/现实对比 | ❌ | ✅ |
| 交互式 UI | ❌ | ✅ |
| 组件成本分解 | 部分 | ✅ 完整 |

---

## 11. 使用指南

### 11.1 运行模式
```bash
python q2-4.py
```

选择分析模式：
1. **Quick test**：100 Mt 默认场景
2. **Custom scenario**：自定义质量和时间
3. **Carbon-focused test**：多场景碳排放对比
4. **Interactive mode**：引导式输入

### 11.2 输出文件
分析完成后在 `scenario_XXMt/` 目录生成：
- `scenario_report.txt`：完整分析报告

---

## 参考文献
1. V5 模型文档：`comprehensive_transport_model_v3.md`
2. MCM 2026 Problem B 原题
3. SpaceX Starship 技术规格
4. IPCC 碳排放因子数据库
