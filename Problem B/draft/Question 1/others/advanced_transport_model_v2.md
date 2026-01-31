# 运输系统高级优化模型说明 (Model V2)

## 0. 模型演进与核心理念
本模型在基础混合方案 (Scenario 1c) 的基础上，引入了**动态性 (Dynamics)**、**物理约束 (Physical Constraints)** 和**随机性 (Stochasticity)**，以解决 2050 年前完成 1 亿吨运输任务的现实可行性问题。

核心升级点：
1.  **动态扩建**：发射场数量 $N(t)$ 不再是常数，而是遵循 Logistic 增长曲线。
2.  **物理极限约束**：通过发射周期 (Turnaround Time) 模型估算发射频率 $L_{\max}$ 的理论上限。
3.  **载荷差异化**：基于火箭方程 ($\Delta v$ Analysis) 严格区分地面发射与锚点发射的载荷能力。
4.  **成本修正**：引入基建资本支出 (CAPEX) 和时间价值 (NPV)。
5.  **不确定性分析**：对火箭技术参数采用 Monte Carlo 模拟以评估风险。

---

## 1. 动态扩建模型：发射场 Logistic 增长
由于现有 10 个发射场无法满足需求，必须在全球范围内扩建重型火箭发射设施。假设基础设施建设遵循生物种群增长规律（初期缓慢、中期加速、后期饱和）。

### 1.1 增长方程
设 $N(t)$ 为第 $t$ 年的可用发射场数量：
$$ N(t) = \frac{K}{1 + \left( \frac{K - N_0}{N_0} \right) e^{-r t}} $$

其中：
*   **$N_0$**：初始数量 (10)。
*   **$K$**：环境承载力 (Carrying Capacity)。考虑到赤道地理位置、地缘政治与空域限制，设定 $K \approx 50 \sim 80$。
*   **$r$**：增长率 (Growth Rate)。反映全人类动员建设的速度（类似阿波罗计划时期的动员率）。

### 1.2 动态运力积分
地面火箭系统的年运力 $T_R(t)$ 随时间变化：
$$ T_R(t) = N(t) \cdot L_{\max} \cdot p_B $$

总运输任务约束变为积分形式：
$$ \int_0^{Y} \left[ T_E + T_R(t) \right] \, dt \ge M_{\text{tot}} $$

---

## 2. 发射频率估算：周转时间模型
为了科学估算 $L_{\max}$，建立单发射台的周转时间模型。

### 2.1 理论周期
$$ t_{\text{cycle}} = t_{\text{refurb}} + t_{\text{pad}} + t_{\text{weather}} + t_{\text{fail}} $$
$$ L_{\max} = \frac{365 \cdot \eta}{t_{\text{cycle}}} $$

*   $t_{\text{refurb}}$：回收翻修时间（关键变量）。
*   $t_{\text{pad}}$：发射台占用时间。
*   $\eta$：系统可用率 (0.9)。

### 2.2 技术情景假设 (Scenarios)
由于无法精确预测 2050 年的技术细节，定义三种技术成熟度情景：

| 情景 | 技术特征 | $t_{\text{cycle}}$ (天) | $L_{\max}$ (次/年) | 备注 |
|---|---|---|---|---|
| **A (Conservative)** | 现有技术优化 | ~14 | ~20 | 类似 Falcon 9 现状 |
| **B (Moderate)** | 快速复用成熟 | ~4 | ~80 | Starship 初期目标 |
| **C (Aggressive)** | 航空化运营 | ~1 | ~300 | 理想目标 |

---

## 3. 载荷差异建模：基于 $\Delta v$ 的物理推导
严谨区分地面发射载荷 $p_B$ 与锚点发射载荷 $p_A$。

### 3.1 齐奥尔科夫斯基方程分析
$$ p = m_{\text{dry}} \cdot \left( e^{\frac{\Delta v}{I_{sp} g_0}} - 1 \right)^{-1} \approx C \cdot e^{-\Delta v} $$

*   **地面发射 (Earth $\to$ Moon)**：需克服深重力井。$\Delta v_{\text{Earth}} \approx 12.6 \text{ km/s}$。
*   **锚点发射 (Anchor $\to$ Moon)**：利用锚点高线速度 ($~7.3 \text{ km/s}$) 甩出，仅需微量变轨。$\Delta v_{\text{Anchor}} \approx 1.5 \text{ km/s}$。

### 3.2 载荷增益系数
定义增益系数 $\beta$：
$$ \beta = \frac{p_A}{p_B} \approx e^{\frac{\Delta v_{\text{Earth}} - \Delta v_{\text{Anchor}}}{v_e}} $$
根据估算，$\beta \approx 4 \sim 8$。
*   若地面载荷 $p_B = 125t$，则锚点载荷 $p_A \approx 500t \sim 1000t$。
*   这不仅增加了系统吞吐量，更大幅降低了单位燃料成本 ($c_E \ll c_R$)。

---

## 4. 不确定性分析：Monte Carlo 模拟
针对火箭载荷 $p_B$ 的不确定性进行随机模拟。

### 4.1 概率分布
假设 2050 年服役的重型火箭载荷服从均匀分布：
$$ p_B \sim U(100, 150) $$

### 4.2 模拟流程
运行 $N=10,000$ 次模拟，每次随机抽取 $p_B$，计算完工时间 $Y$ 和总成本 $C$。输出结果的统计特征（均值、95% 置信区间），以评估方案的鲁棒性。

---

## 5. 修正后的成本模型
引入基建成本与资金时间价值。

### 5.1 总成本函数
$$ C_{\text{total}} = C_{\text{ops}} + C_{\text{capex}} + C_{\text{env}} $$

1.  **运营成本 ($C_{\text{ops}}$)**：
    $$ \int_0^Y \left( c_E \cdot \dot{x}(t) + c_R \cdot (1-\alpha(t))\dot{M}(t) \right) e^{-\rho t} \, dt $$
    其中 $\rho$ 为折现率。

2.  **基建资本支出 ($C_{\text{capex}}$)**：
    $$ C_{\text{capex}} = C_{\text{site}} \cdot (N_{\text{final}} - N_0) + F_E $$
    *   $C_{\text{site}}$：单座发射场建设成本 (~$30B)。
    *   $N_{\text{final}}$：最终扩建数量。

---

## 6. 模型求解逻辑 (V2)
1.  **设定情景**：选择技术情景 (A/B/C) 确定 $L_{\max}$。
2.  **反向求解**：给定目标 $Y_{\max} = 24$ 年，利用积分方程反求所需的发射场承载力 $K$。
    $$ \text{Find } K \text{ such that } \int_0^{24} T_{\text{total}}(N(t; K)) \, dt = M_{\text{tot}} $$
3.  **成本评估**：基于求得的 $K$ 计算 CAPEX 和 OPEX。
4.  **风险分析**：叠加 Monte Carlo 模拟，给出成功概率分析。

---

## 7. 结论与建议接口
*   如果 $K$ 远超地理限制（如需 1000 个发射场），则判定方案不可行。
*   比较不同技术情景下 $K$ 的敏感性，论证“提高重复使用频率比单纯多建发射场更有效”。
