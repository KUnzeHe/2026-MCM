# Problem B 论文写作提纲（对齐 1a / 1b / 1c，且满足 Y ≤ 24 年约束）

> 目的：先把“论文骨架”搭起来，后续把推导、参数表、数值结果、图表逐段填充。
> 
> 你们当前已有：
> - 1a/1b 的理论说明（single_mode_models.md）
> - 1c 的综合模型代码与 9 张图（fig1c_01 ~ fig1c_09）
> - 关键约束：运输任务需在 **最多 24 年** 内完成。

---

## 0. Summary Sheet / Abstract（摘要页）

- **任务**：运输 $M_{tot}=10^8$ 吨材料至月球，比较三种方案：
  - 1a：Space Elevator Only
  - 1b：Rockets Only
  - 1c：Mixed (Elevator + Rockets)
- **核心约束**：完工时间 $Y \le 24$ 年
- **核心方法**：吞吐能力约束 + 发射场 Logistic 增长 + NPV 成本 + Pareto/敏感性/蒙特卡洛
- **摘要需给出**（建议三行内给数值结论）：
  - 在 $Y=24$ 年下的最优分配（电梯占比、火箭占比）
  - 总成本 NPV（$\$T 量级）
  - 关键敏感参数排序（通常由火箭 OPEX 主导）
- **Keywords（3–6 个）**：Space Elevator, Logistic Growth, NPV, Pareto Front, Sensitivity, Monte Carlo

---

## 1. Introduction（引言）

### 1.1 Problem Background（背景）
- 月球殖民建设的物流规模巨大（100 million metric tons）
- 电梯系统提供低边际成本、稳定吞吐；火箭提供灵活扩展、但边际成本高

### 1.2 Restatement of the Problem（问题重述）
- 明确题目要求 1–5 条：
  1) 比较 1a/1b/1c 成本与工期
  2) 非完美运行（故障/摆动/失败）对结果影响
  3) 一年水需求与补给运输
  4) 环境影响与减排建议
  5) 写一页推荐信
- 强调本文额外设定：**必须满足 $Y \le 24$ 年**（作为核心设计约束）

### 1.3 Our Work（工作概览）
- 给出模型路线图（建议画一个流程图）：
  - 参数定义 → 三方案建模 → 1c 优化 → 鲁棒性/敏感性 → 水补给 → 环境与建议
- 指明主要输出：1c 的 9 张分析图（见 Fig 1c-01 ~ Fig 1c-09）

---

## 2. Assumptions and Notations（假设与符号）

### 2.1 Assumptions（假设）
建议写成“可辩护、可放松”的形式：
- 电梯系统在运输开始前已建成，港口数量固定
- 电梯链路吞吐由瓶颈决定：$T_{chain}=\min(T_E, T_{anchor})$
- 火箭侧发射场数量随时间增长，采用 Logistic：$N(t)$
- 单位成本参数、折现率为基准值，可做敏感性分析
- （鲁棒性章节再放松）可用率/故障通过参数或随机抽样体现

### 2.2 Notations（符号表）
建议做成表格（符号 / 含义 / 单位 / 基准值），至少包含：
- $M_{tot}, Y, x$
- $T_E, T_{anchor}, T_{chain}$
- $N(t), N_0, K, r$
- $L_{site}, p_B, c_R, C_{site}$
- $F_E, c_E, \rho$

---

## 3. Model I — Space Elevator Only（方案 1a：纯电梯）

### 3.1 System Structure & Bottleneck（结构与瓶颈）
- 地面 → 电梯 → GEO/锚点 → 转运火箭 → 月球
- 串联系统的瓶颈吞吐：
  - $T_{chain} = \min(T_E, T_{anchor})$

### 3.2 Timeline Model（工期）
- 连续化：$Y_{1a} = M_{tot}/T_{chain}$
- 可选：离散批次修正（若你们需要强调发射批次/整次发射）

### 3.3 Cost Model（成本）
- $C_{1a} = F_E + c_E\,M_{tot}$

### 3.4 Discussion（讨论）
- 优点：单位成本低、稳定
- 缺点：吞吐受限，工期可能过长（与 24 年约束对比）

---

## 4. Model II — Rockets Only（方案 1b：纯火箭）

### 4.1 Static Baseline（静态基准，用于对比）
- 年吞吐：$T_R = N_{sites}\,L_{max}\,p_B$
- 工期：$Y\approx M_{tot}/T_R$

### 4.2 Dynamic Capacity Growth（动态增长：核心）
- 发射场数量 Logistic：
  - $N(t)=\frac{K}{1+\left(\frac{K-N_0}{N_0}\right)e^{-rt}}$
- 累积运量约束：
  - $\int_0^Y N(t)\,L_{site}\,p_B\,dt = M_{tot}$

### 4.3 Cost Model with NPV（贴现成本）
- CAPEX：发射场扩建 $C_{site}(N_{final}-N_0)$
- OPEX：贴现积分（或用等效利用率近似）

### 4.4 Discussion（讨论）
- 优点：扩展灵活，可在 24 年内达成
- 缺点：OPEX 巨大，成本通常由 $c_R$ 主导

---

## 5. Model III — Mixed Optimization（方案 1c：混合优化）

### 5.1 Decision Variable & Constraint（决策变量与约束）
- 分配变量：电梯运 $x$，火箭运 $M_{tot}-x$
- 时间约束：$Y\le 24$
- 容量约束：
  - $x\le T_{chain}Y$
  - $M_{tot}-x \le \int_0^Y N(t)L_{site}p_B\,dt$

### 5.2 Objective Function（目标函数）
- 最小化：$C_{total}=C_{capex}+C_{opex}$（NPV）

### 5.3 Solving Strategy（求解策略）
- 说明你们的数值策略：
  - 扫描 $Y\in[14,24]$（或从最小可行时间开始到 24）
  - 每个 $Y$ 下求最优分配（通常电梯优先）
  - 形成 Pareto 前沿（在 24 年内呈现“成本—工期”权衡）

### 5.4 Results & Visualizations（结果与图表组织）
建议按图叙述（每图一段“发现+解释”）：
- Fig 1c-01：Three-scenario time–cost panorama（24 年内 Pareto）
- Fig 1c-02：Allocation pie + Gantt（固定 $Y=24$ 的最优分配与并行策略）
- Fig 1c-03：Capacity evolution（电梯恒定 vs 火箭增长）
- Fig 1c-04：Cumulative progress（里程碑：25%/50%/75%）
- Fig 1c-05：Cost waterfall（CAPEX/OPEX 分解）
- Fig 1c-06：Pareto deep analysis（2×2 面板：前沿/分解/份额/边际节省）
- Fig 1c-07：Sensitivity tornado（关键参数敏感性排序）
- Fig 1c-08：Monte Carlo（载荷不确定性下的稳健性）
- Fig 1c-09：Radar comparison（固定 $Y=24$ 的三场景多指标对比）

---

## 6. Robustness to Imperfect Operations（鲁棒性：非完美运行，题目第2条）

### 6.1 Sources of Uncertainty（不确定性来源）
- 火箭：载荷波动、失败率、天气窗口
- 电梯：摆动/检修停机、锚点转运故障

### 6.2 Modeling Approach（建模方式）
- 方法 A：有效吞吐折减（可用率 $\eta$）
- 方法 B：随机抽样（Monte Carlo）

### 6.3 Findings（结论）
- 给出：可行率、成本区间、对 24 年约束的影响

---

## 7. One-Year Water Needs（1 年水需求与补给，题目第3条）

### 7.1 Water Demand Estimation（需求估算）
- 人均日需求 × 100,000 人 × 365 天 → 年需求质量（吨）

### 7.2 Delivery Planning（运输规划）
- 将水作为额外任务质量 $M_{water}$
- 用同一模型评估：新增成本与所需时间（或运营期单独核算）

### 7.3 Results（结果）
- 建议：水补给优先走低边际成本链路（电梯/锚点转运）

---

## 8. Environmental Impact（环境影响，题目第4条）

### 8.1 Metric Definition（指标定义）
- 火箭发射次数、燃料消耗、单位吨排放（可用相对指标）

### 8.2 Comparison（方案比较）
- 在 $Y\le 24$ 下火箭占比高 → 环境负担更大

### 8.3 Mitigation Strategy（减排策略）
- 提升电梯链路吞吐（提高 $T_E$ 或 $T_{anchor}$）
- 降低火箭单位 OPEX / 提升可复用效率
- 若政策允许，适度放宽工期可降低高频火箭需求

---

## 9. Model Evaluation & Discussion（模型评估与讨论）

### 9.1 Strengths（优点）
- 容量瓶颈 + 动态增长 + NPV 的统一框架
- 输出可解释：可行边界、成本结构、敏感性排序

### 9.2 Weaknesses（不足）
- 未显式刻画排队/窗口/调度，采用平均化近似
- 参数来源的未来不确定性仍大（但可通过敏感性/MC缓解）

### 9.3 Extensions（可扩展）
- 多目标优化：成本 + 风险 + 排放
- 更真实的故障-维修过程（随机过程/马尔可夫）

---

## 10. One-Page Letter to MCM Agency（给机构的一页建议信，题目第5条）

- 开头：重申任务与 24 年约束
- 推荐：在 24 年内最稳健、成本可控的方案（结合你们结果）
- 依据：
  - 成本主要由火箭 OPEX 驱动（敏感性图支撑）
  - 电梯贡献受吞吐瓶颈限制（容量演化图支撑）
  - 风险与环境影响（鲁棒性与环境章节支撑）
- 行动建议：
  - 优先投资方向（降低 $c_R$ / 提升链路吞吐 / 提升可靠性）

---

## References & Appendices（参考文献与附录）

- References：成本、折现率、运载能力、发射场建设成本、排放系数的来源
- Appendices：推导细节、参数表、代码说明、额外图表
