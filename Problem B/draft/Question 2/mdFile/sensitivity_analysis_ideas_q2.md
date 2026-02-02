# Sensitivity Analysis Plan — Question 2 (Reliability & Robustness)

## 1. Purpose
Q2 introduces stochastic disruptions (failures, weather delays, maintenance). Sensitivity analysis here should answer:
- Which uncertainty sources most degrade **success probability by deadline**, and which mainly increase **expected cost**?
- How sensitive is the recommended “adaptive policy” (shift to elevator when rockets degrade) to its threshold settings?
- What are the tails: how do worst-case outcomes change when assumptions shift?

## 2. Core outputs to track (metrics)
Recommend reporting these sensitivity responses:
- $P_{succ}(Y)$: probability of meeting deadline (e.g., 2050)
- $E[C]$: expected cost (or expected NPV)
- $\mathrm{VaR}_{95}(C)$ or $\mathrm{CVaR}_{95}(C)$: tail-risk cost
- $E[Y]$ or distribution of completion time (if simulated)
- Expected re-flight mass fraction / loss fraction $P_{loss}$

## 3. Key uncertain parameters to vary
### 3.1 Rocket reliability + downtime
- Per-launch failure probability $P_f$
- Stand-down duration $T_{down}$
- Weather/window loss fraction $\delta_{window}$
- Maintenance loss $\delta_{maint}$

### 3.2 Elevator system reliability
- Elevator failure rate $\lambda_E$
- Mean repair time $t_{repair}$
- Catastrophic outage probability $P_{cat}$

### 3.3 Performance uncertainty
- Payload uncertainty (e.g., $p_B \sim U(100,150)$ tons/launch)
- Availability multipliers $\beta_R^{eff},\beta_E^{eff}$

### 3.4 Policy parameters (if adaptive allocation is used)
- Threshold $\lambda_R$ / $P_f$ at which allocation shifts
- Ramp rate / responsiveness (how quickly allocation changes)

## 4. Sensitivity methods
### 4.1 One-at-a-time + Tornado for success probability
- Vary each parameter within credible bounds.
- Response: change in $P_{succ}(2050)$.

### 4.2 Stress-test scenarios (story-driven)
Define a few interpretable “worlds”:
- Optimistic: low $P_f$, short $T_{down}$, good weather
- Baseline
- Pessimistic: high $P_f$, long $T_{down}$, frequent weather loss
- “Elevator shock”: higher $\lambda_E$ or non-zero $P_{cat}$

### 4.3 Sensitivity surfaces (interaction effects)
Two-way sweeps reveal nonlinearity:
- Heatmap: $P_{succ}$ over $(P_f, T_{down})$
- Heatmap: $P_{succ}$ over $(\lambda_E, t_{repair})$
- Heatmap: cost tail risk over $(P_f, p_B)$

### 4.4 Distributional plots (communicate robustness)
Because Q2 is stochastic, plots of *distributions* are essential:
- CDF of completion time under baseline vs stressed
- Violin/box plot of cost under baseline vs stressed

## 5. Suggested figures to add to the paper (Q2 sensitivity)
### Figure Q2-S1: Success probability curve vs rocket failure
- x-axis: $P_f$ (or failure rate)
- y-axis: $P_{succ}(2050)$
- Multiple lines for different $T_{down}$.

### Figure Q2-S2: Heatmap of $P_{succ}$ over $(P_f, T_{down})$
- Shows the “cliff” where reliability collapses.

### Figure Q2-S3: Cost risk distribution under scenarios
- Box/violin of $C$ for baseline/optimistic/pessimistic.

### Figure Q2-S4: Policy threshold sensitivity
- Plot outcome (e.g., $P_{succ}$ and $E[C]$) vs switching threshold.
- Goal: show recommended threshold is not overly tuned.

## 6. How to connect to narrative
Expected qualitative message to verify:
- Deadline feasibility is highly sensitive to **rocket failure probability** and **stand-down duration**.
- Elevator reliability matters as a systemic stabilizer, but catastrophic outage probability is a key red-flag.
- Adaptive policy should reduce tail risk, not only mean cost.

## 7. Deliverable checklist (no code yet)
- Define baseline distributions and justification.
- Choose parameter ranges with citations or engineering plausibility.
- Decide which risk metric to emphasize (VaR or CVaR) for judges.
- Specify the 3–4 scenario cases and the figure list.
