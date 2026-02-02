# Sensitivity Analysis Plan — Question 1 (Optimization Under Ideal Conditions)

## 1. Purpose (What sensitivity answers)
Q1’s optimization output (cost–time Pareto frontier and the “knee point”) depends on growth and capacity assumptions. Sensitivity analysis should answer:
- Which assumptions most strongly change **NPV**, **completion time**, and **optimal modal split** (elevator share vs rocket share)?
- Which parameters shift the **shape** of the Pareto frontier (knee location) vs. only shift it up/down?
- Under what conditions does the 24-year target become feasible/robust vs. intrinsically fragile?

## 2. Core outputs to track (metrics)
Recommend tracking these as sensitivity “responses”:
- $C_{NPV}$: total Net Present Value (or total cost if NPV not used)
- $Y$: completion time / makespan
- $x^*$: optimal total mass assigned to elevator over horizon (or average elevator fraction)
- Slack ratio: $\frac{C_E(Y)+C_R(Y)}{M_{tot}}$ (how close to knife-edge)
- Reliability proxy (even in Q1): margin-to-demand (since Q2 will formalize probability)

## 3. Parameters to perturb (what we vary)
### 3.1 “Schedule & throughput” parameters (usually dominant)
- Deadline / horizon $Y$ (e.g., 20–60 years)
- Elevator throughput ceiling $T_E$ (harbor count / operational efficiency)
- Anchor transfer capacity $T_{R,anchor}$ (if modeled)

### 3.2 Rocket infrastructure growth parameters
- Logistic growth rate $r$ (mobilization speed)
- Carrying capacity $K$ (site ceiling)
- Initial sites $N_0$ (baseline infrastructure)

### 3.3 Per-launch performance and operations
- Payload to Moon per launch $p_B$ (or its range/mean)
- Turnaround/cycle time components (if represented explicitly)

### 3.4 Economic parameters
- Unit costs $c_R, c_E$ (or their multipliers)
- Discount rate $\rho$ (NPV sensitivity)
- CAPEX terms (elevator fixed cost $F_E$, site cost $C_{site}$)

## 4. Sensitivity methods (how we evaluate)
### 4.1 One-at-a-time (OAT) + Tornado plot (fast, interpretable)
- Vary one parameter ±10% / ±20% around baseline.
- Plot change in $C_{NPV}$ and change in $x^*$.
- Output figure: Tornado chart ranked by impact on $C_{NPV}$.

### 4.2 Scenario grid / heatmaps (reveals interaction effects)
Two-parameter sweeps are crucial because capacity constraints are nonlinear.
Recommended grids:
- Heatmap of $C_{NPV}$ over $(Y, T_E)$.
- Heatmap of optimal rocket fraction over $(r, K)$.
- Heatmap of feasibility/slack over $(Y, p_B)$.

### 4.3 Pareto-front shifts (storytelling figure)
- Recompute Pareto frontier under a few “what-if” scenarios:
  - Baseline
  - High $T_E$ (more harbors)
  - Slow rocket growth (low $r$)
  - High discount rate (high $\rho$)
- Overlay frontiers to show *how the knee point moves*.

### 4.4 Local elasticity / normalized sensitivity coefficients
For each parameter $\theta$:
- Report $S = \frac{\Delta C/C}{\Delta \theta/\theta}$ for cost, and similarly for $x^*$.
This gives a compact quantitative summary table.

## 5. Suggested figures to add to the paper (Q1 sensitivity)
### Figure Q1-S1: Tornado (NPV sensitivity)
- Bars: ±20% change in { $T_E$, $Y$, $r$, $K$, $p_B$, $c_R$, $F_E$, $\rho$ }
- Response: $C_{NPV}$.

### Figure Q1-S2: Heatmap — cost vs schedule & elevator capacity
- Axes: $Y$ (years) vs $T_E$ (tons/year)
- Color: $C_{NPV}$
- Contours: fixed rocket share or fixed slack.

### Figure Q1-S3: Pareto overlay (baseline vs 3 stress cases)
- Show knee-point shift and explain why.

### Figure Q1-S4: “Knife-edge” margin plot
- Plot slack ratio vs $Y$ for baseline and pessimistic payload.

## 6. How to connect to narrative (what we will write)
Key expected conclusion to test:
- Sensitivity will likely show **deadline $Y$** and **elevator throughput $T_E$** dominate NPV and feasibility.
- Rocket growth parameters $(r,K)$ matter primarily when $Y$ is short (capacity crunch regime).
- Discount rate $\rho$ changes NPV ranking more than feasibility.

## 7. Deliverable checklist (no code yet)
- List baseline parameter values used in the main results.
- Define perturbation ranges and justify (engineering realism).
- Decide 3–4 “headline” scenarios for Pareto overlay.
- Decide figure formats (tornado + 1–2 heatmaps + overlay).
