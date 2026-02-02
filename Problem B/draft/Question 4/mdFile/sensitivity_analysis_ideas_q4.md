# Sensitivity Analysis Plan — Question 4 (Environmental Impact / SEIS)

## 1. Purpose
Q4’s verdict (Elevator A+ vs Rocket F, payback time, Kessler risk) should be supported by sensitivity analysis because environmental outcomes depend on carbon intensity assumptions and on orbital-risk model parameters. Sensitivity should answer:
- How stable are the **carbon debt** and **break-even time** conclusions under plausible changes in emission factors?
- How sensitive is orbital risk escalation to launch cadence and cascade parameters?
- Does the SEIS grade change under reasonable weighting/normalization choices?

## 2. Core outputs (metrics)
Track these responses:
- Carbon debt $E_{debt}$ (Mt CO$_2$)
- Net annual improvement $e_{net}=e_{avoid}-e_{op}$
- Break-even time $T_{BE}=E_{debt}/e_{net}$ (or $\infty$ if $e_{net}\le0$)
- Orbital risk indicators: peak risk $R_{max}$, time-to-threshold
- SEIS total score + grade (A+ … F)

## 3. Parameters to vary
### 3.1 Emission factors (atmospheric)
- Rocket CO$_2$ per launch (or per ton-to-orbit) factor
- Electricity carbon intensity for elevator operations (grid mix vs renewables)
- Construction footprint assumptions for elevator infrastructure
- Carbon price (if included in an economic-environment coupling)

### 3.2 Program architecture parameters
- Elevator share during construction (allocation fraction)
- Total delivered mass (demand uncertainty) and schedule (affects launch cadence)

### 3.3 Orbital debris dynamics parameters
Using a generic dynamic: $dR/dt = \alpha L(t) + \beta R^2 - \gamma R$,
- $\alpha$: launches-to-risk coupling
- $\beta$: cascade amplification strength
- $\gamma$: mitigation/decay strength
- Risk threshold defining “Kessler warning” (policy choice)

### 3.4 SEIS aggregation choices
- Weight set $(w_C,w_T,w_R)$
- Normalization method (min–max vs z-score vs piecewise scoring)

## 4. Sensitivity methods
### 4.1 Tornado chart (for $T_{BE}$ and SEIS)
- Vary key emission factors and allocation fraction.
- Responses: $T_{BE}$ and SEIS.

### 4.2 Two-way sweeps (most persuasive)
- Heatmap: $T_{BE}$ over (elevator share, rocket emission factor)
- Heatmap: $R_{max}$ over (launch cadence multiplier, $\beta$)

### 4.3 Regime/phase diagrams (orbital risk)
- Show stable vs runaway regions in $(L,\beta,\gamma)$ space.
- Judges like “hard constraint” visuals.

### 4.4 Weight-robustness (SEIS fairness)
- Sample multiple plausible weight sets.
- Report grade frequency: %A, %B, … under weight uncertainty.
- Goal: show ranking (elevator dominates) is robust.

## 5. Suggested figures to add to the paper (Q4 sensitivity)
### Figure Q4-S1: Payback time sensitivity heatmap
- x-axis: elevator fraction during construction
- y-axis: rocket CO$_2$ factor multiplier
- color: $T_{BE}$; mark infeasible ($\infty$) region.

### Figure Q4-S2: Kessler risk surface
- x-axis: launch cadence multiplier
- y-axis: cascade parameter $\beta$ (or mitigation $\gamma$)
- color: $R_{max}$ or time-to-threshold.

### Figure Q4-S3: Tornado for SEIS components
- Inputs: emission factors, $\beta$, $\gamma$, elevator share
- Output: SEIS score.

### Figure Q4-S4: SEIS grade robustness under weights
- Bar chart: fraction of weight samples where each scenario gets each grade.

## 6. Narrative linkage
This section should argue:
- The **ranking** is robust: rocket-heavy strategies remain poor under wide parameter ranges.
- The **payback time** is sensitive to construction-phase rocket dependence, motivating early elevator capacity.
- Orbital risk is nonlinear: small increases in cadence or cascade strength can trigger runaway.

## 7. Deliverable checklist (no code yet)
- Define baseline emission factors and their plausible ranges.
- Define orbital-risk threshold used for “warning”.
- Decide on 1–2 SEIS weight families (e.g., carbon-focused vs orbit-focused).
- Select 3–4 plots that directly defend the headline verdict.
