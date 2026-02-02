# Sensitivity Analysis Plan — Question 3 (Water Sustainability)

## 1. Purpose
Q3’s conclusion (“98% recycling is required”) should be defended with sensitivity analysis because the water system depends on a few high-leverage assumptions. Sensitivity should answer:
- How robust is the **98% recycling threshold** to changes in demand assumptions?
- How sensitive is the “cost chasm” to transport cost assumptions?
- How does required **strategic reserve** scale with disruption duration and recycling performance?

## 2. Core outputs (metrics)
Track these responses:
- Net import $D_{net}(\eta)$ (tons/year)
- Annual cost by mode: $C_R^{water}(\eta)$, $C_E^{water}(\eta)$
- Capacity tax: $\phi_E(\eta)=D_{net}(\eta)/T_E$
- Required reserve: $S_{safe}=D_{net}(\eta)\,\tau$
- Sustainability threshold: smallest $\eta$ such that $\phi_E(\eta)\le \phi_{max}$ (e.g., 10%)

## 3. Parameters to vary
### 3.1 Demand-side uncertainty
- Population $P$ (e.g., 80k–150k)
- Per-capita gross demand $d$ (depends on agriculture/industry assumptions)
- Growth trajectory of population (if staged arrival is modeled)

### 3.2 Recycling system performance
- Recycling efficiency $\eta$ (core variable)
- Recycling degradation / downtime (effective $\eta_{eff}$)

### 3.3 Transport system coupling
- Elevator throughput $T_E$ (available capacity)
- Unit transport costs for water: $c_R^{water}, c_E^{water}$

### 3.4 Reserve design parameters
- Buffer time $\tau$ (e.g., 3–12 months)
- Target service level (if mapping reserve to probability is desired)

## 4. Sensitivity methods
### 4.1 Threshold sensitivity (most important)
Define an operational criterion, e.g.:
- “Water should consume no more than 10% of elevator annual capacity”
Then compute the required $\eta$ threshold under varied $(P,d,T_E)$.

### 4.2 Tornado chart (for required $\eta$)
- Vary each parameter ±20%.
- Response: minimum $\eta$ needed to keep $\phi_E \le 0.10$.

### 4.3 Heatmaps (show structural constraints)
- Heatmap of $\phi_E$ over $(\eta, P)$ or $(\eta, d)$.
- Heatmap of required reserve $S_{safe}$ over $(\eta, \tau)$.

### 4.4 Scenario comparisons
- “Low-demand lifestyle” vs “high-industrialization” colony.
- “High-elevator-capacity” vs “capacity constrained”.

## 5. Suggested figures to add to the paper (Q3 sensitivity)
### Figure Q3-S1: Capacity tax surface
- x-axis: $\eta$
- y-axis: population $P$ (or demand multiplier)
- color: $\phi_E$
- Add contour at $\phi_E=1$ (physical infeasibility) and at $\phi_E=0.10$ (comfortable).

### Figure Q3-S2: Required recycling threshold vs population
- x-axis: $P$
- y-axis: required $\eta$ for $\phi_E \le 0.10$.

### Figure Q3-S3: Reserve requirement vs disruption duration
- x-axis: $\tau$ (months)
- y-axis: $S_{safe}$ (tons)
- Multiple lines for $\eta=90\%, 95\%, 98\%$.

### Figure Q3-S4: Tornado for threshold $\eta$
- Inputs: $P,d,T_E,c_E^{water}$
- Output: required $\eta$.

## 6. Narrative linkage
The sensitivity section should explicitly show:
- The 98% requirement is not a “tuned” value; it is a structural consequence of large $D_{gross}$ and finite $T_E$.
- Reserve scales linearly with disruption time and net import.
- If demand assumptions increase (more agriculture/industry), required recycling rises further.

## 7. Deliverable checklist (no code yet)
- Choose a clear operational criterion for “sustainable” (capacity tax limit).
- Document baseline values for $P,d,T_E$.
- Specify variation ranges and why they’re plausible.
- Finalize 3–4 plots that directly defend the headline claim.
