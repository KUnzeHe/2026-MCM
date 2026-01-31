from __future__ import annotations

from dataclasses import dataclass
from math import ceil, inf
from typing import Optional


@dataclass(frozen=True)
class Params:
    """Model parameters for the mixed transport plan.

    Units
    - Mass: tons
    - Time: years
    - Cost: currency
    """

    # Total demand and deadline
    M_tot: float
    Y_max: float

    # Elevator chain (Earth -> Elevator -> Anchor -> Transfer rocket -> Moon)
    T_E: float  # Elevator throughput (tons/year)
    N_anchor: int  # Number of anchor launch sites
    L_anchor: int  # Max launches per site per year (anchor transfer rockets)
    p_A: float  # Payload per anchor transfer launch (tons/launch)

    # Direct rockets (Earth -> Moon)
    N_sites: int  # Number of ground launch sites
    L_max: int  # Max launches per site per year
    p_B: float  # Payload per direct launch (tons/launch)

    # Costs (linear + fixed for elevator chain)
    F_E: float  # Fixed cost of enabling elevator chain
    c_E: float  # Unit cost of elevator chain (per ton)
    c_R: float  # Unit cost of direct rockets (per ton)


@dataclass(frozen=True)
class Solution:
    """Optimization result.

    - feasible: whether a plan meets Y_max
    - x: mass assigned to elevator chain
    - y: binary indicator for using elevator chain (1 if x>0 else 0)
    - cost: total cost
    - Y_total: makespan of the mixed plan
    - Y_A: completion time of elevator chain branch
    - Y_B: completion time of direct rocket branch
    - nA: number of anchor transfer launches used
    - nB: number of direct launches used
    """

    feasible: bool
    x: float
    y: float
    cost: float
    Y_total: float
    Y_A: float
    Y_B: float
    nA: int
    nB: int
    reason: Optional[str] = None


def _rate(n_sites: int, l_max: int) -> int:
    """Total launches per year from multiple sites."""
    return max(0, n_sites * l_max)


def _time_elevator_chain(x: float, params: Params) -> float:
    """Completion time for the elevator chain branch with batch launches.

    Two bottlenecks:
    - Elevator throughput (continuous)
    - Anchor transfer launches (discrete batches)
    """
    if x <= 0:
        return 0.0
    if params.T_E <= 0:
        return inf
    rate_anchor = _rate(params.N_anchor, params.L_anchor)
    if rate_anchor <= 0:
        return inf
    return max(
        x / params.T_E,
        ceil(x / params.p_A) / rate_anchor,
    )


def _time_direct_rocket(mass: float, params: Params) -> float:
    """Completion time for direct rockets with batch launches."""
    if mass <= 0:
        return 0.0
    rate_direct = _rate(params.N_sites, params.L_max)
    if rate_direct <= 0:
        return inf
    return ceil(mass / params.p_B) / rate_direct


def _cost(x: float, params: Params) -> float:
    """Total cost with fixed cost only if elevator chain is used."""
    fixed = params.F_E if x > 0 else 0.0
    return fixed + params.c_E * x + params.c_R * (params.M_tot - x)


def optimize(params: Params) -> Solution:
    """Analytical optimization using boundary analysis.

    Instead of enumerating all combinations (which is O(N^2) and too slow for large inputs),
    we calculate the feasible mass interval [x_min, x_max] derived from Y_max.

    Constraints on x (elevator mass):
    1. Elevator branch must finish within Y_max:
       x / T_E <= Y_max
       ceil(x / p_A) / Rate_A <= Y_max  =>  x <= floor(Rate_A * Y_max) * p_A
    2. Rocket branch must finish within Y_max:
       ceil((M - x) / p_B) / Rate_B <= Y_max => M - x <= floor(Rate_B * Y_max) * p_B

    Cost function is linear for x > 0: C(x) = F_E + c_E*x + c_R*(M-x).
    Discontinuity at x = 0 (C(0) has no F_E).
    Optimal x is always at a boundary or 0.
    """
    rate_anchor = _rate(params.N_anchor, params.L_anchor)
    rate_direct = _rate(params.N_sites, params.L_max)

    # 1. Calculate Upper Bound for x (Elevator Capacity Limit)
    # Must satisfy continuous flow AND discrete launch constraint
    max_launches_A = int(rate_anchor * params.Y_max)
    capacity_A = max_launches_A * params.p_A
    continuous_limit_A = params.T_E * params.Y_max
    
    x_upper = min(params.M_tot, capacity_A, continuous_limit_A)

    # 2. Calculate Lower Bound for x (Rocket Capacity Constraint)
    # If rockets alone can't carry M_tot within Y_max, x must be at least...
    max_launches_B = int(rate_direct * params.Y_max)
    capacity_B = max_launches_B * params.p_B
    
    # We need M_tot - x <= capacity_B  =>  x >= M_tot - capacity_B
    x_lower = max(0.0, params.M_tot - capacity_B)

    # 3. Feasibility Check
    if x_lower > x_upper:
        return Solution(False, 0.0, 0.0, inf, inf, inf, inf, 0, 0, 
                        reason=f"Infeasible: Min elevator mass {x_lower:,.0f} > Max capacity {x_upper:,.0f}")

    # 4. Identify Candidates
    candidates = {x_lower, x_upper}
    
    # Special case: x = 0 is allowed only if x_lower == 0 (meaning rockets can handle all)
    if x_lower == 0:
        candidates.add(0.0)

    best: Optional[Solution] = None

    for x in candidates:
        # Calculate resulting times (re-verify constraints)
        yA = _time_elevator_chain(x, params)
        yB = _time_direct_rocket(params.M_tot - x, params)
        y_total = max(yA, yB)

        # Double check time constraint (floating point tolerance could be needed, but integers are safe here)
        if y_total > params.Y_max + 1e-9:
             continue
        
        # Calculate launches for reporting
        nA = ceil(x / params.p_A) if x > 0 else 0
        nB = ceil((params.M_tot - x) / params.p_B) if (params.M_tot - x) > 0 else 0

        cost = _cost(x, params)
        
        sol = Solution(
            True, x, 1.0 if x > 0 else 0.0, cost, y_total, yA, yB, nA, nB
        )

        if best is None or cost < best.cost:
            best = sol

    if best is None:
        return Solution(False, 0.0, 0.0, inf, inf, inf, inf, 0, 0, reason="No candidate met time constraint")
    
    return best


if __name__ == "__main__":
    # Example usage (replace with your parameters)
    params = Params(
        M_tot=1.0e8,  # total mass (tons)
        Y_max=24.0,  # deadline (years)
        T_E=5.37e5,  # elevator throughput (tons/year)
        N_anchor=3,  # anchor launch sites
        L_anchor=700,  # launches per site per year
        p_A=125.0,  # payload per anchor launch
        N_sites=60,  # ground launch sites
        L_max=700,  # launches per site per year
        p_B=125.0,  # payload per direct launch
        F_E=0.0,  # fixed cost for elevator chain
        c_E=2.7*10**3,  # unit cost for elevator chain
        c_R=7.2*10**5  # unit cost for direct rockets
    )

    sol = optimize(params)
    if not sol.feasible:
        print("No feasible plan:", sol.reason)
    else:
        print("Best plan")
        print(f"x (elevator chain mass) = {sol.x:,.0f} t")
        print(f"Cost = {sol.cost:,.2f}")
        print(f"Y_total = {sol.Y_total:.4f} years")
        print(f"Y_A = {sol.Y_A:.4f}, Y_B = {sol.Y_B:.4f}")
        print(f"Launches: nA={sol.nA}, nB={sol.nB}")
