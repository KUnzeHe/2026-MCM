from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from scipy.integrate import quad
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List

# Explicitly setting font for matplotlib to avoid issues with CJK characters if environment supports it, 
# otherwise falling back to default.
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

@dataclass(frozen=True)
class GrowthParams:
    """Parameters for the Logistic growth of Rocket Infrastructure."""
    K: float = 80.0       # Carrying capacity (max sites globally)
    N0: float = 10.0      # Initial number of sites
    r: float = 0.15       # Growth rate (approx 0.15 reaches saturation in ~25-30 years)

@dataclass(frozen=True)
class ModelParams:
    """
    Global Model Parameters based on Comprehensive Optimization Framework V4.
    
    Units:
    - Mass: tons
    - Cost: USD
    - Time: years
    """
    # Demand
    M_tot: float = 1.0e8  # 100 million tons
    
    # Financial
    discount_rate: float = 0.03 # 3% discount rate
    
    # Elevator System Parameters
    T_E: float = 5.375e5  # Annual throughput capacity (tons/year)
    F_E: float = 100e9    # Fixed CAPEX for Elevator ($100 Billion)
    c_E: float = 2.7e3    # OPEX per ton ($2,700/ton)
    
    # Rocket System Parameters
    c_R: float = 7.2e5    # OPEX per ton ($720,000/ton)
    C_site: float = 2.0e9 # CAPEX per new launch site (Estimate: $2B per complex)
    
    # Rocket Performance (Turnaround & Payload)
    # L_max derives from turnaround time. 
    # Reference code uses 700 launches/year (~2 per day), likely assuming rapid reuse.
    # Scenario C (1 day turnaround) -> 365. 
    # We will use 700 to match the reference snippet provided by user.
    L_site_annual: float = 700.0 # Launches per site per year
    p_B: float = 150.0    # Payload per launch (tons) - Starship class
    
    # Growth Model for Ground Infrastructure
    growth: GrowthParams = GrowthParams()

@dataclass
class OptimizationResult:
    Y: float              # Completion Time (Years)
    x_opt: float          # Mass allocated to Elevator (tons)
    mR_opt: float         # Mass allocated to Rockets (tons)
    cost_total: float     # Total NPV Cost
    cost_capex: float
    cost_opex: float
    feasible: bool
    N_final: float        # Final number of launch sites built
    message: str = ""

class TransportOptimizationModel:
    def __init__(self, params: ModelParams):
        self.p = params
    
    def N_t(self, t: float) -> float:
        """Logistic growth function for number of launch sites."""
        K, N0, r = self.p.growth.K, self.p.growth.N0, self.p.growth.r
        # Logistic function: N(t) = K / (1 + ((K-N0)/N0) * exp(-r*t))
        # Avoid overflow in exp
        try:
            exp_term = np.exp(-r * t)
        except OverflowError:
            exp_term = 0.0
            
        denom = 1 + ((K - N0) / N0) * exp_term
        return K / denom

    def rocket_capacity_rate(self, t: float) -> float:
        """Instantaneous rocket fleet capacity in tons/year at time t."""
        # Rate = N(t) * Launches_per_site * tons_per_launch
        return self.N_t(t) * self.p.L_site_annual * self.p.p_B

    def max_rocket_capacity_cumulative(self, Y: float) -> float:
        """Total mass rockets can move from t=0 to t=Y."""
        # Integral of N(t) * L * p from 0 to Y
        val, _ = quad(self.rocket_capacity_rate, 0, Y)
        return val

    def max_elevator_capacity_cumulative(self, Y: float) -> float:
        """Total mass elevator can move from t=0 to t=Y."""
        # Simple linear accumulation: T_E * Y
        return self.p.T_E * Y

    def calculate_opex_npv(self, duration: float, linear_rate: float, unit_cost: float) -> float:
        """
        Calculate NPV of OPEX for a process running at constant 'linear_rate' (tons/yr)
        with 'unit_cost' ($/ton) for 'duration' years.
        
        Integral_0^T (rate * cost * e^(-rho*t)) dt
        = rate * cost * [ (1 - e^(-rho*T)) / rho ]
        """
        rho = self.p.discount_rate
        if rho == 0:
            return linear_rate * unit_cost * duration
        
        annual_cash_flow = linear_rate * unit_cost
        discount_factor = (1 - np.exp(-rho * duration)) / rho
        return annual_cash_flow * discount_factor
    
    def calculate_rocket_opex_npv(self, Y: float, required_mass: float) -> float:
        """
        Calculate OPEX NPV for rockets.
        Approximation: We assume the rocket usage is spread somewhat by curve, 
        but to simplify NPV calculation for optimization Loop, we can use an average weighted time 
        or simply integrate flow rate proportional to capacity.
        
        Strictly: m_dot_R(t) needs to be determined.
        Strategy: Use available capacity proportionally or fully until done.
        Given 'required_mass' over total time 'Y', we assume utilization matches the growth curve 
        scaled to meet the demand, OR we run at max capacity until Y.
        
        Constraint: \\int_0^Y m_dot_R(t) dt = required_mass
        Max Capacity: \\int_0^Y C_R(t) dt = C_tot
        
        Utilization ratio u = required_mass / C_tot (must be <= 1)
        Actual flow m_dot(t) = u * C_R(t)
        
        Cost = \\int_0^Y c_R * u * C_R(t) * e^(-rho*t) dt
             = u * c_R * \\int_0^Y C_R(t) e^(-rho*t) dt
        """
        max_cap = self.max_rocket_capacity_cumulative(Y)
        if max_cap == 0:
             return float('inf')
        
        utilization = required_mass / max_cap
        
        # Define integrand: Capacity(t) * discount(t)
        def cost_integrand(t):
            return self.rocket_capacity_rate(t) * np.exp(-self.p.discount_rate * t)
        
        integral_val, _ = quad(cost_integrand, 0, Y)
        
        return utilization * self.p.c_R * integral_val

    def solve(self, Y: float) -> OptimizationResult:
        """
        Find optimal allocation for a fixed deadline Y.
        Strategy: Greedy for Elevator (cheaper).
        """
        # 1. Capabilites at time Y
        cap_E = self.max_elevator_capacity_cumulative(Y)
        cap_R = self.max_rocket_capacity_cumulative(Y)
        
        # 2. Check Feasibility
        if cap_E + cap_R < self.p.M_tot:
            return OptimizationResult(
                Y=Y, x_opt=0, mR_opt=0, cost_total=float('inf'), 
                cost_capex=0, cost_opex=0, feasible=False, N_final=self.N_t(Y),
                message="Capacity insufficient"
            )
            
        # 3. Allocation (Greedy for Elevator)
        # Allocate as much as possible to elevator, up to Demand or Capacity
        x = min(self.p.M_tot, cap_E)
        m_R = self.p.M_tot - x
        
        # 4. CAPEX Calculation
        # Elevator Fixed Cost
        capex_E = self.p.F_E if x > 0 else 0
        
        # Rocket Infrastructure Cost
        # We assume we expand according to the logistic curve up to time Y regardless of utilization
        # (Infrastructure leads capability). 
        # Alternatively, we could throttle infrastructure, but the growth model suggests "available" capacity.
        # Let's assume we pay for the sites we have available at year Y.
        N_final = self.N_t(Y)
        N_new = max(0, N_final - self.p.growth.N0)
        capex_R = N_new * self.p.C_site
        
        capex_total = capex_E + capex_R
        
        # 5. OPEX Calculation (NPV)
        
        # Elevator OPEX
        # Assume constant flow rate T_E (or x/Y if x < cap_E)
        # Effective rate
        eff_rate_E = x / Y 
        # But technically since T_E is a hard cap, if x = T_E * Y, rate is T_E.
        opex_E = self.calculate_opex_npv(Y, eff_rate_E, self.p.c_E)
        
        # Rocket OPEX
        opex_R = self.calculate_rocket_opex_npv(Y, m_R)
        
        opex_total = opex_E + opex_R
        
        return OptimizationResult(
            Y=Y,
            x_opt=x,
            mR_opt=m_R,
            cost_total=capex_total + opex_total,
            cost_capex=capex_total,
            cost_opex=opex_total,
            feasible=True,
            N_final=N_final
        )

def run_analysis():
    # Setup Parameters
    params = ModelParams()
    model = TransportOptimizationModel(params)
    
    # 1. Logistic Growth Visualization
    times = np.linspace(0, 50, 100)
    sites = [model.N_t(t) for t in times]
    rates = [model.rocket_capacity_rate(t)/1e6 for t in times] # Million tons/year
    
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(times, sites, 'b-', label='Launch Sites (N)', linewidth=2)
    ax1.set_xlabel('Year (from 2050)')
    ax1.set_ylabel('Number of Launch Sites', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    ax2.plot(times, rates, 'r--', label='Rocket Fleet Capacity (Mt/yr)', linewidth=2)
    ax2.set_ylabel('Annual Transport Capacity (Million Tons)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    plt.title('Infrastructure Growth & Capacity Dynamics (Logistic Model)')
    plt.savefig('1c/image/Question1/infrastructure_growth_v4.png')
    print("Generated infrastructure_growth_v4.png")
    
    # 2. Pareto Optimization Analysis
    # Sweep Y from 15 to 50 years
    Y_range = np.linspace(15, 60, 46)
    results = []
    
    print("\n--- Running Optimization Sweep ---")
    print(f"{'Years':<10} | {'Elevator %':<12} | {'Rocket %':<10} | {'Cost ($T)':<10} | {'Status'}")
    
    best_sol = None
    
    for Y in Y_range:
        res = model.solve(Y)
        results.append(res)
        
        if res.feasible:
            cost_trillion = res.cost_total / 1e12
            e_pct = (res.x_opt / params.M_tot) * 100
            r_pct = (res.mR_opt / params.M_tot) * 100
            print(f"{Y:<10.1f} | {e_pct:<12.1f} | {r_pct:<10.1f} | {cost_trillion:<10.3f} | OK")
            
            # Simple heuristic for "Best Balance": Knee point 
            # Often where marginal cost of speed becomes insane.
            # Or just min cost within reasonable time.
            # Here we just track valid solutions.
        else:
            print(f"{Y:<10.1f} | {'-':<12} | {'-':<10} | {'-':<10} | Infeasible")

    # Filter feasible
    feas_results = [r for r in results if r.feasible]
    
    if not feas_results:
        print("No feasible solutions found in range.")
        return

    # Extract Data for Plotting
    Y_vals = [r.Y for r in feas_results]
    Costs = [r.cost_total / 1e12 for r in feas_results] # Trillions
    Ele_mass = [r.x_opt / 1e6 for r in feas_results]    # Million tons
    Roc_mass = [r.mR_opt / 1e6 for r in feas_results]   # Million tons
    
    # Plot Pareto Front (Time vs Cost)
    fig2, (ax_cost, ax_mass) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Cost Curve
    ax_cost.plot(Y_vals, Costs, 'o-', color='darkgreen', linewidth=2)
    ax_cost.set_title('Pareto Front: Time vs Total Cost (NPV)')
    ax_cost.set_xlabel('Project Duration (Years)')
    ax_cost.set_ylabel('Total Cost (Trillion USD)')
    ax_cost.grid(True)
    
    # Annotate min cost
    min_cost = min(Costs)
    min_idx = Costs.index(min_cost)
    ax_cost.annotate(f'Min Cost: ${min_cost:.2f}T\n@ {Y_vals[min_idx]:.0f} yrs',
                     xy=(Y_vals[min_idx], min_cost),
                     xytext=(Y_vals[min_idx], min_cost + 1),
                     arrowprops=dict(facecolor='black', shrink=0.05))

    # Mass Distribution Stacked Plot
    ax_mass.stackplot(Y_vals, Ele_mass, Roc_mass, labels=['Elevator', 'Rocket'], 
                      colors=['#3498db', '#e74c3c'], alpha=0.8)
    ax_mass.set_title('Optimal modal split vs Time Constraint')
    ax_mass.set_xlabel('Project Duration (Years)')
    ax_mass.set_ylabel('Transported Mass (Million Tons)')
    ax_mass.legend(loc='upper left')
    ax_mass.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('1c/image/Question1/optimization_results_v4.png')
    print("Generated optimization_results_v4.png")

if __name__ == "__main__":
    run_analysis()
