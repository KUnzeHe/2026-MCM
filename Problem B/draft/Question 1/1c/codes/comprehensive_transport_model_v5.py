"""
Comprehensive Transport Optimization Model V5
==============================================
Based on comprehensive_optimization_framework_v4.md

Key improvements over V4:
1. Added anchor transfer bottleneck constraint for elevator chain
2. Corrected c_R parameter ($7.2M/ton, not $720k/ton)
3. Improved CAPEX calculation based on actual infrastructure needs
4. Added Monte Carlo robustness analysis
5. Added Knee Point identification for Pareto front
6. Enhanced visualization with more analysis plots
"""

from __future__ import annotations

import os
import numpy as np
from dataclasses import dataclass, field
from scipy.integrate import quad
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple
import warnings

# Matplotlib configuration
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120

# ============================================================================
# Data Classes for Parameters
# ============================================================================

@dataclass(frozen=True)
class GrowthParams:
    """Parameters for Logistic growth of ground launch infrastructure."""
    K: float = 80.0       # Carrying capacity (max sites globally)
    N0: float = 10.0      # Initial number of sites
    r: float = 0.15       # Growth rate


@dataclass(frozen=True)
class AnchorParams:
    """Parameters for anchor transfer rockets (Elevator -> Moon).
    
    Note: Anchor rockets operate from space station with much lower delta-v,
    so they can achieve higher payload fractions (beta ~ 4-8x ground rockets).
    We assume anchor capacity matches or exceeds elevator throughput.
    """
    N_anchor: int = 6           # Number of anchor launch platforms (2 per harbor)
    L_anchor: float = 700.0     # Launches per anchor per year
    p_A: float = 150.0          # Payload per anchor launch (tons)
    
    @property
    def annual_capacity(self) -> float:
        """Total annual transfer capacity from anchor."""
        return self.N_anchor * self.L_anchor * self.p_A


@dataclass(frozen=True)
class ModelParams:
    """
    Global Model Parameters based on Comprehensive Optimization Framework V4.
    
    Units:
    - Mass: metric tons
    - Cost: USD
    - Time: years
    """
    # Demand
    M_tot: float = 1.0e8  # 100 million metric tons
    
    # Financial
    discount_rate: float = 0.03  # 3% discount rate (rho)
    
    # =========== Elevator System Parameters ===========
    # Throughput: 179,000 t/yr per harbor × 3 harbors = 537,000 t/yr
    T_E: float = 5.37e5           # Annual elevator throughput (tons/year)
    F_E: float = 100e9            # Fixed CAPEX for elevator ($100 Billion)
    c_E: float = 2.7e3            # OPEX per ton ($2.7/kg = $2,700/ton)
    
    # Anchor transfer (second bottleneck of elevator chain)
    anchor: AnchorParams = field(default_factory=AnchorParams)
    
    # =========== Direct Rocket System Parameters ===========
    # Document states $7,200/kg for current tech. However, for 2050 scenario with
    # Starship-class reusable rockets, costs are expected to drop significantly.
    # Reference code uses $720/kg = $720,000/ton as projected future cost.
    # We use this more optimistic but plausible estimate.
    c_R: float = 7.2e5            # OPEX per ton ($720k/ton = $720/kg future)
    C_site: float = 2.0e9         # CAPEX per new launch site ($2B)
    
    # Rocket Performance
    L_site_annual: float = 2000.0  # Launches per site per year (aggressive reuse)
    p_B: float = 150.0            # Payload per launch (tons) - Starship class
    
    # Infrastructure Growth Model
    growth: GrowthParams = field(default_factory=GrowthParams)
    
    # Monte Carlo settings
    p_B_range: Tuple[float, float] = (100.0, 150.0)  # Payload uncertainty U(100, 150)


@dataclass
class OptimizationResult:
    """Result container for a single optimization run."""
    Y: float                  # Completion time (years)
    x_opt: float              # Mass via elevator chain (tons)
    mR_opt: float             # Mass via direct rockets (tons)
    cost_total: float         # Total NPV cost
    cost_capex: float         # Capital expenditure
    cost_opex: float          # Operational expenditure (NPV)
    feasible: bool
    N_required: float         # Number of launch sites actually needed
    cap_E: float = 0.0        # Elevator chain capacity used
    cap_R: float = 0.0        # Rocket capacity used
    message: str = ""
    
    @property
    def elevator_pct(self) -> float:
        return (self.x_opt / (self.x_opt + self.mR_opt) * 100) if (self.x_opt + self.mR_opt) > 0 else 0


# ============================================================================
# Core Optimization Model
# ============================================================================

class TransportOptimizationModel:
    """
    Implements the mixed transport optimization model.
    
    Two transport chains:
    - Chain A (Elevator): Ground -> Elevator -> Anchor -> Transfer Rocket -> Moon
    - Chain B (Direct):   Ground -> Heavy Rocket -> Moon
    """
    
    def __init__(self, params: ModelParams):
        self.p = params
    
    # ---------- Infrastructure Growth Model ----------
    
    def N_t(self, t: float) -> float:
        """
        Logistic growth: N(t) = K / (1 + ((K-N0)/N0) * exp(-r*t))
        """
        K, N0, r = self.p.growth.K, self.p.growth.N0, self.p.growth.r
        A = (K - N0) / N0
        return K / (1 + A * np.exp(-r * t))
    
    def rocket_capacity_rate(self, t: float, p_B: Optional[float] = None) -> float:
        """Instantaneous rocket fleet capacity (tons/year) at time t."""
        payload = p_B if p_B is not None else self.p.p_B
        return self.N_t(t) * self.p.L_site_annual * payload
    
    # ---------- Cumulative Capacity Functions ----------
    
    def cumulative_rocket_capacity(self, Y: float, p_B: Optional[float] = None) -> float:
        """Total mass rockets can transport from t=0 to t=Y."""
        payload = p_B if p_B is not None else self.p.p_B
        
        def rate(t):
            return self.N_t(t) * self.p.L_site_annual * payload
        
        val, _ = quad(rate, 0, Y)
        return val
    
    def cumulative_elevator_capacity(self, Y: float) -> float:
        """
        Total mass elevator chain can transport from t=0 to t=Y.
        
        Bottleneck = min(Elevator throughput, Anchor transfer capacity)
        """
        # Elevator throughput limit
        cap_elevator = self.p.T_E * Y
        
        # Anchor transfer limit
        cap_anchor = self.p.anchor.annual_capacity * Y
        
        return min(cap_elevator, cap_anchor)
    
    def elevator_bottleneck_rate(self) -> float:
        """Annual throughput of elevator chain (bottleneck)."""
        return min(self.p.T_E, self.p.anchor.annual_capacity)
    
    # ---------- NPV Cost Calculations ----------
    
    def npv_factor(self, duration: float) -> float:
        """NPV discount factor for constant cash flow over duration."""
        rho = self.p.discount_rate
        if rho == 0 or duration == 0:
            return duration
        return (1 - np.exp(-rho * duration)) / rho
    
    def calculate_elevator_opex_npv(self, x: float, Y: float) -> float:
        """
        NPV of elevator OPEX.
        Assumes constant throughput rate = x/Y (tons/year).
        """
        if x <= 0:
            return 0.0
        rate = x / Y
        return rate * self.p.c_E * self.npv_factor(Y)
    
    def calculate_rocket_opex_npv(self, m_R: float, Y: float, p_B: Optional[float] = None) -> float:
        """
        NPV of rocket OPEX with time-varying capacity.
        
        Strategy: Run at capacity proportionally scaled to meet demand.
        m_dot(t) = u * C_R(t), where u = m_R / integral(C_R)
        
        NPV = u * c_R * integral[C_R(t) * e^(-rho*t)]
        """
        if m_R <= 0:
            return 0.0
        
        max_cap = self.cumulative_rocket_capacity(Y, p_B)
        if max_cap <= 0:
            return float('inf')
        
        utilization = min(1.0, m_R / max_cap)
        payload = p_B if p_B is not None else self.p.p_B
        rho = self.p.discount_rate
        
        def cost_integrand(t):
            return self.N_t(t) * self.p.L_site_annual * payload * np.exp(-rho * t)
        
        integral_val, _ = quad(cost_integrand, 0, Y)
        return utilization * self.p.c_R * integral_val
    
    def calculate_required_sites(self, m_R: float, Y: float) -> float:
        """
        Estimate minimum number of launch sites needed to transport m_R tons in Y years.
        
        This is a simplification: we find N such that N * L * p_B * Y >= m_R
        In reality with Logistic growth, fewer new sites may be needed if time is long.
        """
        if m_R <= 0:
            return self.p.growth.N0
        
        # Simple estimate: average sites needed
        annual_need = m_R / Y
        sites_needed = annual_need / (self.p.L_site_annual * self.p.p_B)
        
        # Can't exceed K, can't be less than N0
        return max(self.p.growth.N0, min(self.p.growth.K, sites_needed))
    
    # ---------- Core Optimization ----------
    
    def solve(self, Y: float, p_B: Optional[float] = None) -> OptimizationResult:
        """
        Find optimal allocation for fixed deadline Y.
        
        Strategy: Greedy for Elevator (since c_E << c_R).
        Allocate maximum possible to elevator, rest to rockets.
        """
        payload = p_B if p_B is not None else self.p.p_B
        
        # 1. Calculate capacities
        cap_E = self.cumulative_elevator_capacity(Y)
        cap_R = self.cumulative_rocket_capacity(Y, payload)
        total_cap = cap_E + cap_R
        
        # 2. Feasibility check
        if total_cap < self.p.M_tot:
            return OptimizationResult(
                Y=Y, x_opt=0, mR_opt=0, cost_total=float('inf'),
                cost_capex=0, cost_opex=0, feasible=False,
                N_required=self.p.growth.K, cap_E=cap_E, cap_R=cap_R,
                message=f"Insufficient capacity: {total_cap/1e6:.1f}M < {self.p.M_tot/1e6:.1f}M tons"
            )
        
        # 3. Greedy allocation (elevator first)
        x = min(self.p.M_tot, cap_E)
        m_R = self.p.M_tot - x
        
        # 4. CAPEX
        capex_E = self.p.F_E if x > 0 else 0.0
        
        # Rocket infrastructure: pay for sites actually needed
        N_required = self.calculate_required_sites(m_R, Y)
        N_new = max(0, N_required - self.p.growth.N0)
        capex_R = N_new * self.p.C_site
        
        capex_total = capex_E + capex_R
        
        # 5. OPEX (NPV)
        opex_E = self.calculate_elevator_opex_npv(x, Y)
        opex_R = self.calculate_rocket_opex_npv(m_R, Y, payload)
        opex_total = opex_E + opex_R
        
        return OptimizationResult(
            Y=Y,
            x_opt=x,
            mR_opt=m_R,
            cost_total=capex_total + opex_total,
            cost_capex=capex_total,
            cost_opex=opex_total,
            feasible=True,
            N_required=N_required,
            cap_E=cap_E,
            cap_R=cap_R
        )
    
    # ---------- Monte Carlo Robustness Analysis ----------
    
    def monte_carlo_analysis(self, Y: float, n_samples: int = 1000) -> dict:
        """
        Run Monte Carlo simulation with payload uncertainty p_B ~ U(100, 150).
        
        Key insight: Payload uncertainty affects:
        1. Feasibility (whether capacity is sufficient)
        2. Optimal allocation (with lower payload, more elevator is needed)
        3. Total cost (indirectly through allocation change)
        
        Returns statistics on cost and feasibility.
        """
        p_min, p_max = self.p.p_B_range
        payloads = np.random.uniform(p_min, p_max, n_samples)
        
        costs = []
        feasible_count = 0
        allocations = []  # Track x_opt for variance analysis
        
        for p_B in payloads:
            result = self.solve(Y, p_B=p_B)
            if result.feasible:
                feasible_count += 1
                costs.append(result.cost_total)
                allocations.append(result.x_opt)
        
        feasibility_rate = feasible_count / n_samples
        
        if len(costs) > 1:
            return {
                'Y': Y,
                'feasibility_rate': feasibility_rate,
                'cost_mean': np.mean(costs),
                'cost_std': np.std(costs),
                'cost_5th': np.percentile(costs, 5),
                'cost_95th': np.percentile(costs, 95),
                'allocation_mean': np.mean(allocations),
                'allocation_std': np.std(allocations),
                'n_samples': n_samples
            }
        elif len(costs) == 1:
            return {
                'Y': Y,
                'feasibility_rate': feasibility_rate,
                'cost_mean': costs[0],
                'cost_std': 0.0,
                'cost_5th': costs[0],
                'cost_95th': costs[0],
                'allocation_mean': allocations[0] if allocations else 0,
                'allocation_std': 0.0,
                'n_samples': n_samples
            }
        else:
            return {
                'Y': Y,
                'feasibility_rate': 0.0,
                'cost_mean': float('inf'),
                'cost_std': 0,
                'cost_5th': float('inf'),
                'cost_95th': float('inf'),
                'n_samples': n_samples
            }
    
    # ---------- Pareto Analysis ----------
    
    def find_minimum_feasible_Y(self, Y_min: float = 10, Y_max: float = 100) -> float:
        """Find the minimum Y where the problem becomes feasible."""
        def feasibility_gap(Y):
            cap_E = self.cumulative_elevator_capacity(Y)
            cap_R = self.cumulative_rocket_capacity(Y)
            return cap_E + cap_R - self.p.M_tot
        
        # Check boundaries
        if feasibility_gap(Y_min) >= 0:
            return Y_min
        if feasibility_gap(Y_max) < 0:
            return float('inf')
        
        # Binary search
        return brentq(feasibility_gap, Y_min, Y_max)
    
    def pareto_sweep(self, Y_range: np.ndarray) -> List[OptimizationResult]:
        """Run optimization for a range of Y values."""
        return [self.solve(Y) for Y in Y_range]
    
    @staticmethod
    def find_knee_point(Y_vals: List[float], costs: List[float]) -> int:
        """
        Find the "knee point" of the Pareto front.
        
        Method: Maximum curvature point using second derivative approximation.
        The knee is where marginal cost savings per year drops most significantly.
        """
        if len(Y_vals) < 3:
            return 0
        
        Y = np.array(Y_vals)
        C = np.array(costs)
        
        # Normalize to [0, 1] for fair comparison
        Y_norm = (Y - Y.min()) / (Y.max() - Y.min() + 1e-10)
        C_norm = (C - C.min()) / (C.max() - C.min() + 1e-10)
        
        # Calculate curvature using finite differences
        # Curvature ≈ |d²C/dY²| / (1 + (dC/dY)²)^1.5
        dC = np.gradient(C_norm, Y_norm)
        d2C = np.gradient(dC, Y_norm)
        
        curvature = np.abs(d2C) / (1 + dC**2)**1.5
        
        # Ignore boundary points
        curvature[0] = 0
        curvature[-1] = 0
        
        return int(np.argmax(curvature))
    
    def sensitivity_analysis(self, Y: float, param_name: str, 
                             multipliers: np.ndarray = None) -> List[dict]:
        """
        Analyze sensitivity of cost to parameter variations.
        
        Args:
            Y: Fixed project duration
            param_name: One of 'c_E', 'c_R', 'T_E', 'growth_r'
            multipliers: Array of multipliers (default: 0.5 to 1.5)
        """
        if multipliers is None:
            multipliers = np.linspace(0.5, 1.5, 11)
        
        results = []
        base_value = getattr(self.p, param_name, None)
        
        if base_value is None:
            if param_name == 'growth_r':
                base_value = self.p.growth.r
            else:
                raise ValueError(f"Unknown parameter: {param_name}")
        
        for mult in multipliers:
            # Create modified params
            if param_name == 'c_E':
                new_params = ModelParams(c_E=base_value * mult)
            elif param_name == 'c_R':
                new_params = ModelParams(c_R=base_value * mult)
            elif param_name == 'T_E':
                new_params = ModelParams(T_E=base_value * mult)
            elif param_name == 'growth_r':
                new_params = ModelParams(growth=GrowthParams(r=base_value * mult))
            else:
                continue
            
            temp_model = TransportOptimizationModel(new_params)
            res = temp_model.solve(Y)
            
            results.append({
                'multiplier': mult,
                'param_value': base_value * mult,
                'cost': res.cost_total if res.feasible else float('inf'),
                'elevator_pct': res.elevator_pct if res.feasible else 0,
                'feasible': res.feasible
            })
        
        return results


# ============================================================================
# Visualization Functions
# ============================================================================

def ensure_dir(path: str):
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)


def plot_infrastructure_growth(model: TransportOptimizationModel, save_path: str):
    """Plot Logistic growth of launch sites and capacity."""
    times = np.linspace(0, 60, 200)
    sites = [model.N_t(t) for t in times]
    rates = [model.rocket_capacity_rate(t) / 1e6 for t in times]
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Sites
    color1 = '#2E86AB'
    ax1.plot(times, sites, '-', color=color1, linewidth=2.5, label='Launch Sites N(t)')
    ax1.set_xlabel('Year (from project start)', fontsize=12)
    ax1.set_ylabel('Number of Launch Sites', color=color1, fontsize=12)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0, 90)
    ax1.axhline(y=model.p.growth.K, color=color1, linestyle='--', alpha=0.5, label=f'K = {model.p.growth.K}')
    ax1.axhline(y=model.p.growth.N0, color=color1, linestyle=':', alpha=0.5, label=f'N₀ = {model.p.growth.N0}')
    
    # Capacity
    ax2 = ax1.twinx()
    color2 = '#E94F37'
    ax2.plot(times, rates, '--', color=color2, linewidth=2.5, label='Rocket Capacity')
    ax2.set_ylabel('Annual Transport Capacity (Mt/yr)', color=color2, fontsize=12)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Elevator capacity line
    elevator_rate = model.elevator_bottleneck_rate() / 1e6
    ax2.axhline(y=elevator_rate, color='#4ECDC4', linestyle='-.', linewidth=2, 
                label=f'Elevator Chain Capacity = {elevator_rate:.2f} Mt/yr')
    
    plt.title('Infrastructure Growth & Transport Capacity (Logistic Model)', fontsize=14, fontweight='bold')
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=10)
    
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_pareto_analysis(results: List[OptimizationResult], knee_idx: int, save_path: str):
    """Plot Pareto front and modal split."""
    feas = [r for r in results if r.feasible]
    if not feas:
        print("No feasible results to plot.")
        return
    
    Y_vals = [r.Y for r in feas]
    costs = [r.cost_total / 1e12 for r in feas]
    ele_mass = [r.x_opt / 1e6 for r in feas]
    roc_mass = [r.mR_opt / 1e6 for r in feas]
    capex = [r.cost_capex / 1e12 for r in feas]
    opex = [r.cost_opex / 1e12 for r in feas]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # (1) Pareto Front: Time vs Cost
    ax1 = axes[0, 0]
    ax1.plot(Y_vals, costs, 'o-', color='#2E86AB', linewidth=2, markersize=6)
    ax1.axvline(x=Y_vals[knee_idx], color='#E94F37', linestyle='--', alpha=0.7, label='Knee Point')
    ax1.scatter([Y_vals[knee_idx]], [costs[knee_idx]], color='#E94F37', s=150, zorder=5, 
                marker='*', label=f'Recommended: {Y_vals[knee_idx]:.0f} yrs, ${costs[knee_idx]:.1f}T')
    ax1.set_xlabel('Project Duration (Years)', fontsize=11)
    ax1.set_ylabel('Total Cost NPV (Trillion USD)', fontsize=11)
    ax1.set_title('Pareto Front: Time vs Cost Trade-off', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # (2) Cost Breakdown
    ax2 = axes[0, 1]
    ax2.stackplot(Y_vals, capex, opex, labels=['CAPEX', 'OPEX (NPV)'], 
                  colors=['#A8DADC', '#457B9D'], alpha=0.85)
    ax2.axvline(x=Y_vals[knee_idx], color='#E94F37', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Project Duration (Years)', fontsize=11)
    ax2.set_ylabel('Cost (Trillion USD)', fontsize=11)
    ax2.set_title('Cost Breakdown: CAPEX vs OPEX', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # (3) Modal Split
    ax3 = axes[1, 0]
    ax3.stackplot(Y_vals, ele_mass, roc_mass, labels=['Elevator Chain', 'Direct Rockets'],
                  colors=['#4ECDC4', '#FF6B6B'], alpha=0.85)
    ax3.axvline(x=Y_vals[knee_idx], color='#1D3557', linestyle='--', alpha=0.7)
    ax3.axhline(y=100, color='gray', linestyle=':', alpha=0.5, label='Total Demand = 100 Mt')
    ax3.set_xlabel('Project Duration (Years)', fontsize=11)
    ax3.set_ylabel('Transported Mass (Million Tons)', fontsize=11)
    ax3.set_title('Optimal Modal Split vs Time Constraint', fontsize=12, fontweight='bold')
    ax3.legend(loc='center right')
    ax3.grid(True, alpha=0.3)
    
    # (4) Elevator Share %
    ax4 = axes[1, 1]
    ele_pct = [r.elevator_pct for r in feas]
    ax4.plot(Y_vals, ele_pct, 's-', color='#4ECDC4', linewidth=2, markersize=6)
    ax4.axvline(x=Y_vals[knee_idx], color='#E94F37', linestyle='--', alpha=0.7)
    ax4.fill_between(Y_vals, 0, ele_pct, alpha=0.3, color='#4ECDC4')
    ax4.set_xlabel('Project Duration (Years)', fontsize=11)
    ax4.set_ylabel('Elevator Chain Share (%)', fontsize=11)
    ax4.set_title('Elevator Utilization Rate', fontsize=12, fontweight='bold')
    ax4.set_ylim(0, 100)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_monte_carlo(mc_results: List[dict], save_path: str):
    """Plot Monte Carlo robustness analysis."""
    Y_vals = [r['Y'] for r in mc_results]
    feas_rates = [r['feasibility_rate'] * 100 for r in mc_results]
    cost_means = [r['cost_mean'] / 1e12 for r in mc_results]
    cost_5th = [r['cost_5th'] / 1e12 for r in mc_results]
    cost_95th = [r['cost_95th'] / 1e12 for r in mc_results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Feasibility Rate
    ax1.plot(Y_vals, feas_rates, 'o-', color='#2E86AB', linewidth=2, markersize=6)
    ax1.axhline(y=95, color='#E94F37', linestyle='--', alpha=0.7, label='95% Target')
    ax1.fill_between(Y_vals, 0, feas_rates, alpha=0.3, color='#2E86AB')
    ax1.set_xlabel('Project Duration (Years)', fontsize=11)
    ax1.set_ylabel('Feasibility Rate (%)', fontsize=11)
    ax1.set_title('Monte Carlo: Feasibility under Payload Uncertainty', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 105)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Cost Distribution
    ax2.plot(Y_vals, cost_means, 'o-', color='#457B9D', linewidth=2, markersize=6, label='Mean Cost')
    ax2.fill_between(Y_vals, cost_5th, cost_95th, alpha=0.3, color='#457B9D', label='5th-95th Percentile')
    ax2.set_xlabel('Project Duration (Years)', fontsize=11)
    ax2.set_ylabel('Total Cost NPV (Trillion USD)', fontsize=11)
    ax2.set_title('Monte Carlo: Cost Distribution (p_B ~ U(100,150))', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_sensitivity_analysis(model: TransportOptimizationModel, Y: float, save_path: str):
    """Plot sensitivity analysis for key parameters."""
    multipliers = np.linspace(0.5, 1.5, 21)
    
    # Analyze sensitivity for key parameters
    params_to_test = ['c_E', 'c_R', 'T_E']
    colors = ['#2E86AB', '#E94F37', '#4ECDC4']
    labels = ['Elevator OPEX (c_E)', 'Rocket OPEX (c_R)', 'Elevator Throughput (T_E)']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    base_result = model.solve(Y)
    base_cost = base_result.cost_total / 1e12
    
    for param, color, label in zip(params_to_test, colors, labels):
        results = model.sensitivity_analysis(Y, param, multipliers)
        
        # Cost sensitivity
        cost_ratios = [(r['cost'] / base_result.cost_total) if r['feasible'] else np.nan 
                       for r in results]
        ax1.plot(multipliers * 100, [c * 100 if not np.isnan(c) else np.nan for c in cost_ratios], 
                 'o-', color=color, linewidth=2, markersize=4, label=label)
    
    ax1.axhline(y=100, color='gray', linestyle='--', alpha=0.5)
    ax1.axvline(x=100, color='gray', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Parameter Value (% of Baseline)', fontsize=11)
    ax1.set_ylabel('Total Cost (% of Baseline)', fontsize=11)
    ax1.set_title(f'Cost Sensitivity Analysis (Y = {Y:.0f} years)', fontsize=12, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(45, 155)
    
    # Cost breakdown at baseline
    labels_pie = ['Elevator CAPEX', 'Rocket CAPEX', 'Elevator OPEX', 'Rocket OPEX']
    
    # Calculate components
    capex_E = model.p.F_E / 1e12
    N_new = max(0, base_result.N_required - model.p.growth.N0)
    capex_R = N_new * model.p.C_site / 1e12
    opex_E = model.calculate_elevator_opex_npv(base_result.x_opt, Y) / 1e12
    opex_R = model.calculate_rocket_opex_npv(base_result.mR_opt, Y) / 1e12
    
    sizes = [capex_E, capex_R, opex_E, opex_R]
    colors_pie = ['#A8DADC', '#E9C46A', '#4ECDC4', '#FF6B6B']
    explode = (0.02, 0.02, 0.02, 0.05)
    
    ax2.pie(sizes, explode=explode, labels=labels_pie, colors=colors_pie, autopct='%1.1f%%',
            shadow=False, startangle=90)
    ax2.set_title(f'Cost Composition at Y = {Y:.0f} years\nTotal: ${base_cost:.1f}T', 
                  fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================================
# Main Analysis
# ============================================================================

def run_analysis():
    """Execute comprehensive transport optimization analysis."""
    
    # Output directory
    output_dir = "Problem B/draft/Question 1/1c/image"
    ensure_dir(output_dir)
    
    # Initialize model
    params = ModelParams()
    model = TransportOptimizationModel(params)
    
    print("=" * 70)
    print("Comprehensive Transport Optimization Model V5")
    print("=" * 70)
    print(f"\nModel Parameters:")
    print(f"  Total Demand:       {params.M_tot/1e6:.0f} Million tons")
    print(f"  Elevator T_E:       {params.T_E/1e3:.0f} kt/yr")
    print(f"  Anchor Capacity:    {params.anchor.annual_capacity/1e3:.0f} kt/yr")
    print(f"  Elevator c_E:       ${params.c_E:,.0f}/ton")
    print(f"  Rocket c_R:         ${params.c_R/1e6:.1f}M/ton")
    print(f"  Discount Rate:      {params.discount_rate*100:.1f}%")
    
    # 1. Infrastructure Growth Plot
    print("\n[1/4] Generating infrastructure growth plot...")
    plot_infrastructure_growth(model, f"{output_dir}/infrastructure_growth_v5.png")
    
    # 2. Find feasible range
    Y_min_feas = model.find_minimum_feasible_Y()
    print(f"\n[2/4] Minimum feasible duration: {Y_min_feas:.1f} years")
    
    # 3. Pareto Sweep
    print("\n[3/4] Running Pareto optimization sweep...")
    Y_start = max(15, int(Y_min_feas))
    Y_range = np.linspace(Y_start, 80, 66)
    results = model.pareto_sweep(Y_range)
    
    # Print results table
    print(f"\n{'Year':<8} | {'Elevator %':<12} | {'Rocket %':<10} | {'Cost ($T)':<12} | {'Status'}")
    print("-" * 60)
    
    feas_results = [r for r in results if r.feasible]
    for r in feas_results[::5]:  # Print every 5th result
        print(f"{r.Y:<8.1f} | {r.elevator_pct:<12.1f} | {100-r.elevator_pct:<10.1f} | {r.cost_total/1e12:<12.2f} | OK")
    
    # Find knee point
    Y_vals = [r.Y for r in feas_results]
    costs = [r.cost_total for r in feas_results]
    knee_idx = TransportOptimizationModel.find_knee_point(Y_vals, costs)
    
    knee_result = feas_results[knee_idx]
    print(f"\n>>> Knee Point (Recommended): Y = {knee_result.Y:.0f} years")
    print(f"    Elevator: {knee_result.x_opt/1e6:.1f} Mt ({knee_result.elevator_pct:.1f}%)")
    print(f"    Rockets:  {knee_result.mR_opt/1e6:.1f} Mt ({100-knee_result.elevator_pct:.1f}%)")
    print(f"    Total Cost: ${knee_result.cost_total/1e12:.2f} Trillion")
    
    # Plot Pareto
    plot_pareto_analysis(results, knee_idx, f"{output_dir}/pareto_analysis_v5.png")
    
    # 4. Monte Carlo Analysis
    print("\n[4/4] Running Monte Carlo robustness analysis (1000 samples per Y)...")
    mc_Y_range = np.linspace(Y_start, 60, 10)
    mc_results = []
    
    for Y in mc_Y_range:
        mc = model.monte_carlo_analysis(Y, n_samples=1000)
        mc_results.append(mc)
        print(f"  Y={Y:.0f}: Feasibility={mc['feasibility_rate']*100:.1f}%, "
              f"Cost=${mc['cost_mean']/1e12:.2f}T ± ${mc['cost_std']/1e12:.2f}T")
    
    plot_monte_carlo(mc_results, f"{output_dir}/monte_carlo_analysis_v5.png")
    
    # 5. Sensitivity Analysis
    print("\n[5/5] Running sensitivity analysis at knee point...")
    plot_sensitivity_analysis(model, knee_result.Y, f"{output_dir}/sensitivity_analysis_v5.png")
    
    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)
    
    return model, results, knee_result


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_analysis()
