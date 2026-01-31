"""
Q2 Visualization Module
========================
Comprehensive visualization for transport system reliability and carbon emissions analysis.
This module imports from q2-4.py and generates all visualization outputs.

Usage:
    python q2_visualization.py

Output:
    All figures saved to: Question 2/image/
"""

from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
from matplotlib.gridspec import GridSpec
from dataclasses import dataclass, field
from scipy.integrate import quad
from scipy.optimize import brentq
from typing import Optional, List, Tuple, Dict, Any
import warnings

# Matplotlib configuration
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10

# ============================================================================
# Import/Copy necessary classes from q2-4.py
# ============================================================================

@dataclass(frozen=True)
class ReliabilityParams:
    """Parameters for non-ideal working conditions."""
    beta_E: float = 0.92
    lambda_E: float = 0.015
    t_repair_E: float = 14.0
    C_E_main: float = 500e6
    C_E_fix: float = 50e6
    P_cat_E: float = 0.0005
    eta_energy: float = 0.92
    beta_R: float = 0.92
    P_f_R: float = 0.02
    C_rocket_loss: float = 100e6
    C_cargo_loss: float = 50e6
    T_down_R: float = 7.0
    C_R_maint: float = 50e6
    delta_window: float = 0.03
    delta_maint: float = 0.03
    use_demand_amplification: bool = True


@dataclass(frozen=True)
class GrowthParams:
    K: float = 80.0
    N0: float = 10.0
    r: float = 0.15


@dataclass(frozen=True)
class AnchorParams:
    N_anchor: int = 6
    L_anchor: float = 700.0
    p_A: float = 150.0
    
    @property
    def annual_capacity(self) -> float:
        return self.N_anchor * self.L_anchor * self.p_A


@dataclass
class CarbonParams:
    CO2_per_launch: float = 2500.0
    carbon_price: float = 150.0
    CO2_elevator_per_ton: float = 0.1
    CO2_elevator_construction: float = 5e6
    CO2_launch_site_construction: float = 100000


@dataclass
class ModelParams:
    M_tot: float = 1.0e8
    discount_rate: float = 0.03
    T_E: float = 5.37e5
    F_E: float = 100e9
    c_E: float = 2.7e3
    anchor: AnchorParams = field(default_factory=AnchorParams)
    c_R: float = 3.0e5
    C_site: float = 3.0e9
    L_site_annual: float = 2000.0
    p_B: float = 150.0
    growth: GrowthParams = field(default_factory=GrowthParams)
    reliability: ReliabilityParams = field(default_factory=ReliabilityParams)
    carbon: CarbonParams = field(default_factory=CarbonParams)
    use_ideal: bool = False


@dataclass
class ComponentCosts:
    elevator_capex: float = 0.0
    elevator_opex: float = 0.0
    rocket_capex: float = 0.0
    rocket_opex: float = 0.0
    elevator_carbon_cost: float = 0.0
    rocket_carbon_cost: float = 0.0
    
    @property
    def elevator_total(self) -> float:
        return self.elevator_capex + self.elevator_opex + self.elevator_carbon_cost
    
    @property
    def rocket_total(self) -> float:
        return self.rocket_capex + self.rocket_opex + self.rocket_carbon_cost
    
    @property
    def total_capex(self) -> float:
        return self.elevator_capex + self.rocket_capex
    
    @property
    def total_opex(self) -> float:
        return self.elevator_opex + self.rocket_opex
    
    @property
    def total_carbon_cost(self) -> float:
        return self.elevator_carbon_cost + self.rocket_carbon_cost
    
    @property
    def total_cost(self) -> float:
        return self.elevator_total + self.rocket_total


@dataclass
class CarbonEmissions:
    elevator_operational: float = 0.0
    rocket_operational: float = 0.0
    elevator_construction: float = 0.0
    rocket_construction: float = 0.0
    
    @property
    def elevator_total(self) -> float:
        return self.elevator_operational + self.elevator_construction
    
    @property
    def rocket_total(self) -> float:
        return self.rocket_operational + self.rocket_construction
    
    @property
    def total_operational(self) -> float:
        return self.elevator_operational + self.rocket_operational
    
    @property
    def total_construction(self) -> float:
        return self.elevator_construction + self.rocket_construction
    
    @property
    def total_emissions(self) -> float:
        return self.elevator_total + self.rocket_total


@dataclass
class OptimizationResult:
    Y: float
    x_opt: float
    mR_opt: float
    costs: ComponentCosts
    emissions: CarbonEmissions
    feasible: bool
    N_required: float
    cap_E: float = 0.0
    cap_R: float = 0.0
    message: str = ""
    is_ideal: bool = False
    
    @property
    def elevator_pct(self) -> float:
        if (self.x_opt + self.mR_opt) > 0:
            return (self.x_opt / (self.x_opt + self.mR_opt) * 100)
        return 0.0
    
    @property
    def rocket_pct(self) -> float:
        if (self.x_opt + self.mR_opt) > 0:
            return (self.mR_opt / (self.x_opt + self.mR_opt) * 100)
        return 0.0
    
    @property
    def cost_total(self) -> float:
        return self.costs.total_cost
    
    @property
    def cost_capex(self) -> float:
        return self.costs.total_capex
    
    @property
    def cost_opex(self) -> float:
        return self.costs.total_opex
    
    @property
    def emissions_total(self) -> float:
        return self.emissions.total_emissions
    
    @property
    def carbon_intensity(self) -> float:
        total_mass = self.x_opt + self.mR_opt
        if total_mass > 0:
            return self.emissions_total / total_mass
        return 0.0


class TransportOptimizationModel:
    """Implements the mixed transport optimization model."""
    
    def __init__(self, params: ModelParams):
        self.p = params
    
    def calculate_elevator_availability(self) -> float:
        if self.p.use_ideal:
            return 1.0
        rel = self.p.reliability
        beta_E_base = rel.beta_E
        downtime_failures = rel.lambda_E * rel.t_repair_E
        beta_E_failures = 1 - (downtime_failures / 365)
        beta_E_cat = 1 - rel.P_cat_E
        beta_E_effective = beta_E_base * beta_E_failures * beta_E_cat
        return max(0.1, beta_E_effective)
    
    def calculate_rocket_availability(self) -> float:
        if self.p.use_ideal:
            return 1.0
        rel = self.p.reliability
        beta_R_base = 1 - (rel.delta_window + rel.delta_maint)
        downtime_failures = rel.P_f_R * rel.T_down_R
        beta_R_failures = 1 - (downtime_failures / 365)
        beta_R_effective = beta_R_base * beta_R_failures
        return max(0.2, beta_R_effective)
    
    def get_effective_demand(self, Y: float) -> float:
        if self.p.use_ideal:
            return self.p.M_tot
        rel = self.p.reliability
        if rel.use_demand_amplification:
            P_f_combined = 0.3 * rel.lambda_E + 0.7 * rel.P_f_R
            return self.p.M_tot / (1 - P_f_combined)
        return self.p.M_tot
    
    def N_t(self, t: float) -> float:
        K, N0, r = self.p.growth.K, self.p.growth.N0, self.p.growth.r
        A = (K - N0) / N0
        return K / (1 + A * np.exp(-r * t))
    
    def rocket_capacity_rate(self, t: float, p_B: Optional[float] = None, 
                           adjusted: bool = True) -> float:
        payload = p_B if p_B is not None else self.p.p_B
        base_rate = self.N_t(t) * self.p.L_site_annual * payload
        if adjusted and not self.p.use_ideal:
            beta_R = self.calculate_rocket_availability()
            return base_rate * beta_R
        return base_rate
    
    def cumulative_rocket_capacity(self, Y: float, p_B: Optional[float] = None,
                                 adjusted: bool = True) -> float:
        payload = p_B if p_B is not None else self.p.p_B
        def rate(t):
            return self.rocket_capacity_rate(t, payload, adjusted)
        if Y <= 0:
            return 0.0
        val, _ = quad(rate, 0, Y)
        return val
    
    def cumulative_elevator_capacity(self, Y: float, adjusted: bool = True) -> float:
        if Y <= 0:
            return 0.0
        if adjusted and not self.p.use_ideal:
            beta_E = self.calculate_elevator_availability()
            effective_T_E = self.p.T_E * beta_E
        else:
            effective_T_E = self.p.T_E
        cap_elevator = effective_T_E * Y
        if adjusted and not self.p.use_ideal:
            beta_R = self.calculate_rocket_availability()
            effective_anchor_cap = self.p.anchor.annual_capacity * beta_R
        else:
            effective_anchor_cap = self.p.anchor.annual_capacity
        cap_anchor = effective_anchor_cap * Y
        return min(cap_elevator, cap_anchor)
    
    def find_minimum_feasible_Y(self, Y_min: float = 0.01, Y_max: float = 100) -> float:
        def feasibility_gap(Y):
            cap_E = self.cumulative_elevator_capacity(Y, adjusted=not self.p.use_ideal)
            cap_R = self.cumulative_rocket_capacity(Y, adjusted=not self.p.use_ideal)
            M_eff = self.get_effective_demand(Y)
            return cap_E + cap_R - M_eff
        
        if feasibility_gap(Y_min) >= 0:
            return Y_min
        if feasibility_gap(Y_max) < 0:
            return float('inf')
        try:
            return brentq(feasibility_gap, Y_min, Y_max)
        except ValueError:
            for Y_test in np.linspace(Y_min, Y_max, 200):
                if feasibility_gap(Y_test) >= 0:
                    return Y_test
            return float('inf')
    
    def npv_factor(self, duration: float) -> float:
        rho = self.p.discount_rate
        if rho == 0 or duration == 0:
            return duration
        return (1 - np.exp(-rho * duration)) / rho
    
    def get_adjusted_elevator_opex(self) -> float:
        if self.p.use_ideal:
            return self.p.c_E
        rel = self.p.reliability
        c_E_energy = self.p.c_E / rel.eta_energy
        failure_cost_per_ton = (rel.lambda_E * rel.C_E_fix) / self.p.T_E
        return c_E_energy + failure_cost_per_ton
    
    def get_adjusted_rocket_opex(self) -> float:
        if self.p.use_ideal:
            return self.p.c_R
        rel = self.p.reliability
        risk_cost_per_ton = (rel.P_f_R * (rel.C_rocket_loss + rel.C_cargo_loss)) / self.p.p_B
        maint_cost_per_ton = rel.C_R_maint / (self.p.L_site_annual * self.p.p_B)
        return self.p.c_R + risk_cost_per_ton + maint_cost_per_ton
    
    def calculate_elevator_opex_npv(self, x: float, Y: float) -> float:
        if x <= 0 or Y <= 0:
            return 0.0
        if not self.p.use_ideal:
            maint_npv = self.p.reliability.C_E_main * self.npv_factor(Y)
        else:
            maint_npv = 0.0
        rate = x / Y
        c_E_adj = self.get_adjusted_elevator_opex()
        return rate * c_E_adj * self.npv_factor(Y) + maint_npv
    
    def calculate_rocket_opex_npv(self, m_R: float, Y: float, 
                                p_B: Optional[float] = None) -> float:
        if m_R <= 0 or Y <= 0:
            return 0.0
        max_cap = self.cumulative_rocket_capacity(Y, p_B, adjusted=True)
        if max_cap <= 0:
            return float('inf')
        utilization = min(1.0, m_R / max_cap)
        payload = p_B if p_B is not None else self.p.p_B
        rho = self.p.discount_rate
        c_R_adj = self.get_adjusted_rocket_opex()
        def cost_integrand(t):
            return self.rocket_capacity_rate(t, payload, adjusted=False) * np.exp(-rho * t)
        integral_val, _ = quad(cost_integrand, 0, Y)
        return utilization * c_R_adj * integral_val
    
    def calculate_required_sites(self, m_R: float, Y: float) -> float:
        if m_R <= 0 or Y <= 0:
            return self.p.growth.N0
        if not self.p.use_ideal:
            beta_R = self.calculate_rocket_availability()
            effective_rate = self.p.L_site_annual * beta_R
        else:
            effective_rate = self.p.L_site_annual
        annual_need = m_R / Y
        sites_needed = annual_need / (effective_rate * self.p.p_B)
        return max(self.p.growth.N0, min(self.p.growth.K, sites_needed))
    
    def calculate_rocket_launches(self, m_R: float, Y: float) -> float:
        if m_R <= 0 or Y <= 0:
            return 0.0
        if not self.p.use_ideal:
            beta_R = self.calculate_rocket_availability()
            effective_launch_rate = self.p.L_site_annual * beta_R
        else:
            effective_launch_rate = self.p.L_site_annual
        sites_needed = self.calculate_required_sites(m_R, Y)
        total_launches = sites_needed * effective_launch_rate * Y
        return total_launches
    
    def calculate_carbon_emissions(self, x: float, m_R: float, Y: float, 
                                  N_required: float) -> CarbonEmissions:
        emissions = CarbonEmissions()
        emissions.elevator_operational = x * self.p.carbon.CO2_elevator_per_ton
        if x > 0:
            emissions.elevator_construction = self.p.carbon.CO2_elevator_construction
        if m_R > 0:
            total_launches = self.calculate_rocket_launches(m_R, Y)
            emissions.rocket_operational = total_launches * self.p.carbon.CO2_per_launch
        N_new = max(0, N_required - self.p.growth.N0)
        emissions.rocket_construction = N_new * self.p.carbon.CO2_launch_site_construction
        return emissions
    
    def calculate_carbon_costs(self, emissions: CarbonEmissions) -> Tuple[float, float]:
        elevator_carbon_cost = (emissions.elevator_operational + emissions.elevator_construction) * self.p.carbon.carbon_price
        rocket_carbon_cost = (emissions.rocket_operational + emissions.rocket_construction) * self.p.carbon.carbon_price
        return elevator_carbon_cost, rocket_carbon_cost
    
    def solve(self, Y: float, p_B: Optional[float] = None) -> OptimizationResult:
        payload = p_B if p_B is not None else self.p.p_B
        if Y <= 0:
            return OptimizationResult(
                Y=Y, x_opt=0, mR_opt=0, 
                costs=ComponentCosts(),
                emissions=CarbonEmissions(),
                feasible=False,
                N_required=self.p.growth.N0, cap_E=0, cap_R=0,
                message=f"Time must be positive (got {Y})", 
                is_ideal=self.p.use_ideal
            )
        M_eff = self.get_effective_demand(Y)
        cap_E = self.cumulative_elevator_capacity(Y, adjusted=not self.p.use_ideal)
        cap_R = self.cumulative_rocket_capacity(Y, payload, adjusted=not self.p.use_ideal)
        total_cap = cap_E + cap_R
        if total_cap < M_eff:
            return OptimizationResult(
                Y=Y, x_opt=0, mR_opt=0, 
                costs=ComponentCosts(),
                emissions=CarbonEmissions(),
                feasible=False,
                N_required=self.p.growth.K, cap_E=cap_E, cap_R=cap_R,
                message=f"Insufficient capacity", 
                is_ideal=self.p.use_ideal
            )
        x = min(M_eff, cap_E)
        m_R = M_eff - x
        N_required = self.calculate_required_sites(m_R, Y)
        costs = ComponentCosts()
        costs.elevator_capex = self.p.F_E if x > 0 else 0.0
        N_new = max(0, N_required - self.p.growth.N0)
        costs.rocket_capex = N_new * self.p.C_site
        if not self.p.use_ideal:
            costs.rocket_capex *= 1.10
        costs.elevator_opex = self.calculate_elevator_opex_npv(x, Y)
        costs.rocket_opex = self.calculate_rocket_opex_npv(m_R, Y, payload)
        emissions = self.calculate_carbon_emissions(x, m_R, Y, N_required)
        costs.elevator_carbon_cost, costs.rocket_carbon_cost = self.calculate_carbon_costs(emissions)
        return OptimizationResult(
            Y=Y, x_opt=x, mR_opt=m_R,
            costs=costs,
            emissions=emissions,
            feasible=True, N_required=N_required,
            cap_E=cap_E, cap_R=cap_R, is_ideal=self.p.use_ideal
        )

# Matplotlib configuration
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['font.size'] = 10

# Color schemes
COLORS = {
    'ideal': '#2E86AB',      # Blue
    'real': '#E94F37',       # Red
    'elevator': '#4ECDC4',   # Teal
    'rocket': '#FF6B6B',     # Coral
    'capex': '#A8DADC',      # Light blue
    'opex': '#457B9D',       # Dark blue
    'carbon': '#F4A261',     # Orange
    'construction': '#E9C46A',  # Yellow
    'operational': '#2A9D8F',   # Green
    'highlight': '#E63946',  # Bright red
}

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "image")


def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")


def run_analysis(total_mass: float = 1.0e8, Y: float = 24.0):
    """
    Run the analysis and return results for both ideal and real conditions.
    
    Args:
        total_mass: Total mass to transport (tons)
        Y: Project duration (years)
    
    Returns:
        dict with ideal_result, real_result, ideal_model, real_model
    """
    # Suppress debug output
    import io
    from contextlib import redirect_stdout
    
    # Create models
    ideal_params = ModelParams(use_ideal=True, M_tot=total_mass)
    real_params = ModelParams(use_ideal=False, M_tot=total_mass)
    
    ideal_model = TransportOptimizationModel(ideal_params)
    real_model = TransportOptimizationModel(real_params)
    
    # Suppress debug prints
    with redirect_stdout(io.StringIO()):
        ideal_result = ideal_model.solve(Y)
        real_result = real_model.solve(Y)
        
        # Find minimum feasible times
        Y_min_ideal = ideal_model.find_minimum_feasible_Y(Y_min=0.01, Y_max=100)
        Y_min_real = real_model.find_minimum_feasible_Y(Y_min=0.01, Y_max=100)
    
    # Calculate additional metrics
    ideal_launches = ideal_model.calculate_rocket_launches(ideal_result.mR_opt, Y)
    real_launches = real_model.calculate_rocket_launches(real_result.mR_opt, Y)
    
    return {
        'ideal_result': ideal_result,
        'real_result': real_result,
        'ideal_model': ideal_model,
        'real_model': real_model,
        'ideal_launches': ideal_launches,
        'real_launches': real_launches,
        'Y_min_ideal': Y_min_ideal,
        'Y_min_real': Y_min_real,
        'Y': Y,
        'total_mass': total_mass,
    }


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_radar_comparison(data: dict, save_path: str):
    """
    Plot radar chart comparing ideal vs real conditions across multiple dimensions.
    """
    ideal = data['ideal_result']
    real = data['real_result']
    
    # Define metrics (normalized to 0-1 scale for visualization)
    categories = [
        'Completion\nTime',
        'Total\nCost',
        'Carbon\nEmissions',
        'Elevator\nUtilization',
        'Rocket\nLaunches',
        'Carbon\nIntensity'
    ]
    
    # Get raw values
    ideal_values = [
        data['Y_min_ideal'],
        ideal.cost_total / 1e12,
        ideal.emissions_total / 1e6,
        ideal.elevator_pct,
        data['ideal_launches'] / 1e6,
        ideal.carbon_intensity
    ]
    
    real_values = [
        data['Y_min_real'],
        real.cost_total / 1e12,
        real.emissions_total / 1e6,
        real.elevator_pct,
        data['real_launches'] / 1e6,
        real.carbon_intensity
    ]
    
    # Normalize values (0-1 scale based on max)
    max_values = [max(i, r) * 1.2 for i, r in zip(ideal_values, real_values)]
    ideal_norm = [v / m if m > 0 else 0 for v, m in zip(ideal_values, max_values)]
    real_norm = [v / m if m > 0 else 0 for v, m in zip(real_values, max_values)]
    
    # Number of variables
    N = len(categories)
    
    # Compute angle for each category
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the loop
    
    ideal_norm += ideal_norm[:1]
    real_norm += real_norm[:1]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Draw ideal condition
    ax.plot(angles, ideal_norm, 'o-', linewidth=2.5, label='Ideal Conditions', 
            color=COLORS['ideal'], markersize=8)
    ax.fill(angles, ideal_norm, alpha=0.25, color=COLORS['ideal'])
    
    # Draw real condition
    ax.plot(angles, real_norm, 's-', linewidth=2.5, label='Real Conditions', 
            color=COLORS['real'], markersize=8)
    ax.fill(angles, real_norm, alpha=0.25, color=COLORS['real'])
    
    # Set category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=11, fontweight='bold')
    
    # Add value annotations
    for i, (angle, iv, rv) in enumerate(zip(angles[:-1], ideal_values, real_values)):
        # Ideal value
        ax.annotate(f'{iv:.2f}', xy=(angle, ideal_norm[i]), 
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, color=COLORS['ideal'], fontweight='bold')
        # Real value
        ax.annotate(f'{rv:.2f}', xy=(angle, real_norm[i]), 
                   xytext=(5, -10), textcoords='offset points',
                   fontsize=9, color=COLORS['real'], fontweight='bold')
    
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['25%', '50%', '75%', '100%'], color='gray', size=9)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.1), fontsize=11)
    plt.title('Ideal vs Real Conditions: Multi-Dimensional Comparison\n', 
              fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def plot_cost_breakdown_grouped_bar(data: dict, save_path: str):
    """
    Plot grouped bar chart comparing cost components between ideal and real conditions.
    Uses dual panels to handle large scale differences.
    """
    ideal = data['ideal_result']
    real = data['real_result']
    
    # Cost categories - separate small and large components
    small_categories = ['Elevator\nCAPEX', 'Elevator\nOPEX', 'Elevator\nCarbon',
                        'Rocket\nCAPEX', 'Rocket\nCarbon']
    large_categories = ['Rocket\nOPEX', 'TOTAL']
    
    # Values in billions
    ideal_small = [
        ideal.costs.elevator_capex / 1e9,
        ideal.costs.elevator_opex / 1e9,
        ideal.costs.elevator_carbon_cost / 1e9,
        ideal.costs.rocket_capex / 1e9,
        ideal.costs.rocket_carbon_cost / 1e9,
    ]
    real_small = [
        real.costs.elevator_capex / 1e9,
        real.costs.elevator_opex / 1e9,
        real.costs.elevator_carbon_cost / 1e9,
        real.costs.rocket_capex / 1e9,
        real.costs.rocket_carbon_cost / 1e9,
    ]
    
    ideal_large = [ideal.costs.rocket_opex / 1e9, ideal.cost_total / 1e9]
    real_large = [real.costs.rocket_opex / 1e9, real.cost_total / 1e9]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), 
                                    gridspec_kw={'width_ratios': [5, 2]})
    
    width = 0.35
    
    # ========== Left panel: Small components ==========
    x1 = np.arange(len(small_categories))
    bars1_small = ax1.bar(x1 - width/2, ideal_small, width, label='Ideal Conditions', 
                          color=COLORS['ideal'], alpha=0.85, edgecolor='white', linewidth=1.5)
    bars2_small = ax1.bar(x1 + width/2, real_small, width, label='Real Conditions', 
                          color=COLORS['real'], alpha=0.85, edgecolor='white', linewidth=1.5)
    
    # Add value labels
    for bar, val in zip(bars1_small, ideal_small):
        ax1.annotate(f'${val:.1f}B',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar, val in zip(bars2_small, real_small):
        ax1.annotate(f'${val:.1f}B',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax1.set_ylabel('Cost (Billion USD)', fontsize=12)
    ax1.set_xlabel('Cost Component', fontsize=12)
    ax1.set_title('Infrastructure & Carbon Costs\n(Smaller Components)', fontsize=13, fontweight='bold')
    ax1.set_xticks(x1)
    ax1.set_xticklabels(small_categories, fontsize=10)
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add change percentages
    for i, (iv, rv) in enumerate(zip(ideal_small, real_small)):
        if iv > 0:
            pct_change = (rv - iv) / iv * 100
            color = COLORS['highlight'] if pct_change > 0 else COLORS['operational']
            ax1.annotate(f'{pct_change:+.1f}%',
                        xy=(i, max(iv, rv) + max(ideal_small) * 0.08),
                        ha='center', va='bottom', fontsize=8, fontweight='bold',
                        color=color)
    
    # ========== Right panel: Large components (Rocket OPEX & Total) ==========
    x2 = np.arange(len(large_categories))
    bars1_large = ax2.bar(x2 - width/2, ideal_large, width, label='Ideal Conditions', 
                          color=COLORS['ideal'], alpha=0.85, edgecolor='white', linewidth=1.5)
    bars2_large = ax2.bar(x2 + width/2, real_large, width, label='Real Conditions', 
                          color=COLORS['real'], alpha=0.85, edgecolor='white', linewidth=1.5)
    
    # Convert to Trillion for readability
    for bar, val in zip(bars1_large, ideal_large):
        ax2.annotate(f'${val/1000:.2f}T',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    for bar, val in zip(bars2_large, real_large):
        ax2.annotate(f'${val/1000:.2f}T',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Add change percentages
    for i, (iv, rv) in enumerate(zip(ideal_large, real_large)):
        if iv > 0:
            pct_change = (rv - iv) / iv * 100
            ax2.annotate(f'{pct_change:+.1f}%',
                        xy=(i + width/2 + 0.1, rv),
                        ha='left', va='center', fontsize=11, fontweight='bold',
                        color=COLORS['highlight'])
    
    ax2.set_ylabel('Cost (Billion USD)', fontsize=12)
    ax2.set_xlabel('Cost Component', fontsize=12)
    ax2.set_title('Rocket Operations & Total\n(Dominant Components)', fontsize=13, fontweight='bold')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(large_categories, fontsize=10)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add note about scale difference
    fig.text(0.5, 0.01, 
             '⚠️ Note: Right panel values are ~400× larger than left panel. Rocket OPEX dominates total cost.',
             ha='center', fontsize=10, fontstyle='italic', color='gray')
    
    plt.suptitle('Cost Breakdown: Ideal vs Real Conditions', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def plot_carbon_emissions_stacked(data: dict, save_path: str):
    """
    Plot carbon emissions breakdown with improved visualization for scale differences.
    Uses main plot + inset for small components.
    """
    ideal = data['ideal_result']
    real = data['real_result']
    
    categories = ['Ideal\nConditions', 'Real\nConditions']
    
    # Emissions in million tons CO2
    elevator_construction = [
        ideal.emissions.elevator_construction / 1e6,
        real.emissions.elevator_construction / 1e6
    ]
    elevator_operational = [
        ideal.emissions.elevator_operational / 1e6,
        real.emissions.elevator_operational / 1e6
    ]
    rocket_construction = [
        ideal.emissions.rocket_construction / 1e6,
        real.emissions.rocket_construction / 1e6
    ]
    rocket_operational = [
        ideal.emissions.rocket_operational / 1e6,
        real.emissions.rocket_operational / 1e6
    ]
    
    totals = [ideal.emissions_total / 1e6, real.emissions_total / 1e6]
    
    fig = plt.figure(figsize=(16, 8))
    
    # Main layout: 3 columns
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.2, 1], wspace=0.3)
    
    # ========== Left panel: Main stacked bar (total view) ==========
    ax1 = fig.add_subplot(gs[0])
    
    x = np.arange(len(categories))
    width = 0.5
    
    # Only show rocket operational as it dominates
    bars_rocket_op = ax1.bar(x, rocket_operational, width, label='Rocket Operational', 
                              color=COLORS['rocket'], alpha=0.85)
    
    # Add total labels
    for i, (total, rk_op) in enumerate(zip(totals, rocket_operational)):
        ax1.annotate(f'{total:.1f} Mt CO₂\n(Rocket Op: {rk_op/total*100:.1f}%)',
                    xy=(i, total + 10),
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax1.set_ylabel('Carbon Emissions (Million tons CO₂)', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=12)
    ax1.set_title('Total Carbon Emissions\n(Rocket Operations Dominant)', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    
    # ========== Middle panel: Small components detail ==========
    ax2 = fig.add_subplot(gs[1])
    
    # Bar chart for small components
    small_categories = ['Elev.\nConstr.', 'Elev.\nOper.', 'Rocket\nConstr.']
    ideal_small = [elevator_construction[0], elevator_operational[0], rocket_construction[0]]
    real_small = [elevator_construction[1], elevator_operational[1], rocket_construction[1]]
    
    x2 = np.arange(len(small_categories))
    width2 = 0.35
    
    bars_i = ax2.bar(x2 - width2/2, ideal_small, width2, label='Ideal', 
                     color=COLORS['ideal'], alpha=0.85, edgecolor='white', linewidth=1.5)
    bars_r = ax2.bar(x2 + width2/2, real_small, width2, label='Real', 
                     color=COLORS['real'], alpha=0.85, edgecolor='white', linewidth=1.5)
    
    # Add value labels
    for bar, val in zip(bars_i, ideal_small):
        if val > 0:
            ax2.annotate(f'{val:.3f}',
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar, val in zip(bars_r, real_small):
        if val > 0:
            ax2.annotate(f'{val:.3f}',
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax2.set_ylabel('Carbon Emissions (Mt CO₂)', fontsize=12)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(small_categories, fontsize=10)
    ax2.set_title('Non-Operational Emissions\n(Zoomed: <1% of Total)', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add percentage note
    total_small_ideal = sum(ideal_small)
    total_small_real = sum(real_small)
    ax2.text(0.5, 0.95, f'Ideal: {total_small_ideal/totals[0]*100:.2f}% | Real: {total_small_real/totals[1]*100:.2f}% of total',
             transform=ax2.transAxes, ha='center', va='top', fontsize=9, fontstyle='italic',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # ========== Right panel: Carbon intensity comparison ==========
    ax3 = fig.add_subplot(gs[2])
    
    intensities = [ideal.carbon_intensity, real.carbon_intensity]
    colors_bar = [COLORS['ideal'], COLORS['real']]
    
    bars = ax3.barh(categories, intensities, color=colors_bar, alpha=0.85, 
                    edgecolor='white', linewidth=2, height=0.5)
    
    # Add value labels
    for bar, val in zip(bars, intensities):
        ax3.annotate(f'{val:.3f} tCO₂/t',
                    xy=(val, bar.get_y() + bar.get_height()/2),
                    xytext=(5, 0),
                    textcoords="offset points",
                    ha='left', va='center', fontsize=11, fontweight='bold')
    
    # Add change indicator
    change_pct = (intensities[1] - intensities[0]) / intensities[0] * 100
    ax3.annotate(f'+{change_pct:.1f}%',
                xy=(intensities[1] + 0.3, 1),
                ha='left', va='center', fontsize=12, fontweight='bold',
                color=COLORS['highlight'])
    
    ax3.set_xlabel('Carbon Intensity\n(tCO₂ per ton payload)', fontsize=11)
    ax3.set_title('Carbon Intensity\nComparison', fontsize=13, fontweight='bold')
    ax3.set_xlim(0, max(intensities) * 1.4)
    ax3.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add overall note
    fig.text(0.5, -0.02, 
             '📊 Note: Rocket operational emissions account for >99% of total. Middle panel shows zoomed view of construction emissions.',
             ha='center', fontsize=10, fontstyle='italic', color='gray')
    
    plt.suptitle('Carbon Emissions Analysis', fontsize=15, fontweight='bold', y=1.02)
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def plot_reliability_breakdown(data: dict, save_path: str):
    """
    Plot horizontal bar chart showing how reliability factors reduce effective capacity.
    """
    ideal_model = data['ideal_model']
    real_model = data['real_model']
    
    # Get reliability parameters
    rel = real_model.p.reliability
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # ========== Elevator System ==========
    beta_E_base = rel.beta_E
    downtime_failures = rel.lambda_E * rel.t_repair_E / 365
    beta_E_fail = 1 - downtime_failures
    beta_E_cat = 1 - rel.P_cat_E
    beta_E_effective = real_model.calculate_elevator_availability()
    
    # Components that reduce availability
    elev_components = ['Base\nAvailability', 'After\nFailures', 'After\nCatastrophic', 'Effective']
    elev_values = [1.0, beta_E_base, beta_E_base * beta_E_fail, beta_E_effective]
    elev_losses = [0, 1 - beta_E_base, beta_E_base * downtime_failures, 
                   beta_E_base * beta_E_fail * rel.P_cat_E]
    
    y_pos = np.arange(len(elev_components))
    
    # Remaining availability
    bars1 = ax1.barh(y_pos, elev_values, color=COLORS['elevator'], alpha=0.85,
                     edgecolor='white', linewidth=1.5, label='Available')
    
    # Add loss indicators
    for i, (val, loss) in enumerate(zip(elev_values, elev_losses)):
        if loss > 0:
            ax1.annotate(f'-{loss*100:.2f}%',
                        xy=(val, i),
                        xytext=(5, 0),
                        textcoords="offset points",
                        ha='left', va='center', fontsize=10, 
                        color=COLORS['highlight'], fontweight='bold')
        ax1.annotate(f'{val*100:.2f}%',
                    xy=(val/2, i),
                    ha='center', va='center', fontsize=11, fontweight='bold',
                    color='white')
    
    ax1.set_xlim(0, 1.15)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(elev_components, fontsize=11)
    ax1.set_xlabel('Availability Factor', fontsize=12)
    ax1.set_title('Elevator System: Availability Factor Decomposition\n', 
                  fontsize=13, fontweight='bold')
    ax1.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    
    # ========== Rocket System ==========
    delta_window = rel.delta_window
    delta_maint = rel.delta_maint
    beta_R_base = 1 - (delta_window + delta_maint)
    downtime_R = rel.P_f_R * rel.T_down_R / 365
    beta_R_fail = 1 - downtime_R
    beta_R_effective = real_model.calculate_rocket_availability()
    
    rock_components = ['Ideal\n(100%)', 'After\nWindow Loss', 'After\nMaintenance', 
                       'After\nFailures', 'Effective']
    rock_values = [1.0, 1 - delta_window, beta_R_base, 
                   beta_R_base * beta_R_fail, beta_R_effective]
    rock_losses = [0, delta_window, delta_maint, downtime_R * beta_R_base, 0]
    
    y_pos2 = np.arange(len(rock_components))
    
    bars2 = ax2.barh(y_pos2, rock_values, color=COLORS['rocket'], alpha=0.85,
                     edgecolor='white', linewidth=1.5, label='Available')
    
    for i, (val, loss) in enumerate(zip(rock_values, rock_losses)):
        if loss > 0:
            ax2.annotate(f'-{loss*100:.2f}%',
                        xy=(val, i),
                        xytext=(5, 0),
                        textcoords="offset points",
                        ha='left', va='center', fontsize=10,
                        color=COLORS['highlight'], fontweight='bold')
        ax2.annotate(f'{val*100:.2f}%',
                    xy=(val/2, i),
                    ha='center', va='center', fontsize=11, fontweight='bold',
                    color='white')
    
    ax2.set_xlim(0, 1.15)
    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels(rock_components, fontsize=11)
    ax2.set_xlabel('Availability Factor', fontsize=12)
    ax2.set_title('Rocket System: Availability Factor Decomposition\n', 
                  fontsize=13, fontweight='bold')
    ax2.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def plot_cost_waterfall(data: dict, save_path: str):
    """
    Plot waterfall chart showing cost changes from ideal to real conditions.
    Uses dual panels to handle scale differences between small and large changes.
    """
    ideal = data['ideal_result']
    real = data['real_result']
    
    # Calculate individual changes in billions
    changes_dict = {
        'Elevator OPEX': (real.costs.elevator_opex - ideal.costs.elevator_opex) / 1e9,
        'Rocket CAPEX': (real.costs.rocket_capex - ideal.costs.rocket_capex) / 1e9,
        'Rocket OPEX': (real.costs.rocket_opex - ideal.costs.rocket_opex) / 1e9,
        'Carbon Cost': (real.costs.total_carbon_cost - ideal.costs.total_carbon_cost) / 1e9,
    }
    
    total_change = (real.cost_total - ideal.cost_total) / ideal.cost_total * 100
    
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.5], wspace=0.25)
    
    # ========== Left panel: Percentage contribution breakdown ==========
    ax1 = fig.add_subplot(gs[0])
    
    labels = list(changes_dict.keys())
    values = list(changes_dict.values())
    total_increase = sum(values)
    percentages = [v / total_increase * 100 for v in values]
    
    # Horizontal bar chart for percentage contribution
    colors_bar = [COLORS['highlight'] if v > 0 else COLORS['operational'] for v in values]
    y_pos = np.arange(len(labels))
    
    bars = ax1.barh(y_pos, percentages, color=colors_bar, alpha=0.85, 
                    edgecolor='white', linewidth=1.5)
    
    # Add value labels (both percentage and absolute)
    for i, (bar, pct, val) in enumerate(zip(bars, percentages, values)):
        # Percentage inside bar
        ax1.annotate(f'{pct:.1f}%',
                    xy=(pct/2, i),
                    ha='center', va='center', fontsize=11, fontweight='bold',
                    color='white')
        # Absolute value outside
        ax1.annotate(f'+${val:.1f}B',
                    xy=(pct + 2, i),
                    ha='left', va='center', fontsize=10, fontweight='bold')
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=11)
    ax1.set_xlabel('Contribution to Cost Increase (%)', fontsize=12)
    ax1.set_title('Cost Increase by Component\n(Percentage Breakdown)', fontsize=13, fontweight='bold')
    ax1.set_xlim(0, 110)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add note about dominance
    ax1.text(0.5, -0.12, f'Rocket OPEX accounts for {percentages[2]:.1f}% of all cost increases',
             transform=ax1.transAxes, ha='center', fontsize=10, fontstyle='italic',
             color=COLORS['highlight'])
    
    # ========== Right panel: Visual waterfall (simplified) ==========
    ax2 = fig.add_subplot(gs[1])
    
    # Simplified waterfall with just start, dominant change, and end
    waterfall_labels = ['Ideal\nTotal', 'Other\nChanges', 'Rocket\nOPEX↑', 'Real\nTotal']
    other_changes = sum(values) - values[2]  # All changes except Rocket OPEX
    waterfall_values = [
        ideal.cost_total / 1e9,          # Start
        other_changes,                    # Other changes (small)
        values[2],                        # Rocket OPEX (dominant)
        real.cost_total / 1e9             # End
    ]
    
    # Calculate cumulative positions
    cumulative = [0, waterfall_values[0], waterfall_values[0] + other_changes, 0]
    
    x = np.arange(len(waterfall_labels))
    
    # Plot bars
    colors_wf = [COLORS['ideal'], COLORS['operational'], COLORS['highlight'], COLORS['real']]
    
    for i, (label, val) in enumerate(zip(waterfall_labels, waterfall_values)):
        if i == 0:  # Start bar
            ax2.bar(i, val, color=colors_wf[i], alpha=0.85, edgecolor='white', linewidth=2)
            ax2.annotate(f'${val/1000:.2f}T', xy=(i, val), ha='center', va='bottom',
                        fontsize=11, fontweight='bold')
        elif i == 3:  # End bar
            ax2.bar(i, val, color=colors_wf[i], alpha=0.85, edgecolor='white', linewidth=2)
            ax2.annotate(f'${val/1000:.2f}T', xy=(i, val), ha='center', va='bottom',
                        fontsize=11, fontweight='bold')
        elif i == 1:  # Other changes (small)
            bottom = cumulative[i]
            ax2.bar(i, val, bottom=bottom, color=colors_wf[i], alpha=0.85, 
                   edgecolor='white', linewidth=2)
            ax2.annotate(f'+${val:.1f}B\n({val/sum(waterfall_values[1:3])*100:.1f}%)', 
                        xy=(i, bottom + val + 500), ha='center', va='bottom',
                        fontsize=9, fontweight='bold')
            # Connector line
            ax2.hlines(y=bottom + val, xmin=i-0.4, xmax=i+0.5, color='gray', 
                      linestyle='--', alpha=0.5, linewidth=1)
        else:  # Rocket OPEX (dominant)
            bottom = cumulative[i]
            ax2.bar(i, val, bottom=bottom, color=colors_wf[i], alpha=0.85,
                   edgecolor='white', linewidth=2)
            ax2.annotate(f'+${val/1000:.2f}T\n({val/sum(waterfall_values[1:3])*100:.1f}%)', 
                        xy=(i, bottom + val/2), ha='center', va='center',
                        fontsize=11, fontweight='bold', color='white')
            # Connector line
            ax2.hlines(y=bottom + val, xmin=i-0.4, xmax=i+0.5, color='gray',
                      linestyle='--', alpha=0.5, linewidth=1)
    
    # Connector from start to first change
    ax2.hlines(y=waterfall_values[0], xmin=0.4, xmax=0.6, color='gray', 
              linestyle='--', alpha=0.5, linewidth=1)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(waterfall_labels, fontsize=10)
    ax2.set_ylabel('Cost (Billion USD)', fontsize=12)
    ax2.set_title('Cost Waterfall: From Ideal to Real\n', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add total change annotation
    ax2.annotate(f'Total Increase: +{total_change:.1f}%\n(+${total_increase/1000:.2f}T)',
                xy=(0.5, 0.95), xycoords='axes fraction',
                ha='center', va='top', fontsize=13, fontweight='bold',
                color=COLORS['highlight'],
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.suptitle('Cost Waterfall Analysis: Ideal → Real Conditions', 
                fontsize=15, fontweight='bold', y=1.02)
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def plot_mass_allocation_pie(data: dict, save_path: str):
    """
    Plot pie charts comparing mass allocation between ideal and real conditions.
    """
    ideal = data['ideal_result']
    real = data['real_result']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # Ideal conditions
    ideal_sizes = [ideal.x_opt / 1e6, ideal.mR_opt / 1e6]
    ideal_labels = [f'Elevator\n{ideal.x_opt/1e6:.1f} Mt\n({ideal.elevator_pct:.1f}%)',
                    f'Rocket\n{ideal.mR_opt/1e6:.1f} Mt\n({ideal.rocket_pct:.1f}%)']
    colors1 = [COLORS['elevator'], COLORS['rocket']]
    
    wedges1, texts1 = ax1.pie(ideal_sizes, colors=colors1, 
                              startangle=90, explode=(0.02, 0.02),
                              wedgeprops=dict(width=0.6, edgecolor='white', linewidth=2))
    
    # Add center text
    ax1.text(0, 0, f'Total\n{(ideal.x_opt + ideal.mR_opt)/1e6:.1f} Mt', 
            ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Add labels
    ax1.legend(wedges1, ideal_labels, loc='center left', bbox_to_anchor=(0.85, 0.5),
               fontsize=11, frameon=False)
    ax1.set_title('Ideal Conditions\nMass Allocation', fontsize=14, fontweight='bold')
    
    # Real conditions
    real_sizes = [real.x_opt / 1e6, real.mR_opt / 1e6]
    real_labels = [f'Elevator\n{real.x_opt/1e6:.1f} Mt\n({real.elevator_pct:.1f}%)',
                   f'Rocket\n{real.mR_opt/1e6:.1f} Mt\n({real.rocket_pct:.1f}%)']
    
    wedges2, texts2 = ax2.pie(real_sizes, colors=colors1,
                              startangle=90, explode=(0.02, 0.02),
                              wedgeprops=dict(width=0.6, edgecolor='white', linewidth=2))
    
    ax2.text(0, 0, f'Total\n{(real.x_opt + real.mR_opt)/1e6:.1f} Mt', 
            ha='center', va='center', fontsize=14, fontweight='bold')
    
    ax2.legend(wedges2, real_labels, loc='center left', bbox_to_anchor=(0.85, 0.5),
               fontsize=11, frameon=False)
    ax2.set_title('Real Conditions\nMass Allocation', fontsize=14, fontweight='bold')
    
    # Add comparison note
    elev_change = (real.x_opt - ideal.x_opt) / ideal.x_opt * 100 if ideal.x_opt > 0 else 0
    rock_change = (real.mR_opt - ideal.mR_opt) / ideal.mR_opt * 100 if ideal.mR_opt > 0 else 0
    
    fig.text(0.5, 0.02, 
             f'Elevator: {elev_change:+.1f}%  |  Rocket: {rock_change:+.1f}%  |  '
             f'Demand Amplification: +{((real.x_opt + real.mR_opt) - (ideal.x_opt + ideal.mR_opt))/(ideal.x_opt + ideal.mR_opt)*100:.1f}%',
             ha='center', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def plot_capacity_curves(data: dict, save_path: str):
    """
    Plot cumulative capacity curves comparing ideal and real conditions over time.
    """
    ideal_model = data['ideal_model']
    real_model = data['real_model']
    total_mass = data['total_mass']
    
    # Time range
    Y_range = np.linspace(0.1, 50, 200)
    
    # Calculate capacities (suppress debug output)
    import io
    from contextlib import redirect_stdout
    
    ideal_elev_cap = []
    ideal_rock_cap = []
    real_elev_cap = []
    real_rock_cap = []
    
    with redirect_stdout(io.StringIO()):
        for Y in Y_range:
            # Ideal
            ideal_elev_cap.append(ideal_model.cumulative_elevator_capacity(Y, adjusted=False) / 1e6)
            ideal_rock_cap.append(ideal_model.cumulative_rocket_capacity(Y, adjusted=False) / 1e6)
            # Real
            real_elev_cap.append(real_model.cumulative_elevator_capacity(Y, adjusted=True) / 1e6)
            real_rock_cap.append(real_model.cumulative_rocket_capacity(Y, adjusted=True) / 1e6)
    
    ideal_total = np.array(ideal_elev_cap) + np.array(ideal_rock_cap)
    real_total = np.array(real_elev_cap) + np.array(real_rock_cap)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Left plot: Total capacity comparison
    ax1.plot(Y_range, ideal_total, '-', color=COLORS['ideal'], linewidth=2.5, 
             label='Ideal Total Capacity')
    ax1.plot(Y_range, real_total, '-', color=COLORS['real'], linewidth=2.5, 
             label='Real Total Capacity')
    ax1.fill_between(Y_range, real_total, ideal_total, alpha=0.3, color=COLORS['highlight'],
                     label='Capacity Loss')
    
    ax1.axhline(y=total_mass/1e6, color='gray', linestyle='--', linewidth=2,
                label=f'Target Demand ({total_mass/1e6:.0f} Mt)')
    
    # Mark minimum feasible times
    ax1.axvline(x=data['Y_min_ideal'], color=COLORS['ideal'], linestyle=':', alpha=0.7)
    ax1.axvline(x=data['Y_min_real'], color=COLORS['real'], linestyle=':', alpha=0.7)
    
    ax1.annotate(f"Ideal min:\n{data['Y_min_ideal']:.1f} yr",
                xy=(data['Y_min_ideal'], total_mass/1e6),
                xytext=(-40, 20), textcoords='offset points',
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color=COLORS['ideal']))
    
    ax1.annotate(f"Real min:\n{data['Y_min_real']:.1f} yr",
                xy=(data['Y_min_real'], total_mass/1e6),
                xytext=(40, 20), textcoords='offset points',
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color=COLORS['real']))
    
    ax1.set_xlabel('Project Duration (Years)', fontsize=12)
    ax1.set_ylabel('Cumulative Capacity (Million tons)', fontsize=12)
    ax1.set_title('Total Transport Capacity Over Time\n', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(0, 50)
    ax1.set_ylim(0, max(ideal_total) * 1.1)
    
    # Right plot: Component breakdown
    ax2.stackplot(Y_range, 
                  [ideal_elev_cap, ideal_rock_cap],
                  labels=['Ideal Elevator', 'Ideal Rocket'],
                  colors=[COLORS['elevator'], COLORS['rocket']],
                  alpha=0.5)
    
    ax2.plot(Y_range, np.array(real_elev_cap), '--', color=COLORS['elevator'], 
             linewidth=2, label='Real Elevator')
    ax2.plot(Y_range, real_total, '--', color='darkred', 
             linewidth=2, label='Real Total')
    
    ax2.axhline(y=total_mass/1e6, color='gray', linestyle='--', linewidth=2)
    
    ax2.set_xlabel('Project Duration (Years)', fontsize=12)
    ax2.set_ylabel('Cumulative Capacity (Million tons)', fontsize=12)
    ax2.set_title('Capacity Breakdown: Elevator vs Rocket\n', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim(0, 50)
    ax2.set_ylim(0, max(ideal_total) * 1.1)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def plot_demand_amplification(data: dict, save_path: str):
    """
    Plot diagram explaining demand amplification effect.
    """
    ideal = data['ideal_result']
    real = data['real_result']
    rel = data['real_model'].p.reliability
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Target demand box
    target = data['total_mass'] / 1e6
    ax.add_patch(FancyBboxPatch((0.5, 7), 3, 1.5, boxstyle="round,pad=0.1",
                                facecolor=COLORS['ideal'], alpha=0.85, edgecolor='black'))
    ax.text(2, 7.75, f'Target Demand\n{target:.0f} Mt', ha='center', va='center',
           fontsize=12, fontweight='bold', color='white')
    
    # Arrow down
    ax.annotate('', xy=(2, 5.5), xytext=(2, 6.9),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # Failure factors
    ax.add_patch(FancyBboxPatch((0.5, 4), 3.5, 1.3, boxstyle="round,pad=0.1",
                                facecolor='lightyellow', alpha=0.9, edgecolor='orange'))
    ax.text(2.25, 5, 'Failure Amplification Factors', ha='center', va='top',
           fontsize=11, fontweight='bold')
    ax.text(2.25, 4.4, f'• Elevator failures: λ = {rel.lambda_E:.2f}/yr\n'
                       f'• Rocket failures: P_f = {rel.P_f_R:.2f}',
           ha='center', va='center', fontsize=10)
    
    # Arrow down
    ax.annotate('', xy=(2, 2.5), xytext=(2, 3.9),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # Effective demand box
    effective = (real.x_opt + real.mR_opt) / 1e6
    amp_pct = (effective - target) / target * 100
    ax.add_patch(FancyBboxPatch((0.5, 1), 3, 1.5, boxstyle="round,pad=0.1",
                                facecolor=COLORS['real'], alpha=0.85, edgecolor='black'))
    ax.text(2, 1.75, f'Effective Demand\n{effective:.1f} Mt (+{amp_pct:.1f}%)', 
           ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    
    # Right side: breakdown
    ax.add_patch(FancyBboxPatch((5, 3), 4.5, 5.5, boxstyle="round,pad=0.1",
                                facecolor='white', alpha=0.9, edgecolor='gray'))
    ax.text(7.25, 8, 'Demand Amplification Formula', ha='center', va='center',
           fontsize=12, fontweight='bold')
    
    ax.text(7.25, 7, r'$M_{eff} = \frac{M_{target}}{1 - P_f^{combined}}$', 
           ha='center', va='center', fontsize=14, fontfamily='serif')
    
    P_combined = 0.3 * rel.lambda_E + 0.7 * rel.P_f_R
    ax.text(7.25, 5.8, f'Combined failure probability:\n'
                       f'$P_f^{{combined}} = 0.3 × {rel.lambda_E:.2f} + 0.7 × {rel.P_f_R:.2f}$\n'
                       f'$= {P_combined:.4f}$',
           ha='center', va='center', fontsize=11)
    
    ax.text(7.25, 4.2, f'Amplification factor:\n'
                       f'$\\frac{{1}}{{1 - {P_combined:.4f}}} = {1/(1-P_combined):.4f}$',
           ha='center', va='center', fontsize=11)
    
    ax.text(7.25, 3.3, f'Extra mass needed:\n{effective - target:.2f} Mt',
           ha='center', va='center', fontsize=11, fontweight='bold',
           color=COLORS['highlight'])
    
    plt.title('Demand Amplification Effect Explained\n', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def plot_comprehensive_dashboard(data: dict, save_path: str):
    """
    Create a comprehensive 3x3 dashboard with all key visualizations.
    """
    ideal = data['ideal_result']
    real = data['real_result']
    
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # ========== Row 1 ==========
    
    # 1. Key Metrics Cards
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis('off')
    
    metrics = [
        ('Min Time (Ideal)', f"{data['Y_min_ideal']:.1f} yr", COLORS['ideal']),
        ('Min Time (Real)', f"{data['Y_min_real']:.1f} yr", COLORS['real']),
        ('Time Increase', f"+{(data['Y_min_real']-data['Y_min_ideal'])/data['Y_min_ideal']*100:.1f}%", COLORS['highlight']),
        ('Cost (Ideal)', f"${ideal.cost_total/1e12:.2f}T", COLORS['ideal']),
        ('Cost (Real)', f"${real.cost_total/1e12:.2f}T", COLORS['real']),
        ('Cost Increase', f"+{(real.cost_total-ideal.cost_total)/ideal.cost_total*100:.1f}%", COLORS['highlight']),
    ]
    
    for i, (label, value, color) in enumerate(metrics):
        row = i // 2
        col = i % 2
        ax1.add_patch(FancyBboxPatch((col*0.5 + 0.02, 0.65 - row*0.35), 0.46, 0.28,
                                     boxstyle="round,pad=0.02", facecolor=color, alpha=0.15,
                                     edgecolor=color, linewidth=2))
        ax1.text(col*0.5 + 0.25, 0.85 - row*0.35, value, ha='center', va='center',
                fontsize=14, fontweight='bold', color=color)
        ax1.text(col*0.5 + 0.25, 0.72 - row*0.35, label, ha='center', va='center',
                fontsize=10, color='gray')
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_title('Key Performance Metrics', fontsize=12, fontweight='bold', pad=10)
    
    # 2. Cost Breakdown Bar
    ax2 = fig.add_subplot(gs[0, 1])
    
    categories = ['Elev\nCAPEX', 'Elev\nOPEX', 'Rock\nCAPEX', 'Rock\nOPEX', 'Carbon']
    ideal_vals = [ideal.costs.elevator_capex/1e9, ideal.costs.elevator_opex/1e9,
                  ideal.costs.rocket_capex/1e9, ideal.costs.rocket_opex/1e9,
                  ideal.costs.total_carbon_cost/1e9]
    real_vals = [real.costs.elevator_capex/1e9, real.costs.elevator_opex/1e9,
                 real.costs.rocket_capex/1e9, real.costs.rocket_opex/1e9,
                 real.costs.total_carbon_cost/1e9]
    
    x = np.arange(len(categories))
    width = 0.35
    ax2.bar(x - width/2, ideal_vals, width, label='Ideal', color=COLORS['ideal'], alpha=0.85)
    ax2.bar(x + width/2, real_vals, width, label='Real', color=COLORS['real'], alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, fontsize=9)
    ax2.set_ylabel('Cost ($B)', fontsize=10)
    ax2.set_title('Cost Components Comparison', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Carbon Emissions Pie
    ax3 = fig.add_subplot(gs[0, 2])
    
    emissions_data = [
        real.emissions.elevator_construction / 1e6,
        real.emissions.rocket_construction / 1e6,
        real.emissions.rocket_operational / 1e6
    ]
    emission_labels = ['Elevator\nConstruction', 'Rocket\nConstruction', 'Rocket\nOperational']
    emission_colors = [COLORS['elevator'], COLORS['construction'], COLORS['operational']]
    
    wedges, texts, autotexts = ax3.pie(emissions_data, labels=emission_labels, 
                                        colors=emission_colors, autopct='%1.1f%%',
                                        startangle=90, explode=(0.02, 0.02, 0.02))
    ax3.set_title(f'Carbon Emissions Breakdown\n(Total: {real.emissions_total/1e6:.1f} Mt CO₂)', 
                  fontsize=12, fontweight='bold')
    
    # ========== Row 2 ==========
    
    # 4. Mass Allocation Comparison
    ax4 = fig.add_subplot(gs[1, 0])
    
    categories = ['Ideal', 'Real']
    elev_mass = [ideal.x_opt/1e6, real.x_opt/1e6]
    rock_mass = [ideal.mR_opt/1e6, real.mR_opt/1e6]
    
    ax4.bar(categories, elev_mass, label='Elevator', color=COLORS['elevator'], alpha=0.85)
    ax4.bar(categories, rock_mass, bottom=elev_mass, label='Rocket', color=COLORS['rocket'], alpha=0.85)
    
    for i, (e, r) in enumerate(zip(elev_mass, rock_mass)):
        ax4.text(i, e/2, f'{e:.1f} Mt', ha='center', va='center', fontsize=10, 
                fontweight='bold', color='white')
        ax4.text(i, e + r/2, f'{r:.1f} Mt', ha='center', va='center', fontsize=10,
                fontweight='bold', color='white')
    
    ax4.set_ylabel('Mass (Million tons)', fontsize=10)
    ax4.set_title('Mass Allocation', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. Availability Factors
    ax5 = fig.add_subplot(gs[1, 1])
    
    avail_labels = ['Elevator', 'Rocket']
    ideal_avail = [1.0, 1.0]
    real_avail = [data['real_model'].calculate_elevator_availability(),
                  data['real_model'].calculate_rocket_availability()]
    
    x = np.arange(len(avail_labels))
    ax5.bar(x - width/2, ideal_avail, width, label='Ideal (100%)', color=COLORS['ideal'], alpha=0.85)
    ax5.bar(x + width/2, real_avail, width, label='Real', color=COLORS['real'], alpha=0.85)
    
    for i, (iv, rv) in enumerate(zip(ideal_avail, real_avail)):
        ax5.text(i + width/2, rv + 0.02, f'{rv*100:.1f}%', ha='center', fontsize=10, fontweight='bold')
    
    ax5.set_xticks(x)
    ax5.set_xticklabels(avail_labels, fontsize=10)
    ax5.set_ylabel('Availability Factor', fontsize=10)
    ax5.set_title('System Availability', fontsize=12, fontweight='bold')
    ax5.set_ylim(0, 1.15)
    ax5.legend(fontsize=9)
    ax5.grid(axis='y', alpha=0.3)
    
    # 6. Carbon Intensity
    ax6 = fig.add_subplot(gs[1, 2])
    
    intensities = [ideal.carbon_intensity, real.carbon_intensity]
    bars = ax6.barh(['Ideal', 'Real'], intensities, 
                    color=[COLORS['ideal'], COLORS['real']], alpha=0.85, height=0.5)
    
    for bar, val in zip(bars, intensities):
        ax6.text(val + 0.05, bar.get_y() + bar.get_height()/2, f'{val:.3f} tCO₂/t',
                va='center', fontsize=11, fontweight='bold')
    
    change = (intensities[1] - intensities[0]) / intensities[0] * 100
    ax6.set_title(f'Carbon Intensity\n(+{change:.1f}% increase)', fontsize=12, fontweight='bold')
    ax6.set_xlabel('tCO₂ per ton payload', fontsize=10)
    ax6.set_xlim(0, max(intensities) * 1.4)
    ax6.grid(axis='x', alpha=0.3)
    
    # ========== Row 3 ==========
    
    # 7. Rocket Launches
    ax7 = fig.add_subplot(gs[2, 0])
    
    launches = [data['ideal_launches']/1e6, data['real_launches']/1e6]
    bars = ax7.bar(['Ideal', 'Real'], launches, color=[COLORS['ideal'], COLORS['real']], alpha=0.85)
    
    for bar, val in zip(bars, launches):
        ax7.text(bar.get_x() + bar.get_width()/2, val + 0.5, f'{val:.1f}M',
                ha='center', fontsize=11, fontweight='bold')
    
    ax7.set_ylabel('Launches (Millions)', fontsize=10)
    ax7.set_title('Total Rocket Launches Required', fontsize=12, fontweight='bold')
    ax7.grid(axis='y', alpha=0.3)
    
    # 8. Cost Structure Donut
    ax8 = fig.add_subplot(gs[2, 1])
    
    # Real condition cost breakdown
    cost_data = [
        real.costs.elevator_capex/1e9,
        real.costs.elevator_opex/1e9,
        real.costs.rocket_capex/1e9,
        real.costs.rocket_opex/1e9,
        real.costs.total_carbon_cost/1e9
    ]
    cost_labels = ['Elev CAPEX', 'Elev OPEX', 'Rock CAPEX', 'Rock OPEX', 'Carbon']
    cost_colors = [COLORS['elevator'], '#3AAFA9', COLORS['rocket'], '#FF8585', COLORS['carbon']]
    
    wedges, texts = ax8.pie(cost_data, colors=cost_colors, startangle=90,
                            wedgeprops=dict(width=0.5, edgecolor='white'))
    ax8.legend(wedges, [f'{l}: ${v:.0f}B' for l, v in zip(cost_labels, cost_data)],
              loc='center', fontsize=9, frameon=False)
    ax8.set_title(f'Cost Structure (Real)\nTotal: ${real.cost_total/1e9:.0f}B', 
                  fontsize=12, fontweight='bold')
    
    # 9. Summary Statistics
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    summary_text = f"""
    ANALYSIS SUMMARY
    ─────────────────────────────────
    Mission: {data['total_mass']/1e6:.0f} Mt at Y = {data['Y']:.0f} years
    
    TIME IMPACT:
    • Min feasible time: +{(data['Y_min_real']-data['Y_min_ideal'])/data['Y_min_ideal']*100:.1f}%
    
    COST IMPACT:
    • Total cost increase: +{(real.cost_total-ideal.cost_total)/ideal.cost_total*100:.1f}%
    • CAPEX increase: +{(real.cost_capex-ideal.cost_capex)/ideal.cost_capex*100:.1f}%
    • OPEX increase: +{(real.cost_opex-ideal.cost_opex)/ideal.cost_opex*100:.1f}%
    
    ENVIRONMENTAL IMPACT:
    • CO₂ emissions increase: +{(real.emissions_total-ideal.emissions_total)/ideal.emissions_total*100:.1f}%
    • Carbon intensity: {real.carbon_intensity:.3f} tCO₂/t
    
    CAPACITY IMPACT:
    • Elevator capacity: -{(1-data['real_model'].calculate_elevator_availability())*100:.1f}%
    • Rocket capacity: -{(1-data['real_model'].calculate_rocket_availability())*100:.1f}%
    """
    
    ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle('Transport System Analysis: Comprehensive Dashboard\n'
                 'Ideal vs Real Conditions Comparison',
                 fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


def plot_scenario_comparison(save_path: str):
    """
    Plot comparison across multiple mission scenarios.
    """
    import io
    from contextlib import redirect_stdout
    
    scenarios = [
        {'name': '1 Mt', 'mass': 1e6},
        {'name': '10 Mt', 'mass': 10e6},
        {'name': '50 Mt', 'mass': 50e6},
        {'name': '100 Mt', 'mass': 100e6},
    ]
    
    results = []
    
    for scenario in scenarios:
        ideal_params = ModelParams(use_ideal=True, M_tot=scenario['mass'])
        real_params = ModelParams(use_ideal=False, M_tot=scenario['mass'])
        
        ideal_model = TransportOptimizationModel(ideal_params)
        real_model = TransportOptimizationModel(real_params)
        
        with redirect_stdout(io.StringIO()):
            Y_min_ideal = ideal_model.find_minimum_feasible_Y(Y_min=0.001, Y_max=100)
            Y_min_real = real_model.find_minimum_feasible_Y(Y_min=0.001, Y_max=100)
            
            # Solve at minimum real time
            if Y_min_real < float('inf'):
                Y_solve = Y_min_real * 1.2  # 20% buffer
                ideal_result = ideal_model.solve(Y_solve)
                real_result = real_model.solve(Y_solve)
                
                results.append({
                    'name': scenario['name'],
                    'mass': scenario['mass'],
                    'Y_min_ideal': Y_min_ideal,
                    'Y_min_real': Y_min_real,
                    'ideal_cost': ideal_result.cost_total / 1e9,
                    'real_cost': real_result.cost_total / 1e9,
                    'ideal_emissions': ideal_result.emissions_total / 1e6,
                    'real_emissions': real_result.emissions_total / 1e6,
                })
    
    if not results:
        print("No feasible scenarios found")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    names = [r['name'] for r in results]
    x = np.arange(len(names))
    width = 0.35
    
    # 1. Minimum Time
    ax1 = axes[0, 0]
    ideal_times = [r['Y_min_ideal'] for r in results]
    real_times = [r['Y_min_real'] for r in results]
    ax1.bar(x - width/2, ideal_times, width, label='Ideal', color=COLORS['ideal'], alpha=0.85)
    ax1.bar(x + width/2, real_times, width, label='Real', color=COLORS['real'], alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    ax1.set_ylabel('Minimum Feasible Time (Years)')
    ax1.set_title('Minimum Completion Time by Mission Size')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Total Cost
    ax2 = axes[0, 1]
    ideal_costs = [r['ideal_cost'] for r in results]
    real_costs = [r['real_cost'] for r in results]
    ax2.bar(x - width/2, ideal_costs, width, label='Ideal', color=COLORS['ideal'], alpha=0.85)
    ax2.bar(x + width/2, real_costs, width, label='Real', color=COLORS['real'], alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names)
    ax2.set_ylabel('Total Cost (Billion USD)')
    ax2.set_title('Total Cost by Mission Size')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Carbon Emissions
    ax3 = axes[1, 0]
    ideal_emissions = [r['ideal_emissions'] for r in results]
    real_emissions = [r['real_emissions'] for r in results]
    ax3.bar(x - width/2, ideal_emissions, width, label='Ideal', color=COLORS['ideal'], alpha=0.85)
    ax3.bar(x + width/2, real_emissions, width, label='Real', color=COLORS['real'], alpha=0.85)
    ax3.set_xticks(x)
    ax3.set_xticklabels(names)
    ax3.set_ylabel('Carbon Emissions (Mt CO₂)')
    ax3.set_title('Carbon Emissions by Mission Size')
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Cost Increase Percentage
    ax4 = axes[1, 1]
    cost_increase = [(r['real_cost'] - r['ideal_cost']) / r['ideal_cost'] * 100 for r in results]
    bars = ax4.bar(names, cost_increase, color=COLORS['highlight'], alpha=0.85)
    
    for bar, val in zip(bars, cost_increase):
        ax4.text(bar.get_x() + bar.get_width()/2, val + 0.5, f'{val:.1f}%',
                ha='center', fontsize=10, fontweight='bold')
    
    ax4.set_ylabel('Cost Increase (%)')
    ax4.set_title('Real vs Ideal Cost Increase')
    ax4.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Multi-Scenario Analysis: Mission Size Impact', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================================
# Main Execution
# ============================================================================

def main():
    """Generate all visualizations."""
    
    print("=" * 70)
    print("Q2 VISUALIZATION MODULE")
    print("Generating comprehensive visualization outputs...")
    print("=" * 70)
    
    # Setup
    ensure_output_dir()
    
    # Run analysis with default parameters
    print("\n[1/10] Running analysis (100 Mt, 24 years)...")
    data = run_analysis(total_mass=1.0e8, Y=24.0)
    
    if not data['ideal_result'].feasible or not data['real_result'].feasible:
        print("Warning: Results not feasible at Y=24. Trying with larger Y...")
        data = run_analysis(total_mass=1.0e8, Y=30.0)
    
    # Generate all visualizations
    print("\n[2/10] Creating radar comparison chart...")
    plot_radar_comparison(data, os.path.join(OUTPUT_DIR, "01_radar_comparison.png"))
    
    print("\n[3/10] Creating cost breakdown chart...")
    plot_cost_breakdown_grouped_bar(data, os.path.join(OUTPUT_DIR, "02_cost_breakdown.png"))
    
    print("\n[4/10] Creating carbon emissions chart...")
    plot_carbon_emissions_stacked(data, os.path.join(OUTPUT_DIR, "03_carbon_emissions.png"))
    
    print("\n[5/10] Creating reliability breakdown chart...")
    plot_reliability_breakdown(data, os.path.join(OUTPUT_DIR, "04_reliability_breakdown.png"))
    
    print("\n[6/10] Creating cost waterfall chart...")
    plot_cost_waterfall(data, os.path.join(OUTPUT_DIR, "05_cost_waterfall.png"))
    
    print("\n[7/10] Creating mass allocation chart...")
    plot_mass_allocation_pie(data, os.path.join(OUTPUT_DIR, "06_mass_allocation.png"))
    
    print("\n[8/10] Creating capacity curves chart...")
    plot_capacity_curves(data, os.path.join(OUTPUT_DIR, "07_capacity_curves.png"))
    
    print("\n[9/10] Creating demand amplification diagram...")
    plot_demand_amplification(data, os.path.join(OUTPUT_DIR, "08_demand_amplification.png"))
    
    print("\n[10/10] Creating comprehensive dashboard...")
    plot_comprehensive_dashboard(data, os.path.join(OUTPUT_DIR, "09_comprehensive_dashboard.png"))
    
    print("\n[BONUS] Creating multi-scenario comparison...")
    plot_scenario_comparison(os.path.join(OUTPUT_DIR, "10_scenario_comparison.png"))
    
    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE!")
    print(f"All figures saved to: {OUTPUT_DIR}")
    print("=" * 70)
    
    # List generated files
    print("\nGenerated files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith('.png'):
            print(f"  - {f}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
