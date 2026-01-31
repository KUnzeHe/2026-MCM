"""
Transport System Analysis: Separate Component Breakdown
========================================================
Shows detailed breakdown for:
1. Elevator system costs and mass
2. Rocket system costs and mass  
3. Combined totals and changes
4. Carbon emissions analysis
"""

from __future__ import annotations

import os
import numpy as np
from dataclasses import dataclass, field
from scipy.integrate import quad
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Dict, Any
import warnings

# Matplotlib configuration
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120

# ============================================================================
# Data Classes
# ============================================================================

@dataclass(frozen=True)
class ReliabilityParams:
    """Parameters for non-ideal working conditions.
    
    References:
    - Elevator: Based on large infrastructure availability (power grids, pipelines)
    - Rocket: Based on SpaceX Falcon 9/Starship reliability data
    """
    # Elevator reliability
    # Reference: Large infrastructure typically achieves 95-99% availability
    # Source: Power grid availability ~99.9%, oil pipelines ~98%
    beta_E: float = 0.92          # Availability factor (improved from 0.85)
    lambda_E: float = 0.015       # Annual failure rate (1.5%, similar to offshore platforms)
    t_repair_E: float = 14.0      # Days to repair (reasonable for major systems)
    C_E_main: float = 500e6       # Annual maintenance cost ($) ~0.5% of CAPEX
    C_E_fix: float = 50e6         # Cost to fix a failure ($)
    P_cat_E: float = 0.0005       # Catastrophic failure probability (0.05%, reduced)
    eta_energy: float = 0.92      # Energy efficiency factor (grid-scale efficiency)
    
    # Rocket reliability
    # Reference: SpaceX Falcon 9 success rate ~98%, Starship target ~99%
    # Source: spacexstats.com, NASA safety requirements
    beta_R: float = 0.92          # Availability factor (improved with maturity)
    P_f_R: float = 0.02           # Launch failure probability (2%, conservative for new system)
    C_rocket_loss: float = 100e6  # Cost of lost rocket ($) - Starship target ~$10M, use 100M conservative
    C_cargo_loss: float = 50e6    # Cost of lost cargo ($)
    T_down_R: float = 7.0         # Days downtime after failure (rapid turnaround goal)
    C_R_maint: float = 50e6       # Annual maintenance per site ($)
    delta_window: float = 0.03    # Launch window limitation (3%, improved with technology)
    delta_maint: float = 0.03     # Maintenance downtime (3%)
    
    use_demand_amplification: bool = True


@dataclass(frozen=True)
class GrowthParams:
    """Parameters for Logistic growth of ground launch infrastructure."""
    K: float = 80.0       # Carrying capacity
    N0: float = 10.0      # Initial number of sites
    r: float = 0.15       # Growth rate


@dataclass(frozen=True)
class AnchorParams:
    """Parameters for anchor transfer rockets."""
    N_anchor: int = 6           # Number of anchor launch platforms
    L_anchor: float = 700.0     # Launches per anchor per year
    p_A: float = 150.0          # Payload per anchor launch (tons)
    
    @property
    def annual_capacity(self) -> float:
        return self.N_anchor * self.L_anchor * self.p_A


@dataclass
class CarbonParams:
    """Parameters for carbon emissions calculation.
    
    References:
    - Rocket emissions: Based on propellant combustion calculations
    - Carbon price: EU ETS and projected 2040-2050 prices
    - Construction: Life cycle assessment studies
    """
    # Carbon emissions per rocket launch (tons CO2)
    # Reference: Falcon 9 ~425 tCO2, Starship (CH4/LOX) ~2000-3000 tCO2
    # Source: Everyday Astronaut, Dallas et al. (2020) "Environmental Impact of Rockets"
    CO2_per_launch: float = 2500.0  # Starship-class vehicle
    
    # Carbon price ($ per ton CO2)
    # Reference: EU ETS ~€80/t (2024), projected $150-250/t by 2050
    # Source: World Bank Carbon Pricing Dashboard
    carbon_price: float = 150.0
    
    # Elevator carbon emissions (tons CO2 per ton payload)
    # Even with renewables: maintenance, backup systems, embodied energy
    # Reference: Rail freight ~0.03 kg CO2/ton-km, scaled for space elevator
    CO2_elevator_per_ton: float = 0.1  # Small but non-zero
    
    # Construction carbon emissions
    # Reference: Large infrastructure ~500-1000 tCO2 per $1M construction
    # Source: ICE Database, Circular Ecology
    CO2_elevator_construction: float = 5e6   # tons CO2 for elevator construction (5 Mt)
    CO2_launch_site_construction: float = 100000  # tons CO2 per launch site (100k)


@dataclass
class ModelParams:
    """Global Model Parameters."""
    # Demand - NOW ADJUSTABLE
    M_tot: float = 1.0e8  # 100 million metric tons - DEFAULT VALUE
    
    # Financial
    discount_rate: float = 0.03  # 3% discount rate
    
    # =========== Elevator System ===========
    T_E: float = 5.37e5           # Annual elevator throughput
    F_E: float = 100e9            # Fixed CAPEX
    c_E: float = 2.7e3            # OPEX per ton
    
    # Anchor transfer
    anchor: AnchorParams = field(default_factory=AnchorParams)
    
    # =========== Direct Rocket System ===========
    # Reference: SpaceX Starship target $10-50/kg to orbit, currently ~$2000/kg
    # For Earth-Moon transport, multiply by ~3-5x
    # Source: SpaceX presentations, NASA cost estimates
    c_R: float = 3.0e5            # OPEX per ton ($300k/t, reduced from $720k)
    C_site: float = 3.0e9         # CAPEX per new launch site (Starbase-class facility)
    
    # Rocket Performance
    L_site_annual: float = 2000.0  # Launches per site per year
    p_B: float = 150.0            # Payload per launch
    
    # Infrastructure Growth Model
    growth: GrowthParams = field(default_factory=GrowthParams)
    
    # Reliability Parameters
    reliability: ReliabilityParams = field(default_factory=ReliabilityParams)
    
    # Carbon Emissions Parameters
    carbon: CarbonParams = field(default_factory=CarbonParams)
    
    # Control flags
    use_ideal: bool = False


@dataclass
class ComponentCosts:
    """Detailed cost breakdown for each component."""
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
    """Detailed carbon emissions breakdown."""
    # Direct emissions from operations (tons CO2)
    elevator_operational: float = 0.0
    rocket_operational: float = 0.0
    
    # Construction emissions (tons CO2)
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
    """Result container with component breakdown."""
    Y: float                  # Completion time
    x_opt: float              # Mass via elevator (tons)
    mR_opt: float             # Mass via rockets (tons)
    costs: ComponentCosts     # Detailed cost breakdown
    emissions: CarbonEmissions  # Carbon emissions breakdown
    feasible: bool
    N_required: float         # Launch sites needed
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
    def cost_carbon(self) -> float:
        return self.costs.total_carbon_cost
    
    @property
    def emissions_total(self) -> float:
        return self.emissions.total_emissions
    
    @property
    def emissions_operational(self) -> float:
        return self.emissions.total_operational
    
    @property
    def emissions_construction(self) -> float:
        return self.emissions.total_construction
    
    @property
    def carbon_intensity(self) -> float:
        """Carbon intensity in tons CO2 per ton payload."""
        total_mass = self.x_opt + self.mR_opt
        if total_mass > 0:
            return self.emissions_total / total_mass
        return 0.0


# ============================================================================
# Transport Optimization Model
# ============================================================================

class TransportOptimizationModel:
    """Implements the mixed transport optimization model."""
    
    def __init__(self, params: ModelParams):
        self.p = params
    
    # ---------- Reliability Calculations ----------
    
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
    
    # ---------- Infrastructure Growth ----------
    
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
    
    # ---------- Cumulative Capacity ----------
    
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
    
    # ---------- Optimization ----------
    
    def find_minimum_feasible_Y(self, Y_min: float = 0.01, Y_max: float = 100) -> float:
        """Find minimum feasible completion time."""
        
        def feasibility_gap(Y):
            cap_E = self.cumulative_elevator_capacity(Y, adjusted=not self.p.use_ideal)
            cap_R = self.cumulative_rocket_capacity(Y, adjusted=not self.p.use_ideal)
            M_eff = self.get_effective_demand(Y)
            gap = cap_E + cap_R - M_eff
            return gap
        
        # Add debug information
        print(f"DEBUG: Searching minimum feasible time for M={self.p.M_tot/1e6:.2f}Mt")
        print(f"DEBUG: Search range: {Y_min:.3f}-{Y_max} years")
        
        # Test a few points for debugging
        test_points = [0.001, 0.01, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20, Y_max]
        for test_Y in test_points:
            if test_Y > Y_max:
                continue
            gap = feasibility_gap(test_Y)
            print(f"DEBUG: Y={test_Y:.3f}y, capacity gap={gap/1e6:.2f}Mt ({'FEASIBLE' if gap>=0 else 'INFEASIBLE'})")
        
        # Check if even the smallest time is feasible
        if feasibility_gap(Y_min) >= 0:
            print(f"DEBUG: {Y_min:.3f} year is already feasible")
            # Try to find even smaller if possible
            try:
                # Search from a very small value to Y_min
                very_small = 0.001  # ~0.365 days
                if feasibility_gap(very_small) >= 0:
                    # If even 0.001 year is feasible, find exact minimum
                    return brentq(feasibility_gap, very_small, Y_min)
                else:
                    # Binary search between very_small and Y_min
                    return brentq(feasibility_gap, very_small, Y_min)
            except ValueError:
                # If brentq fails, try linear search
                for Y_test in np.linspace(0.001, Y_min, 100):
                    if feasibility_gap(Y_test) >= 0:
                        print(f"DEBUG: Linear search found minimum: {Y_test:.4f} years")
                        return Y_test
                return Y_min
            except Exception as e:
                print(f"DEBUG: Error in finding smaller time: {e}")
                return Y_min
        
        # Check if even Y_max is infeasible
        if feasibility_gap(Y_max) < 0:
            print(f"DEBUG: Even {Y_max} years is infeasible")
            return float('inf')
        
        # Binary search for minimum feasible time
        try:
            result = brentq(feasibility_gap, Y_min, Y_max)
            print(f"DEBUG: Found minimum feasible time: {result:.4f} years")
            return result
        except ValueError as e:
            print(f"DEBUG: Brentq failed: {e}")
            # Fallback: linear search with finer resolution
            for Y_test in np.linspace(Y_min, Y_max, 200):
                if feasibility_gap(Y_test) >= 0:
                    print(f"DEBUG: Linear search found: {Y_test:.4f} years")
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
    
    # ---------- Carbon Emissions Calculations ----------
    
    def calculate_rocket_launches(self, m_R: float, Y: float) -> float:
        """Calculate total number of rocket launches needed."""
        if m_R <= 0 or Y <= 0:
            return 0.0
        
        # Annual rocket mass requirement
        annual_rocket_mass = m_R / Y
        
        # Calculate required launches per year
        if not self.p.use_ideal:
            beta_R = self.calculate_rocket_availability()
            effective_launch_rate = self.p.L_site_annual * beta_R
        else:
            effective_launch_rate = self.p.L_site_annual
        
        # Sites needed (this is already calculated elsewhere, but recalc for clarity)
        sites_needed = self.calculate_required_sites(m_R, Y)
        
        # Total launches = sites * launches/site/year * years
        total_launches = sites_needed * effective_launch_rate * Y
        
        return total_launches
    
    def calculate_carbon_emissions(self, x: float, m_R: float, Y: float, 
                                  N_required: float) -> CarbonEmissions:
        """Calculate carbon emissions for both elevator and rocket transport."""
        emissions = CarbonEmissions()
        
        # Elevator emissions (operational - assumed to be zero with renewable energy)
        emissions.elevator_operational = x * self.p.carbon.CO2_elevator_per_ton
        
        # Elevator construction emissions (if elevator is used)
        if x > 0:
            emissions.elevator_construction = self.p.carbon.CO2_elevator_construction
        
        # Rocket operational emissions
        if m_R > 0:
            # Calculate total launches needed
            total_launches = self.calculate_rocket_launches(m_R, Y)
            emissions.rocket_operational = total_launches * self.p.carbon.CO2_per_launch
        
        # Rocket construction emissions (for new launch sites)
        N_new = max(0, N_required - self.p.growth.N0)
        emissions.rocket_construction = N_new * self.p.carbon.CO2_launch_site_construction
        
        return emissions
    
    def calculate_carbon_costs(self, emissions: CarbonEmissions) -> Tuple[float, float]:
        """Calculate carbon costs for elevator and rocket systems."""
        elevator_carbon_cost = (emissions.elevator_operational + emissions.elevator_construction) * self.p.carbon.carbon_price
        rocket_carbon_cost = (emissions.rocket_operational + emissions.rocket_construction) * self.p.carbon.carbon_price
        
        return elevator_carbon_cost, rocket_carbon_cost
    
    def solve(self, Y: float, p_B: Optional[float] = None) -> OptimizationResult:
        payload = p_B if p_B is not None else self.p.p_B
        
        # Check for zero or negative time
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
        
        # Effective demand
        M_eff = self.get_effective_demand(Y)
        
        # Capacities
        cap_E = self.cumulative_elevator_capacity(Y, adjusted=not self.p.use_ideal)
        cap_R = self.cumulative_rocket_capacity(Y, payload, adjusted=not self.p.use_ideal)
        total_cap = cap_E + cap_R
        
        # Feasibility check with debugging
        print(f"DEBUG solve(Y={Y:.3f}): M_eff={M_eff/1e6:.2f}Mt, cap_E={cap_E/1e6:.2f}Mt, cap_R={cap_R/1e6:.2f}Mt, total={total_cap/1e6:.2f}Mt")
        
        if total_cap < M_eff:
            return OptimizationResult(
                Y=Y, x_opt=0, mR_opt=0, 
                costs=ComponentCosts(),
                emissions=CarbonEmissions(),
                feasible=False,
                N_required=self.p.growth.K, cap_E=cap_E, cap_R=cap_R,
                message=f"Insufficient capacity (need {M_eff/1e6:.3f}Mt, have {total_cap/1e6:.3f}Mt)", 
                is_ideal=self.p.use_ideal
            )
        
        # Allocation
        x = min(M_eff, cap_E)
        m_R = M_eff - x
        
        # Calculate required sites
        N_required = self.calculate_required_sites(m_R, Y)
        
        # Calculate component costs
        costs = ComponentCosts()
        
        # Elevator CAPEX
        costs.elevator_capex = self.p.F_E if x > 0 else 0.0
        
        # Rocket CAPEX
        N_new = max(0, N_required - self.p.growth.N0)
        costs.rocket_capex = N_new * self.p.C_site
        
        if not self.p.use_ideal:
            costs.rocket_capex *= 1.10  # 10% redundancy
        
        # OPEX
        costs.elevator_opex = self.calculate_elevator_opex_npv(x, Y)
        costs.rocket_opex = self.calculate_rocket_opex_npv(m_R, Y, payload)
        
        # Carbon emissions
        emissions = self.calculate_carbon_emissions(x, m_R, Y, N_required)
        
        # Carbon costs
        costs.elevator_carbon_cost, costs.rocket_carbon_cost = self.calculate_carbon_costs(emissions)
        
        return OptimizationResult(
            Y=Y, x_opt=x, mR_opt=m_R,
            costs=costs,
            emissions=emissions,
            feasible=True, N_required=N_required,
            cap_E=cap_E, cap_R=cap_R, is_ideal=self.p.use_ideal
        )
    
    def pareto_sweep(self, Y_range: np.ndarray) -> List[OptimizationResult]:
        return [self.solve(Y) for Y in Y_range]


# ============================================================================
# Comparison Analysis Functions with Component Breakdown
# ============================================================================

def compare_component_breakdown(total_mass: float = 1.0e8, 
                               time_limit: Optional[float] = None,
                               min_years: float = 0.01,
                               max_years: float = 100) -> Dict[str, Any]:
    """Comprehensive comparison with component breakdown."""
    
    print("=" * 80)
    print(f"COMPONENT BREAKDOWN: IDEAL VS REAL CONDITIONS")
    print(f"Total Mass to Transfer: {total_mass/1e6:.2f} million tons")
    print(f"Total Mass in tons: {total_mass:.0f} tons")
    if time_limit:
        print(f"Time Limit: {time_limit:.3f} years")
    print("=" * 80)
    
    # Create models with specified total mass
    ideal_params = ModelParams(use_ideal=True, M_tot=total_mass)
    real_params = ModelParams(use_ideal=False, M_tot=total_mass)
    
    ideal_model = TransportOptimizationModel(ideal_params)
    real_model = TransportOptimizationModel(real_params)
    
    # Find minimum feasible times with explicit parameters
    print(f"\nFinding minimum feasible times (searching {min_years:.3f}-{max_years} years)...")
    Y_min_ideal = ideal_model.find_minimum_feasible_Y(Y_min=min_years, Y_max=max_years)
    Y_min_real = real_model.find_minimum_feasible_Y(Y_min=min_years, Y_max=max_years)
    
    print(f"\n1. TIME REQUIREMENTS:")
    if Y_min_ideal == float('inf'):
        print(f"   Ideal conditions:    Not feasible within {max_years} years")
    else:
        print(f"   Ideal conditions:    {Y_min_ideal:.4f} years")
    
    if Y_min_real == float('inf'):
        print(f"   Real conditions:     Not feasible within {max_years} years")
    else:
        print(f"   Real conditions:     {Y_min_real:.4f} years")
    
    if Y_min_ideal < float('inf') and Y_min_real < float('inf'):
        time_increase = Y_min_real - Y_min_ideal
        time_change_pct = (time_increase / Y_min_ideal * 100) if Y_min_ideal > 0 else 0
        print(f"   Time increase:       {time_increase:.4f} years")
        print(f"   Time change rate:    {time_change_pct:.2f}%")
    else:
        print(f"   Time increase:       N/A (infeasible)")
    
    # Analyze at multiple durations
    if time_limit is not None:
        # Use the specified time limit
        durations = [time_limit]
        print(f"\n2. ANALYSIS AT SPECIFIED TIME LIMIT ({time_limit:.3f} years):")
    else:
        # Use a range of durations based on minimum feasible time
        if Y_min_real < float('inf'):
            start_year = max(Y_min_real * 1.2, 0.1)  # Start slightly above minimum
            # Create appropriate range based on the time scale
            if Y_min_real < 1.0:
                # Small times: create range from min to 5x min
                end_year = min(Y_min_real * 5, 10.0)
                durations = np.linspace(start_year, end_year, 6)
            else:
                # Larger times: create range from min to min+30
                end_year = min(Y_min_real + 30, 60.0)
                durations = np.linspace(start_year, end_year, 6)
        else:
            durations = np.linspace(10, 60, 6)  # Default range if not feasible
        print(f"\n2. ANALYSIS ACROSS MULTIPLE DURATIONS:")
    
    component_results = []
    
    for Y in durations:
        print(f"\n  Analyzing Y={Y:.3f} years...")
        ideal_result = ideal_model.solve(Y)
        real_result = real_model.solve(Y)
        
        if ideal_result.feasible and real_result.feasible:
            print(f"    Both ideal and real conditions are feasible")
            
            # Calculate rocket launches for reference
            ideal_launches = ideal_model.calculate_rocket_launches(ideal_result.mR_opt, Y)
            real_launches = real_model.calculate_rocket_launches(real_result.mR_opt, Y)
            
            # Calculate component changes
            component_changes = {
                'duration': Y,
                
                # Mass breakdown
                'ideal_elevator_mass': ideal_result.x_opt / 1e6,  # Million tons
                'real_elevator_mass': real_result.x_opt / 1e6,
                'ideal_rocket_mass': ideal_result.mR_opt / 1e6,
                'real_rocket_mass': real_result.mR_opt / 1e6,
                'ideal_total_mass': (ideal_result.x_opt + ideal_result.mR_opt) / 1e6,
                'real_total_mass': (real_result.x_opt + real_result.mR_opt) / 1e6,
                
                # Mass percentages
                'ideal_elevator_pct': ideal_result.elevator_pct,
                'real_elevator_pct': real_result.elevator_pct,
                'ideal_rocket_pct': ideal_result.rocket_pct,
                'real_rocket_pct': real_result.rocket_pct,
                
                # Rocket launches
                'ideal_rocket_launches': ideal_launches,
                'real_rocket_launches': real_launches,
                
                # Mass changes
                'elevator_mass_change': (real_result.x_opt - ideal_result.x_opt) / 1e6,
                'rocket_mass_change': (real_result.mR_opt - ideal_result.mR_opt) / 1e6,
                'total_mass_change': ((real_result.x_opt + real_result.mR_opt) - 
                                     (ideal_result.x_opt + ideal_result.mR_opt)) / 1e6,
                
                # Mass change percentages
                'elevator_mass_change_pct': ((real_result.x_opt - ideal_result.x_opt) / 
                                           ideal_result.x_opt * 100) if ideal_result.x_opt > 0 else 0,
                'rocket_mass_change_pct': ((real_result.mR_opt - ideal_result.mR_opt) / 
                                         ideal_result.mR_opt * 100) if ideal_result.mR_opt > 0 else 0,
                'total_mass_change_pct': (((real_result.x_opt + real_result.mR_opt) - 
                                          (ideal_result.x_opt + ideal_result.mR_opt)) / 
                                         (ideal_result.x_opt + ideal_result.mR_opt) * 100) if (ideal_result.x_opt + ideal_result.mR_opt) > 0 else 0,
                
                # Cost breakdown (in billions)
                'ideal_elevator_capex': ideal_result.costs.elevator_capex / 1e9,
                'real_elevator_capex': real_result.costs.elevator_capex / 1e9,
                'ideal_elevator_opex': ideal_result.costs.elevator_opex / 1e9,
                'real_elevator_opex': real_result.costs.elevator_opex / 1e9,
                'ideal_rocket_capex': ideal_result.costs.rocket_capex / 1e9,
                'real_rocket_capex': real_result.costs.rocket_capex / 1e9,
                'ideal_rocket_opex': ideal_result.costs.rocket_opex / 1e9,
                'real_rocket_opex': real_result.costs.rocket_opex / 1e9,
                'ideal_elevator_carbon': ideal_result.costs.elevator_carbon_cost / 1e9,
                'real_elevator_carbon': real_result.costs.elevator_carbon_cost / 1e9,
                'ideal_rocket_carbon': ideal_result.costs.rocket_carbon_cost / 1e9,
                'real_rocket_carbon': real_result.costs.rocket_carbon_cost / 1e9,
                
                # Total costs
                'ideal_elevator_total': ideal_result.costs.elevator_total / 1e9,
                'real_elevator_total': real_result.costs.elevator_total / 1e9,
                'ideal_rocket_total': ideal_result.costs.rocket_total / 1e9,
                'real_rocket_total': real_result.costs.rocket_total / 1e9,
                'ideal_total_cost': ideal_result.cost_total / 1e9,
                'real_total_cost': real_result.cost_total / 1e9,
                
                # Carbon emissions breakdown (million tons CO2)
                'ideal_elevator_emissions_operational': ideal_result.emissions.elevator_operational / 1e6,
                'real_elevator_emissions_operational': real_result.emissions.elevator_operational / 1e6,
                'ideal_rocket_emissions_operational': ideal_result.emissions.rocket_operational / 1e6,
                'real_rocket_emissions_operational': real_result.emissions.rocket_operational / 1e6,
                'ideal_elevator_emissions_construction': ideal_result.emissions.elevator_construction / 1e6,
                'real_elevator_emissions_construction': real_result.emissions.elevator_construction / 1e6,
                'ideal_rocket_emissions_construction': ideal_result.emissions.rocket_construction / 1e6,
                'real_rocket_emissions_construction': real_result.emissions.rocket_construction / 1e6,
                'ideal_total_emissions': ideal_result.emissions_total / 1e6,
                'real_total_emissions': real_result.emissions_total / 1e6,
                'ideal_carbon_intensity': ideal_result.carbon_intensity,
                'real_carbon_intensity': real_result.carbon_intensity,
                
                # Cost changes
                'elevator_cost_change': (real_result.costs.elevator_total - 
                                       ideal_result.costs.elevator_total) / 1e9,
                'rocket_cost_change': (real_result.costs.rocket_total - 
                                      ideal_result.costs.rocket_total) / 1e9,
                'total_cost_change': (real_result.cost_total - 
                                     ideal_result.cost_total) / 1e9,
                
                # Carbon emission changes
                'rocket_emissions_change': (real_result.emissions.rocket_operational - 
                                          ideal_result.emissions.rocket_operational) / 1e6,
                'total_emissions_change': (real_result.emissions_total - 
                                         ideal_result.emissions_total) / 1e6,
                
                # Cost change percentages
                'elevator_cost_change_pct': ((real_result.costs.elevator_total - 
                                            ideal_result.costs.elevator_total) / 
                                           ideal_result.costs.elevator_total * 100) if ideal_result.costs.elevator_total > 0 else 0,
                'rocket_cost_change_pct': ((real_result.costs.rocket_total - 
                                          ideal_result.costs.rocket_total) / 
                                         ideal_result.costs.rocket_total * 100) if ideal_result.costs.rocket_total > 0 else 0,
                'total_cost_change_pct': ((real_result.cost_total - 
                                         ideal_result.cost_total) / 
                                        ideal_result.cost_total * 100) if ideal_result.cost_total > 0 else 0,
                
                # Emissions change percentages
                'rocket_emissions_change_pct': ((real_result.emissions.rocket_operational - 
                                               ideal_result.emissions.rocket_operational) / 
                                              ideal_result.emissions.rocket_operational * 100) if ideal_result.emissions.rocket_operational > 0 else 0,
                'total_emissions_change_pct': ((real_result.emissions_total - 
                                              ideal_result.emissions_total) / 
                                             ideal_result.emissions_total * 100) if ideal_result.emissions_total > 0 else 0,
                
                # Feasibility flags
                'ideal_feasible': ideal_result.feasible,
                'real_feasible': real_result.feasible,
            }
            
            component_results.append(component_changes)
        else:
            print(f"    WARNING: Solution infeasible for {Y:.3f} years")
            print(f"      Ideal: {ideal_result.message}")
            print(f"      Real:  {real_result.message}")
            
            # Still record infeasible results
            component_changes = {
                'duration': Y,
                'ideal_feasible': ideal_result.feasible,
                'real_feasible': real_result.feasible,
                'ideal_message': ideal_result.message,
                'real_message': real_result.message,
                'feasible': False,
            }
            component_results.append(component_changes)
    
    return {
        'ideal_model': ideal_model,
        'real_model': real_model,
        'Y_min_ideal': Y_min_ideal,
        'Y_min_real': Y_min_real,
        'component_results': component_results,
        'total_mass': total_mass,
        'time_limit': time_limit,
    }


def display_detailed_component_table(comparison_data: Dict[str, Any]):
    """Display detailed table with component breakdown including carbon emissions."""
    
    component_results = comparison_data['component_results']
    total_mass = comparison_data.get('total_mass', 1.0e8)
    
    print("\n" + "=" * 120)
    print(f"DETAILED COMPONENT BREAKDOWN (Total Mass: {total_mass/1e6:.0f} Mt)")
    print("=" * 120)
    
    if not component_results:
        print("No results found.")
        return
    
    for result in component_results:
        Y = result['duration']
        
        # Check feasibility
        ideal_feas = result.get('ideal_feasible', True)
        real_feas = result.get('real_feasible', True)
        
        if not (ideal_feas and real_feas):
            print(f"\nDuration: {Y:.3f} years - INFEASIBLE")
            print(f"  Ideal: {result.get('ideal_message', 'Unknown')}")
            print(f"  Real:  {result.get('real_message', 'Unknown')}")
            continue
        
        # Display feasible results
        print(f"\n{'Duration':<12} | {'Component':<15} | {'Ideal':<15} | {'Real':<15} | {'Absolute Δ':<15} | {'% Δ':<10}")
        print("-" * 120)
        
        # MASS BREAKDOWN
        print(f"{Y:<12.3f} | {'MASS (Mt)':<15} | {'':<15} | {'':<15} | {'':<15} | {'':<10}")
        print(f"{'':<12} | {'  Elevator':<15} | {result['ideal_elevator_mass']:<15.3f} | {result['real_elevator_mass']:<15.3f} | {result['elevator_mass_change']:<+15.3f} | {result['elevator_mass_change_pct']:<+10.1f}%")
        print(f"{'':<12} | {'  Rocket':<15} | {result['ideal_rocket_mass']:<15.3f} | {result['real_rocket_mass']:<15.3f} | {result['rocket_mass_change']:<+15.3f} | {result['rocket_mass_change_pct']:<+10.1f}%")
        print(f"{'':<12} | {'  TOTAL':<15} | {result['ideal_total_mass']:<15.3f} | {result['real_total_mass']:<15.3f} | {result['total_mass_change']:<+15.3f} | {result['total_mass_change_pct']:<+10.1f}%")
        
        # ROCKET LAUNCHES
        print(f"{'':<12} | {'LAUNCHES':<15} | {'':<15} | {'':<15} | {'':<15} | {'':<10}")
        print(f"{'':<12} | {'  Rocket Launches':<15} | {result['ideal_rocket_launches']/1e6:<15.2f}M | {result['real_rocket_launches']/1e6:<15.2f}M | {'':<15} | {'':<10}")
        
        # CARBON EMISSIONS (Million tons CO2)
        print(f"{'':<12} | {'CO2 (Mt)':<15} | {'':<15} | {'':<15} | {'':<15} | {'':<10}")
        print(f"{'':<12} | {'  Elevator Op':<15} | {result['ideal_elevator_emissions_operational']:<15.3f} | {result['real_elevator_emissions_operational']:<15.3f} | {'':<15} | {'':<10}")
        print(f"{'':<12} | {'  Elevator Con':<15} | {result['ideal_elevator_emissions_construction']:<15.3f} | {result['real_elevator_emissions_construction']:<15.3f} | {'':<15} | {'':<10}")
        print(f"{'':<12} | {'  Rocket Op':<15} | {result['ideal_rocket_emissions_operational']:<15.3f} | {result['real_rocket_emissions_operational']:<15.3f} | {result['rocket_emissions_change']:<+15.3f} | {result['rocket_emissions_change_pct']:<+10.1f}%")
        print(f"{'':<12} | {'  Rocket Con':<15} | {result['ideal_rocket_emissions_construction']:<15.3f} | {result['real_rocket_emissions_construction']:<15.3f} | {'':<15} | {'':<10}")
        print(f"{'':<12} | {'  TOTAL CO2':<15} | {result['ideal_total_emissions']:<15.3f} | {result['real_total_emissions']:<15.3f} | {result['total_emissions_change']:<+15.3f} | {result['total_emissions_change_pct']:<+10.1f}%")
        
        # CARBON INTENSITY
        print(f"{'':<12} | {'CO2 Intensity':<15} | {'':<15} | {'':<15} | {'':<15} | {'':<10}")
        print(f"{'':<12} | {'  tCO2/t payload':<15} | {result['ideal_carbon_intensity']:<15.3f} | {result['real_carbon_intensity']:<15.3f} | {'':<15} | {'':<10}")
        
        # COST BREAKDOWN - Elevator
        print(f"{'':<12} | {'COSTS ($B)':<15} | {'':<15} | {'':<15} | {'':<15} | {'':<10}")
        print(f"{'':<12} | {'  Elevator CAPEX':<15} | {result['ideal_elevator_capex']:<15.3f} | {result['real_elevator_capex']:<15.3f} | {'':<15} | {'':<10}")
        print(f"{'':<12} | {'  Elevator OPEX':<15} | {result['ideal_elevator_opex']:<15.3f} | {result['real_elevator_opex']:<15.3f} | {'':<15} | {'':<10}")
        print(f"{'':<12} | {'  Elevator CO2 Cost':<15} | {result['ideal_elevator_carbon']:<15.3f} | {result['real_elevator_carbon']:<15.3f} | {'':<15} | {'':<10}")
        print(f"{'':<12} | {'  Elevator TOTAL':<15} | {result['ideal_elevator_total']:<15.3f} | {result['real_elevator_total']:<15.3f} | {result['elevator_cost_change']:<+15.3f} | {result['elevator_cost_change_pct']:<+10.1f}%")
        
        # COST BREAKDOWN - Rocket
        print(f"{'':<12} | {'  Rocket CAPEX':<15} | {result['ideal_rocket_capex']:<15.3f} | {result['real_rocket_capex']:<15.3f} | {'':<15} | {'':<10}")
        print(f"{'':<12} | {'  Rocket OPEX':<15} | {result['ideal_rocket_opex']:<15.3f} | {result['real_rocket_opex']:<15.3f} | {'':<15} | {'':<10}")
        print(f"{'':<12} | {'  Rocket CO2 Cost':<15} | {result['ideal_rocket_carbon']:<15.3f} | {result['real_rocket_carbon']:<15.3f} | {'':<15} | {'':<10}")
        print(f"{'':<12} | {'  Rocket TOTAL':<15} | {result['ideal_rocket_total']:<15.3f} | {result['real_rocket_total']:<15.3f} | {result['rocket_cost_change']:<+15.3f} | {result['rocket_cost_change_pct']:<+10.1f}%")
        
        # COST BREAKDOWN - Total
        print(f"{'':<12} | {'  SYSTEM TOTAL':<15} | {result['ideal_total_cost']:<15.3f} | {result['real_total_cost']:<15.3f} | {result['total_cost_change']:<+15.3f} | {result['total_cost_change_pct']:<+10.1f}%")
        
        print("-" * 120)


def analyze_specific_scenario(total_mass: float, time_limit: Optional[float] = None):
    """Analyze a specific mission scenario."""
    
    # Clear any previous debug messages
    import sys
    sys.stdout.flush()
    
    # Explicit unit conversion
    mass_megatons = total_mass / 1e6
    
    # Create output directory
    output_dir = f"scenario_{mass_megatons:.0f}Mt"
    if time_limit:
        output_dir += f"_{time_limit:.3f}yr".replace('.', 'p')  # Replace . with p for filename
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 80)
    print(f"SCENARIO ANALYSIS")
    print(f"Total Mass: {mass_megatons:.1f} million tons ({total_mass:.0e} tons)")
    if time_limit:
        print(f"Time Limit: {time_limit:.4f} years")
    print("=" * 80)
    
    # Run component comparison
    print("\n[1/3] Running component comparison...")
    comparison_data = compare_component_breakdown(
        total_mass=total_mass, 
        time_limit=time_limit,
        min_years=0.001,
        max_years=100
    )
    
    print("\n[2/3] Displaying detailed component table...")
    display_detailed_component_table(comparison_data)
    
    print("\n[3/3] Generating report...")
    
    # Generate comprehensive report
    with open(f"{output_dir}/scenario_report.txt", "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"SCENARIO ANALYSIS REPORT\n")
        f.write(f"Mission: {mass_megatons:.1f} million tons")
        if time_limit:
            f.write(f" within {time_limit:.4f} years\n")
        else:
            f.write(" (no time limit)\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("CARBON EMISSIONS PARAMETERS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Rocket CO2 per launch: {comparison_data['ideal_model'].p.carbon.CO2_per_launch:.0f} tons\n")
        f.write(f"Carbon price: ${comparison_data['ideal_model'].p.carbon.carbon_price:.0f}/ton CO2\n")
        f.write(f"Elevator CO2 per ton: {comparison_data['ideal_model'].p.carbon.CO2_elevator_per_ton:.1f} tons\n")
        f.write(f"Elevator construction CO2: {comparison_data['ideal_model'].p.carbon.CO2_elevator_construction/1e6:.1f} million tons\n")
        f.write(f"Launch site construction CO2: {comparison_data['ideal_model'].p.carbon.CO2_launch_site_construction/1e3:.0f} thousand tons per site\n\n")
        
        f.write("MINIMUM TIME REQUIREMENTS:\n")
        f.write("-" * 40 + "\n")
        if comparison_data['Y_min_ideal'] == float('inf'):
            f.write(f"Ideal conditions: Not feasible within 100 years\n")
        else:
            f.write(f"Ideal conditions: {comparison_data['Y_min_ideal']:.4f} years\n")
        
        if comparison_data['Y_min_real'] == float('inf'):
            f.write(f"Real conditions:  Not feasible within 100 years\n")
        else:
            f.write(f"Real conditions:  {comparison_data['Y_min_real']:.4f} years\n")
        
        if (comparison_data['Y_min_ideal'] != float('inf') and 
            comparison_data['Y_min_real'] != float('inf')):
            time_diff = comparison_data['Y_min_real'] - comparison_data['Y_min_ideal']
            f.write(f"Time increase:    {time_diff:.4f} years")
            if comparison_data['Y_min_ideal'] > 0:
                f.write(f" ({time_diff/comparison_data['Y_min_ideal']*100:.2f}%)\n")
        
        component_results = comparison_data['component_results']
        feasible_results = [r for r in component_results if r.get('ideal_feasible', True) and r.get('real_feasible', True)]
        
        if feasible_results:
            result = feasible_results[0]
            f.write("\nSOLUTION AT SPECIFIED TIME:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Duration: {result['duration']:.4f} years\n\n")
            
            f.write("MASS DISTRIBUTION:\n")
            f.write(f"  Elevator: {result['ideal_elevator_mass']:.3f} Mt (ideal) → {result['real_elevator_mass']:.3f} Mt (real)\n")
            f.write(f"  Rocket:   {result['ideal_rocket_mass']:.3f} Mt (ideal) → {result['real_rocket_mass']:.3f} Mt (real)\n")
            f.write(f"  Total:    {result['ideal_total_mass']:.3f} Mt (ideal) → {result['real_total_mass']:.3f} Mt (real)\n\n")
            
            f.write("ROCKET LAUNCHES:\n")
            f.write(f"  Ideal: {result['ideal_rocket_launches']/1e6:.2f} million launches\n")
            f.write(f"  Real:  {result['real_rocket_launches']/1e6:.2f} million launches\n\n")
            
            f.write("CARBON EMISSIONS (Million tons CO2):\n")
            f.write(f"  Rocket Operational: {result['ideal_rocket_emissions_operational']:.3f} Mt (ideal) → {result['real_rocket_emissions_operational']:.3f} Mt (real)\n")
            f.write(f"  Rocket Construction: {result['ideal_rocket_emissions_construction']:.3f} Mt (ideal) → {result['real_rocket_emissions_construction']:.3f} Mt (real)\n")
            f.write(f"  Elevator Construction: {result['ideal_elevator_emissions_construction']:.3f} Mt (ideal) → {result['real_elevator_emissions_construction']:.3f} Mt (real)\n")
            f.write(f"  TOTAL CO2: {result['ideal_total_emissions']:.3f} Mt (ideal) → {result['real_total_emissions']:.3f} Mt (real)\n")
            f.write(f"  Carbon Intensity: {result['ideal_carbon_intensity']:.3f} tCO2/t (ideal) → {result['real_carbon_intensity']:.3f} tCO2/t (real)\n\n")
            
            f.write("COST DISTRIBUTION (Billion USD):\n")
            f.write(f"  Elevator Total: ${result['ideal_elevator_total']:.3f}B (ideal) → ${result['real_elevator_total']:.3f}B (real)\n")
            f.write(f"  Rocket Total:   ${result['ideal_rocket_total']:.3f}B (ideal) → ${result['real_rocket_total']:.3f}B (real)\n")
            f.write(f"  System Total:   ${result['ideal_total_cost']:.3f}B (ideal) → ${result['real_total_cost']:.3f}B (real)\n")
            f.write(f"  Cost Increase:  {result['total_cost_change_pct']:+.1f}%\n\n")
            
            f.write("CARBON COST BREAKDOWN (Billion USD):\n")
            f.write(f"  Elevator Carbon Cost: ${result['ideal_elevator_carbon']:.3f}B (ideal) → ${result['real_elevator_carbon']:.3f}B (real)\n")
            f.write(f"  Rocket Carbon Cost:   ${result['ideal_rocket_carbon']:.3f}B (ideal) → ${result['real_rocket_carbon']:.3f}B (real)\n")
            f.write(f"  Total Carbon Cost:    ${result['ideal_elevator_carbon'] + result['ideal_rocket_carbon']:.3f}B (ideal) → ${result['real_elevator_carbon'] + result['real_rocket_carbon']:.3f}B (real)\n")
    
    print(f"Saved report: {output_dir}/scenario_report.txt")
    
    # Display summary
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    
    Y_min_ideal = comparison_data['Y_min_ideal']
    Y_min_real = comparison_data['Y_min_real']
    
    print(f"\nMINIMUM FEASIBLE TIMES:")
    if Y_min_ideal == float('inf'):
        print(f"  Ideal conditions: Not feasible within 100 years")
    else:
        print(f"  Ideal conditions: {Y_min_ideal:.4f} years")
    
    if Y_min_real == float('inf'):
        print(f"  Real conditions:  Not feasible within 100 years")
    else:
        print(f"  Real conditions:  {Y_min_real:.4f} years")
    
    if Y_min_ideal < float('inf') and Y_min_real < float('inf'):
        print(f"  Time increase:     {Y_min_real - Y_min_ideal:.4f} years")
        if Y_min_ideal > 0:
            print(f"  Time change:       {(Y_min_real - Y_min_ideal)/Y_min_ideal*100:.2f}%")
    
    # Get feasible results
    component_results = comparison_data['component_results']
    feasible_results = [r for r in component_results if r.get('ideal_feasible', True) and r.get('real_feasible', True)]
    
    if feasible_results and time_limit:
        result = feasible_results[0]
        print(f"\nAT SPECIFIED TIME ({time_limit:.4f} years):")
        print(f"  Mass distribution:")
        print(f"    Elevator: {result['ideal_elevator_pct']:.1f}% (ideal) → {result['real_elevator_pct']:.1f}% (real)")
        print(f"    Rocket:   {result['ideal_rocket_pct']:.1f}% (ideal) → {result['real_rocket_pct']:.1f}% (real)")
        print(f"  Rocket launches: {result['ideal_rocket_launches']/1e6:.2f}M (ideal) → {result['real_rocket_launches']/1e6:.2f}M (real)")
        print(f"  Carbon emissions: {result['ideal_total_emissions']:.3f} Mt CO2 (ideal) → {result['real_total_emissions']:.3f} Mt CO2 (real)")
        print(f"  Carbon intensity: {result['ideal_carbon_intensity']:.3f} tCO2/t (ideal) → {result['real_carbon_intensity']:.3f} tCO2/t (real)")
        print(f"  Total cost: ${result['ideal_total_cost']:.3f}B (ideal) → ${result['real_total_cost']:.3f}B (real)")
        print(f"  Cost increase: {result['total_cost_change_pct']:+.1f}%")
    
    print(f"\n" + "=" * 80)
    print(f"Analysis complete! Check the '{output_dir}' directory for details.")
    print("=" * 80)
    
    return comparison_data


# ============================================================================
# Main Analysis
# ============================================================================

def main():
    """Run the complete component analysis with user-specified parameters."""
    
    print("=" * 80)
    print("SPACE TRANSPORT OPTIMIZATION ANALYSIS")
    print("=" * 80)
    
    # Get user inputs
    print("\nMISSION PARAMETERS:")
    print("-" * 40)
    
    # Total mass to transfer
    while True:
        try:
            mass_input = input("Enter total mass to transfer (in million tons, default=100): ").strip()
            if not mass_input:
                total_mass = 1.0e8  # 100 million tons default
                break
            mass_mt = float(mass_input)
            if mass_mt <= 0:
                print("Mass must be positive. Try again.")
                continue
            total_mass = mass_mt * 1e6  # Convert to tons
            break
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    # Time limit (optional) - now accepts decimal values
    while True:
        try:
            time_input = input("Enter time limit in years (optional, press Enter to skip): ").strip()
            if not time_input:
                time_limit = None
                break
            time_limit = float(time_input)
            if time_limit <= 0:
                print("Time must be positive. Try again.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a number or press Enter to skip.")
    
    print(f"\nAnalyzing mission: {total_mass/1e6:.0f} million tons", end="")
    if time_limit:
        print(f" within {time_limit:.4f} years")
    else:
        print(" (no time limit specified)")
    
    # Run analysis
    results = analyze_specific_scenario(total_mass, time_limit)
    
    return results


def run_quick_test():
    """Run a quick test with default values."""
    print("\nRunning quick test with default values (100 Mt, no time limit)...")
    return analyze_specific_scenario(total_mass=1.0e8, time_limit=None)


def run_carbon_focused_test():
    """Run tests focused on carbon emissions analysis."""
    print("\n" + "=" * 80)
    print("CARBON EMISSIONS FOCUSED TEST")
    print("=" * 80)
    
    test_scenarios = [
        {"name": "Small mission (1 Mt)", "mass_mt": 1, "time_years": None},
        {"name": "Medium mission (10 Mt)", "mass_mt": 10, "time_years": None},
        {"name": "Large mission (100 Mt)", "mass_mt": 100, "time_years": None},
        {"name": "Fast small mission (1 Mt in 0.5y)", "mass_mt": 1, "time_years": 0.5},
        {"name": "Fast medium mission (10 Mt in 5y)", "mass_mt": 10, "time_years": 5},
    ]
    
    all_results = []
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n[{i}/{len(test_scenarios)}] {scenario['name']}")
        print(f"   Mass: {scenario['mass_mt']} Mt", end="")
        if scenario['time_years']:
            print(f", Time limit: {scenario['time_years']:.3f} years")
        else:
            print(" (no time limit - find minimum)")
        
        result = analyze_specific_scenario(
            total_mass=scenario['mass_mt'] * 1e6,
            time_limit=scenario['time_years']
        )
        all_results.append((scenario['name'], result))
        
        # Pause between tests
        if i < len(test_scenarios):
            input("\nPress Enter to continue to next test...")
    
    # Carbon summary
    print("\n" + "=" * 80)
    print("CARBON EMISSIONS SUMMARY")
    print("=" * 80)
    
    for name, result in all_results:
        component_results = result['component_results']
        feasible_results = [r for r in component_results if r.get('ideal_feasible', True) and r.get('real_feasible', True)]
        
        if feasible_results:
            res = feasible_results[0]
            print(f"\n{name}:")
            print(f"  Rocket CO2: {res['ideal_rocket_emissions_operational']:.3f} Mt (ideal) → {res['real_rocket_emissions_operational']:.3f} Mt (real)")
            print(f"  Total CO2: {res['ideal_total_emissions']:.3f} Mt (ideal) → {res['real_total_emissions']:.3f} Mt (real)")
            print(f"  Carbon cost: ${res['ideal_rocket_carbon']:.3f}B (ideal) → ${res['real_rocket_carbon']:.3f}B (real)")
    
    return all_results


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        print("=" * 80)
        print("SPACE TRANSPORT SYSTEM OPTIMIZER WITH CARBON ANALYSIS")
        print("=" * 80)
        
        print("\nSelect analysis mode:")
        print("1. Quick test (100 Mt, no time limit)")
        print("2. Custom scenario (enter your own parameters)")
        print("3. Carbon-focused test (various scenarios)")
        print("4. Interactive mode (guided input)")
        
        choice = input("\nEnter choice (1-4, default=1): ").strip()
        
        if choice == "2":
            results = main()
        elif choice == "3":
            results = run_carbon_focused_test()
        elif choice == "4":
            results = main()
        else:  # Default to quick test
            results = run_quick_test()
