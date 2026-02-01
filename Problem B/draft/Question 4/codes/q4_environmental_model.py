"""
Environmental Impact Assessment Model for Question 4
=====================================================
Based on environmental_impact_model_revised.md

This module implements:
1. Atmospheric Layer Impact Model (Vertical Dimension)
2. Orbital Environment Risk Model (Spatial Dimension)  
3. Life Cycle Assessment Model (Human Dimension)
4. Space Environment Impact Score (SEIS) Calculation

Key Features:
- Full Life-Cycle Assessment (LCA) framework
- Pre-deployed Infrastructure Assumption for space elevator
- Break-even analysis for environmental payback
- Multi-scenario comparison (Pure Rocket, Pure Elevator, Hybrid)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from scipy.integrate import quad
from typing import Optional, List, Tuple, Dict, Any
import warnings

# ============================================================================
# Data Classes for Parameters
# ============================================================================

@dataclass(frozen=True)
class AtmosphericParams:
    """Parameters for atmospheric layer impact model.
    
    References:
    - Troposphere: Standard GHG emission factors
    - Stratosphere: Dallas et al. (2020) - Environmental Impact of Rockets
    - Mesosphere: Ross & Sheaffer (2014) - Radiative forcing from rocket emissions
    """
    # Weight factors for Environmental Damage Potential (EDP)
    W_troposphere: float = 1.0      # Baseline (ground level effects)
    W_stratosphere: float = 500.0   # Ozone layer vulnerability (10-50 km)
    W_mesosphere: float = 50.0      # Upper atmosphere effects (> 50 km)
    
    # Emission fractions per launch (as fraction of propellant mass)
    CO2_fraction: float = 0.70      # CO2 emission fraction (methane/LOX)
    NOx_fraction: float = 0.005     # NOx emission fraction
    BC_fraction: float = 0.01       # Black carbon fraction (soot)
    Al2O3_fraction: float = 0.005   # Alumina particles (solid boosters)
    H2O_fraction: float = 0.15      # Water vapor fraction
    
    # Stratosphere transit parameters
    transit_time_min: float = 2.0   # Minutes in stratosphere per launch
    stratosphere_deposition: float = 0.01  # Fraction deposited in stratosphere


@dataclass(frozen=True)
class OrbitalRiskParams:
    """Parameters for Kessler Syndrome risk model.
    
    References:
    - Kessler & Cour-Palais (1978) - Collision Frequency
    - ESA Space Debris Office reports
    """
    # Risk equation coefficients: dR/dt = α·N_launch + β·R² - γ·R
    alpha: float = 0.001        # Debris generation rate per launch
    beta: float = 0.0001        # Cascade collision coefficient
    gamma: float = 0.01         # Natural decay rate
    
    # Initial conditions
    R_initial: float = 1.0      # Initial normalized risk index
    
    # Critical thresholds
    R_critical: float = 5.0     # Critical risk threshold (cascade onset)
    
    # Elevator debris mitigation factor
    elevator_mitigation: float = 0.8  # Debris reduction via electrodynamic tether


@dataclass(frozen=True)
class CarbonEmissionFactors:
    """Carbon emission factors for different transport modes.
    
    References:
    - SpaceX environmental reports
    - Dallas et al. (2020) - Rocket Launch Carbon Footprint
    - Life Cycle Assessment standards (ISO 14040/14044)
    """
    # Rocket emissions (tons CO2 per launch)
    CO2_per_rocket_launch: float = 2500.0  # Starship-class (CH4/LOX)
    
    # Rocket payload capacity (tons per launch)
    rocket_payload: float = 150.0
    
    # Elevator operational emissions (tons CO2 per ton payload)
    CO2_elevator_per_ton: float = 0.1  # Electric-powered, near-zero
    
    # Construction carbon footprint
    elevator_construction_Mt: float = 5.0    # Total elevator construction (Mt CO2)
    elevator_transport_factor: float = 0.1   # CO2 per ton transported via elevator
    rocket_transport_factor: float = 16.67   # CO2 per ton transported via rocket
    
    # Launch site construction emissions (tons CO2 per site)
    launch_site_construction: float = 100000.0


@dataclass(frozen=True)
class ColonyParams:
    """Parameters for lunar colony and Earth baseline.
    
    References:
    - World Bank per capita CO2 data
    - NASA/ESA lunar base studies
    """
    # Colony population
    population: int = 100000
    
    # Earth per capita carbon footprint (tons CO2/year)
    earth_per_capita_emission: float = 15.0  # Global weighted average
    
    # Colony operational parameters
    colony_start_year: int = 2050
    construction_start_year: int = 2026
    
    # Total mass to transport (tons)
    total_mass_tons: float = 1.0e8  # 100 million tons


@dataclass
class ScenarioParams:
    """Parameters for a specific transport scenario."""
    name: str
    
    # Transport mode allocation
    elevator_fraction: float = 1.0  # Fraction transported via elevator
    rocket_fraction: float = 0.0    # Fraction transported via rocket
    
    # Infrastructure assumptions
    elevator_predeployed: bool = True  # Is elevator built before transport?
    
    # Operational parameters
    annual_elevator_throughput: float = 5.37e5  # tons/year
    annual_rocket_launches: int = 0  # For operations phase
    
    # Construction phase parameters
    construction_rocket_launches: int = 0  # Launches during construction
    
    def __post_init__(self):
        """Ensure fractions sum to 1."""
        total = self.elevator_fraction + self.rocket_fraction
        if abs(total - 1.0) > 0.01:
            warnings.warn(f"Transport fractions sum to {total}, normalizing.")
            self.elevator_fraction /= total
            self.rocket_fraction /= total


@dataclass
class ModelParams:
    """Global model parameters combining all sub-parameters."""
    atmospheric: AtmosphericParams = field(default_factory=AtmosphericParams)
    orbital: OrbitalRiskParams = field(default_factory=OrbitalRiskParams)
    carbon: CarbonEmissionFactors = field(default_factory=CarbonEmissionFactors)
    colony: ColonyParams = field(default_factory=ColonyParams)


# ============================================================================
# Result Data Classes
# ============================================================================

@dataclass
class AtmosphericImpact:
    """Results from atmospheric impact analysis."""
    EDP_troposphere: float = 0.0
    EDP_stratosphere: float = 0.0
    EDP_mesosphere: float = 0.0
    
    @property
    def EDP_total(self) -> float:
        return self.EDP_troposphere + self.EDP_stratosphere + self.EDP_mesosphere
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "EDP_troposphere": self.EDP_troposphere,
            "EDP_stratosphere": self.EDP_stratosphere,
            "EDP_mesosphere": self.EDP_mesosphere,
            "EDP_total": self.EDP_total
        }


@dataclass
class OrbitalRiskResult:
    """Results from orbital debris risk analysis."""
    initial_risk: float = 0.0
    final_risk: float = 0.0
    peak_risk: float = 0.0
    years_above_critical: float = 0.0
    cascade_triggered: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "initial_risk": self.initial_risk,
            "final_risk": self.final_risk,
            "peak_risk": self.peak_risk,
            "years_above_critical": self.years_above_critical,
            "cascade_triggered": self.cascade_triggered
        }


@dataclass
class LCAResult:
    """Results from Life Cycle Assessment."""
    # Construction phase emissions (Mt CO2)
    E_infrastructure: float = 0.0
    E_transport_construction: float = 0.0
    
    # Operations phase emissions (Mt CO2/year)
    E_operations_annual: float = 0.0
    
    # Environmental benefit (Mt CO2/year)
    colony_reduction_annual: float = 0.0
    
    # Break-even analysis
    break_even_years: float = float('inf')
    is_sustainable: bool = False
    
    @property
    def E_construction_total(self) -> float:
        """Total construction phase emissions (Mt CO2)."""
        return self.E_infrastructure + self.E_transport_construction
    
    @property
    def net_annual_impact(self) -> float:
        """Net annual impact during operations (Mt CO2/year)."""
        return self.E_operations_annual - self.colony_reduction_annual
    
    def cumulative_emissions(self, years_after_2050: float) -> float:
        """Calculate cumulative net emissions at time t years after 2050."""
        return self.E_construction_total + self.net_annual_impact * years_after_2050
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "E_infrastructure_Mt": self.E_infrastructure,
            "E_transport_construction_Mt": self.E_transport_construction,
            "E_construction_total_Mt": self.E_construction_total,
            "E_operations_annual_Mt": self.E_operations_annual,
            "colony_reduction_annual_Mt": self.colony_reduction_annual,
            "net_annual_impact_Mt": self.net_annual_impact,
            "break_even_years": self.break_even_years,
            "is_sustainable": self.is_sustainable
        }


@dataclass
class SEISResult:
    """Space Environment Impact Score results."""
    # Component scores (normalized)
    stratospheric_score: float = 0.0
    orbital_risk_score: float = 0.0
    break_even_score: float = 0.0
    
    # Weights
    w1: float = 0.3  # Stratospheric weight
    w2: float = 0.3  # Orbital risk weight
    w3: float = 0.4  # Break-even weight
    
    @property
    def SEIS(self) -> float:
        """Calculate weighted SEIS score."""
        return (self.w1 * self.stratospheric_score + 
                self.w2 * self.orbital_risk_score + 
                self.w3 * self.break_even_score)
    
    @property
    def grade(self) -> str:
        """Convert SEIS to letter grade."""
        seis = self.SEIS
        if seis < 0.5:
            return "A+"
        elif seis < 1.0:
            return "A"
        elif seis < 2.0:
            return "B"
        elif seis < 4.0:
            return "C"
        elif seis < 7.0:
            return "D"
        else:
            return "F"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "stratospheric_score": self.stratospheric_score,
            "orbital_risk_score": self.orbital_risk_score,
            "break_even_score": self.break_even_score,
            "SEIS": self.SEIS,
            "grade": self.grade
        }


@dataclass
class EnvironmentalAssessmentResult:
    """Complete environmental assessment result for a scenario."""
    scenario_name: str
    atmospheric: AtmosphericImpact
    orbital: OrbitalRiskResult
    lca: LCAResult
    seis: SEISResult
    
    def summary(self) -> Dict[str, Any]:
        """Generate summary dictionary."""
        return {
            "scenario": self.scenario_name,
            "atmospheric": self.atmospheric.to_dict(),
            "orbital": self.orbital.to_dict(),
            "lca": self.lca.to_dict(),
            "seis": self.seis.to_dict()
        }


# ============================================================================
# Core Model Classes
# ============================================================================

class AtmosphericImpactModel:
    """
    Model for calculating atmospheric layer impacts from rocket launches.
    
    The model divides atmosphere into three layers:
    - Troposphere (0-10 km): CO2, NOx effects
    - Stratosphere (10-50 km): Ozone depletion, black carbon
    - Mesosphere (>50 km): Noctilucent clouds, thermal effects
    """
    
    def __init__(self, params: ModelParams):
        self.params = params
        self.atm = params.atmospheric
    
    def calculate_launch_emissions(self, num_launches: int, 
                                   propellant_mass_per_launch: float = 4000.0
                                   ) -> Dict[str, float]:
        """
        Calculate total emissions from a number of launches.
        
        Args:
            num_launches: Number of rocket launches
            propellant_mass_per_launch: Propellant mass per launch (tons)
        
        Returns:
            Dictionary of emission masses by type (tons)
        """
        total_propellant = num_launches * propellant_mass_per_launch
        
        return {
            "CO2": total_propellant * self.atm.CO2_fraction,
            "NOx": total_propellant * self.atm.NOx_fraction,
            "BC": total_propellant * self.atm.BC_fraction,
            "Al2O3": total_propellant * self.atm.Al2O3_fraction,
            "H2O": total_propellant * self.atm.H2O_fraction
        }
    
    def calculate_layer_deposition(self, emissions: Dict[str, float]
                                   ) -> Dict[str, Dict[str, float]]:
        """
        Calculate how emissions are distributed across atmospheric layers.
        
        Returns:
            Nested dict: layer -> species -> mass (tons)
        """
        strat_frac = self.atm.stratosphere_deposition
        
        # Troposphere: Most CO2 and NOx remain
        troposphere = {
            "CO2": emissions["CO2"] * 0.9,
            "NOx": emissions["NOx"] * 0.7,
            "BC": emissions["BC"] * 0.1,
            "Al2O3": emissions["Al2O3"] * 0.1,
            "H2O": emissions["H2O"] * 0.2
        }
        
        # Stratosphere: Black carbon and particles accumulate
        stratosphere = {
            "CO2": emissions["CO2"] * 0.05,
            "NOx": emissions["NOx"] * 0.2,
            "BC": emissions["BC"] * 0.8,  # Most BC deposits here
            "Al2O3": emissions["Al2O3"] * 0.8,
            "H2O": emissions["H2O"] * 0.5
        }
        
        # Mesosphere: Primarily water vapor
        mesosphere = {
            "CO2": emissions["CO2"] * 0.05,
            "NOx": emissions["NOx"] * 0.1,
            "BC": emissions["BC"] * 0.1,
            "Al2O3": emissions["Al2O3"] * 0.1,
            "H2O": emissions["H2O"] * 0.3
        }
        
        return {
            "troposphere": troposphere,
            "stratosphere": stratosphere,
            "mesosphere": mesosphere
        }
    
    def calculate_EDP(self, num_launches: int,
                      propellant_mass_per_launch: float = 4000.0
                      ) -> AtmosphericImpact:
        """
        Calculate Environmental Damage Potential (EDP) for all layers.
        
        EDP_total = Σ_layer Σ_species (M_layer,species × W_layer)
        
        Args:
            num_launches: Number of rocket launches
            propellant_mass_per_launch: Propellant per launch (tons)
        
        Returns:
            AtmosphericImpact with EDP values
        """
        emissions = self.calculate_launch_emissions(num_launches, 
                                                     propellant_mass_per_launch)
        deposition = self.calculate_layer_deposition(emissions)
        
        # Calculate weighted EDP for each layer
        EDP_trop = sum(deposition["troposphere"].values()) * self.atm.W_troposphere
        EDP_strat = sum(deposition["stratosphere"].values()) * self.atm.W_stratosphere
        EDP_meso = sum(deposition["mesosphere"].values()) * self.atm.W_mesosphere
        
        return AtmosphericImpact(
            EDP_troposphere=EDP_trop,
            EDP_stratosphere=EDP_strat,
            EDP_mesosphere=EDP_meso
        )


class OrbitalRiskModel:
    """
    Model for Kessler Syndrome risk assessment.
    
    Risk dynamics: dR/dt = α·N_launch(t) + β·R² - γ·R
    
    Where:
    - α: Debris generation rate per launch
    - β: Cascade collision coefficient  
    - γ: Natural orbital decay rate
    """
    
    def __init__(self, params: ModelParams):
        self.params = params
        self.orb = params.orbital
    
    def risk_derivative(self, R: float, launch_rate: float) -> float:
        """
        Calculate dR/dt at given risk level and launch rate.
        
        Args:
            R: Current risk index
            launch_rate: Launches per year
        
        Returns:
            Rate of change of risk index
        """
        alpha = self.orb.alpha
        beta = self.orb.beta
        gamma = self.orb.gamma
        
        return alpha * launch_rate + beta * R**2 - gamma * R
    
    def simulate_risk_evolution(self, annual_launches: np.ndarray,
                                dt: float = 0.1) -> np.ndarray:
        """
        Simulate risk evolution over time using Euler method.
        
        Args:
            annual_launches: Array of launch rates per year
            dt: Time step (years)
        
        Returns:
            Array of risk values over time
        """
        n_years = len(annual_launches)
        n_steps = int(n_years / dt)
        
        R = np.zeros(n_steps + 1)
        R[0] = self.orb.R_initial
        
        for i in range(n_steps):
            year_idx = min(int(i * dt), n_years - 1)
            launch_rate = annual_launches[year_idx]
            
            dRdt = self.risk_derivative(R[i], launch_rate)
            R[i + 1] = max(0, R[i] + dRdt * dt)
        
        return R
    
    def analyze_scenario(self, total_launches: int, 
                         duration_years: float,
                         elevator_active: bool = False) -> OrbitalRiskResult:
        """
        Analyze orbital risk for a transport scenario.
        
        Args:
            total_launches: Total rocket launches
            duration_years: Duration of operations (years)
            elevator_active: Whether space elevator is active
        
        Returns:
            OrbitalRiskResult with risk metrics
        """
        # Create launch rate schedule (assume uniform distribution)
        n_years = int(np.ceil(duration_years))
        annual_rate = total_launches / duration_years if duration_years > 0 else 0
        annual_launches = np.full(n_years, annual_rate)
        
        # Apply elevator mitigation if active
        if elevator_active:
            annual_launches *= (1 - self.orb.elevator_mitigation)
        
        # Simulate risk evolution
        R = self.simulate_risk_evolution(annual_launches)
        
        # Analyze results
        peak_risk = np.max(R)
        final_risk = R[-1]
        
        # Calculate time above critical threshold
        dt = 0.1
        times_above = np.sum(R > self.orb.R_critical) * dt
        
        cascade_triggered = peak_risk > self.orb.R_critical
        
        return OrbitalRiskResult(
            initial_risk=self.orb.R_initial,
            final_risk=final_risk,
            peak_risk=peak_risk,
            years_above_critical=times_above,
            cascade_triggered=cascade_triggered
        )


class LifeCycleAssessmentModel:
    """
    Life Cycle Assessment (LCA) model for space transport systems.
    
    Phases:
    1. Construction (2026-2050): Infrastructure + initial transport
    2. Operations (2050+): Ongoing supply + colony benefits
    
    Key equation:
    E_net(t) = E_const + ∫E_op(τ)dτ - ∫P_colony·e_earth dτ
    """
    
    def __init__(self, params: ModelParams):
        self.params = params
        self.carbon = params.carbon
        self.colony = params.colony
    
    def calculate_infrastructure_emissions(self, 
                                           elevator_built: bool,
                                           num_launch_sites: int = 0) -> float:
        """
        Calculate construction phase infrastructure emissions.
        
        Args:
            elevator_built: Whether space elevator is constructed
            num_launch_sites: Number of new rocket launch sites
        
        Returns:
            Infrastructure emissions (Mt CO2)
        """
        E_infra = 0.0
        
        if elevator_built:
            E_infra += self.carbon.elevator_construction_Mt
        
        # Launch site construction
        E_infra += (num_launch_sites * self.carbon.launch_site_construction) / 1e6
        
        return E_infra
    
    def calculate_transport_emissions(self, mass_tons: float,
                                      elevator_fraction: float) -> float:
        """
        Calculate emissions from transporting mass to Moon.
        
        Args:
            mass_tons: Total mass to transport (tons)
            elevator_fraction: Fraction transported via elevator
        
        Returns:
            Transport emissions (Mt CO2)
        """
        mass_elevator = mass_tons * elevator_fraction
        mass_rocket = mass_tons * (1 - elevator_fraction)
        
        # Elevator transport emissions
        E_elevator = mass_elevator * self.carbon.elevator_transport_factor / 1e6
        
        # Rocket transport emissions (based on launches needed)
        launches_needed = mass_rocket / self.carbon.rocket_payload
        E_rocket = launches_needed * self.carbon.CO2_per_rocket_launch / 1e6
        
        return E_elevator + E_rocket
    
    def calculate_annual_operations_emissions(self, 
                                              annual_supply_tons: float,
                                              elevator_fraction: float) -> float:
        """
        Calculate annual emissions during operations phase.
        
        Args:
            annual_supply_tons: Annual resupply mass (tons)
            elevator_fraction: Fraction via elevator
        
        Returns:
            Annual operational emissions (Mt CO2/year)
        """
        return self.calculate_transport_emissions(annual_supply_tons, elevator_fraction)
    
    def calculate_colony_reduction(self) -> float:
        """
        Calculate annual CO2 reduction from population relocation.
        
        Returns:
            Annual reduction (Mt CO2/year)
        """
        return (self.colony.population * 
                self.colony.earth_per_capita_emission / 1e6)
    
    def calculate_break_even(self, E_construction: float,
                             E_operations_annual: float,
                             colony_reduction_annual: float) -> Tuple[float, bool]:
        """
        Calculate break-even time and sustainability.
        
        Break-even when: E_const + t·E_op = t·colony_reduction
        Solving: t = E_const / (colony_reduction - E_op)
        
        Args:
            E_construction: Total construction emissions (Mt)
            E_operations_annual: Annual operational emissions (Mt/yr)
            colony_reduction_annual: Annual colony benefit (Mt/yr)
        
        Returns:
            Tuple of (break_even_years, is_sustainable)
        """
        net_annual_benefit = colony_reduction_annual - E_operations_annual
        
        if net_annual_benefit <= 0:
            # Never breaks even
            return float('inf'), False
        
        break_even_years = E_construction / net_annual_benefit
        is_sustainable = break_even_years < 100  # 100-year sustainability threshold
        
        return break_even_years, is_sustainable
    
    def assess_scenario(self, scenario: ScenarioParams) -> LCAResult:
        """
        Perform full LCA for a transport scenario.
        
        Args:
            scenario: Scenario parameters
        
        Returns:
            LCAResult with all metrics
        """
        # Construction phase
        E_infra = self.calculate_infrastructure_emissions(
            elevator_built=scenario.elevator_predeployed,
            num_launch_sites=0  # Assume existing sites
        )
        
        # Transport emissions for 100Mt
        E_transport = self.calculate_transport_emissions(
            mass_tons=self.colony.total_mass_tons,
            elevator_fraction=scenario.elevator_fraction
        )
        
        # Operations phase (assume 1000 tons/year resupply per 10k people)
        annual_supply = self.colony.population / 10000 * 1000  # 10,000 tons/year
        E_ops_annual = self.calculate_annual_operations_emissions(
            annual_supply_tons=annual_supply,
            elevator_fraction=scenario.elevator_fraction
        )
        
        # Colony benefit
        colony_reduction = self.calculate_colony_reduction()
        
        # Break-even analysis
        E_construction_total = E_infra + E_transport
        break_even, is_sustainable = self.calculate_break_even(
            E_construction_total, E_ops_annual, colony_reduction
        )
        
        return LCAResult(
            E_infrastructure=E_infra,
            E_transport_construction=E_transport,
            E_operations_annual=E_ops_annual,
            colony_reduction_annual=colony_reduction,
            break_even_years=break_even,
            is_sustainable=is_sustainable
        )


class SEISCalculator:
    """
    Space Environment Impact Score (SEIS) Calculator.
    
    SEIS = w1·(E_Strat/E_ref) + w2·(Risk_Orbital/R_ref) + w3·(T_BE/T_ref)
    
    Lower score = better environmental performance.
    """
    
    # Reference values for normalization
    E_STRAT_REF = 1e9       # Reference stratospheric impact (tons)
    R_ORBITAL_REF = 5.0     # Reference orbital risk
    T_BE_REF = 50.0         # Reference break-even time (years)
    
    def __init__(self, w1: float = 0.3, w2: float = 0.3, w3: float = 0.4):
        """
        Initialize with component weights.
        
        Args:
            w1: Stratospheric impact weight
            w2: Orbital risk weight  
            w3: Break-even time weight
        """
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
    
    def calculate(self, atmospheric: AtmosphericImpact,
                  orbital: OrbitalRiskResult,
                  lca: LCAResult) -> SEISResult:
        """
        Calculate SEIS from component results.
        
        Args:
            atmospheric: Atmospheric impact results
            orbital: Orbital risk results
            lca: LCA results
        
        Returns:
            SEISResult with scores and grade
        """
        # Normalize stratospheric impact
        strat_score = atmospheric.EDP_stratosphere / self.E_STRAT_REF
        
        # Normalize orbital risk
        orbital_score = orbital.peak_risk / self.R_ORBITAL_REF
        
        # Normalize break-even (cap at 10 for infinite)
        if lca.break_even_years == float('inf'):
            be_score = 10.0
        else:
            be_score = min(10.0, lca.break_even_years / self.T_BE_REF)
        
        return SEISResult(
            stratospheric_score=strat_score,
            orbital_risk_score=orbital_score,
            break_even_score=be_score,
            w1=self.w1,
            w2=self.w2,
            w3=self.w3
        )


# ============================================================================
# Integrated Environmental Assessment
# ============================================================================

class EnvironmentalAssessmentModel:
    """
    Integrated environmental assessment combining all model dimensions.
    """
    
    def __init__(self, params: Optional[ModelParams] = None):
        """
        Initialize with model parameters.
        
        Args:
            params: Model parameters (uses defaults if None)
        """
        self.params = params or ModelParams()
        
        # Initialize sub-models
        self.atmospheric_model = AtmosphericImpactModel(self.params)
        self.orbital_model = OrbitalRiskModel(self.params)
        self.lca_model = LifeCycleAssessmentModel(self.params)
        self.seis_calculator = SEISCalculator()
    
    def assess_scenario(self, scenario: ScenarioParams
                        ) -> EnvironmentalAssessmentResult:
        """
        Perform complete environmental assessment for a scenario.
        
        Args:
            scenario: Scenario parameters
        
        Returns:
            EnvironmentalAssessmentResult with all metrics
        """
        # Calculate number of rocket launches needed
        total_mass = self.params.colony.total_mass_tons
        rocket_mass = total_mass * scenario.rocket_fraction
        num_launches = int(rocket_mass / self.params.carbon.rocket_payload)
        
        # Duration estimate (years to complete transport)
        if scenario.elevator_fraction > 0:
            elevator_years = (total_mass * scenario.elevator_fraction / 
                            scenario.annual_elevator_throughput)
        else:
            elevator_years = 0
        
        rocket_launches_per_year = 50000  # Assume max capacity
        rocket_years = num_launches / rocket_launches_per_year if num_launches > 0 else 0
        
        total_years = max(elevator_years, rocket_years, 1)
        
        # Atmospheric impact
        atmospheric = self.atmospheric_model.calculate_EDP(num_launches)
        
        # Orbital risk
        orbital = self.orbital_model.analyze_scenario(
            total_launches=num_launches,
            duration_years=total_years,
            elevator_active=(scenario.elevator_fraction > 0.5)
        )
        
        # LCA
        lca = self.lca_model.assess_scenario(scenario)
        
        # SEIS
        seis = self.seis_calculator.calculate(atmospheric, orbital, lca)
        
        return EnvironmentalAssessmentResult(
            scenario_name=scenario.name,
            atmospheric=atmospheric,
            orbital=orbital,
            lca=lca,
            seis=seis
        )
    
    def compare_scenarios(self, scenarios: List[ScenarioParams]
                          ) -> List[EnvironmentalAssessmentResult]:
        """
        Assess and compare multiple scenarios.
        
        Args:
            scenarios: List of scenario parameters
        
        Returns:
            List of assessment results
        """
        return [self.assess_scenario(s) for s in scenarios]


# ============================================================================
# Predefined Scenarios (Q1a, Q1b, Q2)
# ============================================================================

def create_pure_elevator_scenario() -> ScenarioParams:
    """Create Pure Elevator scenario (Q1a)."""
    return ScenarioParams(
        name="Pure Elevator (Q1a)",
        elevator_fraction=1.0,
        rocket_fraction=0.0,
        elevator_predeployed=True,
        annual_elevator_throughput=5.37e5
    )


def create_pure_rocket_scenario() -> ScenarioParams:
    """Create Pure Rocket scenario (Q1b)."""
    return ScenarioParams(
        name="Pure Rocket (Q1b)",
        elevator_fraction=0.0,
        rocket_fraction=1.0,
        elevator_predeployed=False,
        annual_elevator_throughput=0
    )


def create_hybrid_scenario(elevator_frac: float = 0.10) -> ScenarioParams:
    """
    Create Hybrid scenario (Q2).
    
    Based on Q2 solution: ~90% rocket, ~10% elevator for robustness.
    """
    return ScenarioParams(
        name="Hybrid (Q2 Solution)",
        elevator_fraction=elevator_frac,
        rocket_fraction=1.0 - elevator_frac,
        elevator_predeployed=True,
        annual_elevator_throughput=5.37e5
    )


def get_standard_scenarios() -> List[ScenarioParams]:
    """Get list of standard scenarios for comparison."""
    return [
        create_pure_elevator_scenario(),
        create_pure_rocket_scenario(),
        create_hybrid_scenario()
    ]


# ============================================================================
# Report Generation
# ============================================================================

def generate_assessment_report(results: List[EnvironmentalAssessmentResult],
                               output_path: Optional[str] = None) -> str:
    """
    Generate formatted text report of environmental assessment.
    
    Args:
        results: List of assessment results
        output_path: Optional file path to save report
    
    Returns:
        Formatted report string
    """
    lines = []
    lines.append("=" * 80)
    lines.append("ENVIRONMENTAL IMPACT ASSESSMENT REPORT")
    lines.append("Space Transport System Comparison")
    lines.append("=" * 80)
    lines.append("")
    
    # Summary table
    lines.append("SCENARIO COMPARISON SUMMARY")
    lines.append("-" * 80)
    lines.append(f"{'Scenario':<25} {'Carbon Debt (Mt)':<18} {'Break-Even (yr)':<18} {'SEIS':<10} {'Grade':<8}")
    lines.append("-" * 80)
    
    for r in results:
        carbon_debt = f"{r.lca.E_construction_total:.1f}"
        if r.lca.break_even_years == float('inf'):
            be_str = "∞"
        else:
            be_str = f"{r.lca.break_even_years:.1f}"
        seis_str = f"{r.seis.SEIS:.2f}"
        grade = r.seis.grade
        
        lines.append(f"{r.scenario_name:<25} {carbon_debt:<18} {be_str:<18} {seis_str:<10} {grade:<8}")
    
    lines.append("-" * 80)
    lines.append("")
    
    # Detailed results
    for r in results:
        lines.append("=" * 80)
        lines.append(f"DETAILED ANALYSIS: {r.scenario_name}")
        lines.append("=" * 80)
        lines.append("")
        
        # Atmospheric
        lines.append("1. ATMOSPHERIC IMPACT (Environmental Damage Potential)")
        lines.append(f"   - Troposphere EDP:  {r.atmospheric.EDP_troposphere:,.0f}")
        lines.append(f"   - Stratosphere EDP: {r.atmospheric.EDP_stratosphere:,.0f}")
        lines.append(f"   - Mesosphere EDP:   {r.atmospheric.EDP_mesosphere:,.0f}")
        lines.append(f"   - TOTAL EDP:        {r.atmospheric.EDP_total:,.0f}")
        lines.append("")
        
        # Orbital
        lines.append("2. ORBITAL DEBRIS RISK (Kessler Syndrome Index)")
        lines.append(f"   - Initial Risk:     {r.orbital.initial_risk:.2f}")
        lines.append(f"   - Peak Risk:        {r.orbital.peak_risk:.2f}")
        lines.append(f"   - Final Risk:       {r.orbital.final_risk:.2f}")
        lines.append(f"   - Cascade Trigger:  {'YES' if r.orbital.cascade_triggered else 'NO'}")
        lines.append("")
        
        # LCA
        lines.append("3. LIFE CYCLE ASSESSMENT")
        lines.append(f"   Construction Phase:")
        lines.append(f"     - Infrastructure: {r.lca.E_infrastructure:.2f} Mt CO2")
        lines.append(f"     - Transport:      {r.lca.E_transport_construction:.2f} Mt CO2")
        lines.append(f"     - TOTAL:          {r.lca.E_construction_total:.2f} Mt CO2")
        lines.append(f"   Operations Phase (annual):")
        lines.append(f"     - Emissions:      {r.lca.E_operations_annual:.4f} Mt CO2/yr")
        lines.append(f"     - Colony Benefit: {r.lca.colony_reduction_annual:.4f} Mt CO2/yr")
        lines.append(f"     - Net Impact:     {r.lca.net_annual_impact:.4f} Mt CO2/yr")
        lines.append(f"   Sustainability:")
        lines.append(f"     - Break-Even:     {r.lca.break_even_years:.1f} years" 
                     if r.lca.break_even_years != float('inf') else "     - Break-Even:     Never")
        lines.append(f"     - Sustainable:    {'YES' if r.lca.is_sustainable else 'NO'}")
        lines.append("")
        
        # SEIS
        lines.append("4. SPACE ENVIRONMENT IMPACT SCORE (SEIS)")
        lines.append(f"   - Stratospheric:    {r.seis.stratospheric_score:.4f}")
        lines.append(f"   - Orbital Risk:     {r.seis.orbital_risk_score:.4f}")
        lines.append(f"   - Break-Even:       {r.seis.break_even_score:.4f}")
        lines.append(f"   - TOTAL SEIS:       {r.seis.SEIS:.2f}")
        lines.append(f"   - GRADE:            {r.seis.grade}")
        lines.append("")
    
    # Conclusions
    lines.append("=" * 80)
    lines.append("CONCLUSIONS AND RECOMMENDATIONS")
    lines.append("=" * 80)
    lines.append("")
    
    # Find best scenario
    best = min(results, key=lambda x: x.seis.SEIS)
    worst = max(results, key=lambda x: x.seis.SEIS)
    
    lines.append(f"1. BEST OPTION: {best.scenario_name}")
    lines.append(f"   - Achieves environmental break-even in {best.lca.break_even_years:.1f} years"
                 if best.lca.break_even_years != float('inf') else "   - Achieves sustainable operations")
    lines.append(f"   - SEIS Score: {best.seis.SEIS:.2f} (Grade: {best.seis.grade})")
    lines.append("")
    
    lines.append(f"2. WORST OPTION: {worst.scenario_name}")
    lines.append(f"   - Never achieves environmental break-even" 
                 if worst.lca.break_even_years == float('inf') 
                 else f"   - Break-even in {worst.lca.break_even_years:.1f} years")
    lines.append(f"   - SEIS Score: {worst.seis.SEIS:.2f} (Grade: {worst.seis.grade})")
    lines.append("")
    
    lines.append("3. KEY FINDINGS:")
    lines.append("   - Space elevator is essential for environmental sustainability")
    lines.append("   - Pure rocket approach has unsustainable carbon footprint")
    lines.append("   - Infrastructure-first policy is critical for long-term success")
    lines.append("")
    
    report = "\n".join(lines)
    
    # Save if path provided
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
    
    return report


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    import os
    
    print("=" * 80)
    print("Environmental Impact Assessment Model - Question 4")
    print("=" * 80)
    print()
    
    # Initialize model
    params = ModelParams()
    model = EnvironmentalAssessmentModel(params)
    
    # Get standard scenarios
    scenarios = get_standard_scenarios()
    
    # Run assessments
    print("Running environmental assessments...")
    results = model.compare_scenarios(scenarios)
    
    # Generate report
    output_dir = os.path.dirname(os.path.abspath(__file__))
    report_dir = os.path.join(os.path.dirname(output_dir), "mdFile")
    os.makedirs(report_dir, exist_ok=True)
    
    report_path = os.path.join(report_dir, "Q4_Environmental_Assessment_Report.txt")
    report = generate_assessment_report(results, report_path)
    
    # Print summary
    print("\n" + "=" * 80)
    print("ASSESSMENT COMPLETE")
    print("=" * 80)
    
    for r in results:
        print(f"\n{r.scenario_name}:")
        print(f"  Carbon Debt: {r.lca.E_construction_total:.1f} Mt CO2")
        print(f"  Break-Even: {r.lca.break_even_years:.1f} years" 
              if r.lca.break_even_years != float('inf') else "  Break-Even: Never")
        print(f"  SEIS Grade: {r.seis.grade}")
    
    print(f"\nFull report saved to: {report_path}")
    print()
    
    # Print numerical results for reference
    print("=" * 80)
    print("NUMERICAL RESULTS FOR PAPER")
    print("=" * 80)
    
    for r in results:
        summary = r.summary()
        print(f"\n{r.scenario_name}:")
        print(f"  LCA Results:")
        print(f"    E_infrastructure: {summary['lca']['E_infrastructure_Mt']:.2f} Mt")
        print(f"    E_transport: {summary['lca']['E_transport_construction_Mt']:.2f} Mt")
        print(f"    E_total_construction: {summary['lca']['E_construction_total_Mt']:.2f} Mt")
        print(f"    Break-even: {summary['lca']['break_even_years']:.1f} years"
              if summary['lca']['break_even_years'] != float('inf') else "    Break-even: ∞")
        print(f"  SEIS: {summary['seis']['SEIS']:.2f} ({summary['seis']['grade']})")
