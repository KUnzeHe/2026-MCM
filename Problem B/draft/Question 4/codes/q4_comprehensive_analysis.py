"""
Question 4: Comprehensive Environmental Impact Analysis
========================================================
综合环境影响评估 - 整合所有Q4分析模块

This comprehensive module integrates:
1. Core Environmental Impact Model (Atmospheric, Orbital, LCA, SEIS)
2. Capacity Constraint Analysis (Construction vs Operation phases)
3. Optimal Phased Strategy Analysis
4. Future Expansion Analysis (Mars, Venus, etc.)
5. Sensitivity and Monte Carlo Analysis

Author: Q4 Analysis Team
Date: 2026
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from scipy.integrate import quad, solve_ivp
from typing import Optional, List, Tuple, Dict, Any, Callable
import os
from datetime import datetime


# ============================================================================
# PART 1: DATA CLASSES AND PARAMETERS
# ============================================================================

@dataclass(frozen=True)
class AtmosphericParams:
    """Parameters for atmospheric impact modeling."""
    troposphere_height: float = 10.0      # km
    stratosphere_height: float = 50.0     # km
    mesosphere_height: float = 85.0       # km
    
    co2_emission_factor: float = 2.5      # kg CO2 / kg propellant
    nox_emission_factor: float = 0.01     # kg NOx / kg propellant
    bc_emission_factor: float = 0.001     # kg black carbon / kg propellant
    h2o_emission_factor: float = 1.2      # kg H2O / kg propellant
    
    ozone_depletion_factor: float = 0.05  # per kg NOx
    radiative_forcing_bc: float = 1.1     # W/m² per Tg BC


@dataclass(frozen=True)
class OrbitalRiskParams:
    """Parameters for orbital debris risk modeling."""
    alpha: float = 1e-5      # Debris generation rate per launch
    beta: float = 2e-3       # Cascade collision coefficient
    gamma: float = 0.02      # Natural orbital decay rate
    
    baseline_debris: float = 1.0    # Baseline debris level (normalized)
    critical_threshold: float = 5.0  # Kessler syndrome threshold


@dataclass(frozen=True)
class CarbonEmissionFactors:
    """Carbon emission factors for different systems."""
    rocket_per_launch: float = 2500.0     # tonnes CO2 per launch
    rocket_payload: float = 150.0         # tonnes payload per launch
    
    elevator_per_ton: float = 0.1         # tonnes CO2 per ton cargo (electric)
    elevator_infrastructure: float = 10e6  # tonnes CO2 for construction
    
    launch_site_construction: float = 1e6  # tonnes CO2 per site


@dataclass(frozen=True)
class ColonyParams:
    """Parameters for lunar colony."""
    target_population: int = 100000
    total_mass_tons: float = 1e8          # 100 Mt total transport
    annual_supply_per_capita: float = 3.0  # tons/person/year operations
    earth_emission_per_capita: float = 15.0  # tonnes CO2/year on Earth


@dataclass
class ScenarioParams:
    """Parameters for a transport scenario."""
    name: str
    elevator_fraction: float
    rocket_fraction: float
    elevator_predeployed: bool = True
    annual_elevator_throughput: float = 5.37e5  # tons/year
    construction_years: float = 24.0
    
    def __post_init__(self):
        total = self.elevator_fraction + self.rocket_fraction
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Fractions must sum to 1.0, got {total}")


@dataclass
class ModelParams:
    """Combined model parameters."""
    atmospheric: AtmosphericParams = field(default_factory=AtmosphericParams)
    orbital: OrbitalRiskParams = field(default_factory=OrbitalRiskParams)
    carbon: CarbonEmissionFactors = field(default_factory=CarbonEmissionFactors)
    colony: ColonyParams = field(default_factory=ColonyParams)


# ============================================================================
# PART 2: RESULT DATA CLASSES
# ============================================================================

@dataclass
class AtmosphericImpact:
    """Results from atmospheric impact assessment."""
    total_emissions: Dict[str, float]
    layer_deposition: Dict[str, Dict[str, float]]
    EDP_score: float  # Environmental Deposition Potential
    stratospheric_impact: float


@dataclass
class OrbitalRiskResult:
    """Results from orbital risk assessment."""
    peak_risk: float
    final_risk: float
    risk_trajectory: np.ndarray
    time_above_threshold: float
    cascade_probability: float


@dataclass
class LCAResult:
    """Life Cycle Assessment results."""
    E_infrastructure: float
    E_transport: float
    E_construction_total: float
    E_operations_annual: float
    colony_reduction_annual: float
    break_even_years: float
    is_sustainable: bool


@dataclass
class SEISResult:
    """Space Environment Impact Score results."""
    stratospheric_score: float
    orbital_score: float
    break_even_score: float
    SEIS: float
    grade: str


@dataclass
class EnvironmentalAssessmentResult:
    """Complete environmental assessment result."""
    scenario: ScenarioParams
    atmospheric: AtmosphericImpact
    orbital: OrbitalRiskResult
    lca: LCAResult
    seis: SEISResult


@dataclass
class CapacityAnalysisResult:
    """Capacity constraint analysis result."""
    elevator_max_24yr: float
    elevator_fraction_max: float
    rocket_required: float
    construction_feasible: bool
    operation_supply_annual: float
    operation_elevator_sufficient: bool
    utilization_rate: float


@dataclass
class PhasedStrategyResult:
    """Phased strategy analysis result."""
    construction_carbon: float
    operation_annual_carbon: float
    annual_reduction: float
    break_even_years: float
    cumulative_benefit_100yr: float
    cumulative_benefit_500yr: float


@dataclass
class ExpansionScenario:
    """Expansion scenario definition."""
    name: str
    colonies: List[Tuple[str, int, float]]  # (name, population, carbon_debt)
    description: str


@dataclass
class ExpansionResult:
    """Expansion analysis result."""
    scenario_name: str
    total_population: int
    total_carbon_debt: float
    annual_reduction: float
    break_even_years: float


# ============================================================================
# PART 3: CORE ENVIRONMENTAL MODELS
# ============================================================================

class AtmosphericImpactModel:
    """Model for calculating atmospheric layer impacts from rocket launches."""
    
    def __init__(self, params: ModelParams):
        self.params = params
        self.atm = params.atmospheric
    
    def calculate_launch_emissions(self, num_launches: int,
                                   propellant_mass_per_launch: float = 4000.0
                                   ) -> Dict[str, float]:
        """Calculate total emissions from rocket launches."""
        total_propellant = num_launches * propellant_mass_per_launch
        
        return {
            'CO2': total_propellant * self.atm.co2_emission_factor / 1000,
            'NOx': total_propellant * self.atm.nox_emission_factor / 1000,
            'BC': total_propellant * self.atm.bc_emission_factor / 1000,
            'H2O': total_propellant * self.atm.h2o_emission_factor / 1000
        }
    
    def calculate_layer_deposition(self, emissions: Dict[str, float]
                                   ) -> Dict[str, Dict[str, float]]:
        """Calculate emissions deposited in each atmospheric layer."""
        # Deposition fractions by layer
        fractions = {
            'troposphere': {'CO2': 0.7, 'NOx': 0.5, 'BC': 0.3, 'H2O': 0.3},
            'stratosphere': {'CO2': 0.25, 'NOx': 0.4, 'BC': 0.5, 'H2O': 0.5},
            'mesosphere': {'CO2': 0.05, 'NOx': 0.1, 'BC': 0.2, 'H2O': 0.2}
        }
        
        deposition = {}
        for layer, layer_fracs in fractions.items():
            deposition[layer] = {
                species: emissions[species] * frac
                for species, frac in layer_fracs.items()
            }
        return deposition
    
    def calculate_EDP(self, num_launches: int,
                      propellant_mass_per_launch: float = 4000.0
                      ) -> AtmosphericImpact:
        """Calculate Environmental Deposition Potential."""
        emissions = self.calculate_launch_emissions(num_launches, propellant_mass_per_launch)
        deposition = self.calculate_layer_deposition(emissions)
        
        # Impact weights
        weights = {'troposphere': 0.2, 'stratosphere': 0.5, 'mesosphere': 0.3}
        species_weights = {'CO2': 1.0, 'NOx': 100.0, 'BC': 1000.0, 'H2O': 0.1}
        
        EDP = 0.0
        for layer, layer_dep in deposition.items():
            layer_impact = sum(
                amount * species_weights.get(species, 1.0)
                for species, amount in layer_dep.items()
            )
            EDP += weights[layer] * layer_impact
        
        stratospheric = sum(deposition['stratosphere'].values())
        
        return AtmosphericImpact(
            total_emissions=emissions,
            layer_deposition=deposition,
            EDP_score=EDP,
            stratospheric_impact=stratospheric
        )


class OrbitalRiskModel:
    """Model for Kessler Syndrome risk assessment."""
    
    def __init__(self, params: ModelParams):
        self.params = params
        self.orb = params.orbital
    
    def risk_derivative(self, R: float, launch_rate: float) -> float:
        """Calculate dR/dt for debris risk."""
        generation = self.orb.alpha * launch_rate
        cascade = self.orb.beta * R ** 2
        decay = self.orb.gamma * R
        return generation + cascade - decay
    
    def simulate_risk_evolution(self, annual_launches: np.ndarray,
                                dt: float = 0.1) -> np.ndarray:
        """Simulate debris risk evolution over time."""
        n_years = len(annual_launches)
        n_steps = int(n_years / dt)
        
        R = np.zeros(n_steps)
        R[0] = self.orb.baseline_debris
        
        for i in range(1, n_steps):
            year_idx = min(int(i * dt), n_years - 1)
            launch_rate = annual_launches[year_idx]
            dR = self.risk_derivative(R[i-1], launch_rate)
            R[i] = max(0, R[i-1] + dR * dt)
        
        return R
    
    def analyze_scenario(self, total_launches: int,
                         duration_years: float,
                         elevator_active: bool = False) -> OrbitalRiskResult:
        """Analyze orbital risk for a scenario."""
        duration_years = max(duration_years, 1.0)
        
        if elevator_active and total_launches == 0:
            return OrbitalRiskResult(
                peak_risk=self.orb.baseline_debris,
                final_risk=self.orb.baseline_debris * 0.9,
                risk_trajectory=np.ones(int(duration_years * 10)) * self.orb.baseline_debris,
                time_above_threshold=0.0,
                cascade_probability=0.0
            )
        
        annual_rate = total_launches / duration_years
        annual_launches = np.ones(int(duration_years)) * annual_rate
        
        trajectory = self.simulate_risk_evolution(annual_launches)
        
        peak_risk = np.max(trajectory)
        final_risk = trajectory[-1]
        time_above = np.sum(trajectory > self.orb.critical_threshold) * 0.1
        
        cascade_prob = 1 - np.exp(-0.1 * max(0, peak_risk - self.orb.critical_threshold))
        
        return OrbitalRiskResult(
            peak_risk=peak_risk,
            final_risk=final_risk,
            risk_trajectory=trajectory,
            time_above_threshold=time_above,
            cascade_probability=cascade_prob
        )


class LifeCycleAssessmentModel:
    """Life Cycle Assessment model for space transport systems."""
    
    def __init__(self, params: ModelParams):
        self.params = params
        self.carbon = params.carbon
        self.colony = params.colony
    
    def calculate_infrastructure_emissions(self,
                                           elevator_built: bool,
                                           num_launch_sites: int = 0) -> float:
        """Calculate infrastructure construction emissions."""
        total = 0.0
        if elevator_built:
            total += self.carbon.elevator_infrastructure
        total += num_launch_sites * self.carbon.launch_site_construction
        return total / 1e6  # Convert to Mt
    
    def calculate_transport_emissions(self, mass_tons: float,
                                      elevator_fraction: float) -> float:
        """Calculate emissions from transporting cargo."""
        elevator_mass = mass_tons * elevator_fraction
        rocket_mass = mass_tons * (1 - elevator_fraction)
        
        elevator_emissions = elevator_mass * self.carbon.elevator_per_ton
        
        num_launches = rocket_mass / self.carbon.rocket_payload
        rocket_emissions = num_launches * self.carbon.rocket_per_launch
        
        return (elevator_emissions + rocket_emissions) / 1e6  # Mt
    
    def calculate_colony_reduction(self) -> float:
        """Calculate annual emission reduction from colony population."""
        return (self.colony.target_population * 
                self.colony.earth_emission_per_capita / 1e6)  # Mt/year
    
    def calculate_break_even(self, E_construction: float,
                             E_operations_annual: float,
                             colony_reduction_annual: float) -> Tuple[float, bool]:
        """Calculate break-even time for carbon payback."""
        net_annual = colony_reduction_annual - E_operations_annual
        
        if net_annual <= 0:
            return float('inf'), False
        
        break_even = E_construction / net_annual
        return break_even, True
    
    def assess_scenario(self, scenario: ScenarioParams) -> LCAResult:
        """Complete LCA assessment for a scenario."""
        E_infra = self.calculate_infrastructure_emissions(
            elevator_built=scenario.elevator_predeployed,
            num_launch_sites=3 if scenario.rocket_fraction > 0 else 0
        )
        
        E_transport = self.calculate_transport_emissions(
            self.colony.total_mass_tons,
            scenario.elevator_fraction
        )
        
        E_construction = E_infra + E_transport
        
        annual_supply = self.colony.target_population * self.colony.annual_supply_per_capita
        E_ops_annual = self.calculate_transport_emissions(annual_supply, 1.0) if scenario.elevator_fraction > 0 else \
                       self.calculate_transport_emissions(annual_supply, 0.0)
        
        colony_reduction = self.calculate_colony_reduction()
        break_even, sustainable = self.calculate_break_even(
            E_construction, E_ops_annual, colony_reduction
        )
        
        return LCAResult(
            E_infrastructure=E_infra,
            E_transport=E_transport,
            E_construction_total=E_construction,
            E_operations_annual=E_ops_annual,
            colony_reduction_annual=colony_reduction,
            break_even_years=break_even,
            is_sustainable=sustainable
        )


class SEISCalculator:
    """Space Environment Impact Score Calculator."""
    
    E_STRAT_REF = 1e9
    R_ORBITAL_REF = 5.0
    T_BE_REF = 100.0
    
    WEIGHTS = {'stratospheric': 0.30, 'orbital': 0.30, 'break_even': 0.40}
    
    GRADES = [
        (0.5, 'A+'), (1.0, 'A'), (1.5, 'B+'), (2.0, 'B'),
        (2.5, 'C+'), (3.0, 'C'), (4.0, 'D'), (float('inf'), 'F')
    ]
    
    @classmethod
    def calculate(cls, atmospheric: AtmosphericImpact,
                  orbital: OrbitalRiskResult,
                  lca: LCAResult) -> SEISResult:
        """Calculate SEIS score."""
        strat_score = min(10.0, atmospheric.stratospheric_impact / (cls.E_STRAT_REF / 1e6))
        orbital_score = min(10.0, orbital.peak_risk / cls.R_ORBITAL_REF)
        
        if lca.break_even_years == float('inf'):
            be_score = 10.0
        else:
            be_score = min(10.0, lca.break_even_years / cls.T_BE_REF)
        
        SEIS = (cls.WEIGHTS['stratospheric'] * strat_score +
                cls.WEIGHTS['orbital'] * orbital_score +
                cls.WEIGHTS['break_even'] * be_score)
        
        grade = 'F'
        for threshold, g in cls.GRADES:
            if SEIS < threshold:
                grade = g
                break
        
        return SEISResult(
            stratospheric_score=strat_score,
            orbital_score=orbital_score,
            break_even_score=be_score,
            SEIS=SEIS,
            grade=grade
        )


class EnvironmentalAssessmentModel:
    """Integrated environmental assessment model."""
    
    def __init__(self, params: Optional[ModelParams] = None):
        self.params = params or ModelParams()
        self.atmospheric_model = AtmosphericImpactModel(self.params)
        self.orbital_model = OrbitalRiskModel(self.params)
        self.lca_model = LifeCycleAssessmentModel(self.params)
    
    def assess_scenario(self, scenario: ScenarioParams) -> EnvironmentalAssessmentResult:
        """Complete environmental assessment for a scenario."""
        # Calculate rocket launches
        rocket_mass = self.params.colony.total_mass_tons * scenario.rocket_fraction
        num_launches = int(rocket_mass / self.params.carbon.rocket_payload)
        
        # Atmospheric impact
        atmospheric = self.atmospheric_model.calculate_EDP(num_launches)
        
        # Orbital risk
        duration = max(scenario.construction_years, 1.0)
        orbital = self.orbital_model.analyze_scenario(
            num_launches, duration,
            elevator_active=(scenario.elevator_fraction > 0)
        )
        
        # LCA
        lca = self.lca_model.assess_scenario(scenario)
        
        # SEIS
        seis = SEISCalculator.calculate(atmospheric, orbital, lca)
        
        return EnvironmentalAssessmentResult(
            scenario=scenario,
            atmospheric=atmospheric,
            orbital=orbital,
            lca=lca,
            seis=seis
        )
    
    def compare_scenarios(self, scenarios: List[ScenarioParams]
                          ) -> List[EnvironmentalAssessmentResult]:
        """Compare multiple scenarios."""
        return [self.assess_scenario(s) for s in scenarios]


# ============================================================================
# PART 4: CAPACITY CONSTRAINT ANALYSIS
# ============================================================================

class CapacityConstraintAnalyzer:
    """Analyze elevator capacity constraints."""
    
    def __init__(self, params: Optional[ModelParams] = None):
        self.params = params or ModelParams()
        self.colony = self.params.colony
    
    def analyze_construction_phase(self,
                                   construction_years: float = 24.0,
                                   elevator_annual: float = 5.37e5
                                   ) -> CapacityAnalysisResult:
        """Analyze capacity constraints during construction."""
        total_mass = self.colony.total_mass_tons
        
        # Maximum elevator transport in construction period
        elevator_max = elevator_annual * construction_years
        elevator_fraction_max = elevator_max / total_mass
        rocket_required = total_mass - elevator_max
        
        # Operation phase analysis
        operation_supply = self.colony.target_population * self.colony.annual_supply_per_capita
        utilization = operation_supply / elevator_annual
        
        return CapacityAnalysisResult(
            elevator_max_24yr=elevator_max,
            elevator_fraction_max=elevator_fraction_max,
            rocket_required=rocket_required,
            construction_feasible=(elevator_fraction_max < 1.0),
            operation_supply_annual=operation_supply,
            operation_elevator_sufficient=(operation_supply <= elevator_annual),
            utilization_rate=utilization
        )
    
    def analyze_supply_demand(self, population: int = 100000) -> Dict[str, float]:
        """Detailed supply demand estimation for operations."""
        # Water (from Q3 model - Baseline scenario)
        water_daily_per_person = 75  # L/day
        recycling_rate = 0.90
        water_annual = population * water_daily_per_person * (1 - recycling_rate) * 365 / 1000
        
        # Food (80% self-sufficiency)
        food_per_person = 800  # kg/year on Earth
        food_import = population * food_per_person * 0.20 / 1000
        
        # Equipment and supplies
        equipment = population * 80 / 1000  # 80 kg/person/year
        energy_materials = population * 20 / 1000  # 20 kg/person/year
        
        # Personnel rotation
        rotation_rate = 0.05
        personnel_cargo = population * rotation_rate * 50 / 1000
        
        total = water_annual + food_import + equipment + energy_materials + personnel_cargo
        
        return {
            'water': water_annual,
            'food': food_import,
            'equipment': equipment,
            'energy_materials': energy_materials,
            'personnel': personnel_cargo,
            'total': total
        }


# ============================================================================
# PART 5: PHASED STRATEGY ANALYSIS
# ============================================================================

class PhasedStrategyAnalyzer:
    """Analyze optimal phased transport strategy."""
    
    def __init__(self, params: Optional[ModelParams] = None):
        self.params = params or ModelParams()
        self.carbon = self.params.carbon
        self.colony = self.params.colony
    
    def analyze_phased_strategy(self,
                                construction_years: float = 24.0,
                                elevator_annual: float = 5.37e5
                                ) -> PhasedStrategyResult:
        """Analyze the optimal phased strategy."""
        total_mass = self.colony.total_mass_tons
        
        # Construction phase
        elev_mass = elevator_annual * construction_years
        rocket_mass = total_mass - elev_mass
        
        # Carbon emissions
        co2_per_ton_elev = self.carbon.elevator_per_ton
        co2_per_ton_rocket = self.carbon.rocket_per_launch / self.carbon.rocket_payload
        
        construction_elev = elev_mass * co2_per_ton_elev / 1e6
        construction_rocket = rocket_mass * co2_per_ton_rocket / 1e6
        construction_total = construction_elev + construction_rocket
        
        # Operation phase (pure elevator)
        operation_supply = self.colony.target_population * self.colony.annual_supply_per_capita
        operation_annual = operation_supply * co2_per_ton_elev / 1e6
        
        # Migration reduction
        annual_reduction = self.colony.target_population * self.colony.earth_emission_per_capita / 1e6
        
        # Break-even
        net_annual = annual_reduction - operation_annual
        break_even = construction_total / net_annual if net_annual > 0 else float('inf')
        
        # Long-term benefits
        benefit_100yr = net_annual * 100 - construction_total
        benefit_500yr = net_annual * 500 - construction_total
        
        return PhasedStrategyResult(
            construction_carbon=construction_total,
            operation_annual_carbon=operation_annual,
            annual_reduction=annual_reduction,
            break_even_years=break_even,
            cumulative_benefit_100yr=benefit_100yr,
            cumulative_benefit_500yr=benefit_500yr
        )
    
    def compare_strategies(self) -> Dict[str, Dict]:
        """Compare different transport strategies."""
        total_mass = self.colony.total_mass_tons
        co2_per_ton_elev = self.carbon.elevator_per_ton
        co2_per_ton_rocket = self.carbon.rocket_per_launch / self.carbon.rocket_payload
        annual_reduction = self.colony.target_population * self.colony.earth_emission_per_capita / 1e6
        operation_supply = self.colony.target_population * self.colony.annual_supply_per_capita
        
        strategies = {}
        
        # Strategy A: Pure Rocket
        carbon_a = total_mass * co2_per_ton_rocket / 1e6
        ops_a = operation_supply * co2_per_ton_rocket / 1e6
        strategies['A: Pure Rocket'] = {
            'construction': carbon_a,
            'operation_annual': ops_a,
            'break_even': float('inf') if ops_a >= annual_reduction else carbon_a / (annual_reduction - ops_a),
            'feasible': True
        }
        
        # Strategy B: Pure Elevator
        carbon_b = total_mass * co2_per_ton_elev / 1e6
        ops_b = operation_supply * co2_per_ton_elev / 1e6
        strategies['B: Pure Elevator'] = {
            'construction': carbon_b,
            'operation_annual': ops_b,
            'break_even': carbon_b / (annual_reduction - ops_b),
            'feasible': False  # Time constraint
        }
        
        # Strategy C: Phased
        phased = self.analyze_phased_strategy()
        strategies['C: Phased ★'] = {
            'construction': phased.construction_carbon,
            'operation_annual': phased.operation_annual_carbon,
            'break_even': phased.break_even_years,
            'feasible': True
        }
        
        return strategies


# ============================================================================
# PART 6: EXPANSION ANALYSIS
# ============================================================================

class ExpansionAnalyzer:
    """Analyze future space colonization expansion effects."""
    
    LEARNING_FACTOR = 0.7  # Carbon debt reduction per subsequent colony
    PER_CAPITA_EMISSION = 15  # tonnes CO2/person/year
    PER_CAPITA_OPERATION = 0.3  # tonnes CO2/person/year (low-carbon ops)
    
    def __init__(self, moon_carbon_debt: float = 1453.4):
        self.moon_debt = moon_carbon_debt
    
    def define_scenarios(self) -> List[ExpansionScenario]:
        """Define expansion scenarios."""
        return [
            ExpansionScenario(
                name='Scenario 1: Moon Only',
                colonies=[('Moon', 100000, self.moon_debt)],
                description='Current plan'
            ),
            ExpansionScenario(
                name='Scenario 2: Moon + Mars',
                colonies=[
                    ('Moon', 100000, self.moon_debt),
                    ('Mars', 200000, self.moon_debt * 0.7)
                ],
                description='Mars colonization in 2060s'
            ),
            ExpansionScenario(
                name='Scenario 3: Moon + Mars + Venus Orbit',
                colonies=[
                    ('Moon', 100000, self.moon_debt),
                    ('Mars', 200000, self.moon_debt * 0.7),
                    ('Venus Orbit', 50000, self.moon_debt * 0.5)
                ],
                description='Venus orbital colony in 2080s'
            ),
            ExpansionScenario(
                name='Scenario 4: Full Solar System',
                colonies=[
                    ('Moon', 100000, self.moon_debt),
                    ('Mars', 500000, self.moon_debt * 0.6),
                    ('Venus Orbit', 100000, self.moon_debt * 0.4),
                    ('Asteroid Belt', 50000, self.moon_debt * 0.3),
                    ('Europa', 30000, self.moon_debt * 0.3)
                ],
                description='22nd century vision'
            )
        ]
    
    def analyze_scenario(self, scenario: ExpansionScenario) -> ExpansionResult:
        """Analyze a single expansion scenario."""
        total_pop = sum(c[1] for c in scenario.colonies)
        total_debt = sum(c[2] for c in scenario.colonies)
        
        annual_reduction = total_pop * self.PER_CAPITA_EMISSION / 1e6
        annual_emission = total_pop * self.PER_CAPITA_OPERATION / 1e6
        net_annual = annual_reduction - annual_emission
        
        break_even = total_debt / net_annual if net_annual > 0 else float('inf')
        
        return ExpansionResult(
            scenario_name=scenario.name,
            total_population=total_pop,
            total_carbon_debt=total_debt,
            annual_reduction=net_annual,
            break_even_years=break_even
        )
    
    def analyze_all_scenarios(self) -> List[ExpansionResult]:
        """Analyze all expansion scenarios."""
        scenarios = self.define_scenarios()
        return [self.analyze_scenario(s) for s in scenarios]
    
    def calculate_improvement(self) -> Dict[str, float]:
        """Calculate improvement from expansion."""
        results = self.analyze_all_scenarios()
        moon_be = results[0].break_even_years
        full_be = results[-1].break_even_years
        
        return {
            'moon_only': moon_be,
            'full_expansion': full_be,
            'improvement_percent': (moon_be - full_be) / moon_be * 100
        }


# ============================================================================
# PART 7: SENSITIVITY ANALYSIS
# ============================================================================

@dataclass
class SensitivityResult:
    """Sensitivity analysis result."""
    parameter_name: str
    base_value: float
    values: np.ndarray
    seis_scores: np.ndarray
    carbon_debt: np.ndarray
    break_even_years: np.ndarray
    
    def get_elasticity(self) -> float:
        """Calculate elasticity at base value."""
        mid = len(self.values) // 2
        if mid == 0 or self.seis_scores[mid] == 0:
            return 0.0
        
        delta_seis = (self.seis_scores[-1] - self.seis_scores[0]) / self.seis_scores[mid]
        delta_param = (self.values[-1] - self.values[0]) / self.base_value
        
        return delta_seis / delta_param if delta_param != 0 else 0.0


class SensitivityAnalyzer:
    """Sensitivity analysis for environmental model."""
    
    def __init__(self, params: Optional[ModelParams] = None):
        self.params = params or ModelParams()
    
    def analyze_elevator_fraction(self, n_points: int = 20) -> SensitivityResult:
        """Analyze sensitivity to elevator fraction."""
        fractions = np.linspace(0.0, 1.0, n_points)
        model = EnvironmentalAssessmentModel(self.params)
        
        seis_scores = []
        carbon_debts = []
        break_evens = []
        
        for frac in fractions:
            scenario = ScenarioParams(
                name=f'Mix {frac:.0%}',
                elevator_fraction=frac,
                rocket_fraction=1 - frac,
                elevator_predeployed=True
            )
            result = model.assess_scenario(scenario)
            seis_scores.append(result.seis.SEIS)
            carbon_debts.append(result.lca.E_construction_total)
            break_evens.append(result.lca.break_even_years)
        
        return SensitivityResult(
            parameter_name='elevator_fraction',
            base_value=0.5,
            values=fractions,
            seis_scores=np.array(seis_scores),
            carbon_debt=np.array(carbon_debts),
            break_even_years=np.array(break_evens)
        )


# ============================================================================
# PART 8: COMPREHENSIVE ANALYSIS RUNNER
# ============================================================================

class ComprehensiveQ4Analysis:
    """Main class to run all Q4 analyses."""
    
    def __init__(self, params: Optional[ModelParams] = None):
        self.params = params or ModelParams()
        self.env_model = EnvironmentalAssessmentModel(self.params)
        self.capacity_analyzer = CapacityConstraintAnalyzer(self.params)
        self.phased_analyzer = PhasedStrategyAnalyzer(self.params)
        self.expansion_analyzer = ExpansionAnalyzer()
        self.sensitivity_analyzer = SensitivityAnalyzer(self.params)
        
        self.results = {}
    
    def run_core_assessment(self) -> List[EnvironmentalAssessmentResult]:
        """Run core environmental assessment."""
        scenarios = [
            ScenarioParams(
                name='Q1a: Pure Elevator',
                elevator_fraction=1.0,
                rocket_fraction=0.0,
                elevator_predeployed=True
            ),
            ScenarioParams(
                name='Q1b: Pure Rocket',
                elevator_fraction=0.0,
                rocket_fraction=1.0,
                elevator_predeployed=False
            ),
            ScenarioParams(
                name='Q2: Hybrid (13%E + 87%R)',
                elevator_fraction=0.129,
                rocket_fraction=0.871,
                elevator_predeployed=True
            )
        ]
        
        results = self.env_model.compare_scenarios(scenarios)
        self.results['core_assessment'] = results
        return results
    
    def run_capacity_analysis(self) -> CapacityAnalysisResult:
        """Run capacity constraint analysis."""
        result = self.capacity_analyzer.analyze_construction_phase()
        supply_demand = self.capacity_analyzer.analyze_supply_demand()
        
        self.results['capacity'] = {
            'analysis': result,
            'supply_demand': supply_demand
        }
        return result
    
    def run_phased_strategy_analysis(self) -> PhasedStrategyResult:
        """Run phased strategy analysis."""
        result = self.phased_analyzer.analyze_phased_strategy()
        comparison = self.phased_analyzer.compare_strategies()
        
        self.results['phased_strategy'] = {
            'analysis': result,
            'comparison': comparison
        }
        return result
    
    def run_expansion_analysis(self) -> List[ExpansionResult]:
        """Run expansion analysis."""
        results = self.expansion_analyzer.analyze_all_scenarios()
        improvement = self.expansion_analyzer.calculate_improvement()
        
        self.results['expansion'] = {
            'scenarios': results,
            'improvement': improvement
        }
        return results
    
    def run_sensitivity_analysis(self) -> SensitivityResult:
        """Run sensitivity analysis."""
        result = self.sensitivity_analyzer.analyze_elevator_fraction()
        self.results['sensitivity'] = result
        return result
    
    def run_complete_analysis(self) -> Dict[str, Any]:
        """Run all analyses."""
        print("=" * 80)
        print("QUESTION 4: COMPREHENSIVE ENVIRONMENTAL IMPACT ANALYSIS")
        print("=" * 80)
        print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        # Phase 1: Core Assessment
        print("=" * 70)
        print("PHASE 1: Core Environmental Assessment")
        print("=" * 70)
        core_results = self.run_core_assessment()
        
        print(f"\n{'Scenario':<30} {'Carbon Debt(Mt)':<16} {'Break-Even':<15} {'SEIS Grade':<10}")
        print("-" * 71)
        for r in core_results:
            be = f"{r.lca.break_even_years:.1f} yr" if r.lca.break_even_years < 10000 else "∞"
            print(f"{r.scenario.name:<30} {r.lca.E_construction_total:<16.1f} {be:<15} {r.seis.grade:<10}")
        
        # Phase 2: Capacity Analysis
        print("\n" + "=" * 70)
        print("PHASE 2: Capacity Constraint Analysis")
        print("=" * 70)
        capacity = self.run_capacity_analysis()
        
        print(f"\n【Construction Phase (2026-2050)】")
        print(f"  Elevator max capacity (24yr): {capacity.elevator_max_24yr/1e6:.1f} Mt")
        print(f"  Max elevator fraction: {capacity.elevator_fraction_max*100:.1f}%")
        print(f"  Rocket required: {capacity.rocket_required/1e6:.1f} Mt ({(1-capacity.elevator_fraction_max)*100:.1f}%)")
        
        print(f"\n【Operation Phase (2050+)】")
        print(f"  Annual supply demand: {capacity.operation_supply_annual/1e3:.0f} kt")
        print(f"  Elevator capacity: 537 kt")
        print(f"  Utilization rate: {capacity.utilization_rate*100:.1f}%")
        print(f"  Status: {'✓ Sufficient' if capacity.operation_elevator_sufficient else '✗ Insufficient'}")
        
        # Phase 3: Phased Strategy
        print("\n" + "=" * 70)
        print("PHASE 3: Optimal Phased Strategy Analysis")
        print("=" * 70)
        phased = self.run_phased_strategy_analysis()
        
        print(f"\n【Recommended Strategy: Construction (Hybrid) + Operation (Pure Elevator)】")
        print(f"\n  Construction Phase:")
        print(f"    Carbon debt: {phased.construction_carbon:.1f} Mt CO2")
        print(f"  Operation Phase:")
        print(f"    Annual carbon: {phased.operation_annual_carbon:.4f} Mt CO2 (near-zero)")
        print(f"    Annual reduction: {phased.annual_reduction:.2f} Mt CO2")
        print(f"\n  Break-even time: {phased.break_even_years:.1f} years")
        print(f"  100-year benefit: {phased.cumulative_benefit_100yr:.1f} Mt CO2 reduction")
        print(f"  500-year benefit: {phased.cumulative_benefit_500yr:.1f} Mt CO2 reduction")
        
        # Phase 4: Expansion Analysis
        print("\n" + "=" * 70)
        print("PHASE 4: Future Expansion Analysis")
        print("=" * 70)
        expansion = self.run_expansion_analysis()
        
        print(f"\n{'Scenario':<35} {'Population':<12} {'Carbon Debt':<15} {'Break-Even':<10}")
        print("-" * 72)
        for r in expansion:
            print(f"{r.scenario_name:<35} {r.total_population:<12,} {r.total_carbon_debt:<15.1f} {r.break_even_years:<10.0f}")
        
        improvement = self.results['expansion']['improvement']
        print(f"\n  Improvement from expansion: {improvement['improvement_percent']:.1f}%")
        print(f"  ({improvement['moon_only']:.0f} yr → {improvement['full_expansion']:.0f} yr)")
        
        # Phase 5: Sensitivity Analysis
        print("\n" + "=" * 70)
        print("PHASE 5: Sensitivity Analysis")
        print("=" * 70)
        sensitivity = self.run_sensitivity_analysis()
        
        print(f"\n  Elevator Fraction Sensitivity:")
        print(f"  {'Fraction':<12} {'Carbon Debt(Mt)':<18} {'Break-Even(yr)':<15} {'SEIS':<10}")
        print("  " + "-" * 55)
        for i in range(0, len(sensitivity.values), 4):
            frac = sensitivity.values[i]
            be = sensitivity.break_even_years[i]
            be_str = f"{be:.1f}" if be < 10000 else "∞"
            print(f"  {frac*100:>6.0f}%      {sensitivity.carbon_debt[i]:<18.1f} {be_str:<15} {sensitivity.seis_scores[i]:<10.2f}")
        
        # Summary
        print("\n" + "=" * 80)
        print("EXECUTIVE SUMMARY")
        print("=" * 80)
        print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│                     OPTIMAL TRANSPORT STRATEGY                               │
├─────────────────────────────────────────────────────────────────────────────┤
│  Phase          │  Period      │  Transport Mode        │  Carbon Impact    │
├─────────────────────────────────────────────────────────────────────────────┤
│  Construction   │  2026-2050   │  13% Elevator + 87% R  │  ~1453 Mt debt    │
│  Operation      │  2050+       │  100% Elevator         │  ≈0 Mt/year       │
└─────────────────────────────────────────────────────────────────────────────┘

KEY FINDINGS:
1. Construction phase rocket use is UNAVOIDABLE due to capacity constraints
2. Operation phase can be 100% zero-carbon with elevator
3. Break-even time: ~989 years (Moon only) → ~330 years (with expansion)
4. Long-term: Space elevator enables sustainable space colonization

RECOMMENDATION:
✓ Accept construction phase carbon debt as necessary infrastructure investment
✓ Plan for 100% elevator operations post-2050
✓ Factor in expansion to Mars/Venus for faster environmental payback
""")
        
        return self.results
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """Generate comprehensive analysis report."""
        if not self.results:
            self.run_complete_analysis()
        
        report_lines = [
            "=" * 80,
            "QUESTION 4: ENVIRONMENTAL IMPACT COMPREHENSIVE ANALYSIS REPORT",
            "=" * 80,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "1. EXECUTIVE SUMMARY",
            "-" * 40,
            f"Recommended Strategy: Phased Approach",
            f"  - Construction (2026-2050): 13% Elevator + 87% Rocket",
            f"  - Operation (2050+): 100% Elevator",
            "",
        ]
        
        # Core results
        if 'core_assessment' in self.results:
            report_lines.extend([
                "2. SCENARIO COMPARISON",
                "-" * 40,
                f"{'Scenario':<30} {'Carbon(Mt)':<12} {'Break-Even':<12} {'Grade':<8}",
            ])
            for r in self.results['core_assessment']:
                be = f"{r.lca.break_even_years:.1f}yr" if r.lca.break_even_years < 10000 else "∞"
                report_lines.append(
                    f"{r.scenario.name:<30} {r.lca.E_construction_total:<12.1f} {be:<12} {r.seis.grade:<8}"
                )
            report_lines.append("")
        
        # Phased strategy
        if 'phased_strategy' in self.results:
            ps = self.results['phased_strategy']['analysis']
            report_lines.extend([
                "3. PHASED STRATEGY DETAILS",
                "-" * 40,
                f"Construction Carbon Debt: {ps.construction_carbon:.1f} Mt",
                f"Operation Annual Carbon: {ps.operation_annual_carbon:.4f} Mt/yr",
                f"Annual Reduction: {ps.annual_reduction:.2f} Mt/yr",
                f"Break-even Time: {ps.break_even_years:.1f} years",
                f"100-year Net Benefit: {ps.cumulative_benefit_100yr:.1f} Mt",
                f"500-year Net Benefit: {ps.cumulative_benefit_500yr:.1f} Mt",
                "",
            ])
        
        # Expansion
        if 'expansion' in self.results:
            imp = self.results['expansion']['improvement']
            report_lines.extend([
                "4. EXPANSION BENEFITS",
                "-" * 40,
                f"Moon-only break-even: {imp['moon_only']:.0f} years",
                f"Full expansion break-even: {imp['full_expansion']:.0f} years",
                f"Improvement: {imp['improvement_percent']:.1f}%",
                "",
            ])
        
        report_lines.extend([
            "5. CONCLUSIONS",
            "-" * 40,
            "• Space elevator is the optimal long-term solution",
            "• Construction phase rocket use is unavoidable (capacity constraint)",
            "• Operation phase enables zero-carbon sustainable transport",
            "• Future expansion accelerates environmental payback",
            "",
            "=" * 80,
        ])
        
        report = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
        
        return report


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main entry point."""
    analysis = ComprehensiveQ4Analysis()
    results = analysis.run_complete_analysis()
    
    # Save report
    output_dir = os.path.dirname(os.path.abspath(__file__))
    report_dir = os.path.join(os.path.dirname(output_dir), "mdFile")
    os.makedirs(report_dir, exist_ok=True)
    
    report_path = os.path.join(report_dir, "Q4_Comprehensive_Analysis_Report.txt")
    analysis.generate_report(report_path)
    print(f"\nReport saved to: {report_path}")
    
    return results


if __name__ == "__main__":
    main()
