"""
Environmental Impact Sensitivity Analysis Module
================================================
Sensitivity analysis and parameter studies for Q4 Environmental Model.

This module provides:
1. Parameter sensitivity analysis
2. Monte Carlo uncertainty quantification
3. Scenario comparison utilities
4. Data export functions
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Callable
import json
import csv
import os

from q4_environmental_model import (
    ModelParams, ScenarioParams, EnvironmentalAssessmentModel,
    EnvironmentalAssessmentResult, LCAResult, SEISResult,
    AtmosphericParams, OrbitalRiskParams, CarbonEmissionFactors, ColonyParams,
    create_pure_elevator_scenario, create_pure_rocket_scenario, create_hybrid_scenario
)


# ============================================================================
# Sensitivity Analysis
# ============================================================================

@dataclass
class SensitivityResult:
    """Result from parameter sensitivity analysis."""
    parameter_name: str
    base_value: float
    values: np.ndarray
    break_even_years: np.ndarray
    seis_scores: np.ndarray
    carbon_debt: np.ndarray
    
    def get_elasticity(self) -> float:
        """Calculate elasticity (% change in output / % change in input)."""
        # Use central values for elasticity calculation
        mid_idx = len(self.values) // 2
        if mid_idx == 0:
            return 0.0
        
        delta_param = (self.values[-1] - self.values[0]) / self.base_value
        delta_seis = (self.seis_scores[-1] - self.seis_scores[0]) / self.seis_scores[mid_idx]
        
        if delta_param == 0:
            return 0.0
        return delta_seis / delta_param


class SensitivityAnalyzer:
    """
    Perform sensitivity analysis on environmental model parameters.
    """
    
    def __init__(self, base_params: Optional[ModelParams] = None):
        """Initialize with base parameters."""
        self.base_params = base_params or ModelParams()
    
    def analyze_parameter(self, 
                          param_name: str,
                          param_range: Tuple[float, float],
                          n_points: int = 20,
                          scenario_factory: Callable[[], ScenarioParams] = None
                          ) -> SensitivityResult:
        """
        Analyze sensitivity to a single parameter.
        
        Args:
            param_name: Name of parameter to vary (e.g., 'carbon.CO2_per_rocket_launch')
            param_range: (min, max) values for parameter
            n_points: Number of sample points
            scenario_factory: Function to create scenario (default: hybrid)
        
        Returns:
            SensitivityResult with analysis data
        """
        if scenario_factory is None:
            scenario_factory = create_hybrid_scenario
        
        values = np.linspace(param_range[0], param_range[1], n_points)
        break_evens = np.zeros(n_points)
        seis_scores = np.zeros(n_points)
        carbon_debts = np.zeros(n_points)
        
        base_value = self._get_param_value(param_name)
        
        for i, val in enumerate(values):
            # Create modified parameters
            params = self._create_modified_params(param_name, val)
            
            # Run assessment
            model = EnvironmentalAssessmentModel(params)
            scenario = scenario_factory()
            result = model.assess_scenario(scenario)
            
            break_evens[i] = result.lca.break_even_years
            seis_scores[i] = result.seis.SEIS
            carbon_debts[i] = result.lca.E_construction_total
        
        return SensitivityResult(
            parameter_name=param_name,
            base_value=base_value,
            values=values,
            break_even_years=break_evens,
            seis_scores=seis_scores,
            carbon_debt=carbon_debts
        )
    
    def _get_param_value(self, param_name: str) -> float:
        """Get current value of a parameter by name."""
        parts = param_name.split('.')
        obj = self.base_params
        
        for part in parts:
            obj = getattr(obj, part)
        
        return obj
    
    def _create_modified_params(self, param_name: str, value: float) -> ModelParams:
        """Create a copy of params with one parameter modified."""
        parts = param_name.split('.')
        
        # Deep copy approach - recreate with modified values
        if parts[0] == 'carbon':
            carbon_dict = {
                'CO2_per_rocket_launch': self.base_params.carbon.CO2_per_rocket_launch,
                'rocket_payload': self.base_params.carbon.rocket_payload,
                'CO2_elevator_per_ton': self.base_params.carbon.CO2_elevator_per_ton,
                'elevator_construction_Mt': self.base_params.carbon.elevator_construction_Mt,
                'elevator_transport_factor': self.base_params.carbon.elevator_transport_factor,
                'rocket_transport_factor': self.base_params.carbon.rocket_transport_factor,
                'launch_site_construction': self.base_params.carbon.launch_site_construction
            }
            carbon_dict[parts[1]] = value
            return ModelParams(
                atmospheric=self.base_params.atmospheric,
                orbital=self.base_params.orbital,
                carbon=CarbonEmissionFactors(**carbon_dict),
                colony=self.base_params.colony
            )
        
        elif parts[0] == 'colony':
            colony_dict = {
                'population': self.base_params.colony.population,
                'earth_per_capita_emission': self.base_params.colony.earth_per_capita_emission,
                'colony_start_year': self.base_params.colony.colony_start_year,
                'construction_start_year': self.base_params.colony.construction_start_year,
                'total_mass_tons': self.base_params.colony.total_mass_tons
            }
            colony_dict[parts[1]] = value
            return ModelParams(
                atmospheric=self.base_params.atmospheric,
                orbital=self.base_params.orbital,
                carbon=self.base_params.carbon,
                colony=ColonyParams(**colony_dict)
            )
        
        elif parts[0] == 'atmospheric':
            atm_dict = {
                'W_troposphere': self.base_params.atmospheric.W_troposphere,
                'W_stratosphere': self.base_params.atmospheric.W_stratosphere,
                'W_mesosphere': self.base_params.atmospheric.W_mesosphere,
                'CO2_fraction': self.base_params.atmospheric.CO2_fraction,
                'NOx_fraction': self.base_params.atmospheric.NOx_fraction,
                'BC_fraction': self.base_params.atmospheric.BC_fraction,
                'Al2O3_fraction': self.base_params.atmospheric.Al2O3_fraction,
                'H2O_fraction': self.base_params.atmospheric.H2O_fraction,
                'transit_time_min': self.base_params.atmospheric.transit_time_min,
                'stratosphere_deposition': self.base_params.atmospheric.stratosphere_deposition
            }
            atm_dict[parts[1]] = value
            return ModelParams(
                atmospheric=AtmosphericParams(**atm_dict),
                orbital=self.base_params.orbital,
                carbon=self.base_params.carbon,
                colony=self.base_params.colony
            )
        
        return self.base_params
    
    def run_full_sensitivity(self) -> Dict[str, SensitivityResult]:
        """
        Run sensitivity analysis on all key parameters.
        
        Returns:
            Dictionary of parameter name -> SensitivityResult
        """
        # Define parameter ranges
        param_configs = [
            ('carbon.CO2_per_rocket_launch', (1500, 4000)),
            ('carbon.elevator_construction_Mt', (2, 10)),
            ('carbon.elevator_transport_factor', (0.05, 0.2)),
            ('colony.population', (50000, 200000)),
            ('colony.earth_per_capita_emission', (8, 25)),
            ('atmospheric.W_stratosphere', (200, 800)),
        ]
        
        results = {}
        for param_name, param_range in param_configs:
            try:
                results[param_name] = self.analyze_parameter(param_name, param_range)
            except Exception as e:
                print(f"Warning: Could not analyze {param_name}: {e}")
        
        return results


# ============================================================================
# Monte Carlo Uncertainty Analysis
# ============================================================================

@dataclass
class MonteCarloResult:
    """Results from Monte Carlo simulation."""
    n_samples: int
    break_even_samples: np.ndarray
    seis_samples: np.ndarray
    carbon_debt_samples: np.ndarray
    
    @property
    def break_even_mean(self) -> float:
        finite = self.break_even_samples[np.isfinite(self.break_even_samples)]
        return np.mean(finite) if len(finite) > 0 else float('inf')
    
    @property
    def break_even_std(self) -> float:
        finite = self.break_even_samples[np.isfinite(self.break_even_samples)]
        return np.std(finite) if len(finite) > 0 else float('inf')
    
    @property
    def seis_mean(self) -> float:
        return np.mean(self.seis_samples)
    
    @property
    def seis_std(self) -> float:
        return np.std(self.seis_samples)
    
    @property
    def carbon_debt_mean(self) -> float:
        return np.mean(self.carbon_debt_samples)
    
    @property
    def carbon_debt_std(self) -> float:
        return np.std(self.carbon_debt_samples)
    
    def percentile(self, metric: str, p: float) -> float:
        """Get percentile of a metric."""
        if metric == 'break_even':
            finite = self.break_even_samples[np.isfinite(self.break_even_samples)]
            return np.percentile(finite, p) if len(finite) > 0 else float('inf')
        elif metric == 'seis':
            return np.percentile(self.seis_samples, p)
        elif metric == 'carbon_debt':
            return np.percentile(self.carbon_debt_samples, p)
        return 0.0


class MonteCarloAnalyzer:
    """
    Monte Carlo uncertainty quantification for environmental model.
    """
    
    def __init__(self, base_params: Optional[ModelParams] = None, seed: int = 42):
        """Initialize with base parameters and random seed."""
        self.base_params = base_params or ModelParams()
        self.rng = np.random.default_rng(seed)
    
    def run_simulation(self, 
                       scenario_factory: Callable[[], ScenarioParams],
                       n_samples: int = 1000,
                       uncertainty_config: Optional[Dict] = None
                       ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation with parameter uncertainty.
        
        Args:
            scenario_factory: Function to create scenario
            n_samples: Number of Monte Carlo samples
            uncertainty_config: Dict of parameter -> (distribution, params)
        
        Returns:
            MonteCarloResult with sample distributions
        """
        if uncertainty_config is None:
            uncertainty_config = self._default_uncertainty()
        
        break_evens = np.zeros(n_samples)
        seis_scores = np.zeros(n_samples)
        carbon_debts = np.zeros(n_samples)
        
        for i in range(n_samples):
            # Sample parameters
            params = self._sample_params(uncertainty_config)
            
            # Run assessment
            model = EnvironmentalAssessmentModel(params)
            scenario = scenario_factory()
            result = model.assess_scenario(scenario)
            
            break_evens[i] = result.lca.break_even_years
            seis_scores[i] = result.seis.SEIS
            carbon_debts[i] = result.lca.E_construction_total
        
        return MonteCarloResult(
            n_samples=n_samples,
            break_even_samples=break_evens,
            seis_samples=seis_scores,
            carbon_debt_samples=carbon_debts
        )
    
    def _default_uncertainty(self) -> Dict:
        """Get default parameter uncertainty specification."""
        return {
            'carbon.CO2_per_rocket_launch': ('normal', (2500, 500)),
            'carbon.elevator_construction_Mt': ('normal', (5.0, 1.0)),
            'colony.earth_per_capita_emission': ('normal', (15.0, 3.0)),
            'atmospheric.W_stratosphere': ('uniform', (400, 600)),
        }
    
    def _sample_params(self, config: Dict) -> ModelParams:
        """Sample a set of parameters from uncertainty distributions."""
        # Start with base params
        carbon_dict = {
            'CO2_per_rocket_launch': self.base_params.carbon.CO2_per_rocket_launch,
            'rocket_payload': self.base_params.carbon.rocket_payload,
            'CO2_elevator_per_ton': self.base_params.carbon.CO2_elevator_per_ton,
            'elevator_construction_Mt': self.base_params.carbon.elevator_construction_Mt,
            'elevator_transport_factor': self.base_params.carbon.elevator_transport_factor,
            'rocket_transport_factor': self.base_params.carbon.rocket_transport_factor,
            'launch_site_construction': self.base_params.carbon.launch_site_construction
        }
        
        colony_dict = {
            'population': self.base_params.colony.population,
            'earth_per_capita_emission': self.base_params.colony.earth_per_capita_emission,
            'colony_start_year': self.base_params.colony.colony_start_year,
            'construction_start_year': self.base_params.colony.construction_start_year,
            'total_mass_tons': self.base_params.colony.total_mass_tons
        }
        
        atm_dict = {
            'W_troposphere': self.base_params.atmospheric.W_troposphere,
            'W_stratosphere': self.base_params.atmospheric.W_stratosphere,
            'W_mesosphere': self.base_params.atmospheric.W_mesosphere,
            'CO2_fraction': self.base_params.atmospheric.CO2_fraction,
            'NOx_fraction': self.base_params.atmospheric.NOx_fraction,
            'BC_fraction': self.base_params.atmospheric.BC_fraction,
            'Al2O3_fraction': self.base_params.atmospheric.Al2O3_fraction,
            'H2O_fraction': self.base_params.atmospheric.H2O_fraction,
            'transit_time_min': self.base_params.atmospheric.transit_time_min,
            'stratosphere_deposition': self.base_params.atmospheric.stratosphere_deposition
        }
        
        # Sample each parameter
        for param_name, (dist, params) in config.items():
            parts = param_name.split('.')
            
            if dist == 'normal':
                value = self.rng.normal(params[0], params[1])
                value = max(0, value)  # Ensure non-negative
            elif dist == 'uniform':
                value = self.rng.uniform(params[0], params[1])
            elif dist == 'lognormal':
                value = self.rng.lognormal(params[0], params[1])
            else:
                continue
            
            # Update appropriate dict
            if parts[0] == 'carbon':
                carbon_dict[parts[1]] = value
            elif parts[0] == 'colony':
                if parts[1] == 'population':
                    value = int(value)
                colony_dict[parts[1]] = value
            elif parts[0] == 'atmospheric':
                atm_dict[parts[1]] = value
        
        return ModelParams(
            atmospheric=AtmosphericParams(**atm_dict),
            orbital=self.base_params.orbital,
            carbon=CarbonEmissionFactors(**carbon_dict),
            colony=ColonyParams(**colony_dict)
        )


# ============================================================================
# Hybrid Scenario Optimization
# ============================================================================

class HybridOptimizer:
    """
    Find optimal elevator/rocket mix for different objectives.
    """
    
    def __init__(self, params: Optional[ModelParams] = None):
        """Initialize with model parameters."""
        self.params = params or ModelParams()
    
    def find_optimal_mix(self, 
                         objective: str = 'seis',
                         elevator_range: Tuple[float, float] = (0.0, 1.0),
                         n_points: int = 50) -> Tuple[float, EnvironmentalAssessmentResult]:
        """
        Find optimal elevator fraction for given objective.
        
        Args:
            objective: 'seis', 'break_even', or 'carbon_debt'
            elevator_range: (min, max) elevator fraction
            n_points: Number of search points
        
        Returns:
            Tuple of (optimal_fraction, result)
        """
        fractions = np.linspace(elevator_range[0], elevator_range[1], n_points)
        best_value = float('inf')
        best_fraction = 0.5
        best_result = None
        
        model = EnvironmentalAssessmentModel(self.params)
        
        for frac in fractions:
            scenario = ScenarioParams(
                name=f"Mix_{frac:.2f}",
                elevator_fraction=frac,
                rocket_fraction=1.0 - frac,
                elevator_predeployed=True,
                annual_elevator_throughput=5.37e5
            )
            
            result = model.assess_scenario(scenario)
            
            if objective == 'seis':
                value = result.seis.SEIS
            elif objective == 'break_even':
                value = result.lca.break_even_years
            elif objective == 'carbon_debt':
                value = result.lca.E_construction_total
            else:
                value = result.seis.SEIS
            
            if value < best_value:
                best_value = value
                best_fraction = frac
                best_result = result
        
        return best_fraction, best_result
    
    def generate_pareto_front(self, 
                              n_points: int = 50
                              ) -> List[Tuple[float, float, float]]:
        """
        Generate Pareto front for time vs carbon tradeoff.
        
        Returns:
            List of (elevator_fraction, carbon_debt, break_even) tuples
        """
        fractions = np.linspace(0.0, 1.0, n_points)
        results = []
        
        model = EnvironmentalAssessmentModel(self.params)
        
        for frac in fractions:
            scenario = ScenarioParams(
                name=f"Mix_{frac:.2f}",
                elevator_fraction=frac,
                rocket_fraction=1.0 - frac,
                elevator_predeployed=True,
                annual_elevator_throughput=5.37e5
            )
            
            result = model.assess_scenario(scenario)
            results.append((
                frac,
                result.lca.E_construction_total,
                result.lca.break_even_years
            ))
        
        return results


# ============================================================================
# Data Export Functions
# ============================================================================

class DataExporter:
    """Export analysis results to various formats."""
    
    @staticmethod
    def to_csv(results: List[EnvironmentalAssessmentResult], 
               filepath: str) -> None:
        """Export results to CSV file."""
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'Scenario', 'E_infrastructure_Mt', 'E_transport_Mt', 
                'E_total_Mt', 'E_ops_annual_Mt', 'Break_Even_Years',
                'SEIS_Score', 'SEIS_Grade', 'Is_Sustainable'
            ])
            
            # Data rows
            for r in results:
                writer.writerow([
                    r.scenario_name,
                    f"{r.lca.E_infrastructure:.2f}",
                    f"{r.lca.E_transport_construction:.2f}",
                    f"{r.lca.E_construction_total:.2f}",
                    f"{r.lca.E_operations_annual:.4f}",
                    f"{r.lca.break_even_years:.1f}" if r.lca.break_even_years != float('inf') else "inf",
                    f"{r.seis.SEIS:.2f}",
                    r.seis.grade,
                    r.lca.is_sustainable
                ])
    
    @staticmethod
    def to_json(results: List[EnvironmentalAssessmentResult],
                filepath: str) -> None:
        """Export results to JSON file."""
        data = [r.summary() for r in results]
        
        # Handle infinity values
        def replace_inf(obj):
            if isinstance(obj, dict):
                return {k: replace_inf(v) for k, v in obj.items()}
            elif isinstance(obj, float) and not np.isfinite(obj):
                return "infinity"
            return obj
        
        data = replace_inf(data)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    @staticmethod
    def sensitivity_to_csv(results: Dict[str, SensitivityResult],
                           filepath: str) -> None:
        """Export sensitivity results to CSV."""
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            for param_name, result in results.items():
                writer.writerow([f"Parameter: {param_name}"])
                writer.writerow(['Value', 'SEIS', 'Break_Even', 'Carbon_Debt'])
                
                for i in range(len(result.values)):
                    be = result.break_even_years[i]
                    writer.writerow([
                        f"{result.values[i]:.4f}",
                        f"{result.seis_scores[i]:.4f}",
                        f"{be:.1f}" if np.isfinite(be) else "inf",
                        f"{result.carbon_debt[i]:.2f}"
                    ])
                writer.writerow([])


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Environmental Impact Sensitivity Analysis")
    print("=" * 80)
    print()
    
    # Initialize analyzers
    sensitivity_analyzer = SensitivityAnalyzer()
    mc_analyzer = MonteCarloAnalyzer()
    optimizer = HybridOptimizer()
    
    # 1. Run sensitivity analysis
    print("1. Running Parameter Sensitivity Analysis...")
    sensitivity_results = sensitivity_analyzer.run_full_sensitivity()
    
    print("\n   Parameter Elasticities:")
    for param, result in sensitivity_results.items():
        elasticity = result.get_elasticity()
        print(f"   - {param}: {elasticity:.3f}")
    
    # 2. Run Monte Carlo for each scenario
    print("\n2. Running Monte Carlo Uncertainty Analysis...")
    scenarios = [
        ('Pure Elevator', create_pure_elevator_scenario),
        ('Pure Rocket', create_pure_rocket_scenario),
        ('Hybrid', create_hybrid_scenario)
    ]
    
    mc_results = {}
    for name, factory in scenarios:
        print(f"   Processing {name}...")
        mc_results[name] = mc_analyzer.run_simulation(factory, n_samples=500)
    
    print("\n   Monte Carlo Results (95% CI):")
    for name, result in mc_results.items():
        print(f"   {name}:")
        print(f"     SEIS: {result.seis_mean:.2f} ± {result.seis_std:.2f}")
        print(f"     Carbon Debt: {result.carbon_debt_mean:.1f} ± {result.carbon_debt_std:.1f} Mt")
    
    # 3. Find optimal hybrid mix
    print("\n3. Finding Optimal Hybrid Mix...")
    opt_frac, opt_result = optimizer.find_optimal_mix(objective='seis')
    print(f"   Optimal elevator fraction for minimum SEIS: {opt_frac:.2%}")
    print(f"   Resulting SEIS: {opt_result.seis.SEIS:.2f} ({opt_result.seis.grade})")
    
    # 4. Generate Pareto front
    print("\n4. Generating Pareto Front...")
    pareto = optimizer.generate_pareto_front(n_points=20)
    
    print("   Elevator% | Carbon Debt | Break-Even")
    print("   " + "-" * 40)
    for frac, carbon, be in pareto[::4]:  # Print every 4th point
        be_str = f"{be:.1f}" if np.isfinite(be) else "∞"
        print(f"   {frac*100:6.1f}%  | {carbon:10.1f} Mt | {be_str} years")
    
    # 5. Export data
    print("\n5. Exporting Results...")
    output_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(output_dir), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    # Export sensitivity results
    DataExporter.sensitivity_to_csv(
        sensitivity_results,
        os.path.join(data_dir, "sensitivity_analysis.csv")
    )
    
    print(f"   Data exported to {data_dir}")
    
    print("\n" + "=" * 80)
    print("Analysis Complete")
    print("=" * 80)
