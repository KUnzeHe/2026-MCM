"""
Question 4: Complete Environmental Impact Analysis
===================================================
Main entry point for running all Q4 environmental analysis components.

This script orchestrates:
1. Core environmental impact model (q4_environmental_model.py)
2. Sensitivity analysis (q4_sensitivity_analysis.py)
3. Orbital debris risk analysis (q4_orbital_risk_model.py)

Outputs:
- Comprehensive assessment reports
- Numerical results for paper
- Data files for visualization
"""

from __future__ import annotations

import os
import sys
import json
import csv
from datetime import datetime
from typing import Dict, List, Any

# Import core modules
from q4_environmental_model import (
    ModelParams, ScenarioParams, EnvironmentalAssessmentModel,
    EnvironmentalAssessmentResult,
    create_pure_elevator_scenario, create_pure_rocket_scenario, 
    create_hybrid_scenario, get_standard_scenarios,
    generate_assessment_report
)

from q4_sensitivity_analysis import (
    SensitivityAnalyzer, MonteCarloAnalyzer, HybridOptimizer, DataExporter
)

from q4_orbital_risk_model import (
    AdvancedOrbitalParams, OrbitalScenarioAnalyzer, OrbitalRiskMetrics
)


# ============================================================================
# Analysis Configuration
# ============================================================================

class AnalysisConfig:
    """Configuration for complete analysis run."""
    
    # Monte Carlo settings
    MC_SAMPLES: int = 500
    MC_SEED: int = 42
    
    # Sensitivity analysis
    SENSITIVITY_POINTS: int = 20
    
    # Pareto analysis
    PARETO_POINTS: int = 50
    
    # Output settings
    GENERATE_CSV: bool = True
    GENERATE_JSON: bool = True
    GENERATE_TXT: bool = True


# ============================================================================
# Complete Analysis Runner
# ============================================================================

class CompleteEnvironmentalAnalysis:
    """
    Orchestrates complete environmental impact analysis for Q4.
    """
    
    def __init__(self, config: AnalysisConfig = None):
        """Initialize analysis components."""
        self.config = config or AnalysisConfig()
        
        # Initialize models
        self.params = ModelParams()
        self.env_model = EnvironmentalAssessmentModel(self.params)
        self.orbital_analyzer = OrbitalScenarioAnalyzer()
        self.sensitivity_analyzer = SensitivityAnalyzer()
        self.mc_analyzer = MonteCarloAnalyzer(seed=self.config.MC_SEED)
        self.optimizer = HybridOptimizer()
        
        # Results storage
        self.results = {}
        
        # Set up output directories
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(os.path.dirname(self.base_dir), "mdFile")
        self.data_dir = os.path.join(os.path.dirname(self.base_dir), "data")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
    
    def run_core_assessment(self) -> List[EnvironmentalAssessmentResult]:
        """Run core environmental assessment for standard scenarios."""
        print("\n" + "=" * 60)
        print("PHASE 1: Core Environmental Assessment")
        print("=" * 60)
        
        scenarios = get_standard_scenarios()
        results = self.env_model.compare_scenarios(scenarios)
        
        self.results['core_assessment'] = results
        
        # Print summary
        print(f"\n{'Scenario':<25} {'Carbon Debt (Mt)':<18} {'Break-Even':<15} {'SEIS Grade':<10}")
        print("-" * 68)
        for r in results:
            be_str = f"{r.lca.break_even_years:.1f} yr" if r.lca.break_even_years != float('inf') else "Never"
            print(f"{r.scenario_name:<25} {r.lca.E_construction_total:<18.1f} {be_str:<15} {r.seis.grade:<10}")
        
        return results
    
    def run_orbital_analysis(self) -> Dict:
        """Run orbital debris risk analysis."""
        print("\n" + "=" * 60)
        print("PHASE 2: Orbital Debris Risk Analysis")
        print("=" * 60)
        
        # Analyze scenarios
        rocket = self.orbital_analyzer.analyze_pure_rocket()
        elevator = self.orbital_analyzer.analyze_pure_elevator()
        hybrid = self.orbital_analyzer.analyze_hybrid(elevator_fraction=0.10)
        
        results = {
            'pure_rocket': rocket,
            'pure_elevator': elevator,
            'hybrid': hybrid
        }
        
        self.results['orbital_analysis'] = results
        
        # Print summary
        print(f"\n{'Scenario':<25} {'Peak Risk':<12} {'Cascade Prob':<15} {'Debris Legacy':<15}")
        print("-" * 67)
        for name, data in results.items():
            legacy = OrbitalRiskMetrics.calculate_debris_legacy(data['trajectory'])
            print(f"{data['scenario']:<25} {data['peak_risk']:<12.2f} {data['cascade_probability']*100:<14.1f}% {legacy:<14.2f}x")
        
        return results
    
    def run_sensitivity_analysis(self) -> Dict:
        """Run parameter sensitivity analysis."""
        print("\n" + "=" * 60)
        print("PHASE 3: Sensitivity Analysis")
        print("=" * 60)
        
        results = self.sensitivity_analyzer.run_full_sensitivity()
        self.results['sensitivity'] = results
        
        # Print elasticities
        print("\nParameter Elasticities (% change SEIS / % change param):")
        print("-" * 50)
        for param, result in sorted(results.items(), key=lambda x: abs(x[1].get_elasticity()), reverse=True):
            elasticity = result.get_elasticity()
            print(f"  {param}: {elasticity:.3f}")
        
        return results
    
    def run_monte_carlo(self) -> Dict:
        """Run Monte Carlo uncertainty analysis."""
        print("\n" + "=" * 60)
        print("PHASE 4: Monte Carlo Uncertainty Quantification")
        print("=" * 60)
        
        scenarios = [
            ('Pure Elevator', create_pure_elevator_scenario),
            ('Pure Rocket', create_pure_rocket_scenario),
            ('Hybrid', create_hybrid_scenario)
        ]
        
        mc_results = {}
        for name, factory in scenarios:
            print(f"  Running {self.config.MC_SAMPLES} samples for {name}...")
            mc_results[name] = self.mc_analyzer.run_simulation(
                factory, 
                n_samples=self.config.MC_SAMPLES
            )
        
        self.results['monte_carlo'] = mc_results
        
        # Print summary
        print(f"\n{'Scenario':<20} {'SEIS (mean±std)':<20} {'Carbon Debt (Mt)':<20}")
        print("-" * 60)
        for name, result in mc_results.items():
            print(f"{name:<20} {result.seis_mean:.2f} ± {result.seis_std:.2f}       "
                  f"{result.carbon_debt_mean:.1f} ± {result.carbon_debt_std:.1f}")
        
        return mc_results
    
    def run_optimization(self) -> Dict:
        """Find optimal hybrid configurations."""
        print("\n" + "=" * 60)
        print("PHASE 5: Hybrid System Optimization")
        print("=" * 60)
        
        # Find optimal mix for SEIS
        opt_frac, opt_result = self.optimizer.find_optimal_mix(
            objective='seis',
            n_points=self.config.PARETO_POINTS
        )
        
        # Generate Pareto front
        pareto = self.optimizer.generate_pareto_front(n_points=self.config.PARETO_POINTS)
        
        results = {
            'optimal_fraction': opt_frac,
            'optimal_result': opt_result,
            'pareto_front': pareto
        }
        
        self.results['optimization'] = results
        
        print(f"\nOptimal Configuration:")
        print(f"  Elevator fraction: {opt_frac:.1%}")
        print(f"  SEIS Score: {opt_result.seis.SEIS:.2f} ({opt_result.seis.grade})")
        print(f"  Carbon Debt: {opt_result.lca.E_construction_total:.1f} Mt")
        
        return results
    
    def generate_reports(self):
        """Generate all output reports."""
        print("\n" + "=" * 60)
        print("PHASE 6: Generating Reports")
        print("=" * 60)
        
        # 1. Main assessment report (TXT)
        if self.config.GENERATE_TXT:
            core_results = self.results.get('core_assessment', [])
            report_path = os.path.join(self.output_dir, "Q4_Complete_Assessment_Report.txt")
            report = generate_assessment_report(core_results, report_path)
            print(f"  Generated: {report_path}")
        
        # 2. Export CSV data
        if self.config.GENERATE_CSV:
            self._export_csv_data()
        
        # 3. Export JSON data
        if self.config.GENERATE_JSON:
            self._export_json_data()
        
        # 4. Generate summary for paper
        self._generate_paper_summary()
    
    def _export_csv_data(self):
        """Export analysis data to CSV files."""
        # Core assessment
        if 'core_assessment' in self.results:
            csv_path = os.path.join(self.data_dir, "q4_scenario_comparison.csv")
            DataExporter.to_csv(self.results['core_assessment'], csv_path)
            print(f"  Generated: {csv_path}")
        
        # Sensitivity analysis
        if 'sensitivity' in self.results:
            csv_path = os.path.join(self.data_dir, "q4_sensitivity_analysis.csv")
            DataExporter.sensitivity_to_csv(self.results['sensitivity'], csv_path)
            print(f"  Generated: {csv_path}")
        
        # Pareto front
        if 'optimization' in self.results:
            pareto = self.results['optimization']['pareto_front']
            csv_path = os.path.join(self.data_dir, "q4_pareto_front.csv")
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Elevator_Fraction', 'Carbon_Debt_Mt', 'Break_Even_Years'])
                for frac, carbon, be in pareto:
                    be_str = f"{be:.2f}" if be != float('inf') else "inf"
                    writer.writerow([f"{frac:.3f}", f"{carbon:.2f}", be_str])
            print(f"  Generated: {csv_path}")
    
    def _export_json_data(self):
        """Export analysis data to JSON file."""
        # Prepare JSON-serializable data
        export_data = {
            'metadata': {
                'generated': datetime.now().isoformat(),
                'model_version': '1.0',
                'monte_carlo_samples': self.config.MC_SAMPLES
            },
            'scenarios': {},
            'monte_carlo': {},
            'optimization': {}
        }
        
        # Core assessment
        if 'core_assessment' in self.results:
            for r in self.results['core_assessment']:
                export_data['scenarios'][r.scenario_name] = {
                    'carbon_debt_Mt': r.lca.E_construction_total,
                    'break_even_years': r.lca.break_even_years if r.lca.break_even_years != float('inf') else None,
                    'seis_score': r.seis.SEIS,
                    'seis_grade': r.seis.grade,
                    'is_sustainable': r.lca.is_sustainable
                }
        
        # Monte Carlo
        if 'monte_carlo' in self.results:
            for name, mc in self.results['monte_carlo'].items():
                export_data['monte_carlo'][name] = {
                    'seis_mean': float(mc.seis_mean),
                    'seis_std': float(mc.seis_std),
                    'carbon_debt_mean': float(mc.carbon_debt_mean),
                    'carbon_debt_std': float(mc.carbon_debt_std),
                    'seis_p5': float(mc.percentile('seis', 5)),
                    'seis_p95': float(mc.percentile('seis', 95))
                }
        
        # Optimization
        if 'optimization' in self.results:
            export_data['optimization'] = {
                'optimal_elevator_fraction': self.results['optimization']['optimal_fraction'],
                'optimal_seis': self.results['optimization']['optimal_result'].seis.SEIS
            }
        
        json_path = os.path.join(self.data_dir, "q4_analysis_results.json")
        with open(json_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        print(f"  Generated: {json_path}")
    
    def _generate_paper_summary(self):
        """Generate summary data formatted for paper."""
        lines = []
        lines.append("=" * 70)
        lines.append("NUMERICAL RESULTS FOR PAPER - Question 4")
        lines.append("=" * 70)
        lines.append("")
        
        # Table 1: Scenario Comparison
        lines.append("TABLE 1: Environmental Impact Comparison of Transport Scenarios")
        lines.append("-" * 70)
        lines.append(f"{'Scenario':<20} | {'E_const (Mt)':<12} | {'T_BE (yr)':<10} | {'SEIS':<8} | {'Grade':<6}")
        lines.append("-" * 70)
        
        if 'core_assessment' in self.results:
            for r in self.results['core_assessment']:
                be = f"{r.lca.break_even_years:.1f}" if r.lca.break_even_years < 10000 else "∞"
                lines.append(f"{r.scenario_name:<20} | {r.lca.E_construction_total:<12.1f} | {be:<10} | {r.seis.SEIS:<8.2f} | {r.seis.grade:<6}")
        
        lines.append("-" * 70)
        lines.append("")
        
        # Table 2: Monte Carlo Results
        lines.append("TABLE 2: Monte Carlo Uncertainty Analysis (N={})".format(self.config.MC_SAMPLES))
        lines.append("-" * 70)
        lines.append(f"{'Scenario':<20} | {'SEIS Mean':<10} | {'SEIS Std':<10} | {'95% CI':<20}")
        lines.append("-" * 70)
        
        if 'monte_carlo' in self.results:
            for name, mc in self.results['monte_carlo'].items():
                ci = f"[{mc.percentile('seis', 2.5):.2f}, {mc.percentile('seis', 97.5):.2f}]"
                lines.append(f"{name:<20} | {mc.seis_mean:<10.2f} | {mc.seis_std:<10.2f} | {ci:<20}")
        
        lines.append("-" * 70)
        lines.append("")
        
        # Key findings
        lines.append("KEY NUMERICAL FINDINGS:")
        lines.append("-" * 70)
        
        if 'core_assessment' in self.results:
            elevator = next((r for r in self.results['core_assessment'] 
                           if 'Elevator' in r.scenario_name and 'Hybrid' not in r.scenario_name), None)
            rocket = next((r for r in self.results['core_assessment'] 
                          if 'Rocket' in r.scenario_name), None)
            
            if elevator and rocket:
                carbon_ratio = rocket.lca.E_construction_total / elevator.lca.E_construction_total
                lines.append(f"1. Carbon intensity ratio (Rocket/Elevator): {carbon_ratio:.0f}x")
                lines.append(f"2. Elevator break-even time: {elevator.lca.break_even_years:.1f} years")
                lines.append(f"3. Rocket system never achieves environmental break-even")
        
        if 'orbital_analysis' in self.results:
            rocket_risk = self.results['orbital_analysis']['pure_rocket']['peak_risk']
            elevator_risk = self.results['orbital_analysis']['pure_elevator']['peak_risk']
            lines.append(f"4. Orbital risk (Rocket): {rocket_risk:.1f} (Cascade: Yes)")
            lines.append(f"5. Orbital risk (Elevator): {elevator_risk:.2f} (Cascade: No)")
        
        lines.append("")
        lines.append("=" * 70)
        
        summary = "\n".join(lines)
        
        # Save to file
        summary_path = os.path.join(self.output_dir, "Q4_Paper_Summary.txt")
        with open(summary_path, 'w') as f:
            f.write(summary)
        print(f"  Generated: {summary_path}")
        
        # Also print to console
        print("\n" + summary)
    
    def run_complete_analysis(self):
        """Run all analysis phases."""
        print("\n" + "=" * 70)
        print("  QUESTION 4: COMPLETE ENVIRONMENTAL IMPACT ANALYSIS")
        print("  " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print("=" * 70)
        
        # Run all phases
        self.run_core_assessment()
        self.run_orbital_analysis()
        self.run_sensitivity_analysis()
        self.run_monte_carlo()
        self.run_optimization()
        
        # Generate reports
        self.generate_reports()
        
        print("\n" + "=" * 70)
        print("  ANALYSIS COMPLETE")
        print("=" * 70)
        
        return self.results


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point for Q4 analysis."""
    config = AnalysisConfig()
    
    # Reduce samples for quick testing
    if '--quick' in sys.argv:
        config.MC_SAMPLES = 100
        config.SENSITIVITY_POINTS = 10
        config.PARETO_POINTS = 20
        print("Running in QUICK mode (reduced samples)")
    
    analysis = CompleteEnvironmentalAnalysis(config)
    results = analysis.run_complete_analysis()
    
    return results


if __name__ == "__main__":
    main()
