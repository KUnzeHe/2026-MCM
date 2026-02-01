"""
Orbital Debris Risk Dynamics Module
====================================
Detailed Kessler Syndrome risk modeling for Q4.

This module provides:
1. Advanced orbital debris dynamics simulation
2. Risk trajectory analysis
3. Mitigation strategy evaluation
4. Temporal evolution modeling
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from scipy.integrate import solve_ivp
from typing import Optional, List, Tuple, Dict, Callable
import warnings

from q4_environmental_model import OrbitalRiskParams, ModelParams


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class DebrisState:
    """State of orbital debris environment at a point in time."""
    time: float                    # Years from start
    risk_index: float              # Normalized risk
    debris_count: float            # Estimated trackable objects
    collision_rate: float          # Collisions per year
    cascade_probability: float     # Probability of cascade onset
    
    def to_dict(self) -> Dict:
        return {
            'time': self.time,
            'risk_index': self.risk_index,
            'debris_count': self.debris_count,
            'collision_rate': self.collision_rate,
            'cascade_probability': self.cascade_probability
        }


@dataclass
class OrbitalTrajectory:
    """Complete trajectory of orbital risk over time."""
    times: np.ndarray
    risk_indices: np.ndarray
    debris_counts: np.ndarray
    collision_rates: np.ndarray
    cascade_probabilities: np.ndarray
    
    @property
    def peak_risk(self) -> float:
        return np.max(self.risk_indices)
    
    @property
    def peak_time(self) -> float:
        idx = np.argmax(self.risk_indices)
        return self.times[idx]
    
    @property
    def final_risk(self) -> float:
        return self.risk_indices[-1]
    
    def get_state_at(self, t: float) -> DebrisState:
        """Interpolate state at time t."""
        idx = np.searchsorted(self.times, t)
        idx = min(idx, len(self.times) - 1)
        
        return DebrisState(
            time=t,
            risk_index=self.risk_indices[idx],
            debris_count=self.debris_counts[idx],
            collision_rate=self.collision_rates[idx],
            cascade_probability=self.cascade_probabilities[idx]
        )


@dataclass(frozen=True)
class AdvancedOrbitalParams:
    """
    Extended parameters for advanced orbital debris modeling.
    
    References:
    - Kessler & Cour-Palais (1978)
    - NASA ORDEM 3.0 model
    - ESA MASTER model
    - Liou et al. (2010) - Active Debris Removal
    """
    # Base risk dynamics (dR/dt = α·N + β·R² - γ·R)
    alpha_base: float = 0.001           # Debris per launch (baseline)
    beta_collision: float = 0.0001      # Cascade coefficient
    gamma_decay: float = 0.01           # Natural decay rate
    
    # Initial conditions (2026 baseline)
    R_initial: float = 1.0              # Normalized initial risk
    N_initial: float = 30000            # Initial trackable debris count
    
    # Launch-specific debris generation
    debris_per_launch: float = 5.0      # Small debris items per launch
    debris_per_failure: float = 500.0   # Debris from launch failure
    failure_probability: float = 0.02   # Launch failure rate
    
    # Altitude distribution (fraction of activity)
    LEO_fraction: float = 0.7           # Low Earth Orbit (<2000 km)
    GEO_fraction: float = 0.1           # Geostationary
    Transit_fraction: float = 0.2       # Transfer orbits
    
    # Critical thresholds
    R_warning: float = 3.0              # Warning threshold
    R_critical: float = 5.0             # Cascade threshold
    R_catastrophic: float = 10.0        # Catastrophic threshold
    
    # Mitigation technology parameters
    active_removal_rate: float = 0.02   # Fraction removed per year (ADR)
    elevator_shielding: float = 0.3     # Risk reduction from elevator tether
    
    # Environmental factors
    solar_cycle_period: float = 11.0    # Years
    solar_amplitude: float = 0.2        # Decay rate variation


# ============================================================================
# Advanced Orbital Risk Model
# ============================================================================

class AdvancedOrbitalModel:
    """
    Advanced Kessler Syndrome dynamics model.
    
    Models the evolution of orbital debris risk including:
    - Non-linear cascade dynamics
    - Solar cycle effects on atmospheric drag
    - Active debris removal mitigation
    - Launch activity patterns
    """
    
    def __init__(self, params: Optional[AdvancedOrbitalParams] = None):
        """Initialize with orbital parameters."""
        self.params = params or AdvancedOrbitalParams()
    
    def debris_generation_rate(self, N_launch: float, R: float) -> float:
        """
        Calculate net debris generation rate.
        
        Args:
            N_launch: Annual launch rate
            R: Current risk index
        
        Returns:
            Net debris generation (items/year)
        """
        p = self.params
        
        # Normal launch debris
        normal_debris = N_launch * p.debris_per_launch
        
        # Failure debris
        failure_debris = N_launch * p.failure_probability * p.debris_per_failure
        
        # Collision-generated debris (quadratic in R)
        collision_debris = p.N_initial * p.beta_collision * R**2
        
        return normal_debris + failure_debris + collision_debris
    
    def decay_rate(self, t: float, N_debris: float) -> float:
        """
        Calculate debris decay rate including solar cycle effects.
        
        Args:
            t: Time (years from start)
            N_debris: Current debris count
        
        Returns:
            Decay rate (items/year)
        """
        p = self.params
        
        # Base decay
        base_decay = p.gamma_decay * N_debris
        
        # Solar cycle modulation
        solar_phase = 2 * np.pi * t / p.solar_cycle_period
        solar_factor = 1 + p.solar_amplitude * np.sin(solar_phase)
        
        return base_decay * solar_factor
    
    def removal_rate(self, N_debris: float, has_ADR: bool = False) -> float:
        """
        Calculate active debris removal rate.
        
        Args:
            N_debris: Current debris count
            has_ADR: Whether ADR technology is deployed
        
        Returns:
            Removal rate (items/year)
        """
        if has_ADR:
            return self.params.active_removal_rate * N_debris
        return 0.0
    
    def risk_dynamics(self, t: float, state: np.ndarray, 
                      launch_schedule: Callable[[float], float],
                      has_ADR: bool = False,
                      has_elevator: bool = False) -> np.ndarray:
        """
        Differential equations for orbital debris dynamics.
        
        State: [R, N_debris]
        - R: Risk index
        - N_debris: Trackable debris count
        
        Args:
            t: Time
            state: [R, N_debris]
            launch_schedule: Function returning launches/year at time t
            has_ADR: Active debris removal deployed
            has_elevator: Space elevator provides shielding
        
        Returns:
            [dR/dt, dN/dt]
        """
        p = self.params
        R, N_debris = state
        
        # Get launch rate
        N_launch = launch_schedule(t)
        
        # Apply elevator shielding effect
        if has_elevator:
            effective_alpha = p.alpha_base * (1 - p.elevator_shielding)
        else:
            effective_alpha = p.alpha_base
        
        # Risk dynamics: dR/dt = α·N_launch + β·R² - γ·R
        dR = (effective_alpha * N_launch + 
              p.beta_collision * R**2 - 
              p.gamma_decay * R)
        
        # Debris dynamics: dN/dt = generation - decay - removal
        dN = (self.debris_generation_rate(N_launch, R) - 
              self.decay_rate(t, N_debris) - 
              self.removal_rate(N_debris, has_ADR))
        
        return np.array([dR, dN])
    
    def simulate(self, 
                 duration: float,
                 launch_schedule: Callable[[float], float],
                 has_ADR: bool = False,
                 has_elevator: bool = False,
                 dt: float = 0.1) -> OrbitalTrajectory:
        """
        Simulate orbital debris evolution over time.
        
        Args:
            duration: Simulation duration (years)
            launch_schedule: Function(t) -> launches per year
            has_ADR: Active debris removal deployed
            has_elevator: Space elevator present
            dt: Time step for output
        
        Returns:
            OrbitalTrajectory with full time evolution
        """
        p = self.params
        
        # Initial state
        y0 = np.array([p.R_initial, p.N_initial])
        
        # Time points - ensure valid time span
        duration = max(duration, dt * 2)  # Ensure at least 2 time steps
        t_eval = np.arange(0, duration + dt/2, dt)  # Slightly adjust to avoid boundary issues
        t_eval = t_eval[t_eval <= duration]  # Ensure all points within span
        
        # Solve ODE
        def dynamics(t, y):
            return self.risk_dynamics(t, y, launch_schedule, has_ADR, has_elevator)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sol = solve_ivp(
                dynamics, 
                (0, duration), 
                y0, 
                t_eval=t_eval,
                method='RK45',
                max_step=0.5
            )
        
        # Extract trajectories
        times = sol.t
        R = np.maximum(0, sol.y[0])
        N = np.maximum(0, sol.y[1])
        
        # Calculate derived quantities
        collision_rates = p.beta_collision * R**2 * N / p.N_initial
        cascade_probs = np.minimum(1.0, (R / p.R_critical)**2)
        
        return OrbitalTrajectory(
            times=times,
            risk_indices=R,
            debris_counts=N,
            collision_rates=collision_rates,
            cascade_probabilities=cascade_probs
        )
    
    def calculate_cascade_probability(self, 
                                      R: float, 
                                      duration: float = 10.0) -> float:
        """
        Calculate probability of cascade onset given current risk level.
        
        Uses Poisson process approximation for rare cascade events.
        
        Args:
            R: Current risk index
            duration: Time horizon (years)
        
        Returns:
            Probability of cascade within duration
        """
        p = self.params
        
        if R < p.R_warning:
            lambda_cascade = 0.001 * R  # Very low rate below warning
        elif R < p.R_critical:
            lambda_cascade = 0.01 * (R - p.R_warning)
        else:
            lambda_cascade = 0.1 * (R - p.R_critical)
        
        # Poisson probability of at least one event
        return 1 - np.exp(-lambda_cascade * duration)


# ============================================================================
# Scenario Analysis
# ============================================================================

class OrbitalScenarioAnalyzer:
    """
    Analyze orbital debris implications of different transport scenarios.
    """
    
    def __init__(self, params: Optional[AdvancedOrbitalParams] = None):
        """Initialize analyzer."""
        self.params = params or AdvancedOrbitalParams()
        self.model = AdvancedOrbitalModel(self.params)
    
    def create_launch_schedule(self,
                               total_launches: int,
                               duration: float,
                               profile: str = 'constant'
                               ) -> Callable[[float], float]:
        """
        Create a launch schedule function.
        
        Args:
            total_launches: Total launches over duration
            duration: Time period (years)
            profile: 'constant', 'rampup', 'rampdown', 'bell'
        
        Returns:
            Function(t) -> launches per year
        """
        annual_rate = total_launches / duration if duration > 0 else 0
        
        if profile == 'constant':
            return lambda t: annual_rate
        
        elif profile == 'rampup':
            # Linear ramp from 0.5x to 1.5x
            return lambda t: annual_rate * (0.5 + t / duration)
        
        elif profile == 'rampdown':
            # Linear ramp from 1.5x to 0.5x
            return lambda t: annual_rate * (1.5 - t / duration)
        
        elif profile == 'bell':
            # Bell curve centered at midpoint
            mid = duration / 2
            sigma = duration / 4
            return lambda t: annual_rate * 2 * np.exp(-((t - mid)**2) / (2 * sigma**2))
        
        return lambda t: annual_rate
    
    def analyze_pure_rocket(self, 
                            total_mass: float = 1e8,
                            payload_per_launch: float = 150,
                            duration: float = 25) -> Dict:
        """
        Analyze orbital risk for pure rocket transport.
        
        Args:
            total_mass: Total mass to transport (tons)
            payload_per_launch: Payload per launch (tons)
            duration: Construction duration (years)
        
        Returns:
            Analysis results dictionary
        """
        total_launches = int(total_mass / payload_per_launch)
        launch_schedule = self.create_launch_schedule(total_launches, duration)
        
        trajectory = self.model.simulate(
            duration=duration,
            launch_schedule=launch_schedule,
            has_ADR=False,
            has_elevator=False
        )
        
        return {
            'scenario': 'Pure Rocket',
            'total_launches': total_launches,
            'duration_years': duration,
            'peak_risk': trajectory.peak_risk,
            'peak_time': trajectory.peak_time,
            'final_risk': trajectory.final_risk,
            'cascade_triggered': trajectory.peak_risk > self.params.R_critical,
            'cascade_probability': self.model.calculate_cascade_probability(
                trajectory.peak_risk, duration
            ),
            'trajectory': trajectory
        }
    
    def analyze_pure_elevator(self, 
                              total_mass: float = 1e8,
                              throughput: float = 5.37e5,
                              duration: float = None) -> Dict:
        """
        Analyze orbital risk for pure elevator transport.
        
        Space elevator generates no launch debris but may provide
        debris mitigation via electrodynamic tether effects.
        """
        if duration is None:
            duration = total_mass / throughput
        
        # Ensure minimum duration for simulation
        duration = max(duration, 1.0)
        
        # Zero launches during operation
        launch_schedule = lambda t: 0
        
        trajectory = self.model.simulate(
            duration=duration,
            launch_schedule=launch_schedule,
            has_ADR=False,
            has_elevator=True  # Elevator provides passive mitigation
        )
        
        return {
            'scenario': 'Pure Elevator',
            'total_launches': 0,
            'duration_years': duration,
            'peak_risk': trajectory.peak_risk,
            'peak_time': trajectory.peak_time,
            'final_risk': trajectory.final_risk,
            'cascade_triggered': False,
            'cascade_probability': self.model.calculate_cascade_probability(
                trajectory.peak_risk, duration
            ),
            'trajectory': trajectory
        }
    
    def analyze_hybrid(self,
                       total_mass: float = 1e8,
                       elevator_fraction: float = 0.1,
                       payload_per_launch: float = 150,
                       elevator_throughput: float = 5.37e5) -> Dict:
        """
        Analyze orbital risk for hybrid transport system.
        """
        rocket_mass = total_mass * (1 - elevator_fraction)
        elevator_mass = total_mass * elevator_fraction
        
        total_launches = int(rocket_mass / payload_per_launch)
        elevator_time = elevator_mass / elevator_throughput
        rocket_time = total_launches / 50000  # Assume max 50k launches/year
        
        duration = max(elevator_time, rocket_time, 10)
        
        launch_schedule = self.create_launch_schedule(total_launches, duration)
        
        trajectory = self.model.simulate(
            duration=duration,
            launch_schedule=launch_schedule,
            has_ADR=False,
            has_elevator=True
        )
        
        return {
            'scenario': f'Hybrid ({elevator_fraction:.0%} elevator)',
            'total_launches': total_launches,
            'duration_years': duration,
            'peak_risk': trajectory.peak_risk,
            'peak_time': trajectory.peak_time,
            'final_risk': trajectory.final_risk,
            'cascade_triggered': trajectory.peak_risk > self.params.R_critical,
            'cascade_probability': self.model.calculate_cascade_probability(
                trajectory.peak_risk, duration
            ),
            'trajectory': trajectory
        }
    
    def compare_mitigation_strategies(self,
                                       total_launches: int = 600000,
                                       duration: float = 25) -> Dict[str, Dict]:
        """
        Compare different debris mitigation strategies.
        
        Returns:
            Dictionary of strategy name -> analysis results
        """
        launch_schedule = self.create_launch_schedule(total_launches, duration)
        results = {}
        
        # Baseline (no mitigation)
        traj = self.model.simulate(duration, launch_schedule, False, False)
        results['No Mitigation'] = {
            'peak_risk': traj.peak_risk,
            'final_risk': traj.final_risk,
            'cascade_probability': self.model.calculate_cascade_probability(
                traj.peak_risk, duration
            )
        }
        
        # With ADR
        traj = self.model.simulate(duration, launch_schedule, True, False)
        results['Active Debris Removal'] = {
            'peak_risk': traj.peak_risk,
            'final_risk': traj.final_risk,
            'cascade_probability': self.model.calculate_cascade_probability(
                traj.peak_risk, duration
            )
        }
        
        # With Elevator (passive mitigation)
        traj = self.model.simulate(duration, launch_schedule, False, True)
        results['Elevator Shielding'] = {
            'peak_risk': traj.peak_risk,
            'final_risk': traj.final_risk,
            'cascade_probability': self.model.calculate_cascade_probability(
                traj.peak_risk, duration
            )
        }
        
        # Both ADR and Elevator
        traj = self.model.simulate(duration, launch_schedule, True, True)
        results['Combined (ADR + Elevator)'] = {
            'peak_risk': traj.peak_risk,
            'final_risk': traj.final_risk,
            'cascade_probability': self.model.calculate_cascade_probability(
                traj.peak_risk, duration
            )
        }
        
        return results


# ============================================================================
# Risk Metrics Calculator
# ============================================================================

class OrbitalRiskMetrics:
    """
    Calculate comprehensive orbital risk metrics for reporting.
    """
    
    @staticmethod
    def calculate_risk_integrated(trajectory: OrbitalTrajectory) -> float:
        """
        Calculate time-integrated risk (risk-years).
        
        Higher value indicates more sustained risk exposure.
        """
        # Use np.trapezoid for newer numpy versions, fallback to trapz for older
        try:
            return np.trapezoid(trajectory.risk_indices, trajectory.times)
        except AttributeError:
            return np.trapz(trajectory.risk_indices, trajectory.times)
    
    @staticmethod
    def calculate_time_above_threshold(trajectory: OrbitalTrajectory,
                                        threshold: float = 3.0) -> float:
        """Calculate years spent above risk threshold."""
        above = trajectory.risk_indices > threshold
        dt = np.diff(trajectory.times)
        return np.sum(dt[above[:-1]])
    
    @staticmethod
    def calculate_max_collision_rate(trajectory: OrbitalTrajectory) -> float:
        """Calculate maximum collision rate achieved."""
        return np.max(trajectory.collision_rates)
    
    @staticmethod
    def calculate_debris_legacy(trajectory: OrbitalTrajectory) -> float:
        """
        Calculate debris legacy - final debris count relative to initial.
        
        Values > 1 indicate net debris accumulation.
        """
        return trajectory.debris_counts[-1] / trajectory.debris_counts[0]
    
    @staticmethod
    def generate_risk_report(trajectory: OrbitalTrajectory,
                             scenario_name: str = "Unknown") -> str:
        """Generate formatted risk report."""
        metrics = OrbitalRiskMetrics
        
        lines = []
        lines.append(f"Orbital Debris Risk Report: {scenario_name}")
        lines.append("=" * 50)
        lines.append(f"Duration: {trajectory.times[-1]:.1f} years")
        lines.append(f"Peak Risk Index: {trajectory.peak_risk:.2f}")
        lines.append(f"Peak Risk Time: {trajectory.peak_time:.1f} years")
        lines.append(f"Final Risk Index: {trajectory.final_risk:.2f}")
        lines.append(f"Integrated Risk: {metrics.calculate_risk_integrated(trajectory):.1f} risk-years")
        lines.append(f"Time Above Warning (R>3): {metrics.calculate_time_above_threshold(trajectory, 3.0):.1f} years")
        lines.append(f"Time Above Critical (R>5): {metrics.calculate_time_above_threshold(trajectory, 5.0):.1f} years")
        lines.append(f"Max Collision Rate: {metrics.calculate_max_collision_rate(trajectory):.4f} /year")
        lines.append(f"Debris Legacy Factor: {metrics.calculate_debris_legacy(trajectory):.2f}x")
        lines.append("")
        
        return "\n".join(lines)


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    import os
    
    print("=" * 80)
    print("Orbital Debris Risk Analysis - Question 4")
    print("=" * 80)
    print()
    
    # Initialize analyzer
    analyzer = OrbitalScenarioAnalyzer()
    
    # Analyze three main scenarios
    print("1. Analyzing Transport Scenarios...")
    
    rocket_result = analyzer.analyze_pure_rocket()
    elevator_result = analyzer.analyze_pure_elevator()
    hybrid_result = analyzer.analyze_hybrid(elevator_fraction=0.10)
    
    # Print scenario comparisons
    print("\n" + "-" * 60)
    print("SCENARIO COMPARISON")
    print("-" * 60)
    print(f"{'Scenario':<25} {'Peak Risk':<12} {'Final Risk':<12} {'Cascade Prob':<12}")
    print("-" * 60)
    
    for result in [rocket_result, elevator_result, hybrid_result]:
        print(f"{result['scenario']:<25} "
              f"{result['peak_risk']:<12.2f} "
              f"{result['final_risk']:<12.2f} "
              f"{result['cascade_probability']*100:<11.1f}%")
    
    # Compare mitigation strategies
    print("\n2. Comparing Mitigation Strategies...")
    print("   (For Pure Rocket scenario)")
    
    mitigation_results = analyzer.compare_mitigation_strategies()
    
    print("\n" + "-" * 60)
    print("MITIGATION STRATEGY COMPARISON")
    print("-" * 60)
    print(f"{'Strategy':<25} {'Peak Risk':<12} {'Final Risk':<12} {'Cascade Prob':<12}")
    print("-" * 60)
    
    for strategy, data in mitigation_results.items():
        print(f"{strategy:<25} "
              f"{data['peak_risk']:<12.2f} "
              f"{data['final_risk']:<12.2f} "
              f"{data['cascade_probability']*100:<11.1f}%")
    
    # Generate detailed reports
    print("\n3. Generating Detailed Risk Reports...")
    
    output_dir = os.path.dirname(os.path.abspath(__file__))
    report_dir = os.path.join(os.path.dirname(output_dir), "mdFile")
    os.makedirs(report_dir, exist_ok=True)
    
    reports = []
    for result in [rocket_result, elevator_result, hybrid_result]:
        report = OrbitalRiskMetrics.generate_risk_report(
            result['trajectory'], 
            result['scenario']
        )
        reports.append(report)
        print(f"\n{report}")
    
    # Save combined report
    combined_report = "\n\n".join(reports)
    report_path = os.path.join(report_dir, "Q4_Orbital_Risk_Analysis.txt")
    with open(report_path, 'w') as f:
        f.write(combined_report)
    
    print(f"\nDetailed reports saved to: {report_path}")
    
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print(f"1. Pure Rocket scenario leads to peak risk of {rocket_result['peak_risk']:.2f}")
    print(f"   with {rocket_result['cascade_probability']*100:.1f}% cascade probability")
    print(f"2. Pure Elevator scenario maintains risk at {elevator_result['peak_risk']:.2f}")
    print(f"   with minimal cascade risk")
    print(f"3. Space elevator provides ~{analyzer.params.elevator_shielding*100:.0f}% risk reduction")
    print(f"   via electrodynamic tether effects")
    print("=" * 80)
