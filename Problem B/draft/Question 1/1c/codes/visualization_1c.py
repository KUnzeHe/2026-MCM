"""
1c Mixed Transport Visualization Module
========================================
Comprehensive visualization for mixed (Elevator + Rocket) transport optimization.
Based on comprehensive_transport_model_v5.py

9-Figure Framework:
1. Three-Scenario Time-Cost Panorama
2. Optimal Allocation Pie Chart + Gantt Timeline
3. Dual-Chain Capacity Evolution
4. Cumulative Transport Comparison
5. Cost Waterfall Chart
6. Pareto Front Deep Analysis (2×2)
7. Sensitivity Tornado Chart
8. Monte Carlo Distribution
9. Scenario Radar Comparison

Author: MCM Team 2026
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.gridspec import GridSpec
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Import core model from V5
from comprehensive_transport_model_v5 import (
    ModelParams, GrowthParams, AnchorParams, OptimizationResult,
    TransportOptimizationModel
)

# ============================================================================
# Configuration
# ============================================================================

# Color scheme (consistent with 1a/1b)
COLORS = {
    'elevator': '#4ECDC4',      # Teal for Elevator Chain
    'rocket': '#FF6B6B',        # Coral for Direct Rockets
    'mixed': '#9B59B6',         # Purple for Mixed/Combined
    'capex': '#A8DADC',         # Light blue for CAPEX
    'opex': '#457B9D',          # Dark blue for OPEX
    'accent': '#E94F37',        # Red accent
    'primary': '#2E86AB',       # Primary blue
    'secondary': '#1D3557',     # Dark navy
    'gold': '#F4A261',          # Gold/Orange
    'grid': '#E5E5E5'           # Grid color
}

# Plot style
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 120
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# Output directory
OUTPUT_DIR = "../image"


# ============================================================================
# Scenario Definitions
# ============================================================================

@dataclass
class Scenario:
    """Scenario configuration for comparative analysis."""
    name: str
    description: str
    params: ModelParams
    color: str


def get_scenarios() -> Dict[str, Scenario]:
    """
    Define three technology scenarios for comparison.
    
    - Conservative: Lower rocket payload (100t), slower growth
    - Moderate: Middle ground (125t payload)
    - Aggressive: High payload (150t), faster growth
    """
    return {
        'conservative': Scenario(
            name='Conservative',
            description='p_B=100t, r=0.10',
            params=ModelParams(
                p_B=100.0,
                growth=GrowthParams(r=0.10)
            ),
            color='#3498DB'  # Blue
        ),
        'moderate': Scenario(
            name='Moderate',
            description='p_B=125t, r=0.15',
            params=ModelParams(
                p_B=125.0,
                growth=GrowthParams(r=0.15)
            ),
            color='#2ECC71'  # Green
        ),
        'aggressive': Scenario(
            name='Aggressive',
            description='p_B=150t, r=0.20',
            params=ModelParams(
                p_B=150.0,
                growth=GrowthParams(r=0.20)
            ),
            color='#E74C3C'  # Red
        )
    }


# ============================================================================
# Utility Functions
# ============================================================================

def ensure_dir(path: str):
    """Ensure output directory exists."""
    os.makedirs(path, exist_ok=True)


def format_cost(value: float) -> str:
    """Format cost value in human-readable format."""
    if value >= 1e12:
        return f"${value/1e12:.1f}T"
    elif value >= 1e9:
        return f"${value/1e9:.1f}B"
    elif value >= 1e6:
        return f"${value/1e6:.1f}M"
    else:
        return f"${value:,.0f}"


def format_mass(value: float) -> str:
    """Format mass value in human-readable format."""
    if value >= 1e6:
        return f"{value/1e6:.1f} Mt"
    elif value >= 1e3:
        return f"{value/1e3:.1f} kt"
    else:
        return f"{value:.0f} t"


# ============================================================================
# Figure 1: Three-Scenario Time-Cost Panorama
# ============================================================================

def plot_fig01_scenario_panorama(save_path: str):
    """
    Figure 1: Compare Pareto fronts across three technology scenarios.
    Shows how different assumptions affect the time-cost trade-off.
    Constraint: Y_max = 24 years.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    scenarios = get_scenarios()
    Y_range = np.linspace(14, 24, 41)  # Max 24 years constraint
    
    knee_points = {}
    
    for key, scenario in scenarios.items():
        model = TransportOptimizationModel(scenario.params)
        results = model.pareto_sweep(Y_range)
        
        # Filter feasible results
        feas = [r for r in results if r.feasible]
        if not feas:
            continue
        
        Y_vals = [r.Y for r in feas]
        costs = [r.cost_total / 1e12 for r in feas]
        
        # Find knee point
        knee_idx = TransportOptimizationModel.find_knee_point(
            [r.Y for r in feas], [r.cost_total for r in feas]
        )
        knee_points[key] = feas[knee_idx]
        
        # Plot Pareto front
        ax.plot(Y_vals, costs, '-', color=scenario.color, linewidth=2.5,
                label=f'{scenario.name} ({scenario.description})', alpha=0.9)
        
        # Mark knee point
        ax.scatter([Y_vals[knee_idx]], [costs[knee_idx]], 
                   color=scenario.color, s=200, marker='*', zorder=5,
                   edgecolors='white', linewidths=2)
    
    # Annotations for knee points
    for key, result in knee_points.items():
        scenario = scenarios[key]
        ax.annotate(f'{result.Y:.0f}y, ${result.cost_total/1e12:.1f}T',
                    xy=(result.Y, result.cost_total/1e12),
                    xytext=(result.Y + 3, result.cost_total/1e12 + 1),
                    fontsize=10, color=scenario.color,
                    arrowprops=dict(arrowstyle='->', color=scenario.color, alpha=0.7))
    
    ax.set_xlabel('Project Duration (Years)', fontsize=12)
    ax.set_ylabel('Total Cost NPV (Trillion USD)', fontsize=12)
    ax.set_title('Figure 1: Three-Scenario Time-Cost Panorama\n(Mixed Elevator + Rocket Transport)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add text box with key findings
    textstr = '\n'.join([
        f"★ {scenarios[k].name}: Y={knee_points[k].Y:.0f}y, Cost=${knee_points[k].cost_total/1e12:.1f}T"
        for k in knee_points
    ])
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================================
# Figure 2: Optimal Allocation Pie Chart + Gantt Timeline
# ============================================================================

def plot_fig02_allocation_gantt(save_path: str, Y_target: float = 24):
    """
    Figure 2: Optimal allocation visualization with:
    - Left: Pie chart of mass distribution
    - Right: Gantt-style timeline of transport activities
    Constraint: Y_max = 24 years.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Use moderate scenario for visualization
    params = ModelParams(p_B=125.0)
    model = TransportOptimizationModel(params)
    result = model.solve(Y_target)
    
    if not result.feasible:
        print(f"Warning: Y={Y_target} not feasible")
        return
    
    # ===== Left: Pie Chart =====
    sizes = [result.x_opt, result.mR_opt]
    labels = [f'Elevator Chain\n{result.x_opt/1e6:.1f} Mt ({result.elevator_pct:.1f}%)',
              f'Direct Rockets\n{result.mR_opt/1e6:.1f} Mt ({100-result.elevator_pct:.1f}%)']
    colors = [COLORS['elevator'], COLORS['rocket']]
    explode = (0.03, 0.03)
    
    wedges, texts, autotexts = ax1.pie(
        sizes, explode=explode, labels=labels, colors=colors,
        autopct='', shadow=False, startangle=90,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    
    # Add center circle for donut effect
    centre_circle = plt.Circle((0, 0), 0.55, fc='white')
    ax1.add_patch(centre_circle)
    
    # Center text
    ax1.text(0, 0, f'100 Mt\nTotal', ha='center', va='center',
             fontsize=14, fontweight='bold')
    
    ax1.set_title(f'Mass Allocation (Y = {Y_target:.0f} years)',
                  fontsize=12, fontweight='bold')
    
    # ===== Right: Gantt Timeline =====
    # Timeline showing transport phases
    
    # Calculate phase durations based on throughput
    elevator_rate = model.elevator_bottleneck_rate() / 1e6  # Mt/yr
    rocket_avg_rate = result.mR_opt / (Y_target * 1e6)  # Approximate avg Mt/yr
    
    elevator_duration = (result.x_opt / 1e6) / elevator_rate
    rocket_duration = Y_target  # Rockets run full duration
    
    # Gantt bars
    bar_height = 0.6
    
    # Elevator chain (starts at 0)
    ax2.barh(2, elevator_duration, height=bar_height, left=0,
             color=COLORS['elevator'], edgecolor='white', linewidth=2,
             label='Elevator Chain')
    
    # Direct rockets (starts at 0, full duration)
    ax2.barh(1, rocket_duration, height=bar_height, left=0,
             color=COLORS['rocket'], edgecolor='white', linewidth=2,
             label='Direct Rockets')
    
    # Add annotations
    ax2.text(elevator_duration/2, 2, f'{result.x_opt/1e6:.1f} Mt\n({elevator_duration:.1f} yrs)',
             ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    ax2.text(rocket_duration/2, 1, f'{result.mR_opt/1e6:.1f} Mt\n({rocket_duration:.0f} yrs)',
             ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    
    # Deadline marker
    ax2.axvline(x=Y_target, color=COLORS['accent'], linestyle='--', linewidth=2,
                label=f'Deadline: Y={Y_target:.0f}')
    
    ax2.set_xlim(0, Y_target * 1.1)
    ax2.set_ylim(0.3, 2.7)
    ax2.set_yticks([1, 2])
    ax2.set_yticklabels(['Direct Rockets', 'Elevator Chain'])
    ax2.set_xlabel('Year', fontsize=11)
    ax2.set_title('Transport Timeline (Gantt View)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================================
# Figure 3: Dual-Chain Capacity Evolution
# ============================================================================

def plot_fig03_capacity_evolution(save_path: str):
    """
    Figure 3: Show how both transport chain capacities evolve over time.
    - Elevator: Constant throughput (bottleneck constrained)
    - Rocket: Logistic growth of launch site capacity
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    params = ModelParams()
    model = TransportOptimizationModel(params)
    
    times = np.linspace(0, 60, 200)
    
    # ===== Top: Instantaneous Rate =====
    elevator_rate = model.elevator_bottleneck_rate() / 1e6  # Mt/yr
    rocket_rates = [model.rocket_capacity_rate(t) / 1e6 for t in times]
    
    ax1.axhline(y=elevator_rate, color=COLORS['elevator'], linewidth=3,
                label=f'Elevator Chain: {elevator_rate:.2f} Mt/yr (constant)')
    ax1.plot(times, rocket_rates, color=COLORS['rocket'], linewidth=3,
             label='Direct Rockets (Logistic growth)')
    
    # Fill intersection region
    ax1.fill_between(times, 0, [min(elevator_rate, r) for r in rocket_rates],
                     alpha=0.1, color=COLORS['mixed'])
    
    # Mark crossover point
    crossover_idx = np.argmin(np.abs(np.array(rocket_rates) - elevator_rate))
    crossover_t = times[crossover_idx]
    ax1.axvline(x=crossover_t, color='gray', linestyle=':', alpha=0.7)
    ax1.annotate(f'Crossover: t={crossover_t:.1f}y',
                 xy=(crossover_t, elevator_rate),
                 xytext=(crossover_t + 5, elevator_rate + 5),
                 fontsize=10, arrowprops=dict(arrowstyle='->', color='gray'))
    
    ax1.set_ylabel('Instantaneous Rate (Mt/year)', fontsize=11)
    ax1.set_title('(a) Transport Capacity Rate Evolution', fontsize=12, fontweight='bold')
    ax1.legend(loc='center right', fontsize=10)
    ax1.set_ylim(0, 30)
    ax1.grid(True, alpha=0.3)
    
    # ===== Bottom: Cumulative Capacity =====
    cum_elevator = [model.cumulative_elevator_capacity(t) / 1e6 for t in times]
    cum_rocket = [model.cumulative_rocket_capacity(t) / 1e6 for t in times]
    cum_total = [e + r for e, r in zip(cum_elevator, cum_rocket)]
    
    ax2.plot(times, cum_elevator, color=COLORS['elevator'], linewidth=3,
             label='Elevator Chain Cumulative')
    ax2.plot(times, cum_rocket, color=COLORS['rocket'], linewidth=3,
             label='Rocket Cumulative')
    ax2.plot(times, cum_total, color=COLORS['mixed'], linewidth=3, linestyle='--',
             label='Total Combined')
    
    # 100 Mt demand line
    ax2.axhline(y=100, color=COLORS['secondary'], linestyle=':', linewidth=2,
                label='Total Demand: 100 Mt')
    
    # Find minimum feasible Y
    feasible_idx = np.argmin(np.abs(np.array(cum_total) - 100))
    feasible_Y = times[feasible_idx]
    ax2.axvline(x=feasible_Y, color=COLORS['accent'], linestyle='--', alpha=0.7)
    ax2.scatter([feasible_Y], [100], color=COLORS['accent'], s=150, marker='*', zorder=5)
    ax2.annotate(f'Min Feasible Y ≈ {feasible_Y:.0f}y',
                 xy=(feasible_Y, 100),
                 xytext=(feasible_Y + 5, 80),
                 fontsize=10, arrowprops=dict(arrowstyle='->', color=COLORS['accent']))
    
    ax2.set_xlabel('Year', fontsize=11)
    ax2.set_ylabel('Cumulative Capacity (Mt)', fontsize=11)
    ax2.set_title('(b) Cumulative Transport Capacity', fontsize=12, fontweight='bold')
    ax2.legend(loc='center right', fontsize=10)
    ax2.set_ylim(0, 200)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Figure 3: Dual-Chain Capacity Evolution', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================================
# Figure 4: Cumulative Transport Comparison
# ============================================================================

def plot_fig04_cumulative_comparison(save_path: str, Y_target: float = 24):
    """
    Figure 4: Compare cumulative mass transported by each chain over time.
    Shows how the allocation strategy unfolds temporally.
    Constraint: Y_max = 24 years.
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    params = ModelParams(p_B=125.0)
    model = TransportOptimizationModel(params)
    result = model.solve(Y_target)
    
    if not result.feasible:
        print(f"Warning: Y={Y_target} not feasible")
        return
    
    times = np.linspace(0, Y_target, 200)
    
    # Calculate cumulative transport assuming constant utilization
    elevator_rate = result.x_opt / Y_target  # Constant rate
    
    cum_elevator = [min(elevator_rate * t, result.x_opt) / 1e6 for t in times]
    
    # Rocket: follows capacity utilization ratio
    utilization = result.mR_opt / model.cumulative_rocket_capacity(Y_target)
    cum_rocket = [model.cumulative_rocket_capacity(t) * utilization / 1e6 for t in times]
    
    cum_total = [e + r for e, r in zip(cum_elevator, cum_rocket)]
    
    # Stacked area plot
    ax.fill_between(times, 0, cum_elevator, alpha=0.7, color=COLORS['elevator'],
                    label=f'Elevator Chain ({result.x_opt/1e6:.1f} Mt)')
    ax.fill_between(times, cum_elevator, cum_total, alpha=0.7, color=COLORS['rocket'],
                    label=f'Direct Rockets ({result.mR_opt/1e6:.1f} Mt)')
    
    # Boundary lines
    ax.plot(times, cum_elevator, color=COLORS['elevator'], linewidth=2)
    ax.plot(times, cum_total, color=COLORS['secondary'], linewidth=2)
    
    # 100 Mt target line
    ax.axhline(y=100, color=COLORS['secondary'], linestyle='--', linewidth=2,
               label='Target: 100 Mt')
    
    # Deadline marker
    ax.axvline(x=Y_target, color=COLORS['accent'], linestyle=':', linewidth=2)
    ax.scatter([Y_target], [100], color=COLORS['accent'], s=200, marker='*', zorder=5)
    
    # Milestone markers (25%, 50%, 75%)
    for pct in [25, 50, 75]:
        target = pct
        idx = np.argmin(np.abs(np.array(cum_total) - target))
        ax.scatter([times[idx]], [target], color='gray', s=80, marker='o', zorder=4)
        ax.annotate(f'{pct}% at t={times[idx]:.1f}y',
                    xy=(times[idx], target),
                    xytext=(times[idx] + 2, target + 5),
                    fontsize=9, alpha=0.7)
    
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Cumulative Mass Transported (Mt)', fontsize=12)
    ax.set_title(f'Figure 4: Cumulative Transport Progress (Y = {Y_target:.0f} years)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, Y_target * 1.05)
    ax.set_ylim(0, 110)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================================
# Figure 5: Cost Waterfall Chart
# ============================================================================

def plot_fig05_cost_waterfall(save_path: str, Y_target: float = 24):
    """
    Figure 5: Waterfall chart showing cost breakdown from zero to total.
    Shows CAPEX and OPEX contributions from each transport chain.
    Constraint: Y_max = 24 years.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    params = ModelParams(p_B=125.0)
    model = TransportOptimizationModel(params)
    result = model.solve(Y_target)
    
    if not result.feasible:
        print(f"Warning: Y={Y_target} not feasible")
        return
    
    # Calculate cost components
    capex_E = params.F_E / 1e12  # Elevator CAPEX
    N_new = max(0, result.N_required - params.growth.N0)
    capex_R = N_new * params.C_site / 1e12  # Rocket CAPEX
    opex_E = model.calculate_elevator_opex_npv(result.x_opt, Y_target) / 1e12
    opex_R = model.calculate_rocket_opex_npv(result.mR_opt, Y_target) / 1e12
    
    # Waterfall data
    categories = ['Start', 'Elevator\nCAPEX', 'Rocket\nCAPEX', 'Elevator\nOPEX', 'Rocket\nOPEX', 'Total']
    values = [0, capex_E, capex_R, opex_E, opex_R, 0]  # Last is placeholder
    
    # Calculate cumulative positions
    cumulative = [0]
    for v in values[1:-1]:
        cumulative.append(cumulative[-1] + v)
    total = cumulative[-1]
    
    # Colors for each bar
    bar_colors = ['white', '#A8DADC', '#E9C46A', COLORS['elevator'], COLORS['rocket'], COLORS['mixed']]
    
    # Draw waterfall bars
    for i, (cat, val, color) in enumerate(zip(categories, values, bar_colors)):
        if i == 0:  # Start (empty)
            continue
        elif i == len(categories) - 1:  # Total bar from 0
            ax.bar(i, total, color=color, edgecolor='black', linewidth=1.5)
            ax.text(i, total + 0.5, f'${total:.1f}T', ha='center', va='bottom',
                    fontsize=11, fontweight='bold')
        else:
            bottom = cumulative[i-1]
            ax.bar(i, val, bottom=bottom, color=color, edgecolor='black', linewidth=1.5)
            ax.text(i, bottom + val/2, f'${val:.2f}T', ha='center', va='center',
                    fontsize=10, color='white' if i in [3, 4] else 'black', fontweight='bold')
            
            # Connector line to next bar
            if i < len(categories) - 2:
                ax.plot([i+0.4, i+0.6], [cumulative[i], cumulative[i]], 
                        color='gray', linestyle='--', linewidth=1.5)
    
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylabel('Cost NPV (Trillion USD)', fontsize=12)
    ax.set_title(f'Figure 5: Cost Waterfall Chart (Y = {Y_target:.0f} years)',
                 fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, total * 1.15)
    
    # Legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='#A8DADC', edgecolor='black', label='Elevator CAPEX'),
        plt.Rectangle((0, 0), 1, 1, facecolor='#E9C46A', edgecolor='black', label='Rocket CAPEX'),
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORS['elevator'], edgecolor='black', label='Elevator OPEX'),
        plt.Rectangle((0, 0), 1, 1, facecolor=COLORS['rocket'], edgecolor='black', label='Rocket OPEX'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================================
# Figure 6: Pareto Front Deep Analysis (2×2)
# ============================================================================

def plot_fig06_pareto_deep(save_path: str):
    """
    Figure 6: 2×2 panel deep analysis of Pareto front:
    - (a) Time vs Cost with knee point
    - (b) CAPEX vs OPEX breakdown
    - (c) Modal split over time
    - (d) Marginal cost analysis
    Constraint: Y_max = 24 years.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    params = ModelParams(p_B=125.0)
    model = TransportOptimizationModel(params)
    
    Y_range = np.linspace(14, 24, 41)  # Max 24 years constraint
    results = model.pareto_sweep(Y_range)
    feas = [r for r in results if r.feasible]
    
    if not feas:
        print("No feasible results!")
        return
    
    Y_vals = [r.Y for r in feas]
    costs = [r.cost_total / 1e12 for r in feas]
    
    # Find knee point
    knee_idx = TransportOptimizationModel.find_knee_point(
        [r.Y for r in feas], [r.cost_total for r in feas]
    )
    knee = feas[knee_idx]
    
    # ===== (a) Pareto Front =====
    ax1 = axes[0, 0]
    ax1.plot(Y_vals, costs, 'o-', color=COLORS['primary'], linewidth=2, markersize=4)
    ax1.scatter([knee.Y], [knee.cost_total/1e12], color=COLORS['accent'], 
                s=300, marker='*', zorder=5, label=f'Knee: ({knee.Y:.0f}y, ${knee.cost_total/1e12:.1f}T)')
    ax1.axvline(x=knee.Y, color=COLORS['accent'], linestyle='--', alpha=0.5)
    
    ax1.set_xlabel('Project Duration (Years)', fontsize=11)
    ax1.set_ylabel('Total Cost NPV (Trillion USD)', fontsize=11)
    ax1.set_title('(a) Pareto Front: Time-Cost Trade-off', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # ===== (b) Cost Breakdown =====
    ax2 = axes[0, 1]
    capex = [r.cost_capex / 1e12 for r in feas]
    opex = [r.cost_opex / 1e12 for r in feas]
    
    ax2.stackplot(Y_vals, capex, opex, labels=['CAPEX', 'OPEX (NPV)'],
                  colors=[COLORS['capex'], COLORS['opex']], alpha=0.85)
    ax2.axvline(x=knee.Y, color=COLORS['accent'], linestyle='--', alpha=0.7)
    
    ax2.set_xlabel('Project Duration (Years)', fontsize=11)
    ax2.set_ylabel('Cost (Trillion USD)', fontsize=11)
    ax2.set_title('(b) CAPEX vs OPEX Breakdown', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # ===== (c) Modal Split =====
    ax3 = axes[1, 0]
    ele_mass = [r.x_opt / 1e6 for r in feas]
    roc_mass = [r.mR_opt / 1e6 for r in feas]
    
    ax3.stackplot(Y_vals, ele_mass, roc_mass, 
                  labels=['Elevator Chain', 'Direct Rockets'],
                  colors=[COLORS['elevator'], COLORS['rocket']], alpha=0.85)
    ax3.axhline(y=100, color='gray', linestyle=':', alpha=0.5, label='Total: 100 Mt')
    ax3.axvline(x=knee.Y, color=COLORS['secondary'], linestyle='--', alpha=0.7)
    
    ax3.set_xlabel('Project Duration (Years)', fontsize=11)
    ax3.set_ylabel('Mass Transported (Mt)', fontsize=11)
    ax3.set_title('(c) Optimal Modal Split', fontsize=12, fontweight='bold')
    ax3.legend(loc='center right')
    ax3.grid(True, alpha=0.3)
    
    # ===== (d) Marginal Cost =====
    ax4 = axes[1, 1]
    
    # Calculate marginal cost: dC/dY
    marginal_costs = np.gradient(costs, Y_vals)
    
    ax4.plot(Y_vals, [-mc for mc in marginal_costs], 'o-', 
             color=COLORS['gold'], linewidth=2, markersize=4)
    ax4.axvline(x=knee.Y, color=COLORS['accent'], linestyle='--', alpha=0.7,
                label=f'Knee Point: Y={knee.Y:.0f}')
    ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    ax4.set_xlabel('Project Duration (Years)', fontsize=11)
    ax4.set_ylabel('Marginal Savings (-dC/dY, $T/year)', fontsize=11)
    ax4.set_title('(d) Marginal Cost Savings Analysis', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    # Annotation: maximum marginal savings
    max_marginal_idx = np.argmax([-mc for mc in marginal_costs])
    ax4.scatter([Y_vals[max_marginal_idx]], [-marginal_costs[max_marginal_idx]],
                color=COLORS['accent'], s=150, marker='*', zorder=5)
    ax4.annotate(f'Max Savings: Y={Y_vals[max_marginal_idx]:.0f}y',
                 xy=(Y_vals[max_marginal_idx], -marginal_costs[max_marginal_idx]),
                 xytext=(Y_vals[max_marginal_idx] + 5, -marginal_costs[max_marginal_idx] + 0.1),
                 fontsize=10, arrowprops=dict(arrowstyle='->', color=COLORS['accent']))
    
    plt.suptitle('Figure 6: Pareto Front Deep Analysis', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================================
# Figure 7: Sensitivity Tornado Chart
# ============================================================================

def plot_fig07_sensitivity_tornado(save_path: str, Y_target: float = 24):
    """
    Figure 7: Tornado chart showing sensitivity of total cost to parameter changes.
    Constraint: Y_max = 24 years.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    params = ModelParams(p_B=125.0)
    model = TransportOptimizationModel(params)
    
    # Base case
    base_result = model.solve(Y_target)
    if not base_result.feasible:
        print(f"Warning: Y={Y_target} not feasible")
        return
    base_cost = base_result.cost_total
    
    # Parameters to analyze
    param_info = [
        ('c_R', 'Rocket OPEX (c_R)', params.c_R),
        ('c_E', 'Elevator OPEX (c_E)', params.c_E),
        ('T_E', 'Elevator Throughput (T_E)', params.T_E),
    ]
    
    # Calculate sensitivity: ±30%
    results = []
    for param_name, label, base_val in param_info:
        # Low case (-30%)
        if param_name == 'c_R':
            low_params = ModelParams(p_B=125.0, c_R=base_val * 0.7)
            high_params = ModelParams(p_B=125.0, c_R=base_val * 1.3)
        elif param_name == 'c_E':
            low_params = ModelParams(p_B=125.0, c_E=base_val * 0.7)
            high_params = ModelParams(p_B=125.0, c_E=base_val * 1.3)
        elif param_name == 'T_E':
            low_params = ModelParams(p_B=125.0, T_E=base_val * 0.7)
            high_params = ModelParams(p_B=125.0, T_E=base_val * 1.3)
        else:
            continue
        
        low_model = TransportOptimizationModel(low_params)
        high_model = TransportOptimizationModel(high_params)
        
        low_result = low_model.solve(Y_target)
        high_result = high_model.solve(Y_target)
        
        low_cost = low_result.cost_total if low_result.feasible else base_cost
        high_cost = high_result.cost_total if high_result.feasible else base_cost
        
        results.append({
            'label': label,
            'low': (low_cost - base_cost) / base_cost * 100,
            'high': (high_cost - base_cost) / base_cost * 100
        })
    
    # Sort by absolute impact
    results.sort(key=lambda x: abs(x['high'] - x['low']), reverse=True)
    
    # Plot tornado
    y_pos = np.arange(len(results))
    
    for i, r in enumerate(results):
        # Bars extending from 0
        ax.barh(i, r['high'], height=0.6, color=COLORS['rocket'], alpha=0.8,
                label='+30%' if i == 0 else '')
        ax.barh(i, r['low'], height=0.6, color=COLORS['elevator'], alpha=0.8,
                label='-30%' if i == 0 else '')
        
        # Value labels
        ax.text(r['high'] + 0.5, i, f'{r["high"]:+.1f}%', va='center', fontsize=10)
        ax.text(r['low'] - 0.5, i, f'{r["low"]:+.1f}%', va='center', ha='right', fontsize=10)
    
    ax.axvline(x=0, color='black', linewidth=1.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([r['label'] for r in results], fontsize=11)
    ax.set_xlabel('Change in Total Cost (%)', fontsize=12)
    ax.set_title(f'Figure 7: Sensitivity Tornado Chart (Y = {Y_target:.0f} years, ±30% variation)',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, axis='x', alpha=0.3)
    
    # Add base cost annotation
    ax.text(0.02, 0.98, f'Base Cost: ${base_cost/1e12:.2f}T', transform=ax.transAxes,
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================================
# Figure 8: Monte Carlo Distribution
# ============================================================================

def plot_fig08_monte_carlo(save_path: str, Y_target: float = 24, n_samples: int = 5000):
    """
    Figure 8: Monte Carlo analysis showing cost distribution under payload uncertainty.
    Constraint: Y_max = 24 years.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    params = ModelParams()  # Default has p_B_range = (100, 150)
    model = TransportOptimizationModel(params)
    
    # Use Y=24 (max constraint), ensure feasibility
    Y_min_feas = model.find_minimum_feasible_Y()
    Y_target = min(24, max(Y_target, Y_min_feas + 2))  # Max 24 years constraint
    
    # Run Monte Carlo
    np.random.seed(42)
    p_min, p_max = params.p_B_range
    payloads = np.random.uniform(p_min, p_max, n_samples)
    
    costs = []
    allocations = []
    valid_payloads = []
    
    for p_B in payloads:
        result = model.solve(Y_target, p_B=p_B)
        if result.feasible:
            costs.append(result.cost_total / 1e12)
            allocations.append(result.elevator_pct)
            valid_payloads.append(p_B)
    
    if len(costs) < 10:
        print(f"Warning: Only {len(costs)} feasible samples. Increasing Y to 60.")
        Y_target = 60
        costs, allocations, valid_payloads = [], [], []
        for p_B in payloads:
            result = model.solve(Y_target, p_B=p_B)
            if result.feasible:
                costs.append(result.cost_total / 1e12)
                allocations.append(result.elevator_pct)
                valid_payloads.append(p_B)
    
    # ===== Left: Cost Distribution =====
    ax1 = axes[0]
    
    # Check if there's variation in costs
    cost_range = max(costs) - min(costs) if costs else 0
    
    if cost_range < 0.001:  # Almost no variation
        # Show bar chart instead of histogram
        ax1.bar(['Total Cost'], [np.mean(costs)], color=COLORS['primary'], 
                edgecolor='white', width=0.5)
        ax1.text(0, np.mean(costs) + 0.1, f'${np.mean(costs):.2f}T', 
                 ha='center', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Cost NPV (Trillion USD)', fontsize=11)
        ax1.set_title(f'(a) Cost (No variation at Y={Y_target:.0f}y)', 
                      fontsize=12, fontweight='bold')
        stats_text = f'All samples converge to\nsame optimal allocation.\nCost = ${np.mean(costs):.2f}T'
    else:
        # Histogram with auto bins
        ax1.hist(costs, bins='auto', color=COLORS['primary'], edgecolor='white',
                 alpha=0.8, density=True)
        
        mean_cost = np.mean(costs)
        std_cost = np.std(costs)
        p5 = np.percentile(costs, 5)
        p95 = np.percentile(costs, 95)
        
        ax1.axvline(x=mean_cost, color=COLORS['accent'], linewidth=2.5,
                    label=f'Mean: ${mean_cost:.2f}T')
        ax1.axvline(x=p5, color=COLORS['gold'], linestyle='--', linewidth=2,
                    label=f'5th pctl: ${p5:.2f}T')
        ax1.axvline(x=p95, color=COLORS['gold'], linestyle='--', linewidth=2,
                    label=f'95th pctl: ${p95:.2f}T')
        
        ax1.set_ylabel('Probability Density', fontsize=11)
        ax1.set_title(f'(a) Cost Distribution (n={len(costs)})', fontsize=12, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=10)
        stats_text = f'μ = ${mean_cost:.2f}T\nσ = ${std_cost:.3f}T\nCV = {std_cost/mean_cost*100:.2f}%'
    
    ax1.set_xlabel('Total Cost NPV (Trillion USD)', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # ===== Right: Allocation vs Payload Scatter =====
    ax2 = axes[1]
    
    if len(valid_payloads) > 0 and len(set(allocations)) > 1:
        scatter = ax2.scatter(valid_payloads, allocations, 
                              c=costs, cmap='viridis', alpha=0.5, s=15)
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Total Cost ($T)', fontsize=10)
        
        # Trend line
        z = np.polyfit(valid_payloads, allocations, 1)
        p = np.poly1d(z)
        x_line = np.linspace(p_min, p_max, 100)
        ax2.plot(x_line, p(x_line), color=COLORS['accent'], linewidth=2, linestyle='--',
                 label=f'Trend: {z[0]:.2f}×p_B + {z[1]:.1f}')
        ax2.legend(loc='upper right', fontsize=10)
    else:
        # All same allocation - show explanation
        ax2.scatter(valid_payloads, allocations, c=COLORS['primary'], alpha=0.5, s=15)
        ax2.axhline(y=np.mean(allocations), color=COLORS['accent'], linewidth=2,
                    label=f'Constant: {np.mean(allocations):.1f}%')
        ax2.text(0.5, 0.5, f'At Y={Y_target:.0f}y,\nElevator capacity\nsufficient for\nall scenarios',
                 transform=ax2.transAxes, ha='center', va='center', fontsize=11,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax2.legend(loc='upper right', fontsize=10)
    
    ax2.set_xlabel('Rocket Payload p_B (tons/launch)', fontsize=11)
    ax2.set_ylabel('Elevator Chain Share (%)', fontsize=11)
    ax2.set_title('(b) Allocation vs Payload Uncertainty', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'Figure 8: Monte Carlo Analysis (Y = {Y_target:.0f} years, p_B ~ U(100, 150))',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================================
# Figure 9: Scenario Radar Comparison
# ============================================================================

def plot_fig09_radar_comparison(save_path: str):
    """
    Figure 9: Radar chart comparing key metrics across scenarios.
    Constraint: Y_max = 24 years (use Y=24 for comparison).
    """
    # Get scenarios and compute results
    scenarios = get_scenarios()
    
    # Metrics to compare (at Y=24, the deadline)
    metrics = ['Total Cost', 'Elevator Share', 
               'Rocket Sites Needed', 'CAPEX/OPEX Ratio', 'Robustness (MC Feas.)']
    
    # Compute results for each scenario at Y=24
    scenario_data = {}
    Y_deadline = 24  # Fixed deadline
    
    for key, scenario in scenarios.items():
        model = TransportOptimizationModel(scenario.params)
        
        # Use fixed Y=24 instead of knee point
        Y_range = np.linspace(14, 24, 41)  # Max 24 years constraint
        results = model.pareto_sweep(Y_range)
        feas = [r for r in results if r.feasible]
        
        if feas:
            # Get result at Y=24 (deadline)
            result_24 = model.solve(Y_deadline)
            
            # Monte Carlo for robustness at Y=24
            mc = model.monte_carlo_analysis(Y_deadline, n_samples=500)
            
            scenario_data[key] = {
                'cost': result_24.cost_total / 1e12,
                'elevator_pct': result_24.elevator_pct,
                'sites_needed': result_24.N_required,
                'capex_opex_ratio': result_24.cost_capex / max(result_24.cost_opex, 1e6),
                'robustness': mc['feasibility_rate'] * 100
            }
    
    # Normalize data for radar chart (0-1 scale)
    def normalize(values, reverse=False):
        min_v, max_v = min(values), max(values)
        if max_v == min_v:
            return [0.5] * len(values)
        norm = [(v - min_v) / (max_v - min_v) for v in values]
        if reverse:
            norm = [1 - n for n in norm]
        return norm
    
    # Create radar data
    keys = list(scenario_data.keys())
    costs_vals = [scenario_data[k]['cost'] for k in keys]
    ele_pcts = [scenario_data[k]['elevator_pct'] for k in keys]
    sites = [scenario_data[k]['sites_needed'] for k in keys]
    ratios = [scenario_data[k]['capex_opex_ratio'] for k in keys]
    robustness = [scenario_data[k]['robustness'] for k in keys]
    
    # Normalize (reverse for cost/sites so smaller is better = higher on radar)
    norm_data = {
        keys[i]: [
            normalize(costs_vals, reverse=True)[i],  # Cheaper = better
            ele_pcts[i] / 100,  # Higher = more elevator use (normalize to 0-1)
            normalize(sites, reverse=True)[i],  # Fewer sites = better
            normalize(ratios, reverse=True)[i],  # Lower ratio = less CAPEX = better
            robustness[i] / 100  # Higher = more robust
        ]
        for i in range(len(keys))
    }
    
    # Radar plot
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    for key in keys:
        scenario = scenarios[key]
        values = norm_data[key] + [norm_data[key][0]]  # Close the loop
        ax.plot(angles, values, 'o-', linewidth=2.5, label=scenario.name, 
                color=scenario.color, markersize=8)
        ax.fill(angles, values, alpha=0.15, color=scenario.color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=9)
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    
    plt.title(f'Figure 9: Scenario Comparison at Y={Y_deadline} Years\n(Normalized Metrics, Higher = Better)',
              fontsize=14, fontweight='bold', y=1.1)
    
    # Add raw values table
    table_text = f"Raw Values at Y={Y_deadline} (Deadline):\n"
    table_text += "-" * 55 + "\n"
    table_text += f"{'Scenario':<15} {'Cost':>10} {'Elevator %':>12} {'Sites':>8}\n"
    for key in keys:
        d = scenario_data[key]
        table_text += f"{scenarios[key].name:<15} ${d['cost']:>8.1f}T {d['elevator_pct']:>10.1f}% {d['sites_needed']:>8.0f}\n"
    
    fig.text(0.02, 0.02, table_text, fontsize=9, family='monospace',
             verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


# ============================================================================
# Main Execution
# ============================================================================

def run_all_visualizations():
    """Generate all 9 figures for 1c comprehensive model."""
    
    ensure_dir(OUTPUT_DIR)
    
    print("=" * 70)
    print("1c Mixed Transport Visualization Module")
    print("=" * 70)
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    # Generate all figures
    print("[1/9] Generating Three-Scenario Panorama...")
    plot_fig01_scenario_panorama(f"{OUTPUT_DIR}/fig1c_01_scenario_panorama.png")
    
    print("[2/9] Generating Allocation + Gantt...")
    plot_fig02_allocation_gantt(f"{OUTPUT_DIR}/fig1c_02_allocation_gantt.png")
    
    print("[3/9] Generating Capacity Evolution...")
    plot_fig03_capacity_evolution(f"{OUTPUT_DIR}/fig1c_03_capacity_evolution.png")
    
    print("[4/9] Generating Cumulative Comparison...")
    plot_fig04_cumulative_comparison(f"{OUTPUT_DIR}/fig1c_04_cumulative_comparison.png")
    
    print("[5/9] Generating Cost Waterfall...")
    plot_fig05_cost_waterfall(f"{OUTPUT_DIR}/fig1c_05_cost_waterfall.png")
    
    print("[6/9] Generating Pareto Deep Analysis...")
    plot_fig06_pareto_deep(f"{OUTPUT_DIR}/fig1c_06_pareto_deep.png")
    
    print("[7/9] Generating Sensitivity Tornado...")
    plot_fig07_sensitivity_tornado(f"{OUTPUT_DIR}/fig1c_07_sensitivity_tornado.png")
    
    print("[8/9] Generating Monte Carlo Distribution...")
    plot_fig08_monte_carlo(f"{OUTPUT_DIR}/fig1c_08_monte_carlo.png")
    
    print("[9/9] Generating Radar Comparison...")
    plot_fig09_radar_comparison(f"{OUTPUT_DIR}/fig1c_09_radar_comparison.png")
    
    print()
    print("=" * 70)
    print("All 9 figures generated successfully!")
    print("=" * 70)


if __name__ == "__main__":
    run_all_visualizations()
