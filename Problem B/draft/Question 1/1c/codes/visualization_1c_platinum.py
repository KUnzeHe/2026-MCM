"""
1c Mixed Transport Visualization (Final "Platinum Quartet")
===========================================================
Optimized visualization suite for Question 1c.
Focuses on narrative clarity, aesthetic unity, and high-impact storytelling.

Charts Generated:
1. Fig1_Strategic_Landscape.png (Scenario Comparison)
2. Fig2_Execution_Plan.png (Allocation & Timeline)
3. Fig3_Capacity_Dynamics.png (Logistic Growth & Crossover)
4. Fig4_Tradeoff_DeepDive.png (Pareto Cost & Split Analysis)
"""

from __future__ import annotations
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.gridspec import GridSpec

# Ensure proper path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from comprehensive_transport_model_v5 import (
    ModelParams, GrowthParams, TransportOptimizationModel, OptimizationResult
)

# ==========================================
# 0. Design System (Golden Trio Unified)
# ==========================================
# Get the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Set output directory relative to script (../image)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "../image")
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 11

COLORS = {
    'elevator': '#2A9D8F',      # Teal (Efficient/Clean)
    'rocket': '#E76F51',        # Coral (Fast/Expensive)
    'mixed': '#264653',         # Charcoal (Combined)
    'highlight': '#F4A261',     # Sandy Brown (Attention)
    'safe': '#E9C46A',          # Yellow (Feasible Zone)
    'text': '#333333',
    'grid': '#DDDDDD'
}

def save_figure(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    print(f"[SAVED] {path}")

# ==========================================
# 1. Fig 1: The Strategic Landscape
# ==========================================
def plot_strategic_landscape():
    print("Generating Fig 1: Strategic Landscape...")
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Define Scenarios
    scenarios = {
        'Conservative\n(Old Tech)': {'p_B': 100, 'r': 0.10, 'style': ':'},
        'Moderate\n(Base Case)': {'p_B': 125, 'r': 0.15, 'style': '-'},
        'Aggressive\n(Starship)': {'p_B': 150, 'r': 0.20, 'style': '--'}
    }
    
    Y_range = np.linspace(15, 60, 50)
    
    for label, cfg in scenarios.items():
        params = ModelParams(p_B=cfg['p_B'], growth=GrowthParams(r=cfg['r']))
        model = TransportOptimizationModel(params)
        
        costs = []
        times = []
        for Y in Y_range:
            res = model.solve(Y)
            if res.feasible:
                costs.append(res.cost_total / 1e12)
                times.append(res.Y)
                
        # Plot Curve
        lw = 3 if 'Moderate' in label else 2
        alpha = 1.0 if 'Moderate' in label else 0.7
        color = COLORS['mixed'] if 'Moderate' in label else COLORS['text']
        
        ax.plot(times, costs, linestyle=cfg['style'], color=color, linewidth=lw, alpha=alpha, label=label)
        
        # Mark Knee Point (Approx)
        if times:
            knee_idx = np.argmin(np.abs(np.array(times) - 24)) # Target 24 years
            if knee_idx < len(times):
                ax.scatter(times[knee_idx], costs[knee_idx], color=COLORS['highlight'], s=80, zorder=5)

    # Highlight Recommended Zone (Deadline 2050 -> 24 years)
    ax.axvline(x=24, color=COLORS['rocket'], linestyle='-', linewidth=2, alpha=0.8)
    ax.text(24.5, ax.get_ylim()[1]*0.9, "Deadline 2050\n(Y=24)", color=COLORS['rocket'], fontweight='bold')
    
    ax.fill_betweenx(ax.get_ylim(), 0, 24, color=COLORS['safe'], alpha=0.1, label='Feasible Time Window')

    ax.set_xlabel('Project Duration (Years)', fontweight='bold')
    ax.set_ylabel('Total Cost (Trillion USD) [NPV]', fontweight='bold')
    ax.set_title('The Price of Haste: Time-Cost Trade-offs', fontsize=14, fontweight='bold')
    ax.set_xlim(14, 50)
    ax.set_ylim(0, 50) # Cap graphical cost to keep flexible scaling
    
    ax.legend(loc='upper right', frameon=True, framealpha=0.9)
    ax.grid(True, linestyle='--', color=COLORS['grid'])
    
    save_figure(fig, 'Fig1_Strategic_Landscape.png')

# ==========================================
# 2. Fig 2: The Execution Plan (Gantt)
# ==========================================
def plot_execution_plan(Y_target=24):
    print("Generating Fig 2: Execution Plan...")
    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(1, 2, width_ratios=[1, 2])
    
    # Run Model
    params = ModelParams(p_B=125.0) # Moderate
    model = TransportOptimizationModel(params)
    res = model.solve(Y_target)
    
    # --- Left: Allocation Donut ---
    ax_pie = fig.add_subplot(gs[0])
    
    sizes = [res.x_opt, res.mR_opt]
    labels = [f"Elevator Chain\n{res.elevator_pct:.1f}%", f"Direct Rockets\n{100-res.elevator_pct:.1f}%"]
    colors = [COLORS['elevator'], COLORS['rocket']]
    
    wedges, texts, autotexts = ax_pie.pie(sizes, labels=labels, colors=colors, autopct='', startangle=90, 
                                          wedgeprops=dict(width=0.4, edgecolor='w'))
    
    # Center Text
    ax_pie.text(0, 0, f"{res.cost_total/1e12:.1f}T\nUSD", ha='center', va='center', fontweight='bold', fontsize=12)
    ax_pie.set_title("Optimal Mass Split\n(Cost Minimization)", fontweight='bold')
    
    # --- Right: Gantt Chart ---
    ax_gantt = fig.add_subplot(gs[1])
    
    # Calculate simple durations
    # Elevators run constantly. Rockets ramp up.
    # We simplify for visualization:
    
    y_pos = [0, 1]
    labels = ['Direct Rockets', 'Elevator Chain']
    
    # Elevator: Steady state from Year 0 to Y
    ax_gantt.barh(1, Y_target, left=0, height=0.5, color=COLORS['elevator'], label='Elevator Operation')
    
    # Rocket: Ramps up. We show full bar but maybe gradient? Simple bar is clearer.
    ax_gantt.barh(0, Y_target, left=0, height=0.5, color=COLORS['rocket'], label='Rocket Campaign')
    
    # Add Infrastructure Phase for Rockets (Logistic ramp visualization)
    # Highlight the first few years as "Ramp Up"
    ax_gantt.barh(0, 5, left=0, height=0.5, color='white', alpha=0.3, hatch='//')
    ax_gantt.text(2.5, 0, "Ramp Up", ha='center', va='center', fontsize=8, color='white', fontweight='bold')
    
    # Deadline
    ax_gantt.axvline(Y_target, color=COLORS['mixed'], linestyle='--', linewidth=2)
    ax_gantt.text(Y_target, 1.6, f"Completion\nYear {Y_target}", ha='center', fontweight='bold')
    
    ax_gantt.set_yticks(y_pos)
    ax_gantt.set_yticklabels(labels, fontweight='bold', fontsize=11)
    ax_gantt.set_xlabel('Project Timeline (Years)')
    ax_gantt.set_title(f'Operational Timeline (Deadline: 2050)', fontweight='bold')
    ax_gantt.set_xlim(0, Y_target+2)
    ax_gantt.grid(axis='x', linestyle='--', color=COLORS['grid'])
    
    # Annotate Volume
    ax_gantt.text(Y_target/2, 1, f"{res.x_opt/1e6:.1f} Mt Transported", ha='center', va='center', color='white', fontweight='bold')
    ax_gantt.text(Y_target/2, 0, f"{res.mR_opt/1e6:.1f} Mt Transported", ha='center', va='center', color='white', fontweight='bold')

    plt.tight_layout()
    save_figure(fig, 'Fig2_Execution_Plan.png')

# ==========================================
# 3. Fig 3: Capacity Dynamics (Logistic)
# ==========================================
def plot_capacity_dynamics():
    print("Generating Fig 3: Capacity Dynamics...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    params = ModelParams(p_B=125.0)
    model = TransportOptimizationModel(params)
    
    times = np.linspace(0, 30, 200)
    
    # --- Top: Daily Rates ---
    # Elevator is constant. Rocket is Logistic.
    elev_rate = model.elevator_bottleneck_rate() / 1e6 # Mt/yr
    rock_rates = [model.rocket_capacity_rate(t)/1e6 for t in times]
    
    ax1.plot(times, [elev_rate]*len(times), color=COLORS['elevator'], linewidth=3, label='Elevator Capacity (Constant)')
    ax1.plot(times, rock_rates, color=COLORS['rocket'], linewidth=3, label='Rocket Capacity (Logistic Growth)')
    
    # Crossover
    cross_idx = np.argwhere(np.diff(np.sign(np.array(rock_rates) - elev_rate))).flatten()
    if len(cross_idx) > 0:
        t_cross = times[cross_idx[0]]
        ax1.scatter(t_cross, elev_rate, color=COLORS['mixed'], s=100, zorder=5)
        ax1.annotate(f"Crossover\nYear {t_cross:.1f}", xy=(t_cross, elev_rate), xytext=(t_cross+2, elev_rate+2),
                     arrowprops=dict(arrowstyle='->', color=COLORS['mixed']), fontweight='bold')

    ax1.set_ylabel('Annual Throughput (Mt/yr)', fontweight='bold')
    ax1.set_title('(A) The Race for Capacity', fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle='--', color=COLORS['grid'])
    
    # --- Bottom: Cumulative ---
    cum_elev = [model.cumulative_elevator_capacity(t)/1e6 for t in times]
    cum_rock = [model.cumulative_rocket_capacity(t)/1e6 for t in times]
    cum_total = np.array(cum_elev) + np.array(cum_rock)
    
    ax2.stackplot(times, cum_elev, cum_rock, labels=['Elevator Cumulative', 'Rocket Cumulative'], 
                  colors=[COLORS['elevator'], COLORS['rocket']], alpha=0.8)
    
    ax2.plot(times, cum_total, color=COLORS['mixed'], linewidth=2, linestyle='--', label='Total Combined')
    
    # Target Line
    ax2.axhline(100, color=COLORS['highlight'], linewidth=2, linestyle='-')
    ax2.text(1, 102, "Target: 100 Mt", color=COLORS['highlight'], fontweight='bold')

    # Intersection
    finish_idx = np.argmax(cum_total >= 100)
    if finish_idx > 0:
        t_finish = times[finish_idx]
        ax2.axvline(t_finish, color=COLORS['mixed'], linestyle=':')
        ax2.text(t_finish, 50, f"  Complete\n  Year {t_finish:.1f}", fontweight='bold')

    ax2.set_ylabel('Cumulative Mass (Mt)', fontweight='bold')
    ax2.set_xlabel('Year', fontweight='bold')
    ax2.set_title('(B) Cumulative Progress to Goal', fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.grid(True, linestyle='--', color=COLORS['grid'])
    
    plt.tight_layout()
    save_figure(fig, 'Fig3_Capacity_Dynamics.png')

# ==========================================
# 4. Fig 4: Trade-off Deep Dive
# ==========================================
def plot_tradeoff_deepdive():
    print("Generating Fig 4: Trade-off Deep Dive...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    params = ModelParams(p_B=125.0)
    model = TransportOptimizationModel(params)
    
    Y_range = np.linspace(15, 40, 40)
    
    feas_Y = []
    mode_split = [] # Elevator %
    costs_capex = []
    costs_opex = []
    
    for Y in Y_range:
        res = model.solve(Y)
        if res.feasible:
            feas_Y.append(Y)
            mode_split.append(res.elevator_pct)
            costs_capex.append(res.cost_capex/1e12)
            costs_opex.append(res.cost_opex/1e12)
            
    # --- Left: Modal Split ---
    ax1 = axes[0]
    ax1.plot(feas_Y, mode_split, color=COLORS['elevator'], linewidth=3)
    ax1.fill_between(feas_Y, 0, mode_split, color=COLORS['elevator'], alpha=0.2, label='Elevator Share')
    ax1.fill_between(feas_Y, mode_split, 100, color=COLORS['rocket'], alpha=0.2, label='Rocket Share')
    
    ax1.set_xlabel('Project Duration (Years)', fontweight='bold')
    ax1.set_ylabel('Transport Share (%)', fontweight='bold')
    ax1.set_title('Optimal Mode Split vs Time', fontweight='bold')
    ax1.set_ylim(0, 100)
    ax1.legend(loc='center right')
    ax1.grid(True, linestyle='--', color=COLORS['grid'])
    
    # Mark Knee Point (24 yrs)
    idx_24 = np.argmin(np.abs(np.array(feas_Y) - 24))
    val_24 = mode_split[idx_24]
    ax1.scatter(feas_Y[idx_24], val_24, color=COLORS['mixed'], s=80, zorder=5)
    ax1.text(feas_Y[idx_24], val_24+5, f"At Y=24:\n{val_24:.0f}% Elevator", ha='center', fontweight='bold')

    # --- Right: Cost Structure (Log Scale Update) ---
    ax2 = axes[1]
    
    # Use stackplot with Log Scale
    ax2.stackplot(feas_Y, costs_capex, costs_opex, labels=['CAPEX (Infra)', 'OPEX (Operations)'], 
                  colors=[COLORS['safe'], COLORS['mixed']])
    
    ax2.set_xlabel('Project Duration (Years)', fontweight='bold')
    ax2.set_ylabel('Total Cost (Trillion USD) [Log Scale]', fontweight='bold')
    ax2.set_title('Cost Composition vs Time (Log Scale)', fontweight='bold')
    
    # Log Scale Settings
    ax2.set_yscale('log')
    ax2.set_ylim(bottom=0.05) # Limit bottom to $50B to make CAPEX visible
    ax2.grid(True, linestyle='--', color=COLORS['grid'], which='both', alpha=0.5)
    
    # Mark Knee Point Cost
    pk_capex = costs_capex[idx_24]
    pk_opex = costs_opex[idx_24]
    pk_total = pk_capex + pk_opex
    
    ax2.axvline(feas_Y[idx_24], color='white', linestyle='--', alpha=0.5)
    
    # Annotate Total (Top)
    ax2.text(feas_Y[idx_24], pk_total*1.15, f"${pk_total:.1f}T", ha='center', va='bottom', fontweight='bold')
    
    # Annotate CAPEX (Bottom - explicit visibility)
    # Use an arrow if it's too squeezed, or a box if space permits
    ax2.annotate(f"Infra\n${pk_capex:.2f}T", 
                 xy=(feas_Y[idx_24], pk_capex/1.5), 
                 xytext=(feas_Y[idx_24], pk_capex/5),
                 ha='center', va='top', color='white', fontsize=10, fontweight='bold',
                 arrowprops=dict(arrowstyle='-', color='white', lw=1))

    ax2.legend(loc='lower left')

    plt.tight_layout()
    save_figure(fig, 'Fig4_Tradeoff_DeepDive.png')


# ==========================================
# Main Execution
# ==========================================
def main():
    print(">>> Starting Platinum Quartet Visualization (1c)...")
    plot_strategic_landscape()
    plot_execution_plan()
    plot_capacity_dynamics()
    plot_tradeoff_deepdive()
    print(">>> All Optimized Figures Generated Successfully!")

if __name__ == "__main__":
    main()
