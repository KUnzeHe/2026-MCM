import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Ensure proper path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from water_supply_analysis import (
    POPULATION, COST_ELEVATOR, COST_ROCKET, 
    CAPACITY_ELEVATOR_TOTAL, DISRUPTION_DAYS_ELEVATOR, 
    sc_baseline, sc_optimized, sc_pessimistic
)

# ==========================================
# 0. Design System (Golden Trio Unified)
# ==========================================
# Get the directory where the script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Set output directory relative to script (../image)
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "../image")
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 12,
    'axes.labelweight': 'bold',
    'lines.linewidth': 2.5,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'figure.dpi': 300,
    'axes.facecolor': '#F8F9FA',
    'figure.facecolor': '#F8F9FA'
})

COLORS = {
    'primary': '#2A9D8F',    # Teal - Water/Life, Sustainable
    'secondary': '#E76F51',  # Coral - Rocket/Cost, Danger
    'tertiary': '#264653',   # Charcoal - Infrastructure, Deep Background
    'neutral': '#E9C46A',    # Sand/Gold - Warning, Critical Levels
    'slate': '#64748B',      # Slate - Structural elements
    'background': '#F8F9FA'  # Off-white background
}

def save_figure(fig, name):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches='tight', facecolor=COLORS['background'])
    print(f"[SAVED] {path}")

# ==========================================
# 1. Plot A: The Feasibility Frontier
# ==========================================
def plot_feasibility_frontier():
    """
    Plot A: The Feasibility Frontier (Dual Axis)
    Unified "Platinum" Style
    """
    print("Generating A_feasibility_frontier_platinum.png...")
    recycling_rates = np.linspace(0.50, 0.999, 200)
    base_daily_demand = 75 # tons/day (Standard ISS-like)
    
    annual_imports = []
    costs_elevator = []
    capacity_occupation = []

    for r in recycling_rates:
        # Net annual import (Tons)
        net_import = POPULATION * (base_daily_demand / 1000) * (1 - r) * 365
        annual_imports.append(net_import)
        
        # Cost (Billion USD) - Elevator only (as Rocket is off-scale)
        costs_elevator.append(net_import * 1000 * COST_ELEVATOR / 1e9)
        
        # Capacity Occupation (%)
        capacity_occupation.append((net_import / CAPACITY_ELEVATOR_TOTAL) * 100)

    fig, ax1 = plt.subplots(figsize=(12, 7))

    # Plot 1: Capacity (Primary Metric)
    color_cap = COLORS['primary']
    ax1.set_xlabel('Recycling Efficiency ($\eta$)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('System Capacity Occupation (%)', color=color_cap, fontsize=12, fontweight='bold')
    # Change x to percentage for readability
    line1, = ax1.plot(recycling_rates * 100, capacity_occupation, color=color_cap, linewidth=3, label='Capacity Utilization')
    ax1.tick_params(axis='y', labelcolor=color_cap)
    
    # Fill safe zone
    ax1.fill_between(recycling_rates * 100, 0, capacity_occupation, where=(np.array(capacity_occupation) < 100), 
                     color=color_cap, alpha=0.1)

    # Threshold Lines
    ax1.axhline(y=100, color=COLORS['tertiary'], linestyle='-', linewidth=2, alpha=0.8)
    ax1.text(52, 102, 'PHYSICAL LIMIT (100% Capacity)', color=COLORS['tertiary'], fontweight='bold')
    
    # Heavy Load Line (Restored)
    ax1.axhline(y=50, color=COLORS['neutral'], linestyle='--', linewidth=2, alpha=0.8)
    ax1.text(52, 52, 'Heavy Load (50%)', color=COLORS['neutral'], fontweight='bold')
    
    # Grid (Explicitly enabled with darker alpha)
    ax1.grid(True, linestyle='--', alpha=0.5, color='#CCCCCC')

    # Dual Axis for Cost
    ax2 = ax1.twinx()
    color_cost = COLORS['secondary']
    ax2.set_ylabel('Annual Resupply Cost (Billion USD)', color=color_cost, fontsize=12, fontweight='bold')
    line2, = ax2.plot(recycling_rates * 100, costs_elevator, color=color_cost, linestyle='--', linewidth=2.5, label='Annual Cost (Elevator)')
    ax2.tick_params(axis='y', labelcolor=color_cost)
    
    # Mark Scenarios
    # Baseline (90%)
    idx_90 = np.abs(recycling_rates - 0.90).argmin()
    ax1.scatter(90, capacity_occupation[idx_90], color=COLORS['tertiary'], s=120, zorder=5, edgecolors='white', linewidth=2)
    ax1.annotate('Baseline (90%)\n~51% Load', xy=(90, capacity_occupation[idx_90]), xytext=(82, 60),
                 arrowprops=dict(facecolor=COLORS['tertiary'], shrink=0.05, alpha=0.8),
                 color=COLORS['tertiary'], fontweight='bold', ha='center')
    
    # Optimized (98%)
    idx_98 = np.abs(recycling_rates - 0.98).argmin()
    ax1.scatter(98, capacity_occupation[idx_98], color=COLORS['neutral'], s=120, zorder=5, edgecolors='white', linewidth=2)
    ax1.annotate('Target (98%)\n~6% Load', xy=(98, capacity_occupation[idx_98]), xytext=(92, 20),
                 arrowprops=dict(facecolor=COLORS['neutral'], shrink=0.05),
                 color=COLORS['tertiary'], fontweight='bold', ha='center')

    # Combined Legend
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    # Moved legend to top-right corner (inside plot) to avoid title overlap
    ax1.legend(lines, labels, loc='upper right', borderaxespad=1.5, ncol=1, 
               frameon=True, framealpha=0.95, facecolor='white', edgecolor=COLORS['slate'])

    plt.title('The Feasibility Frontier: Water Recycling vs. Logistics Load', fontsize=16, pad=20)
    plt.tight_layout()
    save_figure(fig, 'A_feasibility_frontier_platinum.png')


# ==========================================
# 2. Plot B: The Cost Chasm
# ==========================================
def plot_cost_chasm():
    """
    Plot B: The Cost Chasm (Log Scale)
    Rocket vs Elevator costs
    Unified "Platinum" Style
    """
    print("Generating B_cost_chasm_platinum.png...")
    recycling_rates = np.linspace(0.60, 0.999, 200)
    base_daily_demand = 75 
    
    costs_elevator = []
    costs_rocket = []
    
    for r in recycling_rates:
        net_import = POPULATION * (base_daily_demand / 1000) * (1 - r) * 365
        costs_elevator.append(net_import * 1000 * COST_ELEVATOR) # Actual USD
        costs_rocket.append(net_import * 1000 * COST_ROCKET)     # Actual USD

    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Lines
    ax.plot(recycling_rates*100, costs_rocket, color=COLORS['secondary'], linewidth=3, label='Rocket Transport')
    ax.fill_between(recycling_rates*100, costs_elevator, costs_rocket, color=COLORS['secondary'], alpha=0.1) # Cost Chasm area
    
    ax.plot(recycling_rates*100, costs_elevator, color=COLORS['primary'], linewidth=3, label='Space Elevator')
    ax.fill_between(recycling_rates*100, 1e7, costs_elevator, color=COLORS['primary'], alpha=0.1) # Sustainable area
    
    # Key differentiation: Log Scale
    ax.set_yscale('log')
    
    ax.set_xlabel('Recycling Efficiency (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Annual Cost (USD) - Log Scale', fontsize=12, fontweight='bold')
    ax.set_title('The Cost Chasm: Traditional Rocket vs. Space Elevator', fontsize=16, pad=20)
    
    # Format Y axis logs
    ax.grid(True, which="both", ls="-", alpha=0.15, color=COLORS['tertiary'])
    ax.legend(loc='upper right', frameon=True, fancybox=True, framealpha=0.9, fontsize=11)
    
    # Annotations
    ax.text(70, 2e11, 'Rocket Area:\nEconomic Suicide', color=COLORS['secondary'], fontsize=14, fontweight='bold', alpha=0.8)
    ax.text(70, 5e9, 'Elevator Area:\nSustainable', color=COLORS['primary'], fontsize=14, fontweight='bold', alpha=0.8)
    
    # Gap annotation
    arrow_x = 80
    idx = np.abs(recycling_rates*100 - arrow_x).argmin()
    y_rocket = costs_rocket[idx]
    y_elev = costs_elevator[idx]
    
    ax.annotate('', xy=(arrow_x, y_rocket), xytext=(arrow_x, y_elev),
                arrowprops=dict(arrowstyle='<->', color=COLORS['tertiary'], lw=1.5))
    ax.text(arrow_x + 1, (y_rocket * y_elev)**0.5, '25x Cost Reduction', va='center', color=COLORS['tertiary'], fontweight='bold')

    plt.tight_layout()
    save_figure(fig, 'B_cost_chasm_platinum.png')


# ==========================================
# 3. Plot C: Strategic Reserve Accumulation
# ==========================================
def plot_timeline_accumulation():
    """
    Plot C: Strategic Reserve Accumulation Timeline
    Showing the pre-deployment phase.
    Unified "Platinum" Style
    """
    print("Generating C_reserve_timeline_platinum.png...")
    # Parameters for Baseline Scenario (Worst case safety needs)
    net_daily_demand = sc_baseline.get_gross_daily_demand() * (1 - sc_baseline.eta) # Tons/day
    safety_days = DISRUPTION_DAYS_ELEVATOR * 1.5 # 45 days
    target_stock = net_daily_demand * safety_days # ~33,750 tons
    
    # Assuming we use 50% of elevator capacity for accumulation
    daily_transport_cap = (CAPACITY_ELEVATOR_TOTAL / 365) * 0.5 
    
    days_to_accumulate = target_stock / daily_transport_cap
    
    # Timeline Construction
    t_pre = np.linspace(-days_to_accumulate, 0, 100)
    stock_pre = np.linspace(0, target_stock, 100)
    
    t_op = np.linspace(0, 100, 200)
    stock_op = np.ones_like(t_op) * target_stock
    
    # Simulate a disruption
    disruption_start = 40
    disruption_len = 30
    
    for i, t in enumerate(t_op):
        if t >= disruption_start and t < disruption_start + disruption_len:
            days_into = t - disruption_start
            stock_op[i] = target_stock - (net_daily_demand * days_into)
        elif t >= disruption_start + disruption_len:
            days_recov = t - (disruption_start + disruption_len)
            current = target_stock - (net_daily_demand * disruption_len) + (daily_transport_cap * days_recov)
            stock_op[i] = min(target_stock, current)

    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot Pre-deployment (Accumulation)
    ax.plot(t_pre, stock_pre, color=COLORS['primary'], linestyle='--', linewidth=3, label='Pre-deployment Accumulation')
    ax.fill_between(t_pre, 0, stock_pre, color=COLORS['primary'], alpha=0.15)
    
    # Plot Operation (Resilience)
    ax.plot(t_op, stock_op, color=COLORS['tertiary'], linewidth=3, label='Operational Reserve')
    ax.fill_between(t_op, 0, stock_op, color=COLORS['tertiary'], alpha=0.1)
    
    # Critical Level
    min_survival_stock = target_stock * 0.1 # 10% bottom line
    ax.axhline(y=min_survival_stock, color=COLORS['secondary'], linestyle=':', linewidth=2.5, label='Critical Survival Level')

    # Annotations
    # Colony Opens
    ax.annotate('Colony Opens (T=0)', xy=(0, target_stock), xytext=(-20, target_stock*1.15),
                arrowprops=dict(facecolor=COLORS['tertiary'], shrink=0.05), fontweight='bold', ha='center', color=COLORS['tertiary'])
                
    # Disruption
    ax.annotate('Elevator Failure Event', xy=(40, target_stock), xytext=(45, target_stock*1.25),
                arrowprops=dict(facecolor=COLORS['secondary'], shrink=0.05), color=COLORS['secondary'], fontweight='bold')
    
    # Draw Disruption Area
    ax.axvspan(disruption_start, disruption_start + disruption_len, color=COLORS['secondary'], alpha=0.1, label='Disruption Period')

    ax.set_xlabel('Timeline (Days)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Water Strategic Reserve (Tons)', fontsize=12, fontweight='bold')
    ax.set_title(f'Strategic Reserve Timeline: Accumulation & Resilience', fontsize=16, pad=20)
    
    ax.legend(loc='lower left', frameon=True, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_figure(fig, 'C_reserve_timeline_platinum.png')

if __name__ == "__main__":
    print(">>> Starting Platinum Trio Visualization (Q3)...")
    plot_feasibility_frontier()
    plot_cost_chasm()
    plot_timeline_accumulation()
    print(">>> All Optimized Figures Generated Successfully!")
