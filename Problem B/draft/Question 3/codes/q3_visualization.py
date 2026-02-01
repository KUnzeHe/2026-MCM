import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Import the model logic from the main analysis script
# Assuming q3_model.py is in the same directory or adjust path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from water_supply_analysis import (
    POPULATION, COST_ELEVATOR, COST_ROCKET, 
    CAPACITY_ELEVATOR_TOTAL, DISRUPTION_DAYS_ELEVATOR, 
    sc_baseline, sc_optimized, sc_pessimistic
)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'image')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def plot_feasibility_frontier():
    """
    Plot A: The Feasibility Frontier (Dual Axis)
    X: Recycling Rate
    Y1: Capacity Occupation %
    Y2: Annual Cost $B
    """
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

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_cap = 'tab:red'
    ax1.set_xlabel('Recycling Efficiency ($\eta$)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('System Capacity Occupation (%)', color=color_cap, fontsize=12, fontweight='bold')
    ax1.plot(recycling_rates * 100, capacity_occupation, color=color_cap, linewidth=2.5, label='Capacity Load')
    ax1.tick_params(axis='y', labelcolor=color_cap)
    
    # Threshold Lines
    ax1.axhline(y=100, color='black', linestyle='-', linewidth=1.5, alpha=0.8)
    ax1.text(52, 102, 'PHYSICAL LIMIT (100%)', color='black', fontweight='bold')
    
    ax1.axhline(y=50, color='orange', linestyle='--', alpha=0.6)
    ax1.text(52, 52, 'Heavy Load (50%)', color='orange')

    # Dual Axis for Cost
    ax2 = ax1.twinx()
    color_cost = 'tab:blue'
    ax2.set_ylabel('Annual Resupply Cost (Billion USD)', color=color_cost, fontsize=12, fontweight='bold')
    ax2.plot(recycling_rates * 100, costs_elevator, color=color_cost, linestyle='-.', linewidth=2, label='Annual Cost (Elevator)')
    ax2.tick_params(axis='y', labelcolor=color_cost)
    
    # Mark Scenarios
    # Baseline (90%)
    idx_90 = np.abs(recycling_rates - 0.90).argmin()
    ax1.scatter(90, capacity_occupation[idx_90], color=color_cap, s=100, zorder=5)
    ax1.annotate('Baseline (90%)\n~51% Load', xy=(90, capacity_occupation[idx_90]), xytext=(80, 70),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    
    # Optimized (98%)
    idx_98 = np.abs(recycling_rates - 0.98).argmin()
    ax1.scatter(98, capacity_occupation[idx_98], color='green', s=100, zorder=5)
    ax1.annotate('Target (98%)\n~6% Load', xy=(98, capacity_occupation[idx_98]), xytext=(92, 20),
                 arrowprops=dict(facecolor='black', shrink=0.05))

    plt.title('The Feasibility Frontier: Water Recycling vs. Logistics Load', fontsize=14, pad=20)
    fig.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'A_feasibility_frontier.png'), dpi=300)
    print("Generated: A_feasibility_frontier.png")

def plot_cost_chasm():
    """
    Plot B: The Cost Chasm (Log Scale)
    Rocket vs Elevator costs
    """
    recycling_rates = np.linspace(0.60, 0.999, 200)
    base_daily_demand = 75 
    
    costs_elevator = []
    costs_rocket = []
    
    for r in recycling_rates:
        net_import = POPULATION * (base_daily_demand / 1000) * (1 - r) * 365
        costs_elevator.append(net_import * 1000 * COST_ELEVATOR) # Actual USD
        costs_rocket.append(net_import * 1000 * COST_ROCKET)     # Actual USD

    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(recycling_rates*100, costs_rocket, color='#d62728', linewidth=2, label='Rocket Transport')
    ax.plot(recycling_rates*100, costs_elevator, color='#1f77b4', linewidth=2, label='Space Elevator')
    
    # Key differentiation: Log Scale
    ax.set_yscale('log')
    
    # Fill between
    ax.fill_between(recycling_rates*100, costs_elevator, costs_rocket, color='gray', alpha=0.1, label='Economic Value Added')

    ax.set_xlabel('Recycling Efficiency (%)', fontsize=12)
    ax.set_ylabel('Annual Cost (USD) - Log Scale', fontsize=12)
    ax.set_title('The Cost Chasm: Traditional Rocket vs. Space Elevator', fontsize=14)
    
    # Format Y axis logs
    ax.grid(True, which="both", ls="-", alpha=0.2)
    ax.legend(loc='upper right', frameon=True)
    
    # Annotations
    ax.text(70, 1e11, 'Rocket Area:\nEconomic Suicide', color='red', fontsize=12, fontweight='bold', alpha=0.5)
    ax.text(70, 1e9, 'Elevator Area:\nSustainable', color='blue', fontsize=12, fontweight='bold', alpha=0.5)
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'B_cost_chasm.png'), dpi=300)
    print("Generated: B_cost_chasm.png")

def plot_timeline_accumulation():
    """
    Plot C: Strategic Reserve Accumulation Timeline
    Showing the pre-deployment phase.
    """
    # Parameters for Baseline Scenario (Worst case safety needs)
    net_daily_demand = sc_baseline.get_gross_daily_demand() * (1 - sc_baseline.eta) # Tons/day
    safety_days = DISRUPTION_DAYS_ELEVATOR * 1.5 # 45 days
    target_stock = net_daily_demand * safety_days # ~33,750 tons
    
    # Assuming we use 50% of elevator capacity for accumulation to not block everything else
    daily_transport_cap = (CAPACITY_ELEVATOR_TOTAL / 365) * 0.5 
    
    days_to_accumulate = target_stock / daily_transport_cap
    
    # Timeline Construction
    # T=0 is Colony Opening. T negative is pre-deployment.
    t_pre = np.linspace(-days_to_accumulate, 0, 100)
    stock_pre = np.linspace(0, target_stock, 100)
    
    t_op = np.linspace(0, 100, 200)
    stock_op = np.ones_like(t_op) * target_stock
    
    # Simulate a disruption at day 40
    disruption_start = 40
    disruption_len = 30
    
    # Modify stock for disruption
    for i, t in enumerate(t_op):
        if t >= disruption_start and t < disruption_start + disruption_len:
            # Stock drops by daily demand
            days_into = t - disruption_start
            stock_op[i] = target_stock - (net_daily_demand * days_into)
        elif t >= disruption_start + disruption_len:
            # Recovery phase (refill)
            days_recov = t - (disruption_start + disruption_len)
            current = target_stock - (net_daily_demand * disruption_len) + (daily_transport_cap * days_recov)
            stock_op[i] = min(target_stock, current)

    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot Pre-deployment
    ax.plot(t_pre, stock_pre, color='green', linestyle='--', linewidth=2, label='Pre-deployment Accumulation')
    ax.fill_between(t_pre, 0, stock_pre, color='green', alpha=0.1)
    
    # Plot Operation
    ax.plot(t_op, stock_op, color='blue', linewidth=2, label='Operational Reserve')
    ax.fill_between(t_op, 0, stock_op, color='blue', alpha=0.1)
    
    # Critical Level
    min_survival_stock = target_stock * 0.1 # 10% bottom line
    ax.axhline(y=min_survival_stock, color='red', linestyle=':', linewidth=2, label='Critical Survival Level')

    # Annotations
    ax.annotate('Colony Opens (T=0)', xy=(0, target_stock), xytext=(-20, target_stock*1.1),
                arrowprops=dict(facecolor='black', shrink=0.05), fontweight='bold')
                
    ax.annotate('Disruption Event\n(Elevator Cable Failure)', xy=(40, target_stock), xytext=(45, target_stock*1.2),
                arrowprops=dict(facecolor='red', shrink=0.05), color='red')

    ax.set_xlabel('Timeline (Days)', fontsize=12)
    ax.set_ylabel('Water Strategic Reserve (Tons)', fontsize=12)
    ax.set_title(f'Strategic Reserve Timeline: Accumulation & Resilience (Safety Stock: {target_stock/1000:.1f}k tons)', fontsize=14)
    ax.legend(loc='lower left')
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'C_reserve_timeline.png'), dpi=300)
    print("Generated: C_reserve_timeline.png")

if __name__ == "__main__":
    print("Generating visualizations...")
    plot_feasibility_frontier()
    plot_cost_chasm()
    plot_timeline_accumulation()
    print(f"All images saved to {OUTPUT_DIR}")
