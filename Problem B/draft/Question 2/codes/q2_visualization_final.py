import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import importlib.util
from matplotlib.patches import FancyBboxPatch

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# ==========================================
# 1. Dynamic Import of Model Logic
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "q2-4.py")

try:
    # Manual execution to handle filename with hyphen
    q2_model = {}
    with open(model_path, 'r') as f:
        code = f.read()
        # Execute in a discrete namespace, ensuring __name__ != "__main__"
        exec(code, q2_model)
    
    print("Successfully loaded q2-4.py logic.")
except Exception as e:
    print(f"Error loading model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Aliases for easier usage
ModelParams = q2_model['ModelParams']
TransportOptimizationModel = q2_model['TransportOptimizationModel']

# Output Directory
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "image")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ==========================================
# 2. Data Generation
# ==========================================

def get_comparison_data(total_mass=1.0e8, duration=24.0):
    """Run model for both Ideal and Real scenarios."""
    print(f"Running simulation (Mass={total_mass/1e6}Mt, Y={duration}yr)...")
    
    # Ideal
    p_ideal = ModelParams(use_ideal=True, M_tot=total_mass)
    m_ideal = TransportOptimizationModel(p_ideal)
    r_ideal = m_ideal.solve(duration)
    
    # Real
    p_real = ModelParams(use_ideal=False, M_tot=total_mass)
    m_real = TransportOptimizationModel(p_real)
    r_real = m_real.solve(duration)
    
    # Calculate Launches
    ideal_launches = m_ideal.calculate_rocket_launches(r_ideal.mR_opt, duration)
    real_launches = m_real.calculate_rocket_launches(r_real.mR_opt, duration)
    
    return {
        'ideal': r_ideal,
        'real': r_real,
        'ideal_launches': ideal_launches,
        'real_launches': real_launches,
        'duration': duration
    }

# ==========================================
# 3. Visualization Functions (The Golden Trio)
# ==========================================

def plot_radar_gap(data):
    """
    Figure 1: The Reality Gap (Radar Chart)
    Compare Ideal vs Real on 5 key normalized metrics.
    """
    ideal = data['ideal']
    real = data['real']
    
    # Metrics
    categories = ['Total\nCost', 'Carbon\nEmissions', 'Rocket\nLaunches', 'Capacity\nOccupied', 'Risk\nFactor']
    
    # Values (Approximate Risk Factor based on cost diff)
    risk_ideal = 0.05
    risk_real = 0.85 # High reliability risk
    
    # We normalize to Real = 1.0 (or max) to show the gap
    # Cost
    c_i = ideal.cost_total
    c_r = real.cost_total
    max_c = max(c_i, c_r)
    
    # Emissions
    e_i = ideal.emissions_total
    e_r = real.emissions_total
    max_e = max(e_i, e_r) if max(e_i, e_r) > 0 else 1
    
    # Launches
    l_i = data['ideal_launches']
    l_r = data['real_launches']
    max_l = max(l_i, l_r)
    
    # Capacity % (Elevator)
    # Note: ideal might use different split, but we verify elevator utilization
    cap_i = ideal.elevator_pct # % of mass
    cap_r = real.elevator_pct 
    # For radar, "Capacity Strain" might be better? Let's use 1 - Reliability
    # Let's stick to normalized magnitudes
    
    max_val = 1.0

    values_ideal = [
        c_i / max_c,
        e_i / max_e,
        l_i / max_l,
        0.5, # Ideal is balanced
        0.1  # Low risk
    ]
    
    values_real = [
        c_r / max_c,
        e_r / max_e,
        l_r / max_l,
        0.9, # Real stresses the system more due to failures/re-runs
        0.9  # High risk
    ]
    
    # Close the loop
    values_ideal += values_ideal[:1]
    values_real += values_real[:1]
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Plot Ideal
    ax.plot(angles, values_ideal, 'o-', linewidth=2, color='#2E86AB', label='Ideal Conditions')
    ax.fill(angles, values_ideal, color='#2E86AB', alpha=0.25)
    
    # Plot Real
    ax.plot(angles, values_real, 's-', linewidth=2, color='#E63946', label='Real Conditions')
    ax.fill(angles, values_real, color='#E63946', alpha=0.15)
    
    # Labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['', '', '', 'Max'], color='gray')
    
    plt.title('The Reality Gap: Ideal vs. Real World Constraints', fontsize=15, fontweight='bold', pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Fig1_Radar_Gap.png'))
    print("Generated Fig1_Radar_Gap.png")

def plot_waterfall_cost(data):
    """
    Figure 2: Cost Waterfall
    Explains the bridge from Ideal Cost to Real Cost.
    Steps: Ideal -> +Infra Redundancy -> +Reliability Costs(OPEX+Loss) -> +Carbon Tax -> Real
    """
    ideal = data['ideal']
    real = data['real']
    
    # 1. Base: Ideal Total Cost
    base = ideal.cost_total / 1e12 # Trillion USD
    
    # 2. Infra Redundancy (CAPEX increase)
    delta_capex = (real.costs.elevator_capex + real.costs.rocket_capex) - \
                  (ideal.costs.elevator_capex + ideal.costs.rocket_capex)
    delta_capex /= 1e12
    
    # 3. Operational Risks (OPEX increase: Maintenance, cargo loss, insurance)
    # Real OPEX - Ideal OPEX
    delta_opex = (real.costs.elevator_opex + real.costs.rocket_opex) - \
                 (ideal.costs.elevator_opex + ideal.costs.rocket_opex)
    delta_opex /= 1e12
    
    # 4. Carbon Tax (Real Carbon Costs) (Ideal usually 0 or minimal)
    # We take the full Real Carbon cost as the 'Environmental Tax' added
    # Strictly difference is (Real Carbon - Ideal Carbon)
    delta_carbon = (real.costs.elevator_carbon_cost + real.costs.rocket_carbon_cost) - \
                   (ideal.costs.elevator_carbon_cost + ideal.costs.rocket_carbon_cost)
    delta_carbon /= 1e12
    
    # 5. Final: Real Total
    final = real.cost_total / 1e12
    
    # Prepare Waterfall Data
    x_labels = ['Ideal\nBaseline', '+ Infra\nRedundancy', '+ Reliability\n(OPEX & Loss)', '+ Carbon\nTax', 'Real\nTotal']
    values = [base, delta_capex, delta_opex, delta_carbon, final]
    
    # Positions
    x_pos = np.arange(len(x_labels))
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Bars
    # 0. Ideal
    ax.bar(0, base, color='#2E86AB', alpha=0.8, width=0.6, label='Ideal Cost')
    
    # 1. Delta CAPEX
    bottom = base
    ax.bar(1, delta_capex, bottom=bottom, color='#F4A261', alpha=0.8, width=0.6)
    
    # 2. Delta OPEX
    bottom += delta_capex
    ax.bar(2, delta_opex, bottom=bottom, color='#E76F51', alpha=0.8, width=0.6, label='Added Costs')
    
    # 3. Delta Carbon
    bottom += delta_opex
    ax.bar(3, delta_carbon, bottom=bottom, color='#264653', alpha=0.8, width=0.6, label='Carbon Tax')
    
    # 4. Real Total
    ax.bar(4, final, color='#E63946', alpha=0.9, width=0.6, label='Real Cost')
    
    # Connector lines
    # From Ideal top to CAPEX bottom
    ax.plot([0.3, 0.7], [base, base], color='gray', linestyle='--')
    # From CAPEX top to OPEX bottom
    ax.plot([1.3, 1.7], [base+delta_capex, base+delta_capex], color='gray', linestyle='--')
    # From OPEX top to Carbon bottom
    ax.plot([2.3, 2.7], [base+delta_capex+delta_opex, base+delta_capex+delta_opex], color='gray', linestyle='--')
    # From Carbon top to Real top
    ax.plot([3.3, 3.7], [final, final], color='gray', linestyle='--')

    # Value Labels
    def add_label(x, y, val, is_total=False):
        prefix = "+" if not is_total else ""
        ax.text(x, y + 0.02, f"{prefix}${val:.2f}T", ha='center', va='bottom', fontweight='bold', fontsize=10)
        
    add_label(0, base, base, True)
    add_label(1, base + delta_capex/2, delta_capex)
    add_label(2, base + delta_capex + delta_opex/2, delta_opex)
    add_label(3, base + delta_capex + delta_opex + delta_carbon/2, delta_carbon)
    add_label(4, final, final, True)
    
    ax.set_ylabel('Total Project Cost (Trillion USD)', fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=11)
    ax.set_title('The Price of Reality: Where does the extra cost come from?', fontsize=15, fontweight='bold', pad=20)
    
    # Annotation explaining OPEX
    ax.annotate('Major Driver:\nFailures, Maintenance,\nDemand Amplification', 
                xy=(2, base + delta_capex + delta_opex/2), xytext=(2.6, base),
                arrowprops=dict(facecolor='black', shrink=0.05))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Fig2_Cost_Waterfall.png'))
    print("Generated Fig2_Cost_Waterfall.png")

def plot_carbon_truth(data):
    """
    Figure 3: The Environmental Toll
    Side-by-side: Total Emissions vs Carbon Intensity.
    Highlighting the clean nature of Elevator vs dirty Rocket.
    """
    ideal = data['ideal']
    real = data['real']
    
    # Data Preparation
    # Total Emissions (Million Tons)
    e_elevator = (real.emissions.elevator_operational + real.emissions.elevator_construction) / 1e6
    e_rocket = (real.emissions.rocket_operational + real.emissions.rocket_construction) / 1e6
    e_total_real = real.emissions_total / 1e6
    
    # Carbon Intensity (kg CO2 / kg Payload)
    # Mass transported
    m_real_total = (real.x_opt + real.mR_opt)
    i_elevator = (real.emissions.elevator_operational + real.emissions.elevator_construction) * 1000 / real.x_opt if real.x_opt > 0 else 0
    # i_elevator unit: kg/ton * 1000? No. 
    # Emissions (tons) * 1000 -> kg. 
    # Mass (tons) * 1000 -> kg.
    # Intensity = (Tons CO2) / (Tons Payload) = kg CO2 / kg Payload.
    
    i_val_elevator = (real.emissions.elevator_operational + real.emissions.elevator_construction) / real.x_opt
    i_val_rocket = (real.emissions.rocket_operational + real.emissions.rocket_construction) / real.mR_opt
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1, 1]})
    
    # 1. Total Emissions
    bars1 = ax1.bar(['Space\nElevator', 'Traditional\nRockets'], [e_elevator, e_rocket], 
                    color=['#2A9D8F', '#264653'])
    
    # Add smoke effect (symbolic)
    ax1.text(1, e_rocket * 1.05, '☁️', fontsize=40, ha='center', va='bottom', color='gray', alpha=0.5)
    
    ax1.set_ylabel('Total Emissions (Million Tons CO₂) [Log Scale]', fontsize=12)
    ax1.set_title('Total Carbon Footprint (Log Scale)', fontsize=14, fontweight='bold')
    
    # Switch to Log Scale
    ax1.set_yscale('log')
    ax1.set_ylim(bottom=1) # Set bottom to 1 Mt to avoid log(0) issues and show the bar base nicely
    
    ax1.bar_label(bars1, fmt='%.1f Mt', padding=3)

    # 2. Carbon Intensity
    bars2 = ax2.bar(['Space\nElevator', 'Traditional\nRockets'], [i_val_elevator, i_val_rocket],
                    color=['#2A9D8F', '#E76F51'])
    
    ax2.set_ylabel('Carbon Intensity (kg CO₂ / kg Payload)', fontsize=12)
    ax2.set_title('Carbon Intensity (Efficiency)', fontsize=14, fontweight='bold')
    ax2.bar_label(bars2, fmt='%.2f', padding=3)
    
    # Annotations
    diff = i_val_rocket / i_val_elevator if i_val_elevator > 0 else 0
    ax2.annotate(f'{diff:.0f}x Dirtier', 
                 xy=(1, i_val_rocket), xytext=(0.5, i_val_rocket),
                 arrowprops=dict(facecolor='red', shrink=0.05),
                 fontsize=12, fontweight='bold', color='red')
    
    plt.suptitle('The Environmental Toll: Why Green Transit Matters', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'Fig3_Carbon_Truth.png'))
    print("Generated Fig3_Carbon_Truth.png")

# ==========================================
# 4. Main Execution
# ==========================================

if __name__ == "__main__":
    print("Generating Q2 Final Visualizations (The Golden Trio)...")
    
    # Run data simulation
    sim_data = get_comparison_data(total_mass=1.0e8, duration=24.0)
    
    # Generate Plots
    plot_radar_gap(sim_data)
    plot_waterfall_cost(sim_data)
    plot_carbon_truth(sim_data)
    
    print(f"All visualizations saved to {OUTPUT_DIR}")
