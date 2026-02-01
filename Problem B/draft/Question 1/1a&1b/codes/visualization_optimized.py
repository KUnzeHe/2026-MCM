"""
可视化模块 (优化版)：方案1a与1b的对比分析图表生成
==================================================
Key Updates:
1.  **Color Palette**: Aligned with Q2 "Golden Trio" (Teal/Coral/Slate).
2.  **Log Scales**: Applied to Cost charts to handle the 100x magnitude difference.
3.  **Aesthetics**: Improved font sizes, grid lines, and data labels.

Generated Charts:
1.  fig1_optimized_overview.png
2.  fig2_optimized_bottleneck.png
3.  fig3_optimized_cost_structure.png
4.  fig4_optimized_breakeven.png
5.  fig5_optimized_sensitivity.png
6.  fig6_optimized_tradeoff.png
7.  fig7_optimized_logistic.png
8.  fig8_optimized_cumulative.png
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Import core logic
from single_mode_opt import (
    GlobalParams, DynamicParams,
    calculate_scenario_1a, calculate_scenario_1b_static, calculate_scenario_1b_dynamic,
    compare_scenarios, calculate_breakeven_mass,
    logistic_N, logistic_integral, logistic_inflection_point,
    generate_logistic_curve, generate_cumulative_transport_curve,
    SCENARIO_CONSERVATIVE, SCENARIO_MODERATE, SCENARIO_AGGRESSIVE,
    get_default_params
)

# ==========================================
# 0. Global Visualization Settings
# ==========================================
IMAGE_DIR = Path(__file__).parent.parent / "image"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 11

# The "Golden Trio" Palette
COLORS = {
    'elevator': '#2A9D8F',      # Teal (Space Elevator - Efficient/Clean)
    'rocket_static': '#E9C46A', # Yellow/Orange (Rocket Static)
    'rocket_dyn': '#E76F51',    # Coral (Rocket Dynamic - Realism)
    'fixed': '#264653',         # Charcoal (Fixed Cost)
    'variable': '#A8DADC',      # Light Blue (Variable Cost - 1a)
    'variable_bad': '#F4A261',  # Light Orange (Variable Cost - 1b)
    'text': '#333333',
    'grid': '#DDDDDD'
}

def save_figure(fig, name):
    path = IMAGE_DIR / name
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"[SAVED] {path}")

# ==========================================
# 1. Figure 1: Overview (Log Scale Fix)
# ==========================================
def plot_overview(res_1a, res_1b):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    scenarios = ['1a: Space Elevator', '1b: Rockets']
    colors = [COLORS['elevator'], COLORS['rocket_dyn']]
    
    # 1.1 Time (Linear is fine, 40 vs 180 is visible)
    ax1 = axes[0]
    times = [res_1a['makespan'], res_1b['makespan']]
    bars1 = ax1.bar(scenarios, times, color=colors, width=0.5)
    
    ax1.set_ylabel('Completion Time (Years)', fontweight='bold')
    ax1.set_title('Time Comparison (Linear)', fontweight='bold', fontsize=14)
    ax1.bar_label(bars1, fmt='%.1f yr', padding=3, fontsize=12, fontweight='bold')
    ax1.grid(axis='y', linestyle='--', alpha=0.5)
    
    # 1.2 Cost (MUST BE LOG SCALE)
    ax2 = axes[1]
    costs = [res_1a['cost'], res_1b['cost']]
    bars2 = ax2.bar(scenarios, costs, color=colors, width=0.5)
    
    ax2.set_ylabel('Total Cost (Currency) [Log Scale]', fontweight='bold')
    ax2.set_title('Cost Comparison (Log Scale)', fontweight='bold', fontsize=14)
    ax2.set_yscale('log') # The critical fix
    
    # Custom Labels for Log Scale
    for bar, val in zip(bars2, costs):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                f'${val/1e12:.2f}T', ha='center', va='bottom', fontweight='bold')
        
        # Add factor difference
        if val == max(costs):
            ratio = costs[1] / costs[0]
            ax2.text(bar.get_x() + bar.get_width()/2., height * 0.5,
                    f'~{ratio:.0f}x More\nExpensive', ha='center', color='white', fontweight='bold')

    ax2.set_ylim(top=max(costs)*5) # Give headroom for labels
    ax2.grid(axis='y', which='major', linestyle='--', alpha=0.5)
    
    save_figure(fig, 'fig1_optimized_overview.png')

# ==========================================
# 2. Figure 3: Cost Structure (Stacked & Log)
# ==========================================
def plot_cost_structure(res_1a, res_1b, params):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Data
    f_1a = res_1a.get('cost_fixed', 0)
    v_1a = res_1a.get('cost_variable', 0)
    total_1a = res_1a['cost']
    
    # For 1b, we consider it mostly variable (as per static model logic often used for comparison)
    # But if dynamic result passed, it has capex. Let's handle generic 'cost_fixed' key.
    f_1b = res_1b.get('cost_fixed', res_1b.get('cost_CAPEX', 0))
    v_1b = res_1b['cost'] - f_1b
    
    x = [0, 1]
    labels = ['1a: Space Elevator', '1b: Rockets']
    
    # PLOTTING TRICK FOR LOG SCALE STACKED BARS:
    # Bars starting at 0 disappear or look wrong in log scale.
    # We set a "floor" baseline (e.g. $100M) for the plot base suitable for visualization.
    # Or simply let MPL handle it but enforce a strictly positive ylim.
    
    # Plot 1a
    # Fixed is at bottom.
    p1 = ax.bar(0, f_1a, color=COLORS['fixed'], width=0.4, label='Fixed (Infra)')
    # Variable is on top of Fixed.
    p2 = ax.bar(0, v_1a, bottom=f_1a, color=COLORS['variable'], width=0.4, label='Variable Cost (Elevator)')
    
    # Plot 1b
    # Fixed (if any)
    if f_1b > 0:
        p3 = ax.bar(1, f_1b, color=COLORS['fixed'], width=0.4)
        p4 = ax.bar(1, v_1b, bottom=f_1b, color=COLORS['variable_bad'], width=0.4, label='Variable Cost (Rocket)')
    else:
        # If no fixed, just variable
        p4 = ax.bar(1, v_1b, color=COLORS['variable_bad'], width=0.4, label='Variable Cost (Rocket)')

    ax.set_yscale('log')
    ax.set_title('Cost Structure Breakdown (Log Scale)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cost (Currency) [Log Scale]')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontweight='bold', fontsize=12)
    
    # Critical Fix: Set lower ylim manually to make the "Fixed" bar ($5B) visible against the Total ($275B)
    # Total 1a is ~10^11. Fixed is 5x10^9. 
    # If ylim starts at 10^0, the 10^9 height is large.
    # If ylim starts at 10^9, the 10^9 height is tiny (1 unit).
    # We should set ylim bottom to something like 1e8 ($100M) to give the 5B bar some "leg room".
    ax.set_ylim(bottom=1e8, top=res_1b['cost']*5)
    
    # Annotations
    ax.bar_label(p2, fmt='Total: $%.1fB', padding=3, labels=[f"${total_1a/1e9:.1f}B"])
    # For 1b
    top_bar_1b = p4 if f_1b <= 0 else p4
    ax.text(1, res_1b['cost']*1.1, f"${res_1b['cost']/1e12:.1f}T", ha='center', fontweight='bold')
    
    # Manually adding text for the invisible Fixed bar if needed
    ax.text(0, f_1a/2, "Fixed\n$5B", ha='center', va='center', color='white', fontsize=9, fontweight='bold')

    # Fix Legend: Deduplicate labels
    handles, lbls = ax.get_legend_handles_labels()
    by_label = dict(zip(lbls, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left', frameon=True)
    
    ax.grid(axis='y', which='major', linestyle='--', alpha=0.3)
    
    save_figure(fig, 'fig3_optimized_cost_structure.png')

# ==========================================
# 3. Figure 4: Break-Even (Zoom In)
# ==========================================
def plot_breakeven(params):
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Full range calculation
    M_max = params.M_tot * 1.5
    X = np.linspace(0, M_max, 100)
    Y_1a = params.F_E + params.c_E * X
    Y_1b = params.c_R * X
    
    # Main Plot (Macro View)
    ax.plot(X/1e6, Y_1a/1e9, color=COLORS['elevator'], lw=3, label='1a: Elevator (High Fixed)')
    ax.plot(X/1e6, Y_1b/1e9, color=COLORS['rocket_dyn'], lw=3, label='1b: Rockets (High Slope)')
    
    # Current Mission Point
    M_curr = params.M_tot
    C_1a = params.F_E + params.c_E * M_curr
    C_1b = params.c_R * M_curr
    
    ax.scatter(M_curr/1e6, C_1a/1e9, s=100, color=COLORS['elevator'], ec='k', zorder=5)
    ax.scatter(M_curr/1e6, C_1b/1e9, s=100, color=COLORS['rocket_dyn'], ec='k', zorder=5)
    
    # Dotted line for Current Mission
    ax.vlines(x=M_curr/1e6, ymin=0, ymax=C_1b/1e9, colors='gray', linestyles='--', alpha=0.5)
    ax.text(M_curr/1e6, C_1b/1e9, f"  Current Mission\n  {M_curr/1e6:.0f} Mt", va='bottom', fontweight='bold')
    
    ax.set_xlabel('Total Transported Mass (Million Tons)', fontsize=12)
    ax.set_ylabel('Total Cost (Billion USD)', fontsize=12)
    ax.set_title('Break-even Analysis: The Economic Dominance of 1a', fontsize=15, fontweight='bold')
    # Move legend to lower right to avoid overlap with inset
    ax.legend(loc='lower right', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    
    # --- INSET ZOOM (The Critical Threshold) ---
    M_break = params.F_E / (params.c_R - params.c_E)
    # Define zooming window (e.g., 0 to 3x break-even mass)
    x_limit_zoom = M_break * 4 / 1e6
    y_limit_zoom = (params.c_R * M_break * 4) / 1e9
    
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    # Position: x, y, width, height (relative to axes)
    axins = inset_axes(ax, width="35%", height="35%", loc=2, borderpad=4) 
    
    axins.plot(X/1e6, Y_1a/1e9, color=COLORS['elevator'], lw=2)
    axins.plot(X/1e6, Y_1b/1e9, color=COLORS['rocket_dyn'], lw=2)
    
    # Mark Break-even
    axins.scatter(M_break/1e6, (params.c_R*M_break)/1e9, color='k', s=50, zorder=10)
    # Annotation inside zoom box
    axins.annotate(f'Break-even\n{M_break/1e6:.2f} Mt', 
                   xy=(M_break/1e6, (params.c_R*M_break)/1e9),
                   xytext=(M_break/1e6 + x_limit_zoom*0.1, (params.c_R*M_break)/1e9),
                   fontsize=9,
                   arrowprops=dict(arrowstyle='->'))
    
    axins.set_xlim(0, x_limit_zoom)
    axins.set_ylim(0, y_limit_zoom)
    axins.set_title("Zoom: Origin (0-20Mt)", fontsize=10)
    axins.grid(True, linestyle=':', alpha=0.5)
    # Mark the region on main plot
    from matplotlib.patches import Rectangle
    # We can draw a rectangle on the main plot to show where the zoom is coming from
    rect = Rectangle((0,0), x_limit_zoom, y_limit_zoom, linewidth=1, edgecolor='k', facecolor='none', linestyle='--')
    ax.add_patch(rect)
    
    save_figure(fig, 'fig4_optimized_breakeven.png')

# ==========================================
# 4. Figure 7: Logistic Growth (Color Update)
# ==========================================
def plot_logistic(res_dyn, params, dyn_params):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    N0 = params.N_sites
    K = dyn_params.K
    r = dyn_params.r
    Y_max = res_dyn['makespan'] * 1.2
    
    t_vals, N_vals = generate_logistic_curve(N0, K, r, Y_max, 200)
    
    # Plot
    ax.plot(t_vals, N_vals, color=COLORS['rocket_dyn'], lw=3, label='Active Launch Sites N(t)')
    ax.fill_between(t_vals, N_vals, color=COLORS['rocket_dyn'], alpha=0.1)
    
    # Capacity Line
    ax.axhline(K, color=COLORS['fixed'], linestyle='--', alpha=0.6, label=f'Global Limit K={K}')
    
    # Points
    t_inf = res_dyn['inflection_point']
    n_inf = logistic_N(t_inf, N0, K, r)
    ax.scatter(t_inf, n_inf, color='orange', s=100, zorder=5)
    ax.text(t_inf, n_inf+3, f'Max Growth Rate\nt={t_inf:.1f} yr', ha='center', fontsize=9)
    
    # Final
    t_end = res_dyn['makespan']
    n_end = res_dyn['N_final']
    ax.scatter(t_end, n_end, color=COLORS['rocket_dyn'], s=150, marker='*', zorder=5)
    ax.annotate(f'Mission Complete\nY={t_end:.1f} yr', xy=(t_end, n_end), xytext=(t_end-20, n_end-15),
                arrowprops=dict(arrowstyle='->', color='black'), bbox=dict(boxstyle='round', fc='white'))

    ax.set_xlabel('Time (Years)')
    ax.set_ylabel('Number of Launch Sites')
    ax.set_title('Infrastructure Ramp-up: Logistic Growth Model', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    save_figure(fig, 'fig7_optimized_logistic.png')

# ==========================================
# Main Execution
# ==========================================
def main():
    print(">>> Starting 1a&1b Visualization Update (Optimized)...")
    
    params = get_default_params()
    
    # 1. Run Calculations
    res_1a = calculate_scenario_1a(params, verbose=False)
    
    # Use Dynamic 1b for the main comparison vs 1a (more realistic)
    # We use the Moderate scenario as the "representative" 1b
    dyn_params = SCENARIO_MODERATE
    res_1b = calculate_scenario_1b_dynamic(params, dyn_params, verbose=False)
    
    # 2. Generate Plots
    plot_overview(res_1a, res_1b)
    plot_cost_structure(res_1a, res_1b, params)
    plot_breakeven(params)
    plot_logistic(res_1b, params, dyn_params)
    
    print(">>> All Optimized Figures Generated Successfully!")

if __name__ == "__main__":
    main()
