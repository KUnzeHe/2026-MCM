"""
可视化模块：方案1a与1b的对比分析图表生成
============================================

本模块从 single_mode_opt.py 导入所有计算函数，实现完全模块化。
修改 single_mode_opt.py 中的参数或逻辑后，直接运行本脚本即可更新所有图表。

生成的图表:
----------
1. fig1: 双方案对比总览图 (时间 & 成本)
2. fig2: 串联系统瓶颈分析图 (方案1a)
3. fig3: 成本构成分解图 (Fixed vs Variable)
4. fig4: Break-even盈亏平衡分析图
5. fig5: 敏感性分析图 (龙卷风图)
6. fig6: 时间-成本权衡散点图
7. fig7: Logistic增长曲线 (动态模型)
8. fig8: 累积运输量对比 (静态 vs 动态)
9. fig9: 三种技术情景对比 (动态1b)

图片保存位置: ../image/
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# 从 single_mode_opt 导入所有需要的组件 (模块化核心)
# ============================================================================

from single_mode_opt import (
    # 数据类
    GlobalParams,
    DynamicParams,
    
    # 计算函数
    calculate_scenario_1a,
    calculate_scenario_1b_static,
    calculate_scenario_1b_dynamic,
    compare_scenarios,
    calculate_breakeven_mass,
    
    # Logistic辅助函数
    logistic_N,
    logistic_integral,
    logistic_inflection_point,
    generate_logistic_curve,
    generate_cumulative_transport_curve,
    
    # 预设情景
    SCENARIO_CONSERVATIVE,
    SCENARIO_MODERATE,
    SCENARIO_AGGRESSIVE,
    
    # 默认参数
    get_default_params
)


# ============================================================================
# 配置
# ============================================================================

# 图片保存目录
IMAGE_DIR = Path(__file__).parent.parent / "image"

# 配色方案
COLOR_ELEVATOR = "#3498db"      # 蓝色 - 电梯系统
COLOR_ROCKET = "#e74c3c"        # 红色 - 火箭系统 (静态)
COLOR_ROCKET_DYN = "#c0392b"    # 深红 - 火箭系统 (动态)
COLOR_FIXED = "#2c3e50"         # 深蓝 - 固定成本
COLOR_VARIABLE = "#95a5a6"      # 灰色 - 变动成本
COLOR_CONSERVATIVE = "#f39c12"  # 橙色 - 保守情景
COLOR_MODERATE = "#27ae60"      # 绿色 - 稳健情景
COLOR_AGGRESSIVE = "#9b59b6"    # 紫色 - 激进情景


# ============================================================================
# 工具函数
# ============================================================================

def ensure_image_dir():
    """确保图片保存目录存在"""
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] 图片将保存到: {IMAGE_DIR}")


def save_figure(fig, filename: str, dpi: int = 300):
    """保存图片到指定目录"""
    filepath = IMAGE_DIR / filename
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"[SAVED] {filepath}")


# ============================================================================
# 可视化函数 (原有 + 新增)
# ============================================================================

def plot_comparison_overview(res_1a: dict, res_1b: dict, params: GlobalParams):
    """
    图1: 双方案对比总览图
    左侧: 完工时间对比 | 右侧: 总成本对比
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    scenarios = ['1a\n(Space Elevator)', '1b\n(Traditional Rockets)']
    colors = [COLOR_ELEVATOR, COLOR_ROCKET]
    
    # === 左图: 完工时间 ===
    ax1 = axes[0]
    times = [res_1a['makespan'], res_1b['makespan']]
    bars1 = ax1.bar(scenarios, times, color=colors, edgecolor='black', linewidth=1.2)
    ax1.set_ylabel('Completion Time (Years)', fontsize=12)
    ax1.set_title('Timeline Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, max(times) * 1.2)
    
    for bar, val in zip(bars1, times):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.02,
                f'{val:.1f} yrs', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # === 右图: 总成本 ===
    ax2 = axes[1]
    costs = [res_1a['cost'], res_1b['cost']]
    bars2 = ax2.bar(scenarios, costs, color=colors, edgecolor='black', linewidth=1.2)
    ax2.set_ylabel('Total Cost (Currency Units)', fontsize=12)
    ax2.set_title('Cost Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, max(costs) * 1.2)
    
    for bar, val in zip(bars2, costs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(costs)*0.02,
                f'{val:.2e}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    for ax in axes:
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
    
    plt.suptitle('Scenario 1a vs 1b: Overview Comparison (Static Model)', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_figure(fig, 'fig1_comparison_overview.png')
    plt.close(fig)


def plot_bottleneck_analysis(params: GlobalParams):
    """
    图2: 方案1a 串联系统瓶颈分析图
    展示电梯管道和锚点转运两个环节的吞吐能力
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    throughput_elevator = params.T_E
    rate_anchor = params.N_anchor * params.L_anchor
    throughput_anchor = rate_anchor * params.p_A
    throughput_chain = min(throughput_elevator, throughput_anchor)
    
    stages = ['Elevator Pipeline\n(Continuous Flow)', 'Anchor Transfer\n(Discrete Launches)']
    throughputs = [throughput_elevator, throughput_anchor]
    colors = ['#3498db', '#9b59b6']
    
    y_pos = np.arange(len(stages))
    bars = ax.barh(y_pos, throughputs, color=colors, edgecolor='black', linewidth=1.2, height=0.5)
    
    for bar, val in zip(bars, throughputs):
        ax.text(val + max(throughputs)*0.02, bar.get_y() + bar.get_height()/2,
               f'{val:,.0f} t/yr', va='center', fontsize=11, fontweight='bold')
    
    ax.axvline(x=throughput_chain, color='red', linestyle='--', linewidth=2, 
               label=f'System Bottleneck: {throughput_chain:,.0f} t/yr')
    
    bottleneck_idx = 0 if throughput_elevator <= throughput_anchor else 1
    bars[bottleneck_idx].set_edgecolor('red')
    bars[bottleneck_idx].set_linewidth(3)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(stages, fontsize=12)
    ax.set_xlabel('Annual Throughput (Tons/Year)', fontsize=12)
    ax.set_title('Scenario 1a: Serial System Bottleneck Analysis', fontsize=14, fontweight='bold')
    ax.set_xlim(0, max(throughputs) * 1.3)
    ax.legend(loc='lower right', fontsize=10)
    ax.xaxis.grid(True, linestyle='--', alpha=0.7)
    
    ax.text(0.02, 0.98, f'T_chain = min(T_E, T_anchor) = {throughput_chain:,.0f} t/yr',
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    save_figure(fig, 'fig2_bottleneck_analysis.png')
    plt.close(fig)


def plot_cost_breakdown(res_1a: dict, res_1b: dict, params: GlobalParams):
    """
    图3: 成本构成分解图
    堆叠柱状图展示固定成本和变动成本的占比
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    fixed_1a = res_1a.get('cost_fixed', params.F_E)
    variable_1a = res_1a.get('cost_variable', params.c_E * params.M_tot)
    fixed_1b = res_1b.get('cost_fixed', 0)
    variable_1b = res_1b.get('cost_variable', params.c_R * params.M_tot)
    
    scenarios = ['1a (Space Elevator)', '1b (Traditional Rockets)']
    fixed_costs = [fixed_1a, fixed_1b]
    variable_costs = [variable_1a, variable_1b]
    
    x = np.arange(len(scenarios))
    width = 0.5
    
    bars1 = ax.bar(x, fixed_costs, width, label='Fixed Cost (F_E)', color=COLOR_FIXED, edgecolor='black')
    bars2 = ax.bar(x, variable_costs, width, bottom=fixed_costs, label='Variable Cost (c × M)', 
                   color=COLOR_VARIABLE, edgecolor='black')
    
    for i, (f, v) in enumerate(zip(fixed_costs, variable_costs)):
        total = f + v
        if f > 0:
            ax.text(x[i], f/2, f'Fixed:\n{f:.2e}', ha='center', va='center', 
                   fontsize=9, fontweight='bold', color='white')
        ax.text(x[i], f + v/2, f'Variable:\n{v:.2e}', ha='center', va='center', 
               fontsize=9, fontweight='bold')
        ax.text(x[i], total + max(fixed_costs[0]+variable_costs[0], fixed_costs[1]+variable_costs[1])*0.02,
               f'Total: {total:.2e}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_ylabel('Cost (Currency Units)', fontsize=12)
    ax.set_title('Cost Structure Breakdown: Fixed vs Variable Costs', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=12)
    ax.legend(loc='upper right', fontsize=10)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    save_figure(fig, 'fig3_cost_breakdown.png')
    plt.close(fig)


def plot_breakeven_analysis(params: GlobalParams):
    """
    图4: Break-even盈亏平衡分析图
    展示运输质量与成本的关系，找出两种方案的平衡点
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    M_range = np.linspace(0, 2e8, 1000)
    cost_1a = params.F_E + params.c_E * M_range
    cost_1b = params.c_R * M_range
    
    ax.plot(M_range, cost_1a, color=COLOR_ELEVATOR, linewidth=2.5, 
            label=f'1a (Elevator): C = {params.F_E:.1e} + {params.c_E:.0f}×M')
    ax.plot(M_range, cost_1b, color=COLOR_ROCKET, linewidth=2.5, 
            label=f'1b (Rocket): C = {params.c_R:.0f}×M')
    
    M_breakeven = calculate_breakeven_mass(params.F_E, params.c_E, params.c_R)
    
    if M_breakeven < float('inf'):
        C_breakeven = params.c_R * M_breakeven
        
        ax.scatter([M_breakeven], [C_breakeven], color='green', s=150, zorder=5, 
                  marker='*', edgecolors='black', linewidths=1)
        ax.annotate(f'Break-even Point\nM = {M_breakeven:,.0f} t\nC = {C_breakeven:.2e}',
                   xy=(M_breakeven, C_breakeven),
                   xytext=(M_breakeven + 2e7, C_breakeven + max(cost_1a[-1], cost_1b[-1])*0.1),
                   fontsize=10, fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        ax.fill_between(M_range, cost_1a, cost_1b, where=(M_range < M_breakeven), 
                       alpha=0.2, color=COLOR_ROCKET, label='Rocket Cheaper')
        ax.fill_between(M_range, cost_1a, cost_1b, where=(M_range >= M_breakeven), 
                       alpha=0.2, color=COLOR_ELEVATOR, label='Elevator Cheaper')
    
    M_current = params.M_tot
    C_current_1a = params.F_E + params.c_E * M_current
    C_current_1b = params.c_R * M_current
    
    ax.axvline(x=M_current, color='gray', linestyle=':', linewidth=1.5, alpha=0.8)
    ax.scatter([M_current], [C_current_1a], color=COLOR_ELEVATOR, s=100, zorder=5, 
              marker='o', edgecolors='black')
    ax.scatter([M_current], [C_current_1b], color=COLOR_ROCKET, s=100, zorder=5, 
              marker='o', edgecolors='black')
    
    ax.annotate(f'Current Mission\nM = {M_current:.0e} t',
               xy=(M_current, (C_current_1a + C_current_1b)/2),
               xytext=(M_current - 3e7, (C_current_1a + C_current_1b)/2),
               fontsize=10, fontweight='bold',
               arrowprops=dict(arrowstyle='->', color='gray', lw=1.5),
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    ax.set_xlabel('Total Mass to Transport (Tons)', fontsize=12)
    ax.set_ylabel('Total Cost (Currency Units)', fontsize=12)
    ax.set_title('Break-even Analysis: When Does Space Elevator Become Cost-effective?', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlim(0, 2e8)
    ax.set_ylim(0, max(cost_1a[-1], cost_1b[-1]) * 1.1)
    
    formula_text = f'Break-even: M > F_E / (c_R - c_E) = {params.F_E:.1e} / ({params.c_R:.0f} - {params.c_E:.0f})'
    ax.text(0.02, 0.98, formula_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    save_figure(fig, 'fig4_breakeven_analysis.png')
    plt.close(fig)


def plot_sensitivity_analysis(base_params: GlobalParams):
    """
    图5: 敏感性分析图 (龙卷风图)
    展示关键参数变化±50%对完工时间和成本的影响
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    param_names = ['T_E (Elevator\nThroughput)', 'L_anchor (Anchor\nLaunch Rate)', 
                   'N_sites (Ground\nSites)', 'L_max (Ground\nLaunch Rate)']
    
    res_base_1a = calculate_scenario_1a(base_params, verbose=False)
    res_base_1b = calculate_scenario_1b_static(base_params, verbose=False)
    base_time_1a = res_base_1a['makespan']
    base_time_1b = res_base_1b['makespan']
    base_cost_1a = res_base_1a['cost']
    
    variations = [0.5, 1.5]
    sensitivity_time = []
    
    for i, param_name in enumerate(param_names):
        time_range = [0, 0]
        
        for j, var in enumerate(variations):
            if 'T_E' in param_name:
                new_params = GlobalParams(
                    M_tot=base_params.M_tot, T_E=base_params.T_E*var,
                    N_anchor=base_params.N_anchor, L_anchor=base_params.L_anchor,
                    p_A=base_params.p_A, F_E=base_params.F_E, c_E=base_params.c_E,
                    N_sites=base_params.N_sites, L_max=base_params.L_max,
                    p_B=base_params.p_B, c_R=base_params.c_R
                )
                res = calculate_scenario_1a(new_params, verbose=False)
                time_range[j] = (res['makespan'] - base_time_1a) / base_time_1a * 100
                
            elif 'L_anchor' in param_name:
                new_params = GlobalParams(
                    M_tot=base_params.M_tot, T_E=base_params.T_E,
                    N_anchor=base_params.N_anchor, L_anchor=int(base_params.L_anchor*var),
                    p_A=base_params.p_A, F_E=base_params.F_E, c_E=base_params.c_E,
                    N_sites=base_params.N_sites, L_max=base_params.L_max,
                    p_B=base_params.p_B, c_R=base_params.c_R
                )
                res = calculate_scenario_1a(new_params, verbose=False)
                time_range[j] = (res['makespan'] - base_time_1a) / base_time_1a * 100
                
            elif 'N_sites' in param_name:
                new_params = GlobalParams(
                    M_tot=base_params.M_tot, T_E=base_params.T_E,
                    N_anchor=base_params.N_anchor, L_anchor=base_params.L_anchor,
                    p_A=base_params.p_A, F_E=base_params.F_E, c_E=base_params.c_E,
                    N_sites=int(base_params.N_sites*var), L_max=base_params.L_max,
                    p_B=base_params.p_B, c_R=base_params.c_R
                )
                res = calculate_scenario_1b_static(new_params, verbose=False)
                time_range[j] = (res['makespan'] - base_time_1b) / base_time_1b * 100
                
            elif 'L_max' in param_name:
                new_params = GlobalParams(
                    M_tot=base_params.M_tot, T_E=base_params.T_E,
                    N_anchor=base_params.N_anchor, L_anchor=base_params.L_anchor,
                    p_A=base_params.p_A, F_E=base_params.F_E, c_E=base_params.c_E,
                    N_sites=base_params.N_sites, L_max=int(base_params.L_max*var),
                    p_B=base_params.p_B, c_R=base_params.c_R
                )
                res = calculate_scenario_1b_static(new_params, verbose=False)
                time_range[j] = (res['makespan'] - base_time_1b) / base_time_1b * 100
        
        sensitivity_time.append(time_range)
    
    # === 左图: 完工时间敏感性 ===
    ax1 = axes[0]
    y_pos = np.arange(len(param_names))
    
    for i, (name, sens) in enumerate(zip(param_names, sensitivity_time)):
        low, high = sens[0], sens[1]
        color = COLOR_ELEVATOR if i < 2 else COLOR_ROCKET
        ax1.barh(i, high - low, left=low, height=0.6, color=color, edgecolor='black', alpha=0.7)
        ax1.text(low - 2, i, f'{low:.1f}%', va='center', ha='right', fontsize=9)
        ax1.text(high + 2, i, f'{high:.1f}%', va='center', ha='left', fontsize=9)
    
    ax1.axvline(x=0, color='black', linewidth=1.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(param_names, fontsize=10)
    ax1.set_xlabel('Change in Completion Time (%)', fontsize=12)
    ax1.set_title('Sensitivity Analysis: Completion Time', fontsize=14, fontweight='bold')
    ax1.set_xlim(-60, 120)
    ax1.xaxis.grid(True, linestyle='--', alpha=0.5)
    
    elevator_patch = mpatches.Patch(color=COLOR_ELEVATOR, alpha=0.7, label='Elevator Parameters')
    rocket_patch = mpatches.Patch(color=COLOR_ROCKET, alpha=0.7, label='Rocket Parameters')
    ax1.legend(handles=[elevator_patch, rocket_patch], loc='lower right', fontsize=9)
    
    # === 右图: 成本敏感性 ===
    ax2 = axes[1]
    
    cost_params = ['c_E (Elevator\nUnit Cost)', 'c_R (Rocket\nUnit Cost)']
    cost_sensitivity = []
    
    for param in cost_params:
        cost_range = [0, 0]
        for j, var in enumerate(variations):
            if 'c_E' in param:
                new_cost = base_params.F_E + base_params.c_E * var * base_params.M_tot
                cost_range[j] = (new_cost - base_cost_1a) / base_cost_1a * 100
            else:
                base_cost_1b = base_params.c_R * base_params.M_tot
                new_cost = base_params.c_R * var * base_params.M_tot
                cost_range[j] = (new_cost - base_cost_1b) / base_cost_1b * 100
        cost_sensitivity.append(cost_range)
    
    y_pos2 = np.arange(len(cost_params))
    colors2 = [COLOR_ELEVATOR, COLOR_ROCKET]
    
    for i, (name, sens) in enumerate(zip(cost_params, cost_sensitivity)):
        low, high = sens[0], sens[1]
        ax2.barh(i, high - low, left=low, height=0.6, color=colors2[i], edgecolor='black', alpha=0.7)
        ax2.text(low - 2, i, f'{low:.1f}%', va='center', ha='right', fontsize=9)
        ax2.text(high + 2, i, f'{high:.1f}%', va='center', ha='left', fontsize=9)
    
    ax2.axvline(x=0, color='black', linewidth=1.5)
    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels(cost_params, fontsize=10)
    ax2.set_xlabel('Change in Total Cost (%)', fontsize=12)
    ax2.set_title('Sensitivity Analysis: Total Cost', fontsize=14, fontweight='bold')
    ax2.set_xlim(-60, 60)
    ax2.xaxis.grid(True, linestyle='--', alpha=0.5)
    
    plt.suptitle('Parameter Sensitivity Analysis (±50% Variation)', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, 'fig5_sensitivity_analysis.png')
    plt.close(fig)


def plot_time_cost_tradeoff(res_1a: dict, res_1b: dict):
    """
    图6: 时间-成本权衡散点图
    展示两种方案在时间-成本空间中的位置
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    ax.scatter(res_1a['makespan'], res_1a['cost'], color=COLOR_ELEVATOR, s=300, 
              marker='s', edgecolors='black', linewidths=2, zorder=5, label='1a (Space Elevator)')
    ax.scatter(res_1b['makespan'], res_1b['cost'], color=COLOR_ROCKET, s=300, 
              marker='^', edgecolors='black', linewidths=2, zorder=5, label='1b (Traditional Rockets)')
    
    ax.annotate(f'1a: {res_1a["makespan"]:.1f} yrs\n${res_1a["cost"]:.2e}',
               xy=(res_1a['makespan'], res_1a['cost']),
               xytext=(res_1a['makespan'] + 10, res_1a['cost'] * 1.1),
               fontsize=10, fontweight='bold',
               arrowprops=dict(arrowstyle='->', color=COLOR_ELEVATOR),
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    ax.annotate(f'1b: {res_1b["makespan"]:.1f} yrs\n${res_1b["cost"]:.2e}',
               xy=(res_1b['makespan'], res_1b['cost']),
               xytext=(res_1b['makespan'] + 10, res_1b['cost'] * 0.9),
               fontsize=10, fontweight='bold',
               arrowprops=dict(arrowstyle='->', color=COLOR_ROCKET),
               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    ax.plot([res_1a['makespan'], res_1b['makespan']], 
           [res_1a['cost'], res_1b['cost']], 
           'k--', linewidth=1.5, alpha=0.5, label='Trade-off Line')
    
    ax.set_xlabel('Completion Time (Years)', fontsize=12)
    ax.set_ylabel('Total Cost (Currency Units)', fontsize=12)
    ax.set_title('Time-Cost Trade-off: Scenario Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    ax.set_xlim(0, max(res_1a['makespan'], res_1b['makespan']) * 1.3)
    ax.set_ylim(0, max(res_1a['cost'], res_1b['cost']) * 1.2)
    
    plt.tight_layout()
    save_figure(fig, 'fig6_time_cost_tradeoff.png')
    plt.close(fig)


# ============================================================================
# 新增: 动态模型可视化
# ============================================================================

def plot_logistic_growth(params: GlobalParams, dyn_params: DynamicParams, res_dyn: dict):
    """
    图7: Logistic增长曲线
    展示发射场数量随时间的S型增长
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    N0 = params.N_sites
    K = dyn_params.K
    r = dyn_params.r
    Y_max = res_dyn['makespan'] * 1.2
    
    # 生成曲线数据
    t_values, N_values = generate_logistic_curve(N0, K, r, Y_max, n_points=200)
    
    # 绘制主曲线
    ax.plot(t_values, N_values, color=COLOR_ROCKET_DYN, linewidth=2.5, label='N(t) - Launch Sites')
    
    # 标注关键点
    # 初始点
    ax.scatter([0], [N0], color='green', s=150, zorder=5, marker='o', edgecolors='black')
    ax.annotate(f'N₀ = {N0}', xy=(0, N0), xytext=(Y_max*0.05, N0*1.3),
               fontsize=10, fontweight='bold',
               arrowprops=dict(arrowstyle='->', color='green'))
    
    # 拐点
    t_inflection = res_dyn['inflection_point']
    N_inflection = logistic_N(t_inflection, N0, K, r)
    ax.scatter([t_inflection], [N_inflection], color='orange', s=150, zorder=5, marker='D', edgecolors='black')
    ax.axvline(x=t_inflection, color='orange', linestyle=':', alpha=0.7)
    ax.annotate(f'Inflection\nt* = {t_inflection:.1f} yrs', 
               xy=(t_inflection, N_inflection), 
               xytext=(t_inflection + Y_max*0.08, N_inflection),
               fontsize=10, fontweight='bold',
               arrowprops=dict(arrowstyle='->', color='orange'))
    
    # 完工点
    Y_done = res_dyn['makespan']
    N_final = res_dyn['N_final']
    ax.scatter([Y_done], [N_final], color='red', s=200, zorder=5, marker='*', edgecolors='black')
    ax.axvline(x=Y_done, color='red', linestyle='--', alpha=0.7)
    ax.annotate(f'Mission Complete\nY = {Y_done:.1f} yrs\nN(Y) = {N_final:.0f}', 
               xy=(Y_done, N_final), 
               xytext=(Y_done - Y_max*0.15, N_final*0.7),
               fontsize=10, fontweight='bold',
               arrowprops=dict(arrowstyle='->', color='red'),
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # 承载力线
    ax.axhline(y=K, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Carrying Capacity K = {K}')
    
    # 填充增长区域
    ax.fill_between(t_values, N_values, alpha=0.2, color=COLOR_ROCKET_DYN)
    
    ax.set_xlabel('Time (Years)', fontsize=12)
    ax.set_ylabel('Number of Launch Sites N(t)', fontsize=12)
    ax.set_title('Logistic Growth of Launch Infrastructure', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlim(0, Y_max)
    ax.set_ylim(0, K * 1.1)
    
    # 添加公式
    formula = f'N(t) = {K} / (1 + {(K-N0)/N0:.1f} × e^(-{r}t))'
    ax.text(0.02, 0.98, formula, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    save_figure(fig, 'fig7_logistic_growth.png')
    plt.close(fig)


def plot_cumulative_transport(params: GlobalParams, dyn_params: DynamicParams, res_dyn: dict):
    """
    图8: 累积运输量对比 (静态 vs 动态)
    展示两种模型下的累积运输曲线
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    N0 = params.N_sites
    K = dyn_params.K
    r = dyn_params.r
    L_max = res_dyn['L_max_physical']
    p_B = params.p_B
    M_tot = params.M_tot
    
    Y_dyn = res_dyn['makespan']
    Y_static = res_dyn['makespan_static']
    Y_max = max(Y_dyn, Y_static) * 1.2
    
    # 动态模型累积曲线
    t_dyn, cum_dyn = generate_cumulative_transport_curve(N0, K, r, L_max, p_B, Y_max, n_points=200)
    
    # 静态模型累积曲线 (假设运力恒定为初始值)
    static_throughput = N0 * L_max * p_B
    t_static = np.linspace(0, Y_max, 200)
    cum_static = static_throughput * t_static
    
    # 绘制曲线
    ax.plot(t_dyn, cum_dyn, color=COLOR_ROCKET_DYN, linewidth=2.5, label='Dynamic Model (Logistic Growth)')
    ax.plot(t_static, cum_static, color=COLOR_ROCKET, linewidth=2.5, linestyle='--', 
            label='Static Model (Constant Capacity)')
    
    # 目标线
    ax.axhline(y=M_tot, color='black', linestyle=':', linewidth=2, label=f'Target: M_tot = {M_tot:.1e} t')
    
    # 标注完工时间
    ax.axvline(x=Y_dyn, color=COLOR_ROCKET_DYN, linestyle='--', alpha=0.7)
    ax.axvline(x=Y_static, color=COLOR_ROCKET, linestyle='--', alpha=0.7)
    
    ax.scatter([Y_dyn], [M_tot], color=COLOR_ROCKET_DYN, s=150, zorder=5, marker='*', edgecolors='black')
    ax.scatter([Y_static], [M_tot], color=COLOR_ROCKET, s=150, zorder=5, marker='o', edgecolors='black')
    
    # 标注时间差
    time_diff = Y_dyn - Y_static
    pct_increase = (Y_dyn / Y_static - 1) * 100
    
    ax.annotate(f'Dynamic: {Y_dyn:.1f} yrs', xy=(Y_dyn, M_tot), xytext=(Y_dyn + 5, M_tot * 0.85),
               fontsize=10, fontweight='bold', color=COLOR_ROCKET_DYN,
               arrowprops=dict(arrowstyle='->', color=COLOR_ROCKET_DYN))
    ax.annotate(f'Static: {Y_static:.1f} yrs', xy=(Y_static, M_tot), xytext=(Y_static - 15, M_tot * 0.7),
               fontsize=10, fontweight='bold', color=COLOR_ROCKET,
               arrowprops=dict(arrowstyle='->', color=COLOR_ROCKET))
    
    # 填充差异区域
    ax.fill_betweenx([0, M_tot], Y_static, Y_dyn, alpha=0.2, color='red', 
                     label=f'Time Penalty: +{time_diff:.1f} yrs (+{pct_increase:.0f}%)')
    
    ax.set_xlabel('Time (Years)', fontsize=12)
    ax.set_ylabel('Cumulative Mass Transported (Tons)', fontsize=12)
    ax.set_title('Cumulative Transport: Static vs Dynamic Model', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_xlim(0, Y_max)
    ax.set_ylim(0, M_tot * 1.2)
    
    plt.tight_layout()
    save_figure(fig, 'fig8_cumulative_transport.png')
    plt.close(fig)


def plot_scenario_comparison(params: GlobalParams):
    """
    图9: 三种技术情景对比 (动态1b)
    展示保守/稳健/激进情景下的完工时间和成本
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    scenarios = [
        ("Conservative\n(t_cycle=14d)", SCENARIO_CONSERVATIVE, COLOR_CONSERVATIVE),
        ("Moderate\n(t_cycle=4d)", SCENARIO_MODERATE, COLOR_MODERATE),
        ("Aggressive\n(t_cycle=1d)", SCENARIO_AGGRESSIVE, COLOR_AGGRESSIVE),
    ]
    
    names = [s[0] for s in scenarios]
    results = []
    
    for name, dyn_params, _ in scenarios:
        res = calculate_scenario_1b_dynamic(params, dyn_params, verbose=False)
        results.append(res)
    
    times = [r['makespan'] for r in results]
    costs = [r['cost'] for r in results]
    colors = [s[2] for s in scenarios]
    
    # 添加1a作为基准
    res_1a = calculate_scenario_1a(params, verbose=False)
    names.insert(0, "1a\n(Elevator)")
    times.insert(0, res_1a['makespan'])
    costs.insert(0, res_1a['cost'])
    colors.insert(0, COLOR_ELEVATOR)
    
    x = np.arange(len(names))
    width = 0.6
    
    # === 左图: 完工时间 ===
    ax1 = axes[0]
    bars1 = ax1.bar(x, times, width, color=colors, edgecolor='black', linewidth=1.2)
    
    for bar, val in zip(bars1, times):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.02,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax1.set_ylabel('Completion Time (Years)', fontsize=12)
    ax1.set_title('Completion Time by Scenario', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, fontsize=10)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax1.set_axisbelow(True)
    
    # === 右图: 总成本 ===
    ax2 = axes[1]
    bars2 = ax2.bar(x, costs, width, color=colors, edgecolor='black', linewidth=1.2)
    
    for bar, val in zip(bars2, costs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(costs)*0.02,
                f'{val:.2e}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax2.set_ylabel('Total Cost (Currency Units)', fontsize=12)
    ax2.set_title('Total Cost by Scenario', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=10)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax2.set_axisbelow(True)
    
    plt.suptitle('Technology Scenario Comparison: 1a vs Dynamic 1b Variants', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, 'fig9_scenario_comparison.png')
    plt.close(fig)


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主函数: 执行所有可视化任务"""
    print("=" * 70)
    print("  可视化模块: 方案1a与1b对比分析 (含动态模型)")
    print("=" * 70)
    
    # 确保输出目录存在
    ensure_image_dir()
    
    # 从模块获取默认参数 (模块化: 只需修改 single_mode_opt.py 即可)
    params = get_default_params()
    
    # 计算结果
    print("\n[STEP 1] 计算方案结果...")
    
    res_1a = calculate_scenario_1a(params, verbose=True)
    res_1b_static = calculate_scenario_1b_static(params, verbose=True)
    res_1b_dynamic = calculate_scenario_1b_dynamic(params, SCENARIO_MODERATE, verbose=True)
    
    if not all([res_1a, res_1b_static, res_1b_dynamic]):
        print("[ERROR] 计算失败，无法生成可视化。")
        return
    
    # 生成可视化
    print("\n[STEP 2] 生成可视化图表...")
    
    print("  - 图1: 双方案对比总览图 (静态)...")
    plot_comparison_overview(res_1a, res_1b_static, params)
    
    print("  - 图2: 串联系统瓶颈分析图...")
    plot_bottleneck_analysis(params)
    
    print("  - 图3: 成本构成分解图...")
    plot_cost_breakdown(res_1a, res_1b_static, params)
    
    print("  - 图4: Break-even盈亏平衡分析图...")
    plot_breakeven_analysis(params)
    
    print("  - 图5: 敏感性分析图...")
    plot_sensitivity_analysis(params)
    
    print("  - 图6: 时间-成本权衡散点图...")
    plot_time_cost_tradeoff(res_1a, res_1b_static)
    
    print("  - 图7: Logistic增长曲线 (动态模型)...")
    plot_logistic_growth(params, SCENARIO_MODERATE, res_1b_dynamic)
    
    print("  - 图8: 累积运输量对比 (静态 vs 动态)...")
    plot_cumulative_transport(params, SCENARIO_MODERATE, res_1b_dynamic)
    
    print("  - 图9: 三种技术情景对比...")
    plot_scenario_comparison(params)
    
    print("\n" + "=" * 70)
    print("  ✅ 所有可视化已完成!")
    print(f"  📁 图片保存位置: {IMAGE_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
