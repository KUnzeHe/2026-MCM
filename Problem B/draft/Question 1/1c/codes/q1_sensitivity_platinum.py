"""Q1 Sensitivity (Platinum)

Generates a single high-impact sensitivity chart for Question 1c:
- Total cost response to key parameters around the baseline plan.

Output:
- ../image/Fig5_Sensitivity_Core.png

Notes:
- Keeps styling consistent with the existing Q1 platinum figures.
- Intentionally produces only one main figure ("few but strong").
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace

import numpy as np
import matplotlib.pyplot as plt

# Ensure proper path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from comprehensive_transport_model_v5 import ModelParams, TransportOptimizationModel


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "../image")
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 11

COLORS = {
    'elevator': '#2A9D8F',
    'rocket': '#E76F51',
    'mixed': '#264653',
    'highlight': '#F4A261',
    'safe': '#E9C46A',
    'text': '#333333',
    'grid': '#DDDDDD',
    'background': '#F8F9FA',
}


def save_figure(fig: plt.Figure, name: str) -> str:
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches='tight', facecolor=COLORS['background'])
    print(f"[SAVED] {path}")
    return path


def _evaluate_cost(model: TransportOptimizationModel, Y: float) -> float:
    res = model.solve(Y)
    if not res.feasible:
        return float('nan')
    return res.cost_total


def plot_sensitivity_core(Y_target: float = 24.0) -> str:
    """Single-figure sensitivity: cost ratio vs parameter multiplier."""

    # Baseline scenario consistent with the rest of Q1 figures
    base_params = ModelParams(p_B=125.0)
    base_model = TransportOptimizationModel(base_params)
    base_cost = _evaluate_cost(base_model, Y_target)

    multipliers = np.linspace(0.7, 1.3, 13)

    # Choose the few parameters that directly drive the tradeoff
    sweeps = [
        ("c_E", COLORS['elevator'], "Elevator OPEX $c_E"),
        ("c_R", COLORS['rocket'], "Rocket OPEX $c_R"),
        ("T_E", COLORS['mixed'], "Elevator Throughput $T_E"),
    ]

    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    ax.set_facecolor(COLORS['background'])

    for param_name, color, label in sweeps:
        ratios = []
        for m in multipliers:
            p2 = base_params
            if param_name == "c_E":
                p2 = replace(p2, c_E=base_params.c_E * float(m))
            elif param_name == "c_R":
                p2 = replace(p2, c_R=base_params.c_R * float(m))
            elif param_name == "T_E":
                p2 = replace(p2, T_E=base_params.T_E * float(m))
            else:
                raise ValueError(f"Unsupported param: {param_name}")

            c2 = _evaluate_cost(TransportOptimizationModel(p2), Y_target)
            ratios.append(c2 / base_cost if np.isfinite(c2) and base_cost > 0 else np.nan)

        ax.plot(
            multipliers * 100,
            np.array(ratios) * 100,
            marker='o',
            markersize=4,
            color=color,
            linewidth=2.8,
            label=label,
        )

    ax.axhline(100, color=COLORS['grid'], linestyle='--', linewidth=1.2)
    ax.axvline(100, color=COLORS['grid'], linestyle='--', linewidth=1.2)

    ax.set_xlim(68, 132)
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle='--', color=COLORS['grid'], alpha=0.7)

    ax.set_xlabel('Parameter Value (% of Baseline)', fontweight='bold')
    ax.set_ylabel('Total Cost (% of Baseline)', fontweight='bold')
    ax.set_title(f'Q1 Sensitivity: Cost Response at Deadline Plan (Y = {Y_target:.0f} years)', fontweight='bold')

    ax.legend(loc='upper left', frameon=True, framealpha=0.95)

    return save_figure(fig, 'Fig5_Sensitivity_Core.png')


if __name__ == "__main__":
    plot_sensitivity_core(24.0)
