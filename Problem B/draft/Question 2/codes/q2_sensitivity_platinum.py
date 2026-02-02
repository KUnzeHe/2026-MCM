"""Q2 Reliability Sensitivity (Platinum)

Generates ONE key sensitivity figure for Question 2:
- Minimum feasible completion time (years) as a function of rocket reliability
  parameters (launch failure probability and post-failure downtime).

Output:
- ../image/Fig4_Reliability_Sensitivity.png

Why this figure:
- It is intuitive (deadline feasibility vs reliability)
- It matches the model's capacity feasibility constraint
- It avoids producing many charts ("few but strong")

Implementation notes:
- q2 model file is named "q2-4.py" (hyphen), so we load it via exec like
  q2_visualization_final.py does.
"""

from __future__ import annotations

import os
import sys
from dataclasses import replace

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import brentq
from typing import Any


# ==========================================
# 0. Load model logic from q2-4.py
# ==========================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "q2-4.py")

q2_model: dict = {}
with open(MODEL_PATH, "r", encoding="utf-8") as f:
    code = f.read()
    exec(code, q2_model)

ModelParams = q2_model["ModelParams"]
ReliabilityParams = q2_model["ReliabilityParams"]
TransportOptimizationModel = q2_model["TransportOptimizationModel"]


# ==========================================
# 1. Platinum-like design system
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
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
    'grid.alpha': 0.35,
    'grid.linestyle': '--',
    'figure.dpi': 300,
    'axes.facecolor': '#F8F9FA',
    'figure.facecolor': '#F8F9FA',
})

COLORS = {
    'primary': '#2A9D8F',    # Teal
    'secondary': '#E76F51',  # Coral
    'tertiary': '#264653',   # Charcoal
    'neutral': '#E9C46A',    # Sand
    'slate': '#64748B',
    'background': '#F8F9FA',
    'missing': '#CBD5E1',    # light slate (for infeasible)
}

CMAP = LinearSegmentedColormap.from_list(
    "platinum_seq",
    [COLORS['primary'], COLORS['neutral'], COLORS['secondary']],
)


def save_figure(fig: plt.Figure, name: str) -> str:
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches='tight', facecolor=COLORS['background'])
    print(f"[SAVED] {path}")
    return path


# ==========================================
# 2. Sensitivity computation
# ==========================================

def minimum_feasible_years(model: Any, Y_min: float = 0.01, Y_max: float = 100.0) -> float:
    """Quiet minimum feasible Y using brent root finding."""

    def gap(Y: float) -> float:
        cap_E = model.cumulative_elevator_capacity(Y, adjusted=not model.p.use_ideal)
        cap_R = model.cumulative_rocket_capacity(Y, adjusted=not model.p.use_ideal)
        M_eff = model.get_effective_demand(Y)
        return cap_E + cap_R - M_eff

    g_min = gap(Y_min)
    if g_min >= 0:
        return Y_min

    g_max = gap(Y_max)
    if g_max < 0:
        return float('inf')

    try:
        return float(brentq(gap, Y_min, Y_max, maxiter=200))
    except Exception:
        return float('inf')


def plot_reliability_sensitivity_heatmap(
    total_mass: float = 1.0e8,
    Y_deadline: float = 24.0,
    pf_grid: np.ndarray | None = None,
    tdown_grid: np.ndarray | None = None,
) -> str:
    """Heatmap: min feasible Y as function of (P_f_R, T_down_R)."""

    if pf_grid is None:
        pf_grid = np.linspace(0.005, 0.08, 13)
    if tdown_grid is None:
        tdown_grid = np.linspace(1.0, 30.0, 13)

    Z = np.full((len(tdown_grid), len(pf_grid)), np.nan)

    for i, t_down in enumerate(tdown_grid):
        for j, pf in enumerate(pf_grid):
            base = ModelParams(use_ideal=False, M_tot=total_mass)
            rel2 = replace(base.reliability, P_f_R=float(pf), T_down_R=float(t_down))
            p2 = replace(base, reliability=rel2)
            model = TransportOptimizationModel(p2)

            Y_min = minimum_feasible_years(model, Y_min=0.01, Y_max=120.0)
            Z[i, j] = Y_min

    # Mask infinities for plotting
    Z_plot = Z.copy()
    Z_plot[~np.isfinite(Z_plot)] = np.nan

    fig, ax = plt.subplots(figsize=(11.5, 6.2))

    # imshow expects row-major; y is tdown, x is pf
    extent = [pf_grid.min() * 100, pf_grid.max() * 100, tdown_grid.min(), tdown_grid.max()]

    im = ax.imshow(
        Z_plot,
        origin='lower',
        aspect='auto',
        extent=extent,
        cmap=CMAP,
        vmin=np.nanmin(Z_plot),
        vmax=np.nanpercentile(Z_plot, 95),
    )

    # Overlay infeasible region as hatch-like dots (simple scatter)
    infeasible = ~np.isfinite(Z)
    if np.any(infeasible):
        yy, xx = np.where(infeasible)
        ax.scatter(
            pf_grid[xx] * 100,
            tdown_grid[yy],
            s=10,
            color=COLORS['missing'],
            marker='s',
            alpha=0.8,
            label='Infeasible (Y > 120y)'
        )

    # Contour at deadline
    try:
        X, Y = np.meshgrid(pf_grid * 100, tdown_grid)
        cs = ax.contour(X, Y, Z_plot, levels=[Y_deadline], colors=[COLORS['tertiary']], linewidths=2.5)
        ax.clabel(cs, fmt={Y_deadline: f'{Y_deadline:.0f}y deadline'}, inline=True, fontsize=10)
    except Exception:
        pass

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Minimum Feasible Completion Time (Years)', fontweight='bold')

    ax.set_xlabel('Rocket Launch Failure Probability $P_f$ (%)')
    ax.set_ylabel('Downtime After Failure $T_{down}$ (days)')
    ax.set_title('Q2 Sensitivity: Deadline Feasibility vs Rocket Reliability')
    ax.grid(False)

    # Small legend only if we plotted infeasible markers
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc='upper left', frameon=True, framealpha=0.95)

    return save_figure(fig, 'Fig4_Reliability_Sensitivity.png')


if __name__ == '__main__':
    plot_reliability_sensitivity_heatmap()
