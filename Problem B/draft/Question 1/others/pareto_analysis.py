"""
运输系统优化模型 - Pareto 前沿分析
==================================

基于 transport_optimization_readme.md 文档实现的混合运输方案优化程序。
本程序计算不同完工时间约束下的最小成本，并绘制 Pareto 前沿曲线。

模型概述：
---------
- 分支 A（电梯链路）：地面 → 电梯 → 锚点 → 转运火箭 → 月球
- 分支 B（地面直达火箭）：地面 → 传统火箭 → 月球
- 两条分支并行工作，总完工时间由较慢者决定

决策变量：
---------
- x: 通过电梯链路运输的质量（吨）
- M_tot - x: 通过地面火箭运输的质量（吨）

Author: MCM Team
Date: 2026-01-31
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from math import ceil, inf
from typing import Optional, List, Tuple
import warnings


# =============================================================================
# 数据结构定义
# =============================================================================

class TransportParams:
    """
    运输系统参数类
    
    所有参数含义与 readme 文档中的符号定义一致。
    
    单位约定：
    - 质量: 吨 (tons)
    - 时间: 年 (years)
    - 成本: 美元 ($)
    """
    
    def __init__(
        self,
        M_tot: float,      # 总运输质量 (吨)
        T_E: float,        # 电梯年吞吐能力 (吨/年)
        N_anchor: int,     # 锚点数量 (题目: 3 Harbours × 2 anchors)
        L_anchor: int,     # 锚点单站年最大发射次数 (次/年)
        p_A: float,        # 锚点转运火箭单次有效载荷 (吨/次)
        F_E: float,        # 电梯系统固定成本 ($)
        c_E: float,        # 电梯链路单位运输成本 ($/吨)
        N_sites: int,      # 地面发射场数量
        L_max: int,        # 单场年最大发射次数 (次/年)
        p_B: float,        # 地面火箭单次有效载荷 (吨/次)
        c_R: float         # 地面火箭单位运输成本 ($/吨)
    ):
        """
        初始化运输系统参数
        
        Args:
            M_tot: 总运输质量 (吨)
            T_E: 电梯年吞吐能力 (吨/年)
            N_anchor: 锚点数量
            L_anchor: 锚点单站年最大发射次数 (次/年)
            p_A: 锚点转运火箭单次有效载荷 (吨/次)
            F_E: 电梯系统固定成本 ($)
            c_E: 电梯链路单位运输成本 ($/吨)
            N_sites: 地面发射场数量
            L_max: 单场年最大发射次数 (次/年)
            p_B: 地面火箭单次有效载荷 (吨/次)
            c_R: 地面火箭单位运输成本 ($/吨)
        """
        # 基本需求参数
        self.M_tot = M_tot
        
        # 电梯链路参数 (Branch A)
        self.T_E = T_E
        self.N_anchor = N_anchor
        self.L_anchor = L_anchor
        self.p_A = p_A
        self.F_E = F_E
        self.c_E = c_E
        
        # 地面直达火箭参数 (Branch B)
        self.N_sites = N_sites
        self.L_max = L_max
        self.p_B = p_B
        self.c_R = c_R
        
        # 计算派生参数
        self._compute_derived_params()
    
    def _compute_derived_params(self):
        """计算派生参数"""
        # 锚点转运火箭年吞吐能力 (吨/年)
        self.T_R_anchor = self.N_anchor * self.L_anchor * self.p_A
        
        # 电梯链路瓶颈吞吐能力 (吨/年): T_chain = min(T_E, T_R_anchor)
        self.T_chain = min(self.T_E, self.T_R_anchor)
        
        # 地面直达火箭年吞吐能力 (吨/年): T_R = N_sites × L_max × p_B
        self.T_R = self.N_sites * self.L_max * self.p_B
        
        # 理论最短完工时间 (连续近似): Y_min = M_tot / (T_chain + T_R)
        total_throughput = self.T_chain + self.T_R
        if total_throughput > 0:
            self.Y_min_theoretical = self.M_tot / total_throughput
        else:
            self.Y_min_theoretical = inf
        
        # 纯电梯方案完工时间: Y = M_tot / T_chain
        if self.T_chain > 0:
            self.Y_pure_elevator = self.M_tot / self.T_chain
        else:
            self.Y_pure_elevator = inf


class OptimizationResult:
    """
    单次优化结果
    
    记录给定 Y_max 约束下的最优解信息。
    """
    
    def __init__(
        self,
        feasible: bool,          # 是否存在可行解
        Y_max: float,            # 时间约束上限
        x_optimal: float,        # 最优电梯运量 (吨)
        y_indicator: int,        # 是否启用电梯 (0 或 1)
        cost_total: float,       # 总成本 ($)
        cost_fixed: float,       # 固定成本 ($)
        cost_elevator: float,    # 电梯运输成本 ($)
        cost_rocket: float,      # 火箭运输成本 ($)
        Y_actual: float,         # 实际完工时间 (年)
        Y_A: float,              # 电梯分支完工时间 (年)
        Y_B: float,              # 火箭分支完工时间 (年)
        n_launches_anchor: int,  # 锚点火箭发射次数
        n_launches_ground: int,  # 地面火箭发射次数
        reason: Optional[str] = None  # 不可行原因（如果有）
    ):
        self.feasible = feasible
        self.Y_max = Y_max
        self.x_optimal = x_optimal
        self.y_indicator = y_indicator
        self.cost_total = cost_total
        self.cost_fixed = cost_fixed
        self.cost_elevator = cost_elevator
        self.cost_rocket = cost_rocket
        self.Y_actual = Y_actual
        self.Y_A = Y_A
        self.Y_B = Y_B
        self.n_launches_anchor = n_launches_anchor
        self.n_launches_ground = n_launches_ground
        self.reason = reason


class ParetoFrontier:
    """
    Pareto 前沿分析结果
    
    存储一系列 (Y_max, C_min) 数据点，用于绘制 Time-Cost 权衡曲线。
    """
    
    def __init__(self):
        """初始化空的 Pareto 前沿"""
        self.results = []  # List[OptimizationResult]
    
    def add_result(self, result: OptimizationResult):
        """添加一个优化结果"""
        self.results.append(result)
    
    def get_times(self) -> np.ndarray:
        """获取所有可行解的完工时间"""
        return np.array([r.Y_actual for r in self.results if r.feasible])
    
    def get_costs(self) -> np.ndarray:
        """获取所有可行解的总成本"""
        return np.array([r.cost_total for r in self.results if r.feasible])
    
    def get_elevator_fractions(self, M_tot: float) -> np.ndarray:
        """获取电梯分担比例 (x / M_tot)"""
        return np.array([r.x_optimal / M_tot for r in self.results if r.feasible])
    
    def get_knee_point(self) -> Optional[OptimizationResult]:
        """
        寻找"膝点" (Knee Point)
        
        膝点定义：成本下降速率变化最显著的点。
        即 d²C/dY² 最大的位置，表示"性价比"最高的平衡点。
        
        Returns:
            膝点对应的优化结果，若无法计算则返回 None
        """
        feasible_results = [r for r in self.results if r.feasible]
        if len(feasible_results) < 3:
            return None
        
        # 按时间排序
        sorted_results = sorted(feasible_results, key=lambda r: r.Y_actual)
        
        times = np.array([r.Y_actual for r in sorted_results])
        costs = np.array([r.cost_total for r in sorted_results])
        
        # 计算一阶导数 (成本下降率)
        dC_dY = np.diff(costs) / np.diff(times)
        
        # 计算二阶导数 (曲率近似)
        if len(dC_dY) < 2:
            return None
        d2C_dY2 = np.diff(dC_dY) / np.diff(times[:-1])
        
        # 找到曲率变化最大的点（膝点）
        knee_idx = np.argmax(np.abs(d2C_dY2)) + 1  # +1 因为二阶导数少一个点
        
        return sorted_results[knee_idx]


# =============================================================================
# 核心计算函数
# =============================================================================

def compute_time_elevator_chain(x: float, params: TransportParams) -> float:
    """
    计算电梯链路分支的完工时间 Y_A(x)
    
    电梯链路是串联系统，完工时间由两个瓶颈决定：
    1. 电梯吞吐量限制：x / T_E
    2. 锚点转运火箭批次限制：⌈x / p_A⌉ / (N_anchor × L_anchor)
    
    Y_A(x) = max(连续流时间, 离散批次时间)
    
    Args:
        x: 通过电梯链路运输的质量 (吨)
        params: 系统参数
    
    Returns:
        电梯链路完工时间 (年)
    """
    if x <= 0:
        return 0.0
    
    # 检查参数有效性
    if params.T_E <= 0:
        return inf
    
    rate_anchor = params.N_anchor * params.L_anchor
    if rate_anchor <= 0:
        return inf
    
    # 连续流时间 (电梯吞吐量限制)
    time_continuous = x / params.T_E
    
    # 离散批次时间 (锚点转运火箭发射次数限制)
    n_launches = ceil(x / params.p_A)
    time_discrete = n_launches / rate_anchor
    
    # 返回两者中的较大值（瓶颈决定）
    return max(time_continuous, time_discrete)


def compute_time_direct_rocket(mass_rocket: float, params: TransportParams) -> float:
    """
    计算地面直达火箭分支的完工时间 Y_B(M_R)
    
    Y_B = ⌈M_R / p_B⌉ / (N_sites × L_max)
    
    Args:
        mass_rocket: 通过地面火箭运输的质量 (吨)
        params: 系统参数
    
    Returns:
        火箭分支完工时间 (年)
    """
    if mass_rocket <= 0:
        return 0.0
    
    rate_direct = params.N_sites * params.L_max
    if rate_direct <= 0:
        return inf
    
    n_launches = ceil(mass_rocket / params.p_B)
    return n_launches / rate_direct


def compute_total_cost(x: float, params: TransportParams) -> Tuple[float, float, float, float]:
    """
    计算总成本及其分解
    
    C(x, y) = y × F_E + c_E × x + c_R × (M_tot - x)
    
    其中 y = 1 if x > 0 else 0
    
    Args:
        x: 通过电梯链路运输的质量 (吨)
        params: 系统参数
    
    Returns:
        (total_cost, fixed_cost, elevator_cost, rocket_cost)
    """
    y = 1 if x > 0 else 0
    
    fixed_cost = y * params.F_E
    elevator_cost = params.c_E * x
    rocket_cost = params.c_R * (params.M_tot - x)
    total_cost = fixed_cost + elevator_cost + rocket_cost
    
    return total_cost, fixed_cost, elevator_cost, rocket_cost


def optimize_single(params: TransportParams, Y_max: float) -> OptimizationResult:
    """
    给定 Y_max 约束下的单次优化
    
    算法思路（边界分析法）：
    1. 由 Y_max 推导 x 的可行区间 [x_lower, x_upper]
    2. 由于成本函数对 x 是线性的（且 c_E < c_R），最优解在边界
    3. 特殊考虑 x = 0 的情况（避免支付 F_E）
    
    可行区间推导：
    - 上界：x ≤ T_chain × Y_max（电梯链路容量限制）
    - 下界：M_tot - x ≤ T_R × Y_max  =>  x ≥ M_tot - T_R × Y_max
    
    Args:
        params: 系统参数
        Y_max: 完工时间上限 (年)
    
    Returns:
        OptimizationResult 对象
    """
    rate_anchor = params.N_anchor * params.L_anchor
    rate_direct = params.N_sites * params.L_max
    
    # =====================
    # Step 1: 计算 x 的上界
    # =====================
    # 电梯链路容量限制（离散批次）
    max_launches_A = int(rate_anchor * Y_max)
    capacity_A_discrete = max_launches_A * params.p_A
    
    # 电梯连续流限制
    capacity_A_continuous = params.T_E * Y_max
    
    # 上界取三者最小值
    x_upper = min(params.M_tot, capacity_A_discrete, capacity_A_continuous)
    
    # =====================
    # Step 2: 计算 x 的下界
    # =====================
    # 火箭容量限制：M_tot - x ≤ Rate_B × Y_max × p_B
    max_launches_B = int(rate_direct * Y_max)
    capacity_B = max_launches_B * params.p_B
    
    # 下界：x ≥ M_tot - capacity_B
    x_lower = max(0.0, params.M_tot - capacity_B)
    
    # =====================
    # Step 3: 可行性检查
    # =====================
    if x_lower > x_upper + 1e-6:  # 加容差处理浮点误差
        return OptimizationResult(
            feasible=False,
            Y_max=Y_max,
            x_optimal=0.0,
            y_indicator=0,
            cost_total=inf,
            cost_fixed=0.0,
            cost_elevator=0.0,
            cost_rocket=0.0,
            Y_actual=inf,
            Y_A=inf,
            Y_B=inf,
            n_launches_anchor=0,
            n_launches_ground=0,
            reason=f"不可行: 电梯最小需求 {x_lower:,.0f} t > 电梯最大容量 {x_upper:,.0f} t"
        )
    
    # =====================
    # Step 4: 候选解枚举
    # =====================
    # 线性成本下，最优解在边界
    candidates = [x_lower, x_upper]
    
    # 若 x_lower == 0，也考虑纯火箭方案（避免支付 F_E）
    if x_lower == 0:
        candidates.append(0.0)
    
    # 去重
    candidates = list(set(candidates))
    
    best_result = None
    
    for x in candidates:
        # 计算完工时间
        Y_A = compute_time_elevator_chain(x, params)
        Y_B = compute_time_direct_rocket(params.M_tot - x, params)
        Y_actual = max(Y_A, Y_B)
        
        # 验证时间约束（浮点容差）
        if Y_actual > Y_max + 1e-9:
            continue
        
        # 计算成本
        total_cost, fixed_cost, elevator_cost, rocket_cost = compute_total_cost(x, params)
        
        # 计算发射次数
        n_anchor = ceil(x / params.p_A) if x > 0 else 0
        n_ground = ceil((params.M_tot - x) / params.p_B) if (params.M_tot - x) > 0 else 0
        
        result = OptimizationResult(
            feasible=True,
            Y_max=Y_max,
            x_optimal=x,
            y_indicator=1 if x > 0 else 0,
            cost_total=total_cost,
            cost_fixed=fixed_cost,
            cost_elevator=elevator_cost,
            cost_rocket=rocket_cost,
            Y_actual=Y_actual,
            Y_A=Y_A,
            Y_B=Y_B,
            n_launches_anchor=n_anchor,
            n_launches_ground=n_ground
        )
        
        # 更新最优解
        if best_result is None or total_cost < best_result.cost_total:
            best_result = result
    
    if best_result is None:
        return OptimizationResult(
            feasible=False,
            Y_max=Y_max,
            x_optimal=0.0,
            y_indicator=0,
            cost_total=inf,
            cost_fixed=0.0,
            cost_elevator=0.0,
            cost_rocket=0.0,
            Y_actual=inf,
            Y_A=inf,
            Y_B=inf,
            n_launches_anchor=0,
            n_launches_ground=0,
            reason="无候选解满足时间约束"
        )
    
    return best_result


def compute_pareto_frontier(
    params: TransportParams,
    n_points: int = 100,
    Y_min_override: Optional[float] = None,
    Y_max_override: Optional[float] = None
) -> ParetoFrontier:
    """
    计算 Pareto 前沿
    
    遍历从 Y_min 到 Y_pure_elevator 的时间范围，
    对每个 Y_max 求解最小成本，得到 Time-Cost 权衡曲线。
    
    Args:
        params: 系统参数
        n_points: 采样点数量
        Y_min_override: 自定义时间下界（默认使用理论最小值）
        Y_max_override: 自定义时间上界（默认使用纯电梯时间）
    
    Returns:
        ParetoFrontier 对象
    """
    # 确定时间范围
    Y_min = Y_min_override if Y_min_override else params.Y_min_theoretical
    Y_max = Y_max_override if Y_max_override else params.Y_pure_elevator
    
    # 确保范围有效
    if Y_min >= Y_max:
        warnings.warn(f"时间范围无效: Y_min={Y_min:.2f} >= Y_max={Y_max:.2f}")
        Y_max = Y_min * 1.5
    
    # 生成时间采样点
    Y_values = np.linspace(Y_min, Y_max, n_points)
    
    # 存储结果
    frontier = ParetoFrontier()
    
    for Y in Y_values:
        result = optimize_single(params, Y)
        frontier.add_result(result)
    
    return frontier


# =============================================================================
# 可视化函数
# =============================================================================

def plot_pareto_frontier(
    frontier: ParetoFrontier,
    params: TransportParams,
    save_path: Optional[str] = None,
    show_knee: bool = True,
    figsize: Tuple[int, int] = (14, 10)
) -> plt.Figure:
    """
    绘制 Pareto 前沿曲线（多子图）
    
    包含：
    1. Time-Cost 曲线
    2. 电梯运量分配比例
    3. 成本构成分解
    4. 发射次数统计
    
    Args:
        frontier: Pareto 前沿数据
        params: 系统参数
        save_path: 保存路径（可选）
        show_knee: 是否标注膝点
        figsize: 图像尺寸
    
    Returns:
        matplotlib Figure 对象
    """
    feasible_results = [r for r in frontier.results if r.feasible]
    
    if not feasible_results:
        raise ValueError("没有可行解，无法绘图")
    
    # 按时间排序
    feasible_results = sorted(feasible_results, key=lambda r: r.Y_actual)
    
    # 提取数据
    times = np.array([r.Y_actual for r in feasible_results])
    costs_total = np.array([r.cost_total for r in feasible_results])
    costs_fixed = np.array([r.cost_fixed for r in feasible_results])
    costs_elevator = np.array([r.cost_elevator for r in feasible_results])
    costs_rocket = np.array([r.cost_rocket for r in feasible_results])
    x_values = np.array([r.x_optimal for r in feasible_results])
    elevator_fractions = x_values / params.M_tot * 100
    n_anchor = np.array([r.n_launches_anchor for r in feasible_results])
    n_ground = np.array([r.n_launches_ground for r in feasible_results])
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Mixed Transport Plan: Pareto Frontier Analysis\n(Space Elevator + Traditional Rockets)', 
                 fontsize=14, fontweight='bold')
    
    # =====================
    # 子图1: Time-Cost 曲线
    # =====================
    ax1 = axes[0, 0]
    ax1.plot(times, costs_total / 1e12, 'b-', linewidth=2, label='Total Cost')
    ax1.fill_between(times, 0, costs_total / 1e12, alpha=0.3)
    
    # 标注膝点
    if show_knee:
        knee = frontier.get_knee_point()
        if knee:
            ax1.scatter([knee.Y_actual], [knee.cost_total / 1e12], 
                       color='red', s=150, zorder=5, marker='*', label='Knee Point')
            ax1.annotate(f'Knee: ({knee.Y_actual:.1f} yr, ${knee.cost_total/1e12:.2f}T)',
                        xy=(knee.Y_actual, knee.cost_total / 1e12),
                        xytext=(knee.Y_actual + 10, knee.cost_total / 1e12 * 1.1),
                        fontsize=10, arrowprops=dict(arrowstyle='->', color='red'))
    
    ax1.set_xlabel('Completion Time (years)', fontsize=11)
    ax1.set_ylabel('Total Cost (Trillion $)', fontsize=11)
    ax1.set_title('① Time-Cost Trade-off (Pareto Frontier)', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)
    
    # =====================
    # 子图2: 电梯分担比例
    # =====================
    ax2 = axes[0, 1]
    ax2.plot(times, elevator_fractions, 'g-', linewidth=2)
    ax2.fill_between(times, 0, elevator_fractions, alpha=0.3, color='green')
    ax2.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='100% Elevator')
    
    ax2.set_xlabel('Completion Time (years)', fontsize=11)
    ax2.set_ylabel('Elevator Share (%)', fontsize=11)
    ax2.set_title('② Mass Allocation to Elevator Chain', fontsize=12)
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # =====================
    # 子图3: 成本构成堆叠图
    # =====================
    ax3 = axes[1, 0]
    ax3.stackplot(times, 
                  costs_fixed / 1e12, 
                  costs_elevator / 1e12, 
                  costs_rocket / 1e12,
                  labels=['Fixed Cost (F_E)', 'Elevator Op. Cost', 'Rocket Cost'],
                  colors=['#ff9999', '#66b3ff', '#99ff99'],
                  alpha=0.8)
    
    ax3.set_xlabel('Completion Time (years)', fontsize=11)
    ax3.set_ylabel('Cost (Trillion $)', fontsize=11)
    ax3.set_title('③ Cost Breakdown', fontsize=12)
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(left=0)
    
    # =====================
    # 子图4: 发射次数
    # =====================
    ax4 = axes[1, 1]
    ax4.plot(times, n_anchor / 1e3, 'b-', linewidth=2, label='Anchor Launches')
    ax4.plot(times, n_ground / 1e3, 'r-', linewidth=2, label='Ground Launches')
    ax4.plot(times, (n_anchor + n_ground) / 1e3, 'k--', linewidth=1.5, label='Total Launches')
    
    ax4.set_xlabel('Completion Time (years)', fontsize=11)
    ax4.set_ylabel('Number of Launches (thousands)', fontsize=11)
    ax4.set_title('④ Launch Count by Branch', fontsize=12)
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(left=0)
    ax4.set_ylim(bottom=0)
    
    plt.tight_layout()
    
    # 保存图像
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存至: {save_path}")
    
    return fig


def print_summary_table(frontier: ParetoFrontier, params: TransportParams, n_display: int = 10):
    """
    打印关键结果摘要表
    
    Args:
        frontier: Pareto 前沿数据
        params: 系统参数
        n_display: 显示的行数
    """
    feasible_results = [r for r in frontier.results if r.feasible]
    
    if not feasible_results:
        print("没有可行解。")
        return
    
    # 按时间排序并均匀采样
    feasible_results = sorted(feasible_results, key=lambda r: r.Y_actual)
    step = max(1, len(feasible_results) // n_display)
    sampled = feasible_results[::step]
    
    print("\n" + "=" * 100)
    print("Pareto Frontier Summary Table")
    print("=" * 100)
    print(f"{'Y_max':>10} | {'Y_actual':>10} | {'x (elevator)':>15} | {'Elevator %':>10} | "
          f"{'Cost ($T)':>12} | {'n_Anchor':>10} | {'n_Ground':>10}")
    print("-" * 100)
    
    for r in sampled:
        elevator_pct = r.x_optimal / params.M_tot * 100
        print(f"{r.Y_max:>10.2f} | {r.Y_actual:>10.2f} | {r.x_optimal:>15,.0f} | "
              f"{elevator_pct:>9.1f}% | {r.cost_total/1e12:>12.3f} | "
              f"{r.n_launches_anchor:>10,} | {r.n_launches_ground:>10,}")
    
    print("=" * 100)
    
    # 打印膝点信息
    knee = frontier.get_knee_point()
    if knee:
        print(f"\n★ Knee Point (Recommended):")
        print(f"   Time = {knee.Y_actual:.2f} years")
        print(f"   Cost = ${knee.cost_total/1e12:.3f} Trillion")
        print(f"   Elevator Share = {knee.x_optimal/params.M_tot*100:.1f}%")


# =============================================================================
# 主程序
# =============================================================================

def create_baseline_params() -> TransportParams:
    """
    创建基准参数（与 readme 文档一致）
    
    Returns:
        TransportParams 对象
    """
    params = TransportParams(
        # 基本需求
        M_tot=1.0e8,              # 1亿吨
        
        # 电梯链路
        T_E=537_000,              # 179,000 × 3 = 537,000 吨/年
        N_anchor=6,               # 3 Harbours × 2 anchors
        L_anchor=700,             # 700 次/年/站
        p_A=125,                  # 125 吨/次 (太空发射效率高)
        F_E=100e9,                # $100B 固定成本
        c_E=2700,                   # $2.7/kg = $2,700/吨
        
        # 地面直达火箭
        N_sites=60,               # 60 个发射场
        L_max=700,                 # 700 次/年/站
        p_B=125,                  # 125 吨/次 (100-150 取中)
        c_R=7200000                 # $7,200/kg = $7,200,000/吨
    )
    return params


def main():
    """主函数：执行完整的 Pareto 前沿分析"""
    
    print("=" * 60)
    print("运输系统优化模型 - Pareto 前沿分析")
    print("Moon Colony Material Transport Optimization")
    print("=" * 60)
    
    # 1. 创建基准参数
    params = create_baseline_params()
    
    # 2. 打印参数摘要
    print("\n【参数摘要】")
    print(f"总运输量: {params.M_tot:,.0f} 吨")
    print(f"电梯链路吞吐能力: {params.T_chain:,.0f} 吨/年")
    print(f"地面火箭吞吐能力: {params.T_R:,.0f} 吨/年")
    print(f"理论最短时间: {params.Y_min_theoretical:.2f} 年")
    print(f"纯电梯时间: {params.Y_pure_elevator:.2f} 年")
    
    # 3. 计算 Pareto 前沿
    print("\n【计算 Pareto 前沿...】")
    frontier = compute_pareto_frontier(params, n_points=200)
    
    # 4. 打印结果表格
    print_summary_table(frontier, params, n_display=15)
    
    # 5. 绘制图像
    print("\n【生成可视化图表...】")
    fig = plot_pareto_frontier(
        frontier, 
        params,
        save_path="/Users/kunze/Desktop/2026MCM/Problem B/draft/Question 1/1c/image/pareto_frontier.png",
        show_knee=True
    )
    plt.show()
    
    print("\n分析完成！")


if __name__ == "__main__":
    main()
