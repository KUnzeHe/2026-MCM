"""
单一运输系统优化模块 (single_mode_opt.py)
==========================================

本模块实现方案 1a (空间电梯) 与 1b (传统火箭) 的计算逻辑。
包含静态模型与动态修正模型（Logistic增长、周转时间约束）。

理论依据: single_mode_models.md

模块结构:
---------
1. 数据类定义 (GlobalParams, DynamicParams)
2. 辅助函数 (Logistic增长、积分、数值求解)
3. 方案1a计算 (静态)
4. 方案1b计算 (静态 + 动态)
5. 对比与工具函数

使用方式:
---------
>>> from single_mode_opt import GlobalParams, DynamicParams
>>> from single_mode_opt import calculate_scenario_1a, calculate_scenario_1b_dynamic
>>> params = GlobalParams(...)
>>> dyn_params = DynamicParams(...)
>>> result = calculate_scenario_1b_dynamic(params, dyn_params)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil, inf, log, exp
from typing import Callable, Optional
import warnings


# ============================================================================
# 第一部分: 数据类定义
# ============================================================================

@dataclass(frozen=True)
class GlobalParams:
    """定义计算所需的全局参数 (静态模型).
    
    包含项目需求、电梯配置、火箭配置及各类成本系数。
    适用于静态模型计算。
    """
    # ====== 项目需求 ======
    M_tot: float    # 总运输质量 (吨), 题目预估约 1亿吨
    
    # ====== 1a: 电梯系统参数 (串联: 地面->电梯->锚点->火箭->月球) ======
    T_E: float      # 电梯管道年吞吐能力 (吨/年)
    N_anchor: int   # 锚点港口数量 (Galactic Harbours)
    L_anchor: int   # 每个锚点年最大转运发射次数
    p_A: float      # 锚点转运火箭单次有效载荷 (吨/次)
    
    # 成本参数 (电梯)
    F_E: float      # 电梯系统固定建设成本 (Currency)
    c_E: float      # 电梯系统单位运输成本 (Currency/Ton)
    
    # ====== 1b: 传统火箭参数 (地面直接发射) ======
    N_sites: int    # 地面发射场数量 (初始值 N_0)
    L_max: int      # 每个发射场年最大发射次数
    p_B: float      # 地面火箭单次有效载荷 (吨/次)
    
    # 成本参数 (火箭)
    c_R: float      # 地面火箭单位运输成本 (Currency/Ton)


@dataclass(frozen=True)
class DynamicParams:
    """动态修正模型参数 (方案1b专用).
    
    基于 single_mode_models.md 第2.4节的动态修正理论。
    包含周转时间约束和Logistic增长模型参数。
    """
    # ====== 周转时间模型 (2.4.1) ======
    t_cycle: float      # 周转时间 (天): t_refurb + t_pad + t_weather + t_margin
    eta: float = 0.90   # 系统可用率 (0.85 ~ 0.95)
    
    # ====== Logistic增长模型 (2.4.2) ======
    K: int = 80         # 环境承载力: 全球最大发射场数量 [50, 100]
    r: float = 0.3      # 增长率: 基建动员速度 [0.2, 0.5] /年
    
    # ====== 成本参数 (2.4.4) ======
    C_site: float = 3.0e10      # 单个发射场建设成本 ($20-40B)
    rho: float = 0.03           # 折现率 (2% ~ 5%)
    
    @property
    def L_max_physical(self) -> float:
        """基于周转时间计算的物理极限发射频率 (次/年/工位)."""
        return 365.0 * self.eta / self.t_cycle
    
    def get_scenario_label(self) -> str:
        """返回技术情景标签."""
        if self.t_cycle >= 14:
            return "Conservative (Falcon 9 level)"
        elif self.t_cycle >= 4:
            return "Moderate (Starship target)"
        else:
            return "Aggressive (Aviation-like)"


# ============================================================================
# 第二部分: Logistic增长模型辅助函数
# ============================================================================

def logistic_N(t: float, N0: int, K: int, r: float) -> float:
    """计算 Logistic 增长曲线在时刻 t 的值.
    
    公式: N(t) = K / (1 + ((K - N0) / N0) * exp(-r * t))
    
    Args:
        t: 时间 (年)
        N0: 初始发射场数量
        K: 环境承载力
        r: 增长率
    
    Returns:
        t 时刻的发射场数量 (浮点数，实际应取整)
    """
    if N0 <= 0 or K <= 0 or r <= 0:
        raise ValueError("N0, K, r 必须为正数")
    if N0 >= K:
        return float(K)
    
    ratio = (K - N0) / N0
    return K / (1.0 + ratio * exp(-r * t))


def logistic_integral(Y: float, N0: int, K: int, r: float) -> float:
    """计算 Logistic 函数从 0 到 Y 的积分.
    
    公式: ∫₀^Y N(t) dt = (K/r) * ln((N0 * e^(rY) + K - N0) / K)
    
    这是累积运力的核心计算。
    
    Args:
        Y: 积分上限 (年)
        N0: 初始发射场数量
        K: 环境承载力
        r: 增长率
    
    Returns:
        积分值 (发射场·年)
    """
    if Y <= 0:
        return 0.0
    
    # 防止数值溢出: 当 rY 很大时，使用近似公式
    rY = r * Y
    if rY > 700:  # exp(700) 接近 float64 上限
        # 当 Y 很大时，N(t) ≈ K，积分 ≈ K * Y
        # 更精确: ∫N(t)dt ≈ K*Y - (K/r)*ln(K/N0)
        return K * Y - (K / r) * log(K / N0)
    
    numerator = N0 * exp(rY) + (K - N0)
    return (K / r) * log(numerator / K)


def logistic_inflection_point(N0: int, K: int, r: float) -> float:
    """计算 Logistic 曲线的拐点 (增速最快的时刻).
    
    公式: t* = (1/r) * ln((K - N0) / N0)
    
    Returns:
        拐点时间 (年)
    """
    if N0 >= K:
        return 0.0
    return (1.0 / r) * log((K - N0) / N0)


def solve_dynamic_makespan(
    M_tot: float,
    N0: int,
    K: int,
    r: float,
    L_max: float,
    p_B: float,
    tol: float = 1e-6,
    max_iter: int = 100
) -> Optional[float]:
    """数值求解动态完工时间 Y (超越方程).
    
    求解方程: (K * L_max * p_B / r) * ln((N0 * e^(rY) + K - N0) / K) = M_tot
    
    等价于求 Y 使得: cumulative_transport(Y) = M_tot
    
    使用牛顿法求解。
    
    Args:
        M_tot: 总运输质量
        N0: 初始发射场数量
        K: 环境承载力
        r: 增长率
        L_max: 单场年发射次数
        p_B: 单次载荷
        tol: 收敛容差
        max_iter: 最大迭代次数
    
    Returns:
        动态完工时间 Y (年), 若不收敛返回 None
    """
    # 定义累积运输量函数
    def cumulative(Y: float) -> float:
        return L_max * p_B * logistic_integral(Y, N0, K, r)
    
    # 导数 = N(Y) * L_max * p_B
    def derivative(Y: float) -> float:
        return L_max * p_B * logistic_N(Y, N0, K, r)
    
    # 初始猜测: 用静态模型的时间作为起点
    static_throughput = N0 * L_max * p_B
    Y = M_tot / static_throughput if static_throughput > 0 else 100.0
    
    for _ in range(max_iter):
        f_val = cumulative(Y) - M_tot
        f_deriv = derivative(Y)
        
        if abs(f_deriv) < 1e-12:
            warnings.warn("导数接近零，牛顿法可能不收敛")
            break
        
        Y_new = Y - f_val / f_deriv
        
        # 确保 Y > 0
        if Y_new <= 0:
            Y_new = Y / 2.0
        
        if abs(Y_new - Y) < tol:
            return Y_new
        
        Y = Y_new
    
    warnings.warn(f"牛顿法未在 {max_iter} 次迭代内收敛")
    return Y


def calculate_dynamic_cost(
    Y: float,
    N0: int,
    K: int,
    r: float,
    L_max: float,
    p_B: float,
    c_R: float,
    C_site: float,
    rho: float
) -> dict:
    """计算动态成本模型 (CAPEX + OPEX).
    
    CAPEX = C_site * (N_final - N0)
    OPEX = ∫₀^Y c_R * N(t) * L_max * p_B * e^(-ρt) dt
    
    Args:
        Y: 完工时间
        N0: 初始发射场数量
        K, r: Logistic 参数
        L_max, p_B: 发射参数
        c_R: 单位运输成本
        C_site: 单场建设成本
        rho: 折现率
    
    Returns:
        包含 CAPEX, OPEX, Total 的字典
    """
    # CAPEX: 新建发射场成本
    N_final = logistic_N(Y, N0, K, r)
    capex = C_site * max(0, N_final - N0)
    
    # OPEX: 数值积分 (辛普森法则)
    n_steps = 1000
    dt = Y / n_steps
    opex = 0.0
    
    for i in range(n_steps + 1):
        t = i * dt
        N_t = logistic_N(t, N0, K, r)
        annual_cost = c_R * N_t * L_max * p_B * exp(-rho * t)
        
        # 辛普森权重
        if i == 0 or i == n_steps:
            weight = 1.0
        elif i % 2 == 1:
            weight = 4.0
        else:
            weight = 2.0
        
        opex += weight * annual_cost
    
    opex *= dt / 3.0
    
    return {
        "CAPEX": capex,
        "OPEX": opex,
        "Total": capex + opex,
        "N_final": N_final
    }


# ============================================================================
# 第三部分: 方案 1a 计算 (空间电梯 - 静态模型)
# ============================================================================

def calculate_scenario_1a(params: GlobalParams, verbose: bool = True) -> Optional[dict]:
    """计算方案 1a (仅使用电梯系统) 的时间与成本.
    
    逻辑依据: single_mode_models.md 第1节
    系统架构: 串联系统 (Series System).
    瓶颈: 取决于 '电梯管道吞吐' 与 '锚点转运能力' 之间的最小值.
    
    Args:
        params: 全局参数
        verbose: 是否打印详细信息
    
    Returns:
        包含 makespan, cost, bottleneck 等信息的字典
    """
    if verbose:
        print(f"--- 评估方案 1a: 纯空间电梯运输 ---")
    
    # 1. 计算锚点转运系统的年吞吐能力
    rate_anchor_launches = params.N_anchor * params.L_anchor
    throughput_anchor = rate_anchor_launches * params.p_A
    
    if verbose:
        print(f"  [能力分析] 电梯管道吞吐: {params.T_E:,.0f} 吨/年")
        print(f"  [能力分析] 锚点转运吞吐: {throughput_anchor:,.0f} 吨/年 ({rate_anchor_launches} 次发射/年)")
    
    # 2. 识别系统瓶颈 (Bottleneck)
    throughput_chain = min(params.T_E, throughput_anchor)
    
    if verbose:
        print(f"  [系统瓶颈] 实际链条吞吐: {throughput_chain:,.0f} 吨/年")
    
    if throughput_chain <= 0:
        if verbose:
            print("  [错误] 系统吞吐能力为0，无法完成运输。")
        return None

    # 3. 计算完工时间 (Y_1a)
    time_continuous = params.M_tot / params.T_E if params.T_E > 0 else inf
    total_launches_needed = ceil(params.M_tot / params.p_A)
    time_discrete = total_launches_needed / rate_anchor_launches if rate_anchor_launches > 0 else inf
    
    makespan = max(time_continuous, time_discrete)
    bottleneck_location = "Elevator Pipeline" if time_continuous >= time_discrete else "Anchor Transfer"
    
    if verbose:
        print(f"  [瓶颈位置] {bottleneck_location}")
    
    # 4. 计算总成本 (C_1a)
    cost = params.F_E + (params.c_E * params.M_tot)
    
    if verbose:
        print(f"  [计算结果] 需锚点发射次数: {total_launches_needed:,} 次")
        print(f"  [计算结果] 完工时间 (Y): {makespan:.4f} 年")
        print(f"  [计算结果] 总成本 (C): {cost:,.2f}")
    
    return {
        "scenario": "1a (Elevator Only)",
        "makespan": makespan,
        "cost": cost,
        "cost_fixed": params.F_E,
        "cost_variable": params.c_E * params.M_tot,
        "bottleneck_throughput": throughput_chain,
        "bottleneck_location": bottleneck_location,
        "throughput_elevator": params.T_E,
        "throughput_anchor": throughput_anchor,
        "total_launches": total_launches_needed
    }


# ============================================================================
# 第四部分: 方案 1b 计算 (传统火箭)
# ============================================================================

def calculate_scenario_1b_static(params: GlobalParams, verbose: bool = True) -> Optional[dict]:
    """计算方案 1b 静态模型 (假设发射能力从第一天起满负荷).
    
    逻辑依据: single_mode_models.md 第2.2-2.3节
    这是一个乐观下界估计。
    
    Args:
        params: 全局参数
        verbose: 是否打印详细信息
    
    Returns:
        包含 makespan, cost 等信息的字典
    """
    if verbose:
        print(f"\n--- 评估方案 1b (静态): 纯传统火箭运输 ---")
    
    # 1. 计算地面火箭系统的年总吞吐能力
    rate_ground_launches = params.N_sites * params.L_max
    throughput_ground = rate_ground_launches * params.p_B
    
    if verbose:
        print(f"  [能力分析] 地面火箭群吞吐: {throughput_ground:,.0f} 吨/年 ({rate_ground_launches} 次发射/年)")
    
    if throughput_ground <= 0:
        if verbose:
            print("  [错误] 系统吞吐能力为0，无法完成运输。")
        return None

    # 2. 计算完工时间 (Y_1b)
    total_launches_needed = ceil(params.M_tot / params.p_B)
    makespan = total_launches_needed / rate_ground_launches if rate_ground_launches > 0 else inf
    
    # 3. 计算总成本 (C_1b) - 静态模型只考虑运营成本
    cost = params.c_R * params.M_tot
    
    if verbose:
        print(f"  [计算结果] 需地面发射次数: {total_launches_needed:,} 次")
        print(f"  [计算结果] 完工时间 (Y): {makespan:.4f} 年")
        print(f"  [计算结果] 总成本 (C): {cost:,.2f}")
    
    return {
        "scenario": "1b-Static (Rocket Only)",
        "model_type": "static",
        "makespan": makespan,
        "cost": cost,
        "cost_fixed": 0.0,
        "cost_variable": cost,
        "bottleneck_throughput": throughput_ground,
        "total_launches": total_launches_needed,
        "N_sites": params.N_sites,
        "L_max": params.L_max
    }


def calculate_scenario_1b_dynamic(
    params: GlobalParams,
    dyn_params: DynamicParams,
    verbose: bool = True
) -> Optional[dict]:
    """计算方案 1b 动态模型 (考虑Logistic增长和周转时间约束).
    
    逻辑依据: single_mode_models.md 第2.4节
    
    动态修正包括:
    1. 周转时间约束: L_max = 365 * η / t_cycle
    2. Logistic增长: N(t) 从 N0 增长到 K
    3. 积分约束方程求解完工时间
    4. CAPEX + OPEX 成本计算
    
    Args:
        params: 全局参数 (N_sites 作为 N0)
        dyn_params: 动态修正参数
        verbose: 是否打印详细信息
    
    Returns:
        包含动态 makespan, cost, N(t) 等信息的字典
    """
    if verbose:
        print(f"\n--- 评估方案 1b (动态): 纯传统火箭运输 ---")
        print(f"  [技术情景] {dyn_params.get_scenario_label()}")
        print(f"  [周转时间] t_cycle = {dyn_params.t_cycle} 天")
    
    # 1. 根据周转时间计算物理极限发射频率
    L_max_physical = dyn_params.L_max_physical
    
    if verbose:
        print(f"  [物理约束] L_max = 365 × {dyn_params.eta} / {dyn_params.t_cycle} = {L_max_physical:.1f} 次/年/场")
    
    # 2. 提取Logistic参数
    N0 = params.N_sites
    K = dyn_params.K
    r = dyn_params.r
    p_B = params.p_B
    
    # 计算初始与最终吞吐
    initial_throughput = N0 * L_max_physical * p_B
    max_throughput = K * L_max_physical * p_B
    
    if verbose:
        print(f"  [Logistic参数] N0={N0}, K={K}, r={r}")
        print(f"  [初始吞吐] {initial_throughput:,.0f} 吨/年")
        print(f"  [饱和吞吐] {max_throughput:,.0f} 吨/年")
        print(f"  [拐点时间] t* = {logistic_inflection_point(N0, K, r):.2f} 年")
    
    # 3. 数值求解动态完工时间
    makespan = solve_dynamic_makespan(
        M_tot=params.M_tot,
        N0=N0,
        K=K,
        r=r,
        L_max=L_max_physical,
        p_B=p_B
    )
    
    if makespan is None:
        if verbose:
            print("  [错误] 无法求解动态完工时间")
        return None
    
    # 4. 计算动态成本
    cost_detail = calculate_dynamic_cost(
        Y=makespan,
        N0=N0,
        K=K,
        r=r,
        L_max=L_max_physical,
        p_B=p_B,
        c_R=params.c_R,
        C_site=dyn_params.C_site,
        rho=dyn_params.rho
    )
    
    # 5. 计算静态模型作为对比基准
    static_makespan = params.M_tot / initial_throughput if initial_throughput > 0 else inf
    
    if verbose:
        print(f"  [动态完工时间] Y_dyn = {makespan:.2f} 年")
        print(f"  [静态完工时间] Y_static = {static_makespan:.2f} 年 (假设运力恒定)")
        print(f"  [时间增加比例] +{(makespan/static_makespan - 1)*100:.1f}%")
        print(f"  [最终发射场数] N(Y) = {cost_detail['N_final']:.1f}")
        print(f"  [成本分解] CAPEX: {cost_detail['CAPEX']:,.2f}, OPEX: {cost_detail['OPEX']:,.2f}")
        print(f"  [总成本] {cost_detail['Total']:,.2f}")
    
    return {
        "scenario": "1b-Dynamic (Rocket Only)",
        "model_type": "dynamic",
        "makespan": makespan,
        "makespan_static": static_makespan,
        "time_increase_ratio": makespan / static_makespan - 1,
        "cost": cost_detail["Total"],
        "cost_CAPEX": cost_detail["CAPEX"],
        "cost_OPEX": cost_detail["OPEX"],
        "N0": N0,
        "K": K,
        "r": r,
        "N_final": cost_detail["N_final"],
        "L_max_physical": L_max_physical,
        "t_cycle": dyn_params.t_cycle,
        "initial_throughput": initial_throughput,
        "max_throughput": max_throughput,
        "inflection_point": logistic_inflection_point(N0, K, r),
        "scenario_label": dyn_params.get_scenario_label()
    }


# 保持向后兼容的别名
def calculate_scenario_1b(params: GlobalParams, verbose: bool = True) -> Optional[dict]:
    """方案 1b 计算 (静态模型，向后兼容)."""
    return calculate_scenario_1b_static(params, verbose)

# ============================================================================
# 第五部分: 对比与工具函数
# ============================================================================

def compare_scenarios(res_a: dict, res_b: dict, verbose: bool = True) -> dict:
    """对比两种方案的优劣.
    
    Args:
        res_a: 方案 1a 的计算结果
        res_b: 方案 1b 的计算结果
        verbose: 是否打印详细信息
    
    Returns:
        对比结果字典
    """
    if not res_a or not res_b:
        return {}

    comparison = {
        "time_winner": "1a" if res_a["makespan"] < res_b["makespan"] else "1b",
        "time_diff": abs(res_a["makespan"] - res_b["makespan"]),
        "cost_winner": "1a" if res_a["cost"] < res_b["cost"] else "1b",
        "cost_diff": abs(res_a["cost"] - res_b["cost"]),
        "cost_ratio": res_a["cost"] / res_b["cost"] if res_b["cost"] > 0 else inf
    }
    
    if verbose:
        print(f"\n====== 最终对比结论 ======")
        
        if comparison["time_winner"] == "1a":
            print(f"🚀 时间最优: 方案 1a (电梯) 快 {comparison['time_diff']:.2f} 年")
        else:
            print(f"🚀 时间最优: 方案 1b (火箭) 快 {comparison['time_diff']:.2f} 年")
            
        if comparison["cost_winner"] == "1a":
            print(f"💰 成本最优: 方案 1a (电梯) 省 {comparison['cost_diff']:,.2f}")
        else:
            print(f"💰 成本最优: 方案 1b (火箭) 省 {comparison['cost_diff']:,.2f}")
        
        print(f"📊 成本比 (1a/1b): {comparison['cost_ratio']:.4f}")
    
    return comparison


def calculate_breakeven_mass(F_E: float, c_E: float, c_R: float) -> float:
    """计算电梯与火箭的盈亏平衡点质量.
    
    公式: M* = F_E / (c_R - c_E)
    
    当 M_tot > M* 时，电梯更经济。
    """
    if c_R <= c_E:
        return inf  # 火箭更便宜，电梯永远不划算
    return F_E / (c_R - c_E)


def generate_logistic_curve(
    N0: int,
    K: int,
    r: float,
    Y_max: float,
    n_points: int = 100
) -> tuple:
    """生成 Logistic 曲线数据点 (用于可视化).
    
    Args:
        N0, K, r: Logistic 参数
        Y_max: 时间范围
        n_points: 数据点数量
    
    Returns:
        (t_values, N_values) 元组
    """
    t_values = [i * Y_max / n_points for i in range(n_points + 1)]
    N_values = [logistic_N(t, N0, K, r) for t in t_values]
    return t_values, N_values


def generate_cumulative_transport_curve(
    N0: int,
    K: int,
    r: float,
    L_max: float,
    p_B: float,
    Y_max: float,
    n_points: int = 100
) -> tuple:
    """生成累积运输量曲线 (用于可视化).
    
    Returns:
        (t_values, cumulative_values) 元组
    """
    t_values = [i * Y_max / n_points for i in range(n_points + 1)]
    cumulative_values = [L_max * p_B * logistic_integral(t, N0, K, r) for t in t_values]
    return t_values, cumulative_values


# ============================================================================
# 第六部分: 预设技术情景
# ============================================================================

# 预定义的技术情景参数
SCENARIO_CONSERVATIVE = DynamicParams(t_cycle=14, eta=0.85, K=50, r=0.2, C_site=4.0e10, rho=0.05)
SCENARIO_MODERATE = DynamicParams(t_cycle=4, eta=0.90, K=80, r=0.3, C_site=3.0e10, rho=0.03)
SCENARIO_AGGRESSIVE = DynamicParams(t_cycle=1, eta=0.95, K=100, r=0.5, C_site=2.0e10, rho=0.02)


def get_default_params() -> GlobalParams:
    """返回默认全局参数."""
    return GlobalParams(
        M_tot=1.0e8,        # 1亿吨
        
        # 1a 参数
        T_E=5.37e5,         # 电梯年吞吐 53.7万吨
        N_anchor=6,         # 6个锚点
        L_anchor=2000,      # 年发射次数
        p_A=125.0,          # 125吨/次
        F_E=5.0e9,          # 固定成本 50亿
        c_E=2.7e3,          # 单位成本 2700/吨
        
        # 1b 参数
        N_sites=10,         # 10个发射场
        L_max=2000,         # 年发射次数 (静态模型用)
        p_B=125.0,          # 125吨/次
        c_R=7.2e5           # 单位成本 72万/吨
    )


# ============================================================================
# 主程序
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("  单一运输系统优化模块 - 综合评估")
    print("=" * 70)
    
    # 使用默认参数
    params = get_default_params()
    
    # ===== 方案 1a: 空间电梯 =====
    res_1a = calculate_scenario_1a(params)
    
    # ===== 方案 1b: 静态模型 =====
    res_1b_static = calculate_scenario_1b_static(params)
    
    # ===== 方案 1b: 动态模型 (三种情景) =====
    print("\n" + "=" * 70)
    print("  动态模型: 三种技术情景对比")
    print("=" * 70)
    
    for name, dyn_params in [
        ("保守情景 (Falcon 9)", SCENARIO_CONSERVATIVE),
        ("稳健情景 (Starship)", SCENARIO_MODERATE),
        ("激进情景 (航空化)", SCENARIO_AGGRESSIVE)
    ]:
        print(f"\n>>> {name}")
        res = calculate_scenario_1b_dynamic(params, dyn_params)
    
    # ===== 对比分析 =====
    print("\n" + "=" * 70)
    print("  静态模型对比 (1a vs 1b-Static)")
    print("=" * 70)
    compare_scenarios(res_1a, res_1b_static)
    
    # ===== 盈亏平衡点 =====
    breakeven = calculate_breakeven_mass(params.F_E, params.c_E, params.c_R)
    print(f"\n📌 盈亏平衡点: M* = {breakeven:,.0f} 吨")
    print(f"   当 M_tot > {breakeven:,.0f} 吨时，电梯更经济。")
    print(f"   当前任务 M_tot = {params.M_tot:,.0f} 吨 > M*，电梯方案占优。")
