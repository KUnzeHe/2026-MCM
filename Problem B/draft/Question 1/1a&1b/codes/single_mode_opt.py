from __future__ import annotations

from dataclasses import dataclass
from math import ceil, inf

@dataclass(frozen=True)
class GlobalParams:
    """定义计算所需的全局参数.
    
    包含项目需求、电梯配置、火箭配置及各类成本系数。
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
    N_sites: int    # 地面发射场数量
    L_max: int      # 每个发射场年最大发射次数
    p_B: float      # 地面火箭单次有效载荷 (吨/次)
    
    # 成本参数 (火箭)
    c_R: float      # 地面火箭单位运输成本 (Currency/Ton)


def calculate_scenario_1a(params: GlobalParams):
    """计算方案 1a (仅使用电梯系统) 的时间与成本.
    
    逻辑依据: /Problem B/draft/single_mode_models.md 第1节
    系统架构: 串联系统 (Series System).
    瓶颈: 取决于 '电梯管道吞吐' 与 '锚点转运能力' 之间的最小值.
    """
    print(f"--- 评估方案 1a: 纯空间电梯运输 ---")
    
    # 1. 计算锚点转运系统的年吞吐能力
    # 总吞吐 = 锚点数 * 单锚点年频次 * 单次载荷
    rate_anchor_launches = params.N_anchor * params.L_anchor
    throughput_anchor = rate_anchor_launches * params.p_A
    
    print(f"  [能力分析] 电梯管道吞吐: {params.T_E:,.0f} 吨/年")
    print(f"  [能力分析] 锚点转运吞吐: {throughput_anchor:,.0f} 吨/年 ({rate_anchor_launches} 次发射/年)")
    
    # 2. 识别系统瓶颈 (Bottleneck)
    # 串联系统的总吞吐受限于最弱的一环
    throughput_chain = min(params.T_E, throughput_anchor)
    print(f"  [系统瓶颈] 实际链条吞吐: {throughput_chain:,.0f} 吨/年")
    
    if throughput_chain <= 0:
        print("  [错误] 系统吞吐能力为0，无法完成运输。")
        return None

    # 3. 计算完工时间 (Y_1a)
    # 考虑批次效应: 锚点火箭必须整次发射
    # 时间 = max( 电梯传输时间, 锚点批次发射所需时间 )
    
    # 电梯连续时间（受限于管道吞吐）
    time_continuous = params.M_tot / params.T_E if params.T_E > 0 else inf
    
    # 锚点离散时间（受限于批次发射）
    total_launches_needed = ceil(params.M_tot / params.p_A)
    time_discrete = total_launches_needed / rate_anchor_launches if rate_anchor_launches > 0 else inf
    
    makespan = max(time_continuous, time_discrete)
    
    # 识别实际瓶颈位置
    bottleneck_location = "电梯管道" if time_continuous >= time_discrete else "锚点转运"
    print(f"  [瓶颈位置] {bottleneck_location}")
    
    # 4. 计算总成本 (C_1a)
    # C = 固定建设成本 + (单位成本 * 总质量)
    cost = params.F_E + (params.c_E * params.M_tot)
    
    print(f"  [计算结果] 需锚点发射次数: {total_launches_needed:,} 次")
    print(f"  [计算结果] 完工时间 (Y): {makespan:.4f} 年")
    print(f"  [计算结果] 总成本 (C): {cost:,.2f}")
    
    return {
        "scenario": "1a (Elevator Only)",
        "makespan": makespan,
        "cost": cost,
        "bottleneck_throughput": throughput_chain
    }


def calculate_scenario_1b(params: GlobalParams):
    """计算方案 1b (仅使用传统火箭) 的时间与成本.
    
    逻辑依据: /Problem B/draft/single_mode_models.md 第2节
    系统架构: 并行系统 (Parallel System).
    能力: 所有地面发射场同时运作.
    """
    print(f"\n--- 评估方案 1b: 纯传统火箭运输 ---")
    
    # 1. 计算地面火箭系统的年总吞吐能力
    # 总吞吐 = 场地数 * 单场地年频次 * 单次载荷
    rate_ground_launches = params.N_sites * params.L_max
    throughput_ground = rate_ground_launches * params.p_B
    
    print(f"  [能力分析] 地面火箭群吞吐: {throughput_ground:,.0f} 吨/年 ({rate_ground_launches} 次发射/年)")
    
    if throughput_ground <= 0:
        print("  [错误] 系统吞吐能力为0，无法完成运输。")
        return None

    # 2. 计算完工时间 (Y_1b)
    # 纯离散批次计算
    total_launches_needed = ceil(params.M_tot / params.p_B)
    makespan = total_launches_needed / rate_ground_launches if rate_ground_launches > 0 else inf
    
    # 3. 计算总成本 (C_1b)
    # 假设无额外基建固定成本，全为边际发射成本
    # C = 单位成本 * 总质量
    cost = params.c_R * params.M_tot
    
    print(f"  [计算结果] 需地面发射次数: {total_launches_needed:,} 次")
    print(f"  [计算结果] 完工时间 (Y): {makespan:.4f} 年")
    print(f"  [计算结果] 总成本 (C): {cost:,.2f}")
    
    return {
        "scenario": "1b (Rocket Only)",
        "makespan": makespan,
        "cost": cost,
        "bottleneck_throughput": throughput_ground
    }

def compare_scenarios(res_a, res_b):
    """对比两种方案的优劣."""
    if not res_a or not res_b:
        return

    print(f"\n====== 最终对比结论 ======")
    
    # 时间对比
    if res_a["makespan"] < res_b["makespan"]:
        print(f"🚀 时间最优: 方案 1a (电梯) 快 {res_b['makespan'] - res_a['makespan']:.2f} 年")
    else:
        print(f"🚀 时间最优: 方案 1b (火箭) 快 {res_a['makespan'] - res_b['makespan']:.2f} 年")
        
    # 成本对比
    if res_a["cost"] < res_b["cost"]:
        print(f"💰 成本最优: 方案 1a (电梯) 省 {res_b['cost'] - res_a['cost']:,.2f}")
    else:
        print(f"💰 成本最优: 方案 1b (火箭) 省 {res_a['cost'] - res_b['cost']:,.2f}")


if __name__ == "__main__":
    # 示例参数 (参考 mixed_plan_opt.py 中的配置)
    # 注意：这里的参数决定了结果，实际使用时需根据题目具体数据调整
    test_params = GlobalParams(
        M_tot=1.0e8,        # 1亿吨
        
        # 1a 参数 (假设电梯是长期投资，单位成本低)
        T_E=5.37e5,         # 电梯年吞吐 53.7万吨 (题目给定179,000吨/年 * 3个港口)
        N_anchor=3,         # 3个锚点 (Galactic Harbours)
        L_anchor=3650,      # 年发射次数 (约每天10次，全年运转)
        p_A=125.0,          # 125吨/次 (题目范围100-150吨)
        F_E=5.0e9,          # 固定成本 50亿 (电梯基建投资)
        c_E=2.7e3,          # 单位成本 2700/吨 (电梯链路)
        
        # 1b 参数 (传统火箭，单位成本高，发射能力强)
        N_sites=10,         # 10个发射场 (题目提及的候选场地)
        L_max=3650,         # 年发射次数 (约每天10次，全年运转)
        p_B=125.0,          # 125吨/次 (题目范围100-150吨)
        c_R=7.2e5           # 单位成本 72万/吨 (火箭发射昂贵)
    )
    
    res_1a = calculate_scenario_1a(test_params)
    res_1b = calculate_scenario_1b(test_params)
    
    compare_scenarios(res_1a, res_1b)
