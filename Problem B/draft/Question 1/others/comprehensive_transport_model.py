"""
混合运输系统综合优化模型 (Comprehensive Transport Model V3)
============================================================
基于 Logistic 动态扩建、物理约束（周转时间、Delta-v）、Monte Carlo 模拟
"""

import math
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict

# ============================================================================
# 1. 参数定义类
# ============================================================================

class BaseParameters:
    """基础固定参数"""
    M_tot: float = 1e8          # 总运输质量 (吨)
    T_E: float = 537_000        # 电梯年吞吐量 (吨/年), 179k × 3
    F_E: float = 100e9          # 电梯固定成本 ($)
    c_E: float = 50             # 电梯单位成本 ($/kg)
    N_0: int = 10               # 初始发射场数量


class DynamicParameters:
    """动态/可调参数"""
    K: float = 80               # 发射场环境承载力
    r: float = 0.3              # 基建增长率 (/年)
    p_B_min: float = 100        # 地面火箭载荷下限 (吨)
    p_B_max: float = 150        # 地面火箭载荷上限 (吨)
    eta: float = 0.9            # 系统可用率


class PhysicsParameters:
    """物理推导参数"""
    beta: float = 5.0           # 载荷增益系数 (p_A / p_B)
    delta_v_earth: float = 12.6 # 地面发射 Δv (km/s)
    delta_v_anchor: float = 1.5 # 锚点发射 Δv (km/s)
    I_sp: float = 350           # 比冲 (s)
    g_0: float = 9.81           # 重力加速度 (m/s²)


class FinancialParameters:
    """财务参数"""
    C_site: float = 30e9        # 单发射场建设费 ($)
    C_launch: float = 150e6     # 单次发射成本 ($)
    rho: float = 0.03           # 资金贴现率


class TechScenario:
    """技术情景定义"""
    def __init__(self, name: str, t_cycle: float, description: str):
        self.name = name
        self.t_cycle = t_cycle  # 周转时间 (天)
        self.description = description
        self.L_max = self._calculate_L_max()
    
    def _calculate_L_max(self) -> float:
        """计算年最大发射次数"""
        eta = DynamicParameters.eta
        return (365 * eta) / self.t_cycle


# 预定义三种技术情景
SCENARIO_A = TechScenario("Conservative", 14, "现有 Falcon 9 水平")
SCENARIO_B = TechScenario("Moderate", 4, "Starship 预期目标")
SCENARIO_C = TechScenario("Aggressive", 1, "航空化运营理想")


# ============================================================================
# 2. Logistic 动态增长模型
# ============================================================================

class LogisticGrowthModel:
    """发射场 Logistic 增长模型"""
    
    def __init__(self, N_0: int, K: float, r: float):
        self.N_0 = N_0
        self.K = K
        self.r = r
    
    def N(self, t: float) -> float:
        """计算第 t 年的发射场数量"""
        if t < 0:
            return self.N_0
        return self.K / (1 + ((self.K - self.N_0) / self.N_0) * math.exp(-self.r * t))
    
    def cumulative_capacity(self, Y: float, L_max: float, p_B: float, dt: float = 0.1) -> float:
        """
        计算 0 到 Y 年火箭系统的累计运输量
        ∫₀^Y N(t) · L_max · p_B dt
        """
        total = 0.0
        t = 0.0
        while t < Y:
            total += self.N(t) * L_max * p_B * dt
            t += dt
        return total
    
    def find_required_K(self, Y_target: float, M_required: float, 
                        L_max: float, p_B: float, T_E: float) -> float:
        """
        反向求解：给定目标时间 Y_target，求满足运输任务所需的 K
        使用二分法
        """
        # 电梯贡献
        elevator_contribution = T_E * Y_target
        rocket_required = M_required - elevator_contribution
        
        if rocket_required <= 0:
            return self.N_0  # 电梯足够，不需要扩建
        
        # 二分搜索 K
        K_low, K_high = self.N_0, 1000
        tolerance = 1e6  # 吨
        
        while K_high - K_low > 1:
            K_mid = (K_low + K_high) / 2
            self.K = K_mid
            capacity = self.cumulative_capacity(Y_target, L_max, p_B)
            
            if capacity < rocket_required:
                K_low = K_mid
            else:
                K_high = K_mid
        
        return K_high


# ============================================================================
# 3. 载荷物理模型 (Delta-v Analysis)
# ============================================================================

class PayloadPhysicsModel:
    """基于 Δv 的载荷差异建模"""
    
    def __init__(self, params: PhysicsParameters = PhysicsParameters()):
        self.params = params
    
    def calculate_beta(self) -> float:
        """
        计算载荷增益系数 β = p_A / p_B
        基于火箭方程: β ≈ exp((Δv_Earth - Δv_Anchor) / v_e)
        """
        v_e = self.params.I_sp * self.params.g_0 / 1000  # km/s
        delta_delta_v = self.params.delta_v_earth - self.params.delta_v_anchor
        return math.exp(delta_delta_v / v_e)
    
    def get_anchor_payload(self, p_B: float) -> float:
        """给定地面载荷，计算锚点载荷"""
        return self.params.beta * p_B


# ============================================================================
# 4. 成本模型 (CAPEX + OPEX with NPV)
# ============================================================================

class CostModel:
    """综合成本模型"""
    
    def __init__(self, base: BaseParameters = BaseParameters(),
                 fin: FinancialParameters = FinancialParameters()):
        self.base = base
        self.fin = fin
    
    def calculate_capex(self, N_final: int) -> float:
        """计算基建资本支出"""
        new_sites = max(0, N_final - self.base.N_0)
        return self.fin.C_site * new_sites + self.base.F_E
    
    def calculate_opex_npv(self, x_elevator: float, m_rocket: float, 
                           Y: float, dt: float = 1.0) -> float:
        """
        计算运营成本的净现值 (NPV)
        假设运输均匀分布在 Y 年内
        """
        # 年运营成本
        annual_elevator_cost = (x_elevator / Y) * self.base.c_E * 1000  # kg → $
        c_R = self.fin.C_launch / 125_000  # $/kg (假设平均载荷125t)
        annual_rocket_cost = (m_rocket / Y) * c_R * 1000
        
        # NPV 贴现
        npv = 0.0
        for year in range(int(Y)):
            discount = math.exp(-self.fin.rho * year)
            npv += (annual_elevator_cost + annual_rocket_cost) * discount
        
        return npv
    
    def total_cost(self, x_elevator: float, m_rocket: float, 
                   N_final: int, Y: float) -> float:
        """计算总成本"""
        capex = self.calculate_capex(N_final)
        opex = self.calculate_opex_npv(x_elevator, m_rocket, Y)
        return capex + opex


# ============================================================================
# 5. Monte Carlo 模拟器
# ============================================================================

class MonteCarloSimulator:
    """Monte Carlo 不确定性分析"""
    
    def __init__(self, n_simulations: int = 10000):
        self.n_simulations = n_simulations
        self.results = []
    
    def sample_payload(self) -> float:
        """从均匀分布采样火箭载荷 p_B"""
        return random.uniform(DynamicParameters.p_B_min, DynamicParameters.p_B_max)
    
    def sample_growth_rate(self) -> float:
        """采样增长率 (正态分布)"""
        return max(0.1, random.gauss(0.3, 0.1))
    
    def run_simulation(self, scenario: TechScenario, Y_target: float = 24) -> Dict:
        """
        运行 Monte Carlo 模拟
        返回统计结果
        """
        completion_times = []
        total_costs = []
        required_Ks = []
        
        base = BaseParameters()
        cost_model = CostModel()
        
        for _ in range(self.n_simulations):
            # 随机采样参数
            p_B = self.sample_payload()
            r = self.sample_growth_rate()
            
            # 创建增长模型
            growth_model = LogisticGrowthModel(base.N_0, DynamicParameters.K, r)
            
            # 反向求解所需 K
            K_required = growth_model.find_required_K(
                Y_target, base.M_tot, scenario.L_max, p_B, base.T_E
            )
            required_Ks.append(K_required)
            
            # 计算分配
            elevator_capacity = base.T_E * Y_target
            x_elevator = min(elevator_capacity, base.M_tot)
            m_rocket = base.M_tot - x_elevator
            
            # 计算成本
            cost = cost_model.total_cost(x_elevator, m_rocket, int(K_required), Y_target)
            total_costs.append(cost)
        
        # 统计分析
        results = {
            'scenario': scenario.name,
            'Y_target': Y_target,
            'K_mean': np.mean(required_Ks),
            'K_std': np.std(required_Ks),
            'K_95_ci': (np.percentile(required_Ks, 2.5), np.percentile(required_Ks, 97.5)),
            'cost_mean': np.mean(total_costs),
            'cost_std': np.std(total_costs),
            'cost_95_ci': (np.percentile(total_costs, 2.5), np.percentile(total_costs, 97.5)),
            'raw_K': required_Ks,
            'raw_cost': total_costs
        }
        
        self.results.append(results)
        return results


# ============================================================================
# 6. Pareto 前沿分析器
# ============================================================================

class ParetoAnalyzer:
    """时间-成本 Pareto 前沿分析"""
    
    def __init__(self, scenario: TechScenario):
        self.scenario = scenario
        self.pareto_points = []
    
    def compute_pareto_frontier(self, Y_range: Tuple[float, float] = (15, 50), 
                                 n_points: int = 50) -> List[Tuple[float, float]]:
        """
        计算 Pareto 前沿
        返回 (Y, Cost) 点列表
        """
        base = BaseParameters()
        cost_model = CostModel()
        growth_model = LogisticGrowthModel(base.N_0, DynamicParameters.K, DynamicParameters.r)
        p_B = (DynamicParameters.p_B_min + DynamicParameters.p_B_max) / 2
        
        Y_values = np.linspace(Y_range[0], Y_range[1], n_points)
        
        for Y in Y_values:
            # 电梯贡献
            elevator_contribution = base.T_E * Y
            x_elevator = min(elevator_contribution, base.M_tot)
            m_rocket = base.M_tot - x_elevator
            
            # 求所需 K
            if m_rocket > 0:
                K_required = growth_model.find_required_K(
                    Y, base.M_tot, self.scenario.L_max, p_B, base.T_E
                )
            else:
                K_required = base.N_0
            
            # 计算成本
            cost = cost_model.total_cost(x_elevator, m_rocket, int(K_required), Y)
            
            self.pareto_points.append((Y, cost, K_required))
        
        return self.pareto_points
    
    def find_knee_point(self) -> Tuple[float, float, float]:
        """
        寻找 Pareto 前沿的膝点 (Knee Point)
        使用最大曲率法
        """
        if len(self.pareto_points) < 3:
            return self.pareto_points[0]
        
        # 归一化
        Y_vals = np.array([p[0] for p in self.pareto_points])
        C_vals = np.array([p[1] for p in self.pareto_points])
        
        Y_norm = (Y_vals - Y_vals.min()) / (Y_vals.max() - Y_vals.min())
        C_norm = (C_vals - C_vals.min()) / (C_vals.max() - C_vals.min())
        
        # 计算到对角线的距离
        distances = []
        for i in range(len(Y_norm)):
            # 对角线: 从 (0, 1) 到 (1, 0)
            d = abs(Y_norm[i] + C_norm[i] - 1) / math.sqrt(2)
            distances.append(d)
        
        knee_idx = np.argmax(distances)
        return self.pareto_points[knee_idx]


# ============================================================================
# 7. 可视化模块
# ============================================================================

class Visualizer:
    """可视化工具"""
    
    @staticmethod
    def plot_logistic_growth(growth_model: LogisticGrowthModel, Y_max: float = 30):
        """绘制 Logistic 增长曲线"""
        t_vals = np.linspace(0, Y_max, 200)
        N_vals = [growth_model.N(t) for t in t_vals]
        
        plt.figure(figsize=(10, 6))
        plt.plot(t_vals, N_vals, 'b-', linewidth=2)
        plt.axhline(y=growth_model.K, color='r', linestyle='--', label=f'Carrying Capacity K={growth_model.K}')
        plt.axhline(y=growth_model.N_0, color='g', linestyle='--', label=f'Initial N₀={growth_model.N_0}')
        plt.xlabel('Time (years)', fontsize=12)
        plt.ylabel('Number of Launch Sites N(t)', fontsize=12)
        plt.title('Logistic Growth Model for Launch Site Expansion', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('/Users/kunze/Desktop/2026MCM/Problem B/draft/Question 1/1c/image/logistic_growth.png', dpi=150)
        plt.show()
    
    @staticmethod
    def plot_pareto_frontier(pareto_points: List[Tuple], scenario_name: str, knee_point: Tuple = None):
        """绘制 Pareto 前沿"""
        Y_vals = [p[0] for p in pareto_points]
        C_vals = [p[1] / 1e12 for p in pareto_points]  # 转换为万亿美元
        
        plt.figure(figsize=(12, 7))
        plt.plot(Y_vals, C_vals, 'b-o', linewidth=2, markersize=4, label='Pareto Frontier')
        
        if knee_point:
            plt.scatter([knee_point[0]], [knee_point[1] / 1e12], 
                       color='red', s=200, zorder=5, marker='*', label='Knee Point')
            plt.annotate(f'Knee: ({knee_point[0]:.1f} yrs, ${knee_point[1]/1e12:.2f}T)',
                        xy=(knee_point[0], knee_point[1]/1e12),
                        xytext=(knee_point[0]+3, knee_point[1]/1e12+0.5),
                        fontsize=10, arrowprops=dict(arrowstyle='->', color='red'))
        
        plt.xlabel('Completion Time Y (years)', fontsize=12)
        plt.ylabel('Total Cost (Trillion $)', fontsize=12)
        plt.title(f'Time-Cost Pareto Frontier [{scenario_name}]', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'/Users/kunze/Desktop/2026MCM/Problem B/draft/Question 1/1c/image/pareto_frontier_{scenario_name}.png', dpi=150)
        plt.show()
    
    @staticmethod
    def plot_monte_carlo_distribution(results: Dict):
        """绘制 Monte Carlo 结果分布"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # K 分布
        axes[0].hist(results['raw_K'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0].axvline(results['K_mean'], color='red', linestyle='--', linewidth=2, label=f'Mean={results["K_mean"]:.1f}')
        axes[0].axvline(results['K_95_ci'][0], color='orange', linestyle=':', linewidth=2)
        axes[0].axvline(results['K_95_ci'][1], color='orange', linestyle=':', linewidth=2, label='95% CI')
        axes[0].set_xlabel('Required Launch Sites K', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title(f'Distribution of Required K [{results["scenario"]}]', fontsize=14)
        axes[0].legend()
        
        # 成本分布
        costs_trillion = [c / 1e12 for c in results['raw_cost']]
        axes[1].hist(costs_trillion, bins=50, color='coral', edgecolor='black', alpha=0.7)
        axes[1].axvline(results['cost_mean'] / 1e12, color='red', linestyle='--', linewidth=2, 
                        label=f'Mean=${results["cost_mean"]/1e12:.2f}T')
        axes[1].set_xlabel('Total Cost (Trillion $)', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title(f'Distribution of Total Cost [{results["scenario"]}]', fontsize=14)
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(f'/Users/kunze/Desktop/2026MCM/Problem B/draft/Question 1/1c/image/monte_carlo_{results["scenario"]}.png', dpi=150)
        plt.show()
    
    @staticmethod
    def plot_scenario_comparison(all_pareto_points: Dict):
        """多情景 Pareto 前沿对比"""
        plt.figure(figsize=(14, 8))
        colors = {'Conservative': 'blue', 'Moderate': 'green', 'Aggressive': 'red'}
        
        for scenario_name, points in all_pareto_points.items():
            Y_vals = [p[0] for p in points]
            C_vals = [p[1] / 1e12 for p in points]
            plt.plot(Y_vals, C_vals, '-o', color=colors.get(scenario_name, 'gray'),
                    linewidth=2, markersize=4, label=scenario_name)
        
        plt.xlabel('Completion Time Y (years)', fontsize=12)
        plt.ylabel('Total Cost (Trillion $)', fontsize=12)
        plt.title('Pareto Frontier Comparison Across Technology Scenarios', fontsize=14)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('/Users/kunze/Desktop/2026MCM/Problem B/draft/Question 1/1c/image/scenario_comparison.png', dpi=150)
        plt.show()


# ============================================================================
# 8. 主程序入口
# ============================================================================

def main():
    print("=" * 70)
    print("混合运输系统综合优化模型 V3")
    print("=" * 70)
    
    # -------------------------
    # Part 1: 物理参数验证
    # -------------------------
    print("\n[1] 载荷物理模型验证 (Delta-v Analysis)")
    physics = PayloadPhysicsModel()
    beta_calculated = physics.calculate_beta()
    print(f"    计算得到 β = {beta_calculated:.2f}")
    print(f"    若 p_B = 125t，则 p_A ≈ {physics.get_anchor_payload(125):.0f}t")
    
    # -------------------------
    # Part 2: Logistic 增长演示
    # -------------------------
    print("\n[2] Logistic 发射场扩建模型")
    growth = LogisticGrowthModel(N_0=10, K=80, r=0.3)
    print(f"    初始 N(0) = {growth.N(0):.0f}")
    print(f"    5年后 N(5) = {growth.N(5):.1f}")
    print(f"    15年后 N(15) = {growth.N(15):.1f}")
    print(f"    25年后 N(25) = {growth.N(25):.1f}")
    
    Visualizer.plot_logistic_growth(growth)
    
    # -------------------------
    # Part 3: 三情景 Pareto 分析
    # -------------------------
    print("\n[3] 三技术情景 Pareto 前沿分析")
    scenarios = [SCENARIO_A, SCENARIO_B, SCENARIO_C]
    all_pareto = {}
    
    for scenario in scenarios:
        print(f"\n    --- {scenario.name} (t_cycle={scenario.t_cycle}d, L_max={scenario.L_max:.0f}/yr) ---")
        analyzer = ParetoAnalyzer(scenario)
        points = analyzer.compute_pareto_frontier(Y_range=(15, 50), n_points=40)
        knee = analyzer.find_knee_point()
        
        print(f"    膝点: Y={knee[0]:.1f}年, Cost=${knee[1]/1e12:.2f}T, K={knee[2]:.0f}")
        all_pareto[scenario.name] = points
        
        Visualizer.plot_pareto_frontier(points, scenario.name, knee)
    
    # 多情景对比图
    Visualizer.plot_scenario_comparison(all_pareto)
    
    # -------------------------
    # Part 4: Monte Carlo 模拟
    # -------------------------
    print("\n[4] Monte Carlo 不确定性分析 (n=10,000)")
    mc_sim = MonteCarloSimulator(n_simulations=10000)
    
    for scenario in [SCENARIO_B]:  # 选择 Moderate 情景
        print(f"\n    运行情景: {scenario.name}")
        results = mc_sim.run_simulation(scenario, Y_target=24)
        
        print(f"    所需发射场 K: {results['K_mean']:.1f} ± {results['K_std']:.1f}")
        print(f"    95% CI: [{results['K_95_ci'][0]:.1f}, {results['K_95_ci'][1]:.1f}]")
        print(f"    总成本: ${results['cost_mean']/1e12:.2f}T ± ${results['cost_std']/1e12:.2f}T")
        
        Visualizer.plot_monte_carlo_distribution(results)
    
    # -------------------------
    # Part 5: 可行性判定
    # -------------------------
    print("\n[5] 可行性分析总结")
    print("    " + "-" * 60)
    print(f"    {'情景':<15} {'L_max':<10} {'所需K':<12} {'可行性':<15}")
    print("    " + "-" * 60)
    
    for scenario in scenarios:
        analyzer = ParetoAnalyzer(scenario)
        points = analyzer.compute_pareto_frontier(Y_range=(24, 25), n_points=2)
        K_needed = points[0][2]
        feasible = "✓ 可行" if K_needed <= 100 else "✗ 需审视"
        print(f"    {scenario.name:<15} {scenario.L_max:<10.0f} {K_needed:<12.0f} {feasible:<15}")
    
    print("\n" + "=" * 70)
    print("分析完成，图表已保存至 image/ 目录")
    print("=" * 70)


if __name__ == "__main__":
    main()
