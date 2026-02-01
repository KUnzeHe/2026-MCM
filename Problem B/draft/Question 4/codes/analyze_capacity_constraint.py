"""
分析电梯运输能力约束
- 建设期时间
- 运营期是否满足需求
"""

from q4_environmental_model import *
import numpy as np

print('=' * 70)
print('电梯运输能力分析：时间约束 vs 环境约束')
print('=' * 70)
print()

# ============ 基本参数 ============
TOTAL_MASS = 1e8  # 1亿吨
ELEVATOR_ANNUAL = 5.37e5  # 电梯年吞吐量 (3个Harbor × 179,000)
ROCKET_PAYLOAD = 150  # 吨/次
ROCKET_LAUNCHES_PER_YEAR_MAX = 50000  # 假设全球最大发射能力
ROCKET_ANNUAL_MAX = ROCKET_PAYLOAD * ROCKET_LAUNCHES_PER_YEAR_MAX  # 750万吨/年

# 运营期参数
POPULATION = 100000
SUPPLY_PER_PERSON_PER_YEAR = 100  # kg/人/年（保守估计日常补给）

print('【基本参数】')
print(f'  总运输量: {TOTAL_MASS/1e6:.0f} 百万吨 (100 Mt)')
print(f'  电梯年运力: {ELEVATOR_ANNUAL/1e3:.0f} 千吨/年 (537 kt/yr)')
print(f'  火箭年运力(最大): {ROCKET_ANNUAL_MAX/1e6:.1f} 百万吨/年 (7.5 Mt/yr)')
print()

# ============ 建设期时间分析 ============
print('=' * 70)
print('【建设期时间分析】 - 运输1亿吨所需时间')
print('=' * 70)
print()

time_pure_elevator = TOTAL_MASS / ELEVATOR_ANNUAL
time_pure_rocket = TOTAL_MASS / ROCKET_ANNUAL_MAX

print(f'  纯电梯: {time_pure_elevator:.1f} 年  ⚠️ 远超2050截止日期!')
print(f'  纯火箭: {time_pure_rocket:.1f} 年')
print()

# 不同配比的时间
print('不同配比下的建设时间 (假设并行运输):')
print()
print(f'{"电梯%":<10} {"火箭%":<10} {"建设时间(yr)":<15} {"是否可行(<24yr)":<15}')
print('-' * 55)

for elev_frac in [1.0, 0.9, 0.75, 0.5, 0.25, 0.1, 0.0]:
    rocket_frac = 1 - elev_frac
    
    # 并行运输时，时间取决于较慢的那个
    elev_mass = TOTAL_MASS * elev_frac
    rocket_mass = TOTAL_MASS * rocket_frac
    
    elev_time = elev_mass / ELEVATOR_ANNUAL if elev_frac > 0 else 0
    rocket_time = rocket_mass / ROCKET_ANNUAL_MAX if rocket_frac > 0 else 0
    
    # 并行时取最大值
    total_time = max(elev_time, rocket_time)
    
    feasible = "✓ 可行" if total_time <= 24 else "✗ 超时"
    
    print(f'{elev_frac*100:>6.0f}%    {rocket_frac*100:>6.0f}%    {total_time:>10.1f}       {feasible}')

print()

# ============ 关键约束分析 ============
print('=' * 70)
print('【核心矛盾】')
print('=' * 70)
print()
print('电梯年运力 537 kt vs 目标 100,000 kt = 仅能完成 0.54%/年')
print('要在24年内完成，电梯最多承担: 24 × 537 = 12,888 kt = 12.9%')
print()
print('>>> 结论: 即使电梯满负荷运行24年，也只能运输总量的 12.9%')
print('>>> 剩余 87.1% 必须由火箭承担，否则无法按时完成')
print()

# ============ Q2方案的合理性 ============
print('=' * 70)
print('【Q2方案的合理性重新审视】')
print('=' * 70)
print()

# Q2方案: 电梯跑满，剩余用火箭
elev_max_in_24yr = ELEVATOR_ANNUAL * 24  # 电梯24年最大运量
elev_fraction_max = elev_max_in_24yr / TOTAL_MASS

print(f'电梯24年最大运量: {elev_max_in_24yr/1e6:.1f} Mt ({elev_fraction_max*100:.1f}%)')
print(f'火箭必须承担: {(TOTAL_MASS - elev_max_in_24yr)/1e6:.1f} Mt ({(1-elev_fraction_max)*100:.1f}%)')
print()
print('这就是Q2方案的由来:')
print(f'  电梯 ~{elev_fraction_max*100:.0f}% + 火箭 ~{(1-elev_fraction_max)*100:.0f}%')
print('  不是"选择"，而是"约束"!')
print()

# ============ 运营期分析 ============
print('=' * 70)
print('【运营期运输需求 vs 电梯能力】')
print('=' * 70)
print()

# 从Q3的分析来看
print('运营期年度补给需求估算:')
print()

# 水资源需求 (from Q3)
water_daily_per_person = 75  # L/day (中等场景)
recycling_rate = 0.90
water_import_annual = POPULATION * water_daily_per_person * (1-recycling_rate) * 365 / 1000  # 吨

# 食物/物资
food_per_person_annual = 0.3  # 吨/人/年 (假设大部分自给)
other_supplies = 0.2  # 吨/人/年

total_annual_supply = water_import_annual + POPULATION * (food_per_person_annual + other_supplies)

print(f'  水资源(90%回收): {water_import_annual:,.0f} 吨/年')
print(f'  食物/物资: {POPULATION * (food_per_person_annual + other_supplies):,.0f} 吨/年')
print(f'  总计: {total_annual_supply:,.0f} 吨/年')
print()
print(f'电梯年运力: {ELEVATOR_ANNUAL:,.0f} 吨/年')
print(f'运力利用率: {total_annual_supply/ELEVATOR_ANNUAL*100:.1f}%')
print()

if total_annual_supply < ELEVATOR_ANNUAL:
    print('✓ 运营期电梯运力充足!')
    print(f'  剩余运力: {(ELEVATOR_ANNUAL - total_annual_supply):,.0f} 吨/年')
    print('  可用于: 设备更新、扩建物资、人员往返等')
else:
    print('✗ 运营期电梯运力不足!')
    print(f'  缺口: {(total_annual_supply - ELEVATOR_ANNUAL):,.0f} 吨/年')

print()
print('=' * 70)
print('【最终结论】')
print('=' * 70)
print()
print('1. 建设期: 电梯运力不足是硬约束，必须用火箭')
print('   - 电梯24年最多运12.9%，火箭必须运87.1%')
print('   - Q2的混合方案不是"选择"，是"必须"')
print()
print('2. 运营期: 电梯运力充足')
print('   - 年补给需求 ~32万吨 << 电梯运力 53.7万吨')
print('   - 运营期可以100%使用电梯，实现零碳运营')
print()
print('3. 环境评估应分阶段:')
print('   - 建设期(2026-2050): 高碳排放(不可避免)')
print('   - 运营期(2050+): 零碳/低碳(可持续)')
print()
print('>>> 关键洞察: "Carbon Debt"是一次性的建设成本')
print('    长期看，电梯的优势会在运营期体现')
