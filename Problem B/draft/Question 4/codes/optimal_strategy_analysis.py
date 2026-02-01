"""
综合最优方案分析：分阶段运输策略
"""

import numpy as np

print('=' * 70)
print('综合最优方案：分阶段运输策略')
print('=' * 70)
print()

# ============================================================================
# 参数定义
# ============================================================================

# 建设期参数
TOTAL_MASS = 1e8           # 1亿吨
CONSTRUCTION_YEARS = 24    # 2026-2050
ELEVATOR_ANNUAL = 5.37e5   # 电梯年运力

# 环境参数
CO2_PER_TON_ELEVATOR = 0.1     # 吨CO2/吨货物
CO2_PER_TON_ROCKET = 16.67     # 吨CO2/吨货物

# 运营期参数
OPERATION_SUPPLY_ANNUAL = 3e5  # 30万吨/年 (Baseline场景)
POPULATION = 100000
PER_CAPITA_EARTH_EMISSION = 15  # 吨CO2/人/年

# ============================================================================
# 方案定义
# ============================================================================

print('【方案对比】')
print()
print('方案A: 全程纯火箭 (不现实的对照组)')
print('方案B: 全程纯电梯 (时间不允许)')  
print('方案C: 分阶段策略 - 建设期混合 + 运营期纯电梯 ★推荐')
print()

# ============================================================================
# 方案C详细分析：分阶段策略
# ============================================================================

print('=' * 70)
print('【方案C详细分析：分阶段策略】')
print('=' * 70)
print()

# 建设期 (2026-2050)
elev_mass_construction = ELEVATOR_ANNUAL * CONSTRUCTION_YEARS
rocket_mass_construction = TOTAL_MASS - elev_mass_construction
elev_fraction = elev_mass_construction / TOTAL_MASS

print('■ 建设期 (2026-2050): 混合运输')
print('-' * 50)
print(f'  电梯运输量: {elev_mass_construction/1e6:.1f} Mt ({elev_fraction*100:.1f}%)')
print(f'  火箭运输量: {rocket_mass_construction/1e6:.1f} Mt ({(1-elev_fraction)*100:.1f}%)')
print()

# 建设期碳排放
co2_construction_elev = elev_mass_construction * CO2_PER_TON_ELEVATOR / 1e6
co2_construction_rocket = rocket_mass_construction * CO2_PER_TON_ROCKET / 1e6
co2_construction_total = co2_construction_elev + co2_construction_rocket

print('  碳排放:')
print(f'    电梯部分: {co2_construction_elev:.1f} Mt CO2')
print(f'    火箭部分: {co2_construction_rocket:.1f} Mt CO2')
print(f'    建设期总计: {co2_construction_total:.1f} Mt CO2')
print()

# 运营期 (2050+)
print('■ 运营期 (2050+): 纯电梯运输')
print('-' * 50)
print(f'  年度补给需求: {OPERATION_SUPPLY_ANNUAL/1e3:.0f} kt')
print(f'  电梯年运力:   {ELEVATOR_ANNUAL/1e3:.0f} kt')
print(f'  运力利用率:   {OPERATION_SUPPLY_ANNUAL/ELEVATOR_ANNUAL*100:.1f}%')
print()

co2_operation_annual = OPERATION_SUPPLY_ANNUAL * CO2_PER_TON_ELEVATOR / 1e6
print(f'  年度碳排放:   {co2_operation_annual:.4f} Mt CO2/年 (近乎零碳)')
print()

# 移民减排红利
annual_reduction = POPULATION * PER_CAPITA_EARTH_EMISSION / 1e6
print(f'  移民减排红利: {annual_reduction:.2f} Mt CO2/年 (10万人迁离地球)')
print()

# ============================================================================
# 环境回本分析
# ============================================================================

print('=' * 70)
print('【环境回本分析】')
print('=' * 70)
print()

# 年净减排 = 移民红利 - 运营排放
net_annual_benefit = annual_reduction - co2_operation_annual
break_even_years = co2_construction_total / net_annual_benefit

print(f'建设期碳债务:     {co2_construction_total:.1f} Mt CO2')
print(f'运营期年净减排:   {net_annual_benefit:.2f} Mt CO2/年')
print(f'环境回本时间:     {break_even_years:.1f} 年')
print()

# 长期累积效益
years_to_analyze = [50, 100, 200, 500]
print('长期累积环境效益:')
print(f'{"年份":<10} {"累积净减排(Mt)":<20} {"状态":<15}')
print('-' * 45)

for years in years_to_analyze:
    cumulative = net_annual_benefit * years - co2_construction_total
    status = "偿还碳债" if cumulative > 0 else "仍在偿债"
    print(f'{years:<10} {cumulative:>15.1f}      {status}')

print()

# ============================================================================
# 与其他方案对比
# ============================================================================

print('=' * 70)
print('【方案对比总结】')
print('=' * 70)
print()

print(f'{"方案":<25} {"建设期碳排":<15} {"运营期碳排":<15} {"回本时间":<12} {"可行性":<10}')
print('-' * 77)

# 方案A: 纯火箭
co2_a_construction = TOTAL_MASS * CO2_PER_TON_ROCKET / 1e6
co2_a_operation = OPERATION_SUPPLY_ANNUAL * CO2_PER_TON_ROCKET / 1e6
# 运营期排放 > 移民红利，永不回本
print(f'{"A: 全程纯火箭":<25} {co2_a_construction:<15.1f} {co2_a_operation:<15.2f} {"永不回本":<12} {"✓ 可行":<10}')

# 方案B: 纯电梯
co2_b_construction = TOTAL_MASS * CO2_PER_TON_ELEVATOR / 1e6
co2_b_operation = co2_operation_annual
be_b = co2_b_construction / (annual_reduction - co2_b_operation)
print(f'{"B: 全程纯电梯":<25} {co2_b_construction:<15.1f} {co2_b_operation:<15.4f} {be_b:<12.1f} {"✗ 超时":<10}')

# 方案C: 分阶段
print(f'{"C: 分阶段策略 ★":<25} {co2_construction_total:<15.1f} {co2_operation_annual:<15.4f} {break_even_years:<12.1f} {"✓ 最优":<10}')

print()

# ============================================================================
# 最终结论
# ============================================================================

print('=' * 70)
print('【最终结论】')
print('=' * 70)
print()
print('★ 分阶段策略是唯一兼顾「工程可行性」和「环境可持续性」的方案')
print()
print('┌─────────────────────────────────────────────────────────────────┐')
print('│  阶段        │  时间       │  运输方式           │  碳排放      │')
print('├─────────────────────────────────────────────────────────────────┤')
print(f'│  建设期      │  2026-2050  │  13%电梯 + 87%火箭  │  {co2_construction_total:.0f} Mt    │')
print(f'│  运营期      │  2050+      │  100%电梯           │  ≈0 Mt/年   │')
print('└─────────────────────────────────────────────────────────────────┘')
print()
print('关键数据:')
print(f'  • 建设期碳债务: {co2_construction_total:.0f} Mt (不可避免)')
print(f'  • 环境回本时间: {break_even_years:.0f} 年')
print(f'  • 100年后累积减排: {net_annual_benefit * 100 - co2_construction_total:.0f} Mt')
print(f'  • 500年后累积减排: {net_annual_benefit * 500 - co2_construction_total:.0f} Mt')
print()
print('>>> 这是在物理约束下的最优解，平衡了时间、成本和环境三重目标')
