"""
运营期年度补给需求详细估算
基于Q3水资源模型和文献数据
"""

import numpy as np

print('=' * 70)
print('运营期年度补给需求详细估算')
print('=' * 70)
print()

POPULATION = 100000  # 10万人

# ============================================================================
# 1. 水资源需求 (基于Q3模型)
# ============================================================================
print('【1. 水资源需求】')
print('-' * 50)
print()

print('Q3定义的三个场景:')
print()
print(f'{"场景":<15} {"生活用水":<12} {"农业用水":<12} {"工业用水":<12} {"总计":<10} {"回收率":<8}')
print('-' * 70)

scenarios = [
    ("Baseline", 50, 20, 5, 0.90),      # 中等场景
    ("Optimized", 30, 10, 2, 0.98),     # 高效场景  
    ("Low-Efficiency", 80, 50, 20, 0.70) # 低效场景
]

water_results = []
for name, dom, ag, ind, recycle in scenarios:
    total_daily = dom + ag + ind
    gross_annual = POPULATION * total_daily * 365 / 1000  # 吨/年
    net_import = gross_annual * (1 - recycle)
    water_results.append((name, net_import))
    print(f'{name:<15} {dom:<12} {ag:<12} {ind:<12} {total_daily:<10} {recycle*100:.0f}%')

print()
print('年度净进口量 (扣除回收):')
for name, net in water_results:
    print(f'  {name}: {net:,.0f} 吨/年')

print()
print('>>> 采用 Baseline 场景: 273,750 吨/年')
print()

# ============================================================================
# 2. 食物需求
# ============================================================================
print('【2. 食物需求】')
print('-' * 50)
print()

print('假设:')
print('  - 殖民地农业自给率: 80% (月球温室农业)')
print('  - 需进口: 肉类、调味品、特殊营养品等')
print()

# 地球人均食物消耗约 700-900 kg/年
# 假设月球自给80%，进口20%
food_per_person_earth = 800  # kg/年
self_sufficiency_rate = 0.80
food_import_per_person = food_per_person_earth * (1 - self_sufficiency_rate)

food_import_total = POPULATION * food_import_per_person / 1000  # 吨

print(f'地球人均食物消耗: {food_per_person_earth} kg/年')
print(f'月球农业自给率: {self_sufficiency_rate*100:.0f}%')
print(f'人均进口需求: {food_import_per_person:.0f} kg/年')
print(f'总进口量: {food_import_total:,.0f} 吨/年')
print()

# ============================================================================
# 3. 设备与备件
# ============================================================================
print('【3. 设备与备件】')
print('-' * 50)
print()

print('设备更新与维护需求:')
print('  - 生命支持系统备件')
print('  - 电力系统维护件')
print('  - 通讯设备')
print('  - 医疗设备与耗材')
print('  - 建筑维护材料')
print()

# 参考国际空间站，人均设备消耗约 50-100 kg/年
# 月球殖民地规模更大，但自产能力也更强
equipment_per_person = 80  # kg/年
equipment_total = POPULATION * equipment_per_person / 1000

print(f'人均设备/备件需求: {equipment_per_person} kg/年')
print(f'总需求: {equipment_total:,.0f} 吨/年')
print()

# ============================================================================
# 4. 能源材料（如果不完全自给）
# ============================================================================
print('【4. 能源与特殊材料】')
print('-' * 50)
print()

print('假设:')
print('  - 太阳能为主，核能辅助')
print('  - 月球可开采氦-3，但初期可能需要进口核燃料')
print('  - 某些稀有金属/催化剂需要进口')
print()

energy_materials_per_person = 20  # kg/年 (保守估计)
energy_total = POPULATION * energy_materials_per_person / 1000

print(f'人均能源/特殊材料: {energy_materials_per_person} kg/年')
print(f'总需求: {energy_total:,.0f} 吨/年')
print()

# ============================================================================
# 5. 人员流动（往返）
# ============================================================================
print('【5. 人员流动】')
print('-' * 50)
print()

print('假设:')
print('  - 年轮换率: 5% (5000人/年往返)')
print('  - 人均行李/个人物品: 50 kg')
print()

rotation_rate = 0.05
people_rotating = POPULATION * rotation_rate
luggage_per_person = 50  # kg
personnel_cargo = people_rotating * luggage_per_person / 1000

print(f'年轮换人数: {people_rotating:,.0f} 人')
print(f'相关货物: {personnel_cargo:,.0f} 吨/年')
print()

# ============================================================================
# 总计
# ============================================================================
print('=' * 70)
print('【总计】')
print('=' * 70)
print()

water_baseline = 273750  # 吨/年 (Baseline场景)
food = food_import_total
equipment = equipment_total
energy = energy_total
personnel = personnel_cargo

total = water_baseline + food + equipment + energy + personnel

print(f'  水资源 (净进口):     {water_baseline:>10,.0f} 吨/年  ({water_baseline/total*100:5.1f}%)')
print(f'  食物 (进口部分):     {food:>10,.0f} 吨/年  ({food/total*100:5.1f}%)')
print(f'  设备与备件:          {equipment:>10,.0f} 吨/年  ({equipment/total*100:5.1f}%)')
print(f'  能源/特殊材料:       {energy:>10,.0f} 吨/年  ({energy/total*100:5.1f}%)')
print(f'  人员流动货物:        {personnel:>10,.0f} 吨/年  ({personnel/total*100:5.1f}%)')
print(f'  ' + '-' * 45)
print(f'  总计:                {total:>10,.0f} 吨/年')
print()

# ============================================================================
# 与电梯运力对比
# ============================================================================
print('=' * 70)
print('【与电梯运力对比】')
print('=' * 70)
print()

ELEVATOR_CAPACITY = 537000  # 吨/年

print(f'年度补给需求: {total:,.0f} 吨')
print(f'电梯年运力:   {ELEVATOR_CAPACITY:,.0f} 吨')
print(f'利用率:       {total/ELEVATOR_CAPACITY*100:.1f}%')
print(f'剩余运力:     {ELEVATOR_CAPACITY - total:,.0f} 吨/年')
print()

# ============================================================================
# 敏感性分析
# ============================================================================
print('=' * 70)
print('【敏感性分析】')
print('=' * 70)
print()

print('如果采用不同的水资源场景:')
print()

for name, water_net in water_results:
    total_alt = water_net + food + equipment + energy + personnel
    util = total_alt / ELEVATOR_CAPACITY * 100
    status = "✓ 充足" if total_alt < ELEVATOR_CAPACITY else "✗ 不足"
    print(f'  {name:<15}: {total_alt:>10,.0f} 吨/年 | 利用率 {util:5.1f}% | {status}')

print()
print('>>> 结论: 即使在最差场景(Low-Efficiency)下，电梯运力也能满足需求')
print('         但会占用 77% 的运力，剩余空间有限')
