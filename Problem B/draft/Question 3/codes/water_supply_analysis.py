import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Set style for plots
plt.style.use('ggplot')

# ==========================================
# 1. Model Parameters & Constants
# ==========================================

POPULATION = 100000  # 100,000 people

# Transport Costs ($/kg)
COST_ELEVATOR = 200      # Q1/Q2 estimate: $200/kg (High estimate for conservative analysis)
COST_ROCKET = 5000       # Q1/Q2 estimate: $5,000/kg (Optimistic future rocket cost)
                         # Current cost is ~$20,000/kg, Starship might lower it given fuel, but to Moon is higher.

# System Capacity (Tons/Year)
# Problem states "the Galactic Harbor will provide... 179,000 metric tons every year".
# Since there are 3 Galactic Harbours, total system capacity is 3x.
CAPACITY_ELEVATOR_SINGLE = 179000 
NO_OF_HARBOURS = 3
CAPACITY_ELEVATOR_TOTAL = CAPACITY_ELEVATOR_SINGLE * NO_OF_HARBOURS  # 537,000 tons/year

# Reliability Parameters (From Q2)
DISRUPTION_DAYS_ELEVATOR = 30  # Max repair time for elevator cable issue
DISRUPTION_DAYS_ROCKET = 14    # Max stand-down time for rocket failure
SAFETY_FACTOR_Z = 3.0          # 99.87% confidence

# ==========================================
# 2. Water Metabolism Model Class
# ==========================================

class WaterModel:
    def __init__(self, scenario_name, w_dom, w_ag, w_ind, recycling_rate):
        self.name = scenario_name
        self.w_dom = w_dom  # Liters/person/day
        self.w_ag = w_ag
        self.w_ind = w_ind
        self.eta = recycling_rate
        
    def get_gross_daily_demand(self):
        # returns tons/day (1 Liter water approx 1 kg)
        return POPULATION * (self.w_dom + self.w_ag + self.w_ind) / 1000.0

    def get_net_annual_import(self):
        # Net import needed after recycling
        # Leakage/Loss = (1 - eta)
        gross_daily = self.get_gross_daily_demand()
        net_daily = gross_daily * (1 - self.eta)
        return net_daily * 365

    def get_safety_stock(self, disruption_days):
        # Stock needed to survive max disruption
        net_daily = self.get_gross_daily_demand() * (1 - self.eta) 
        # Note: In emergency, we consume reserves. 
        # Reserve = Net Daily Need * Days + Safety Margin
        # Actually, if supply cuts, we lose the 'Import' part. 
        # The internal recycling continues working. 
        # So we only need to stockpile the replacement water that WOULD have arrived.
        return net_daily * (disruption_days * 1.5) # 1.5 as safety margin buffer

# ==========================================
# 3. Scenario Definitions
# ==========================================

# Scenario A: Baseline (Standard ISS-like recycling, moderate farming)
# Needs: 50L (Dom) + 20L (Ag) + 5L (Ind) = 75L/day. Recycling 90%.
sc_baseline = WaterModel("Baseline", 50, 20, 5, 0.90)

# Scenario B: High-Tech / Optimized (Dune-style water discipline)
# Needs: 30L (Dom) + 10L (Ag) + 2L (Ind) = 42L/day. Recycling 98%.
sc_optimized = WaterModel("Optimized", 30, 10, 2, 0.98)

# Scenario C: Indulgent / Low Tech (Early colony style)
# Needs: 80L (Dom) + 50L (Ag) + 20L (Ind) = 150L/day. Recycling 70%.
sc_pessimistic = WaterModel("Low-Efficiency", 80, 50, 20, 0.70)

scenarios = [sc_baseline, sc_optimized, sc_pessimistic]

# ==========================================
# 4. Simulation & Analysis
# ==========================================

if __name__ == "__main__":
    results = []

    print(f"{'Scenario':<15} | {'Recycle':<8} | {'Net Import (t/yr)':<18} | {'Cost (Elevator) $B':<20} | {'Cost (Rocket) $B':<18} | {'Elevator Cap %':<15}")
    print("-" * 110)

    for sc in scenarios:
        net_import_tons = sc.get_net_annual_import()
        
        # Cost calculations ($ Billion)
        cost_el_B = (net_import_tons * 1000 * COST_ELEVATOR) / 1e9
        cost_rk_B = (net_import_tons * 1000 * COST_ROCKET) / 1e9
        
        # Capacity Occupation
        cap_occ_pct = (net_import_tons / CAPACITY_ELEVATOR_TOTAL) * 100
        
        # Safety Stock (using Elevator disruption as worst case for timeline)
        stock_tons = sc.get_safety_stock(DISRUPTION_DAYS_ELEVATOR)
        
        results.append({
            "Scenario": sc.name,
            "Recycling": sc.eta,
            "Import_Tons": net_import_tons,
            "Cost_Elevator_B": cost_el_B,
            "Cost_Rocket_B": cost_rk_B,
            "Capacity_Occ": cap_occ_pct,
            "Safety_Stock": stock_tons
        })
        
        print(f"{sc.name:<15} | {sc.eta*100:5.0f}%   | {net_import_tons:15,.0f}    | ${cost_el_B:18,.2f} | ${cost_rk_B:16,.2f} | {cap_occ_pct:13.2f}%")

    # ==========================================
    # 5. Sensitivity Analysis Plotting (See q3_visualization.py)
    # ==========================================

    # ==========================================
    # 6. Report Generation
    # ==========================================

    print("\n=== ANALYSIS CONCLUSIONS ===")
    baseline = results[0]
    print(f"1. FEASIBILITY: Even with {baseline['Recycling']*100}% recycling (Baseline), importing water via ROCKETS costs ${baseline['Cost_Rocket_B']:.2f} Billion/year.")
    print(f"   This confirms that Rocket-only logistics are economically unsustainable for maintaining a large colony.")
    print(f"2. ELEVATOR ADVANTAGE: The Space Elevator reduces this to ${baseline['Cost_Elevator_B']:.2f} Billion/year.")
    print(f"3. CAPACITY TAX: Baseline water import consumes {baseline['Capacity_Occ']:.1f}% of the Elevator's total annual capacity.")
    print(f"   If recycling drops to 70%, it would consume {results[2]['Capacity_Occ']:.1f}% of capacity, effectively choking the colony's logistics.")
    print(f"4. SAFETY TIMELINE: Before 100k people arrive, a safety stock of {baseline['Safety_Stock']:.0f} tons must be pre-deployed.")
