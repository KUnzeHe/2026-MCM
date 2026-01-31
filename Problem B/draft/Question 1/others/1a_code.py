G = 6.67430*10**-11  # Gravitational constant in m^3 kg^-1 s^-2
M = 5.972*10**24    # Mass of the Earth in kg
m = (1.79*10**8)*3  # Mass of the materials in kg per year
m_total = 10**11 #total mass in kg
R = 6.371*10**6  # Radius of the Earth in m
r = 1*10**5 # Orbit height of Apex Anchors above the Earth's surface in m


mission_time = m_total/m

C_emit = 1.2*10**3 #per rocket/tone
life = 2400 #per day
beta = 1 #reliability of rocket(probability for repairing rocket)


p = 0.004  # Electric Cost per Joule in USD
alpha = 1  # Efficiency of electric transmission

#Total cost of electricity of the elavator
def total_cost_electricity():
    U = G * M * m_total * (1/R - 1/(R + r))  # Gravitational potential energy in Joules
    E = U / alpha  # Total electrical energy required in Joules
    C = p * E  # Total cost in USD
    print("total cost electricity:(usd)")
    return C
def total_carbon_emission():
    carbon_e = C_emit*(mission_time//(life*beta)+1)
    print("total carbon emission:(tones)")
    return carbon_e

print("mission time:", mission_time)
print(total_cost_electricity())
print(total_carbon_emission())
print(G*M*m) 
