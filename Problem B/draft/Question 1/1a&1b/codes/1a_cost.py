G = 6.67430*10**-11  # Gravitational constant in m^3 kg^-1 s^-2
M = 5.972*10**24    # Mass of the Earth in kg
m = 1.79*10**8  # Mass of the materials in kg
R = 6.371*10**6  # Radius of the Earth in m
r = 1*10**5 # Orbit height of Apex Anchors above the Earth's surface in m

p = 0.004  # Electric Cost per Joule in USD
alpha = 1  # Efficiency of electric transmission

#Total cost of electricity of the elavator
def total_cost_electricity():
    U = G * M * m * (1/R - 1/(R + r))  # Gravitational potential energy in Joules
    E = U * alpha  # Total electrical energy required in Joules
    C = p * E  # Total cost in USD
    return C
print(total_cost_electricity())
print(G*M*m) 
