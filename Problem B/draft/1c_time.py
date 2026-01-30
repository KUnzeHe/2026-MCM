M_tot = 1*10**11 # Total mass of materials in kg
T_E = 537000*10**3 # Elevator transport limit

N_site = 10 #Number of launch sites
L_max = 700 #Maximum launches per year per site
p = 125*10**3 #transport limit per rocket in kg

T_R = N_site*L_max*p # Total annual transport capacity of rocket in kg

x = M_tot*T_E / (T_E + T_R) # Annual mass transported by elevator in kg

Y = x / T_E # Minimum years 

print(Y)