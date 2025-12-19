import math

# Inline copy of estimate_power_watts to validate calculation

def estimate_power_watts(speed_kmph, grade, mass_kg, CdA=0.5, Cr=0.004, rho=1.226):
    g = 9.80665
    v_ms = speed_kmph / 3.6
    F_roll = Cr * mass_kg * g * math.cos(math.atan(grade))
    F_climb = mass_kg * g * grade
    F_aero = 0.5 * rho * CdA * v_ms * v_ms
    F_total = F_roll + F_climb + F_aero
    P = F_total * v_ms
    return max(P, 0.0)

if __name__ == '__main__':
    speed_kmph = 24.1035
    grade = 200.0 / (30.0 * 1000.0)
    mass_kg = 78.0
    CdA = 0.5
    Cr = 0.004
    power_w = estimate_power_watts(speed_kmph, grade, mass_kg, CdA=CdA, Cr=Cr)
    hours = 90.0 / 60.0
    kcal_mech = power_w * hours * 0.860421
    kcal_metabolic = kcal_mech / 0.24
    print('power_w:', power_w)
    print('kcal_mech:', kcal_mech)
    print('kcal_metabolic:', kcal_metabolic)
