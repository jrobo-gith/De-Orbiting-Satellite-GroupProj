import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pymsis
import datetime

np.random.seed(0)

G = 6.67430e-11                             # Gravitational constant (m^3 kg^-1 s^-2)
M_EARTH = 5.972e24                          # Mass of Earth (kg)
EARTH_SEMIMAJOR = 6378137.0                 # Radius of Earth (m)
EARTH_SEMIMINOR = 6356752.314245            # Semi-minor axis of Earth (m)
FLATTENING = 1/298.257223563                # Flattening of Earth (dimensionless)
E_SQUARED = FLATTENING * (2 - FLATTENING)   # Eccentricity squared (dimensionless)
ATMOSPHERE_HEIGHT = 120e3                   # Approximate height of the atmosphere (m)
CD = 2.2                                    # Drag coefficient (dimensionless)
A = 1.0                                     # Cross-sectional area of the satellite (m^2)
M_SAT = 500                                 # Mass of the satellite (kg)
RHO_0 = 1.225                               # Air density at sea level (kg/m^3)
H_SCALE = 8500                              # Scale height of the atmosphere (m)

# pymsis.calculate() returns the total mass density as the first of all 11 output variables.

lon = 0
lat = 70
alts = 5 #np.linspace(0, 1000, 1000)
f107 = 150
f107a = 150
ap = 7
aps = [[ap] * 7]

date = np.datetime64("2003-01-01T00:00")
output_midnight = pymsis.calculate(date, lon, lat, alts, f107, f107a, aps)

def atmospheric_density(altitude):
    # White noise as random perturbations of magnitude 0.1*RHO_0
    density = RHO_0 * np.exp(-altitude / H_SCALE)
    white_noise = np.random.normal(0, 0.05* density)
    return density + white_noise
    
def atmospheric_density_true(lat, long, altitude, time = None):
    if time is None:
        time = datetime.datetime.now()
    
    true_density = pymsis.calculate(time, long, lat, altitude)
    
    return true_density[0, 0]

altitude = np.linspace(1, 120e3, 1000)

plt.figure(figsize=(10, 6))
plt.plot(atmospheric_density(altitude), altitude, label='Simulated Density', color='blue')
plt.plot(atmospheric_density_true(0, 0, altitude/1000)[0, :, 0], altitude, label='True Density', color='red')

plt.title('Atmospheric Density Comparison')
plt.ylabel('Altitude (m)')
plt.xlabel('Density (kg/m^3)')
plt.legend()
plt.show()