import numpy as np

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
H_SCALE = 8500
R_EARTH_EQUATOR = 6378e3  # Radius of Earth near equator (m)
R_EARTH_POLES = 6357e3 # Radius of Earth near poles (m)
EARTH_ROTATION_ANGLE = ((2*np.pi)/(23*3600 + 56*60 + 4)) # Accounting for earth's rotation without time t
MU_EARTH = G * M_EARTH