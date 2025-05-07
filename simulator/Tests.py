import matplotlib.pyplot as plt
import numpy as np

import pymsis

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

# for variable in pymsis.Variable:
#     print(variable.name)

# Gives the total mass density as the [0,0]th index of the outputs
print(output_midnight[0, 0])# [:, :, :, :, 0])
