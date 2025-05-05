import pymsis
import numpy as np
import datetime
import os
os.environ["MSIS21_PATH"] = r"C:\Users\jaisa\...\site-packages\pymsis\data\msis21.parm"

#-----------------------------------------------------------------------
# Atmospheric Density Calculation using NRLMSISE-00 Model
# https://en.wikipedia.org/wiki/NRLMSISE-00
# https://www.nrl.navy.mil/our-research/nrlmsise-00

# Python module documentation:
# https://swxtrec.github.io/pymsis/index.html

# Needs to have the msis21.parm file renamed and copied to the 
# following filepath: 'C:\Users\jaisa\AppData\Local\packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311'
def get_density_at_altitude(alt_km, lat=0, lon=0, f107=150, f107a=150, ap=4, time=None):
    """
    Returns atmospheric density at a given altitude using the NRLMSISE-00 model.
    
    Parameters:
        alt_km (float): Altitude in kilometers (e.g., 500)
        lat (float): Latitude in degrees (default 0)
        lon (float): Longitude in degrees (default 0)
        f107 (float): Daily F10.7 solar flux (default 150)
        f107a (float): 81-day average F10.7 (default 150)
        ap (float): Ap geomagnetic index (default 4)
        time (datetime): datetime object (default is now)
    
    Returns:
        float: Atmospheric density in kg/m³
    """
    if time is None:
        time = [datetime.datetime.utcnow()]
    else:
        time = [time]

    alt = np.array([alt_km])
    lat = np.array([lat])
    lon = np.array([lon])
    ap_array = np.full((len(time), 7), ap)

    result = pymsis.calculate(time, alt, lat, lon, f107, f107a, ap_array)

    print("Result shape:", result.shape)
    print("Result contents:", result)
    print(pymsis.species_names)
    # Return density at first time, location, altitude
    return result  # kg/m³

# Testing
density = get_density_at_altitude(500)  # 500 km
print(f"Atmospheric density at 500 km: {density} kg/m³")
