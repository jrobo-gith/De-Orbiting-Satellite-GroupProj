from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pymsis
import datetime
import os

from partials.constants import *

# IF TESTING
np.random.seed(0)

# ------------------------------------------------------------------------------------------------------------------
# ----------------------- FUNCTIONS FOR CALCULATING POSITION IN LATITUDE AND LONGITUDE -----------------------------
# Functions with the suffix '_plot' use vectorized loops (for plotting)

# Calculates the Earth's "radius" at the satellite's position
# (Uses the WGS-84 ellipsoid model)
# https://en.wikipedia.org/wiki/World_Geodetic_System#WGS84
def earth_radius_WGS84(latitude):
    """
    Calculate the radius of the Earth at a given latitude using the WGS-84 ellipsoid model.
    
    Args:
        latitude (float or array-like): Latitude in radians.
    
    Returns:
        float or array-like: Radius of the Earth at the given latitude in meters.
    """
    numerator = (EARTH_SEMIMAJOR**2 * np.cos(latitude))**2 + (EARTH_SEMIMINOR**2 * np.sin(latitude))**2
    denominator = (EARTH_SEMIMAJOR * np.cos(latitude))**2 + (EARTH_SEMIMINOR * np.sin(latitude))**2
    return np.sqrt(numerator / denominator)

def curvature_in_prime_vertical(phi):
    """
    Calculate the curvature in the prime vertical at a given latitude.
    
    Args:
        phi: Latitude in radians. 
    
    Returns:
        float: Curvature in the prime vertical at the given latitude in meters.
    """
    return EARTH_SEMIMAJOR / np.sqrt(1 - E_SQUARED * np.sin(phi)**2)

# Returns latitude and height above the Earth
# Calculates latitude iteratively for the WGS-84 model using Bowring's method
# RESEARCH INTO FERRARI'S METHOD FOR BETTER ACCURACY
def latitude_iterator_and_height_plot(x, y, z):
    """
    Calculate latitude and height above the Earth using Bowring's method.
    
    Args:
        x (array-like): X coordinates in meters.
        y (array-like): Y coordinates in meters.
        z (array-like): Z coordinates in meters.
    
    Returns:
        tuple: Latitude in radians and height above the Earth in meters.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    r = np.sqrt(x**2 + y**2)
    phi = np.arctan(z / (r * (1 - E_SQUARED)))
    phi_new = phi + 100  # ensures entry into the loop

    converged = np.zeros_like(phi, dtype=bool)
    max_iter = 100
    iter_count = 0

    while not np.all(converged) and iter_count < max_iter:
        phi[~converged] = phi_new[~converged]
        N = curvature_in_prime_vertical(phi)
        phi_new[~converged] = np.arctan2(z[~converged] + (N[~converged] * E_SQUARED) * np.sin(phi[~converged]),
                                         r[~converged] - (N[~converged] * E_SQUARED) * np.cos(phi[~converged]))
        converged[~converged] = np.abs(phi_new[~converged] - phi[~converged]) <= 1e-9
        iter_count += 1

    phi_final = phi_new
    N_final = curvature_in_prime_vertical(phi_final)
    height = (r / np.cos(phi_final)) - N_final
    height = np.maximum(height, 0)  # ensure non-negative height

    return phi_final, height

def latitude_iterator_and_height(x, y, z):
    """
    Calculate latitude and height above the Earth using Bowring's method.
    
    Args:
        x (float): X coordinate in meters.
        y (float): Y coordinate in meters.
        z (float): Z coordinate in meters.
    
    Returns:
        tuple: Latitude in radians and height above the Earth in meters.
    """
    r = np.sqrt(x ** 2 + y ** 2)
    with np.errstate(invalid='ignore', divide='ignore'):
        phi = np.arctan(z / (r * (1.0 - E_SQUARED)))
    phi_new = phi + 100
    N = curvature_in_prime_vertical(phi)

    while np.abs(phi_new - phi) > 1e-9:
        phi = phi_new
        N = curvature_in_prime_vertical(phi)
        phi_new = np.arctan2(z + (N * E_SQUARED) * np.sin(phi), r - (N * E_SQUARED) * np.cos(phi))
    
    height = (r / np.cos(phi)) - N
    if height < 0:
        height = 0
    
    return phi_new, height

# Calculates the height of the satellite above an elliptical Earth
# Latitude is calculated as an approximation for a non-spherical Earth
def lat_long_height_plot(x, y, z):
    """
    Calculate latitude, longitude, and height above the Earth using Bowring's method.
    
    Args:
        x (array-like): X coordinates in meters.
        y (array-like): Y coordinates in meters.
        z (array-like): Z coordinates in meters.
    
    Returns:
        tuple: Latitude in radians, longitude in radians, and height above the Earth in meters.
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    longitude = np.arctan2(y, x)
    latitude, height = latitude_iterator_and_height_plot(x, y, z)
    R_earth = earth_radius_WGS84(latitude)
    height = r - R_earth
    return latitude, longitude, height

def lat_long_height(x, y, z):
    """
    Calculate latitude, longitude, and height above the Earth using Bowring's method.
    
    Args:
        x (float): X coordinate in meters.
        y (float): Y coordinate in meters.
        z (float): Z coordinate in meters.
    
    Returns:
        tuple: Latitude in radians, longitude in radians, and height above the Earth in meters.
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    longitude = np.arctan2(y, x)
    latitude, height = latitude_iterator_and_height(x, y, z)
    R_earth = earth_radius_WGS84(latitude)
    height = r - R_earth
    return latitude, longitude, height

# ------------------------------------------------------------------------------------------------------------------
# ------------------------- FUNCTIONS FOR CALCULATING ATMOSPHERIC DENSITY ------------------------------------------
# Calculates atmospheric density at a given altitude
def atmospheric_density(altitude):
    """
    Calculate atmospheric density at a given altitude using a simplified model.
    
    Args:
        altitude (float): Altitude in meters.
    
    Returns:
        float: Atmospheric density in kg/m^3.
    """
    # White noise as random perturbations of magnitude 0.1*RHO_0
    density = RHO_0 * np.exp(-altitude / H_SCALE)
    white_noise = np.random.normal(0, 0.05* density)
    if altitude > 0:
        return density + white_noise
    else:
        return 0

# Python module documentation:
# https://swxtrec.github.io/pymsis/index.html

# Needs to have the msis21.parm file renamed and copied to the 
# following filepath: 'C:\Users\jaisa\AppData\Local\packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311'
# and renamed to 'sitms21.parm' for pymsis to work correctly

# Atmospheric Density Calculation using NRLMSISE-00 Model
# https://en.wikipedia.org/wiki/NRLMSISE-00
# https://www.nrl.navy.mil/our-research/nrlmsise-00
# (Only accurate when using the WGS-84 model for the Earth)
def atmospheric_density_true(lat, long, altitude, time = None):
    """
    Calculate atmospheric density using the NRLMSISE-00 model.
    
    Args:
        lat (float): Latitude in radians.
        long (float): Longitude in radians.
        altitude (float): Altitude in meters.
        time (datetime): Time for the calculation. If None, uses the current time.
    
    Returns:
        float: Atmospheric density in kg/m^3.
    """
    if time is None:
        time = datetime.datetime.now()
    
    true_density = pymsis.calculate(time, long, lat, altitude)
    
    return true_density[0, 0]  # Total mass density is the [0,0]th index of the pymsis.calculate() outputs

# Diff. eqns for satellite motion
def satellite_dynamics(t, y):
    """
    Calculate the dynamics of a satellite in orbit.
    
    Args:
        t (float): Time.
        y (array-like): State vector [x, y, z, vx, vy, vz].
    
    Returns:
        array-like: Derivative of the state vector [vx, vy, vz, ax, ay, az].
    """
    x, y_pos, z, vx, vy, vz = y
    r = np.sqrt(x**2 + y_pos**2 + z**2)
    lat, long, altitude = lat_long_height(x, y_pos, z)

    # Gravity
    F_gravity = -G * M_EARTH / r**2

    # Drag
    # rho = atmospheric_density_true(lat, long, altitude)
    ## SIMPLER ATMOSPHERIC DENSITY FUNCTION:
    rho = atmospheric_density(altitude)
    v = np.sqrt(vx**2 + vy**2 + vz**2)
    F_drag_x = -0.5 * rho * CD * A * v * vx / M_SAT
    F_drag_y = -0.5 * rho * CD * A * v * vy / M_SAT
    F_drag_z = -0.5 * rho * CD * A * v * vz / M_SAT

    # Acceleration
    ax = F_gravity * (x / r) + F_drag_x
    ay = F_gravity * (y_pos / r) + F_drag_y
    az = F_gravity * (z / r) + F_drag_z

    return [vx, vy, vz, ax, ay, az]

# Stops the simulation when altitude <= 0
def stop_condition(t, y):
    """
    Event function to stop the simulation when altitude is less than or equal to zero.
    
    Args:
        t (float): Time.
        y (array-like): State vector [x, y, z, vx, vy, vz].
    
    Returns:
        float: Altitude above the Earth.
    """
    r = np.sqrt(y[0]**2 + y[1]**2 + y[2]**2)
    altitude = lat_long_height(y[0], y[1], y[2])[2]
    return altitude

# Solves diff. eqn system between measurements radar(n) and radar(n+1)
def system_solver(t_span_, initial_conditions):
    """
    Solve the system of equations for satellite dynamics.
    
    Args:
        t_span_ (float): Time span for the simulation.
        initial_conditions (list): Initial conditions [x0, y0, z0, vx0, vy0, vz0].
    
    Returns:
        solution: Solution object containing the results of the integration.
    """
    t_evals = int(np.rint(t_span_/50))
    t_eval = np.linspace(0, t_span_, t_evals)
    
    stop_condition.terminal = True
    stop_condition.direction = -1
    
    # Solve the system of equations using RK45
    solution = solve_ivp(satellite_dynamics, t_span=[0, t_span_], y0=initial_conditions, method='RK45', t_eval=t_eval, events=stop_condition, max_step=50)

    x_vals = solution.y[0]
    y_vals = solution.y[1]
    z_vals = solution.y[2]

    vx_vals = solution.y[3]
    vy_vals = solution.y[4]
    vz_vals = solution.y[5]

    t_vals = solution.t

    script_dir = os.path.dirname(os.path.abspath(__file__))
    sat_file = os.path.join(script_dir, "sat_traj.dat")
    # Write satellite data to a file
    if __name__ != '__main__':
        with open(sat_file, "w") as fp:
            for i in range(solution.y[0].shape[0]):
                fp.write(
                    f"{x_vals[i]:.6f}\t{y_vals[i]:.6f}\t{z_vals[i]:.6f}\t{vx_vals[i]:.6f}\t{vy_vals[i]:.6f}\t{vz_vals[i]:.6f}\t{t_vals[i]:.3f}\n")

    return solution

if __name__ == '__main__':
    # ----------------------------- TESTING -----------------------------
    # Testing the satellite dynamics to make sure it works correctly, just use the above functions to get radar measurements

    # Initial conditions
    altitude_initial = 150e3
    velocity_initial = 8100
    x0 = EARTH_SEMIMAJOR + altitude_initial
    y0 = 0
    z0 = 0
    vx0 = 0
    vy0 = velocity_initial/np.sqrt(2)
    vz0 = velocity_initial/np.sqrt(2)
    initial_conditions = [x0, y0, z0, vx0, vy0, vz0]

    # Time span
    t_span = 500_000_000
    t_eval = np.linspace(0, t_span, 1000)  # Points for evaluation

    # Solve differential equations using RK45
    solution = system_solver(t_span, initial_conditions)

    x_vals = solution.y[0]
    y_vals = solution.y[1]
    z_vals = solution.y[2]
    altitudes = lat_long_height_plot(x_vals, y_vals, z_vals)[2]

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(solution.t, altitudes / 1e3)  # Altitude vs. time
    plt.axhline(ATMOSPHERE_HEIGHT / 1e3, color='r', linestyle='--', label='Atmosphere Boundary')
    plt.title('Satellite Altitude During Re-entry (3D)')
    plt.xlabel('Time (s)')
    plt.ylabel('Altitude (km)')
    plt.legend()
    plt.grid()
    plt.show()

    # 3D plot
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_vals / 1e3, y_vals / 1e3, z_vals / 1e3)
    ax.set_title('Satellite Trajectory in 3D')
    ax.set_xlabel('X Position (km)')
    ax.set_ylabel('Y Position (km)')
    ax.set_zlabel('Z Position (km)')
    plt.show()