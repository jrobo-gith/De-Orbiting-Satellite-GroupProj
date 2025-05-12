import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise
# import numpy as np
import math
from numpy.random import randn
from scipy.integrate import solve_ivp
from math import atan2
from tqdm import tqdm
from visualiser.V3.debug import debug_print
from visualiser.V3.simulator.simulator import lat_long_height, atmospheric_density


# Import constants
from visualiser.V3.partials.constants import *

    
""" Landing stopping condition"""
def stop_condition(t, y):
    # Stop when altitude <= 0
    r = np.linalg.norm(y[:3])
    _, _, altitude = lat_long_height(y[0], y[1], y[2])
    return altitude

""" Define the ODE of the orbit dynamics ================================="""
def ode(t, state_x):
        x, y, z, vx, vy, vz= state_x        
        
        r = np.linalg.norm(state_x[:3])

        altitude = lat_long_height(x, y, z)[2]

        # Gravity
        F_gravity = -G * M_EARTH / r**2

        # Drag
        rho = atmospheric_density(altitude)
        
        v = np.linalg.norm([vx,vy,vz])

        F_drag_x = -0.5 * rho * CD * A * v * vx / M_SAT
        F_drag_y = -0.5 * rho * CD * A * v * vy / M_SAT
        F_drag_z = -0.5 * rho * CD * A * v * vz / M_SAT
        # debug_print("predictor", f"drag: [F_drag_x, F_drag_y, F_drag_z]")
        
        # Acceleration
        ax = F_gravity * (x / r) + F_drag_x
        ay = F_gravity * (y / r) + F_drag_y
        az = F_gravity * (z / r) + F_drag_z

        return np.array([vx, vy, vz, ax, ay, az])


def ode_with_Cd(t, state_x):
        x, y, z, vx, vy, vz, Cd = state_x        
        
        r = np.linalg.norm(state_x[:3])
        altitude = lat_long_height(x, y, z)[2]

        # Gravity
        F_gravity = -G * M_EARTH / r**2

        # Drag
        rho = atmospheric_density(altitude)
        
        v = np.linalg.norm([vx,vy,vz])

        F_drag_x = -0.5 * rho * Cd * A * v * vx / M_SAT
        F_drag_y = -0.5 * rho * Cd * A * v * vy / M_SAT
        F_drag_z = -0.5 * rho * Cd * A * v * vz / M_SAT
        
        # Acceleration
        ax = F_gravity * (x / r) + F_drag_x
        ay = F_gravity * (y / r) + F_drag_y
        az = F_gravity * (z / r) + F_drag_z

        dCd_dt = 0
        return np.array([vx, vy, vz, ax, ay, az, dCd_dt])

""" Define the process model f(x) ======================================="""
def f(state_x, dt):
    """state vector = state_x = [x,y,z,vx, vy, vz]"""
    solution = solve_ivp(ode, t_span=[0, dt], y0=state_x, method='RK45', t_eval=[dt], max_step=dt)
    return solution.y.flatten()

def f_with_Cd(state_x, dt):
    """state vector = state_x = [x,y,z,vx, vy, vz, Cd]"""
    Cd = state_x[-1]
    solution = solve_ivp(ode_with_Cd, t_span=[0, dt], y0=state_x, method='RK45', t_eval=[dt], max_step=dt)
    return solution.y.flatten()

""" Define measurement function h(x) ================================================="""
def h_radar(x):
    """x is the state vector,
    this H function assumes ra`rdar sends positions in global coordinate system"""
    return x[:3] # return x,y,z position if state order is (x,y,z,vx,vy,vz)

def ukf_Q(dim, dt, var_):
    Q = np.zeros((dim, dim))
    Q[np.ix_([0, 3], [0, 3])] = Q_discrete_white_noise(dim=2, dt=dt,var=var_)  # Q matrix for how other noise affect x and vx
    Q[np.ix_([1, 4], [1, 4])] = Q_discrete_white_noise(dim=2, dt=dt,var=var_)  # Q matrix for how other noise affect y and vy
    Q[np.ix_([2, 5], [2, 5])] = Q_discrete_white_noise(dim=2, dt=dt,var=var_)  # Q matrix for how other noise affect z and vz
    return Q

def ukf_Q_7dim(dim, dt, var_, Cd_var):
    Q = np.zeros((dim, dim))
    uncertainty = Q_discrete_white_noise(dim=2, dt=dt,var=var_)
    Q[np.ix_([0, 3], [0, 3])] = uncertainty  # Q matrix for how other noise affect x and vx
    Q[np.ix_([1, 4], [1, 4])] = uncertainty  # Q matrix for how other noise affect y and vy
    Q[np.ix_([2, 5], [2, 5])] = uncertainty  # Q matrix for how other noise affect z and vz
    Q[dim-1, dim-1] = Cd_var * dt
    return Q
