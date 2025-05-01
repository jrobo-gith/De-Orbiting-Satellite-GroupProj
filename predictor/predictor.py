import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import filterpy
import filterpy.kalman as kf
from filterpy.kalman import KalmanFilter
import filterpy.stats as fs
from filterpy.stats import gaussian, rand_student_t
from filterpy.kalman import predict
from filterpy.kalman import update


print('here')
# radar measurements of satellite position are in the form of (distance, elevation, azimuth).
# convert radar measurements to Cartesian coordinates
def spherical_to_cartesian(distance, elevation, azimuth, ref_position):
    """
    distance = straight-line distance between reference (radar) and object (satellite)
    ref_position = e.g. cartesian coordinate of the radar location
    """
    return (x,y,z)

##############################################################################
##############################################################################
"""============= Take first one (or 2? need to get the initial velocity) measurement as the starting state ============="""
##############################################################################
##############################################################################
# read Radar measurement file and take the first measurement
# radar_measurements = pd.read_csv('radar_measurements.csv')
radar1 = radar_reading()
radar2 = radar_reading()
init_measurement = radar_measurements.iloc[0]

##############################################################################
##############################################################################
"""======================== initialise prior belief on the initial state ======================================"""
##############################################################################
##############################################################################
# define a prior distribution using the first measurement
prior_mean = np.array([init_measurement['x'], init_measurement['y'], init_measurement['z'], 
                       init_measurement['vx'], init_measurement['vy'], init_measurement['vz']])



##############################################################################
##############################################################################
"""======================== Design P (prior covariance matrix 6x6)======================================"""
##############################################################################
##############################################################################

### set it as diagonal (6x6) since we don't know the covariance of position and velocity
P = ...
P = np.diag([100, 100, 100, 50, 50, 50])  # initial covariance matrix (assume variables are independent, and variance is wide)
# change the 100 above??? e.g.:
# prior_cov = np.diag([100, 100, 100, 50, 50, 50])  # initial covariance matrix (assume variables are independent, and variance is wide)


##############################################################################
##############################################################################
"""======================== Design F (state transition matrix 6x6)======================================"""
##############################################################################
##############################################################################

### one equation for each state
dt = 0.1            # re-define dt!
F = np.diag([1.]*6)
F[0,3]=dt 
F[1,4]=dt 
F[2,5] =dt


##############################################################################
##############################################################################
"""======================== Design Q (process covariance matrix 6x6)======================================"""
##############################################################################
##############################################################################

Q = ...

##############################################################################
##############################################################################
"""======================== u (Control input) and B (control input model) ======================================"""
##############################################################################
##############################################################################

### the satellite is not controlled by us so u=0 and B=0
# B=0
# u=0

##############################################################################
##############################################################################
"""======================== Predict step ======================================"""
##############################################################################
##############################################################################

P = P+Q
x, P = predict(x, P, F, Q)
# x, P = predict(x, P, F, Q, B, u)  # this is the same as above since B=0, u=0



##############################################################################
##############################################################################
"""======================== get z (radar measurement) ======================================"""
##############################################################################
##############################################################################

### 
z = 1.


##############################################################################
##############################################################################
"""======================== Design H (measurement function) (3x6) ======================================"""
"""======================== (converts predicted state (6x1) into measurement space (3x1) via z=Hx. This gives H as a (3x6) 
=========================== Cartesian into spherical coord.)
=========================== 0s for velocities, i.e. don't need to convert them) ========================"""
##############################################################################
##############################################################################

H = [[..., ..., ..., 0, 0, 0],
     [..., ..., ..., 0, 0, 0],
     [..., ..., ..., 0, 0, 0]]
# predicted_state =  H @ x 


##############################################################################
##############################################################################
"""======================== Residual (measurement - predicted)======================================"""
##############################################################################
##############################################################################

y = z - H @ x


##############################################################################
##############################################################################
"""======================== Design R (measurement covariance matrix 3x3)======================================"""
##############################################################################
##############################################################################

### measurement covariance matrix contains varainces of distance, elevation, azimuth, and covariances between them (assume independent, so 0s)
R = np.diag(1, 2, 3)    # variance of distance, elevation, azimuth





##############################################################################
##############################################################################
"""======================== Update step ======================================"""
##############################################################################
##############################################################################

from filterpy.kalman import update
x, P = update(x, P, z, R, H)
print('x =', x)



##############################################################################
##############################################################################
"""======================== Kalman Filter all ======================================"""
##############################################################################
##############################################################################

### specify dimensions for the state and measurement
sate_filter = KalmanFilter(dim_x = 6, dim_z= 3)

# print("state x = \n", sate_filter.x)
# print("state prior covariance matrix P = \n", sate_filter.P)
# print("process model F = \n", sate_filter.F)
# print("process noise covariance matrix Q = \n", sate_filter.Q)
# print("measurement z = \n", sate_filter.z)
# print("measurement noise covariance matrix R = \n", sate_filter.R)
# print("measurement function H = \n", sate_filter.H)

### create a function for it instead so it can be called easily ===============================
def satellite_filter (x, P, F, Q, R, dt = 0.1):
    kf = KalmanFilter(dim_x = 6, dim_z= 3)
    kf.x = np.array([x[0], x[1], x[2], x[3], x[4], x[5]])
    kf.P[:] = P 
    kf.F = ...
    kf.Q[:] = Q

    kf.H = ...
    kf.R[:] = R
    return kf

### Create a function to run Kalman filter =====================================================
def run_kf (x0, P, Q, R, dt=0.1, zs = None, trajectory_actual = None, do_plot = False):
    """zs = radar measurements"""
    ### get data from the Simulator ===========
    ...
    x0 = ...
    zs = ...

    ### create kalman filter ===============
    kf = satellite_filter(x0, P=P, Q=Q, R=R, dt=dt)

    ### run kalman filter ================
    xs, cov = [], []

    for z in zs:
        kf.predict()
        kf.update(z)
        xs.append(kf.x)
        cov.append(kf.P)

    xs, cov = np.array(xs), np.array(cov)

    ### plot?
    if do_plot:
        ### compare trajectory_filter to trajectory_actual
        ...
    return xs, cov



##############################################################################
##############################################################################
"""======================== Unscented Kalman Filter  ======================================"""
##############################################################################
##############################################################################

"""============= Take first one (or 2? need to get the initial velocity) measurement as the starting state ============="""

def measurement_to_state_space(radar_pos, range, elev, azimuth):
    "convert from radar measurement space to cartesian pos of the satellite"
    x= ...
    y=...
    z=...
    return x, y, z

def hx(radar_pos, x,y,z):
    "convert from cartesian pos of satellite to radar measurement space"
    range=...
    elev=...
    azimuth=...
    return range, elev, azimuth


radar1 = radar_reading()
radar2 = radar_reading()
### unpack 
radar_pos1, range1, elev1, azimuth1, time1 = radar1
radar_pos2, range2, elev2, azimuth2, time2 = radar2

x1, y1, z1 = measurement_to_state_space(radar_pos1, range1, elev1, azimuth1)
x2, y2, z2 = measurement_to_state_space(radar_pos2, range2, elev2, azimuth2)
dt = time2-time1
vx_init = (x2-x1)/dt
vy_init = (y2-y1)/dt
vz_init = (z2-z1)/dt
