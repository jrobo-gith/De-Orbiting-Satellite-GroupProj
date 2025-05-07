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

""" Set constants ==================================================="""

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



def curvature_in_prime_vertical(phi):
            return EARTH_SEMIMAJOR / np.sqrt(1 - E_SQUARED * np.sin(phi)**2)
        
def latitude_iterator_and_height(x, y, z):
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan(z / (r * (1 - E_SQUARED)))
    phi_new = phi + 100
    
    while np.abs(phi_new - phi) > 1e-9:
        phi = phi_new
        N = curvature_in_prime_vertical(phi)
        phi_new = np.arctan2(z + (N * E_SQUARED) * np.sin(phi), r - (N * E_SQUARED) * np.cos(phi))
    
    height = (r / np.cos(phi)) - N
    if height < 0:
        height = 0
    
    return phi_new, height


def earth_radius_WGS84(latitude):
    numerator = (EARTH_SEMIMAJOR**2 * np.cos(latitude))**2 + (EARTH_SEMIMINOR**2 * np.sin(latitude))**2
    denominator = (EARTH_SEMIMAJOR * np.cos(latitude))**2 + (EARTH_SEMIMINOR * np.sin(latitude))**2
    return np.sqrt(numerator / denominator)

def lat_long_height(x, y, z):
    r = np.linalg.norm([x,y,z])
    longitude = np.arctan2(z, np.sqrt(x**2 + y**2))
    latitude, height = latitude_iterator_and_height(x, y, z)
    R_earth = earth_radius_WGS84(latitude)
    height = r - R_earth
    return latitude, longitude, height

def atmospheric_density(altitude):
    # White noise as random perturbations of magnitude 0.1*RHO_0
    density = RHO_0 * np.exp(-altitude / H_SCALE)
    if altitude > 0:
        return density
    else:
        return 0
    
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
        
        # Acceleration
        ax = F_gravity * (x / r) + F_drag_x
        ay = F_gravity * (y / r) + F_drag_y
        az = F_gravity * (z / r) + F_drag_z

        return np.array([vx, vy, vz, ax, ay, az])


""" Define the process model f(x) ======================================="""
def f(state_x, dt):

    """state vector = state_x = [x,y,z,vx, vy, vz]"""

    if dt <= 10.:
        x, y, z, vx, vy, vz= state_x
        _, _, _, ax, ay, az = ode(0, state_x)
        # Use Euler's method for small dt
        x_new = x + vx * dt
        y_new = y + vy * dt
        z_new = z + vz * dt
        vx_new = vx + ax * dt
        vy_new = vy + ay * dt
        vz_new = vz + az * dt
        return np.array([x_new, y_new, z_new, vx_new, vy_new, vz_new])
    else:
        # Use RK4 for larger dt
        solution = solve_ivp(ode, t_span=[0, dt], y0=state_x, method='RK45', t_eval=[dt])
        # print(solution.y)
        return solution.y.flatten()

""" Define measurement function h(x) ================================================="""
def h_radar(x):
    """x is the state vector,
    this H function assumes radar sends positions in global coordinate system"""
    return x[:3] # return x,y,z position if state order is (x,y,z,vx,vy,vz)



""" Define Unscented Kalman Filter ==================================================="""
def satellite_UKF(state0,  fx, hx, dt=1.0):
    """state0 :list. e.g. state0=[EARTH_SEMIMAJOR + 400e3, 1, -1, 1, 7700, 1]
                                [300e3 + EARTH_SEMIMAJOR, 0, 0 , 0, 7800/np.sqrt(2), 7800/np.sqrt(2)]
    """

    """ =============== Generate sigma points """
    ### initialise UKF
    sigmas_generator = MerweScaledSigmaPoints(n=6, alpha=0.1, beta = 2., kappa= -3.)  #kappa = -3.
    ukf = UKF(dim_x=6, dim_z=3, fx = fx, hx = hx, dt = dt, points=sigmas_generator) # take f, h from Jai and Vijay
    # print(ukf.Q)


    """ ============== Define items in UKF """
    ### initial state values of (x,y,z,vx,vy,vz)
    ukf.x = np.array(state0)  # initial state
    ### initial uncertainty of the state
    ukf.P = np.diag([50**2, 50**2, 50**2,
                    5**2, 5**2, 5**2])    # experiment this
    ### uncertainty in the process model
    ukf.Q = np.zeros((6,6))
    ukf.Q[np.ix_([0,3], [0,3])] = Q_discrete_white_noise(dim=2, dt=dt, var=0.01)  # Q matrix for how other noise affect x and vx
    ukf.Q[np.ix_([1,4], [1,4])] = Q_discrete_white_noise(dim=2, dt=dt, var=0.01)  # Q matrix for how other noise affect y and vy
    ukf.Q[np.ix_([2,5], [2,5])] = Q_discrete_white_noise(dim=2, dt=dt, var=0.01)  # Q matrix for how other noise affect z and vz


    # range_std = 10 # meters. change this!!!!!!!!!!!!!!!!!!!!!! (get from radar)
    # elev_std = math.radians(1)  # 1 degree in radians. change this!!!!!!!!!!!!!!!!!!!!!! (get from radar)
    # azim_std = math.radians(1)  # 1 degree in radians. change this!!!!!!!!!!!!!!!!!!!!!1 (get from radar)
    # ukf.R = np.diag([range_std**2, elev_std**2, azim_std**2])

    """### radar measurement noise (for the simple UKF only! change this!!!!!!!!!!!!!!!!!!!!!!)"""
    # x_std = 500  # meters. 
    # y_std = 500  # meters.
    # z_std = 10  # meters. 
    # ukf.R = np.diag([x_std**2, y_std**2, z_std**2])

    range_std = 10 # meters.
    azim_std = 0.002 # radians. (theta)
    elev_std = 0.002  # radians. (phi)
    ukf.R = np.diag([range_std**2, azim_std**2, elev_std**2])

    # ukf.save_x_prior = True
    return ukf
if __name__ == "__main__":
    """Initialise satellite, radar and UKF========================"""
    # np.random.seed(0)  # for reproducibility
    # satellite_pos_true = SatelliteTraj(pos=[R_EARTH + 1500e3, 0, 0], vel=[0, 7600, 0], vel_std=0.)  # true satellite position with noise
    # radar = RadarStation(radar_pos = (0,0,0), x_std=50000, y_std=50, z_std=10)
    ukf = satellite_UKF(state0= [EARTH_SEMIMAJOR + 400e3, 1, -1, 1, 7700, 1] , fx = f, hx = h_radar, dt=1)


    """Generate radar measurement batches and do batch processing in UKF"""
    num_iterations = 10

    xs_true = []
    xs_prior = []
    zs = []
    xs = []
    Ps = []
    ts = [0.0]


    time_duration = 0.0 # initialise total time duration of the filtering process

    for iter in tqdm(range(num_iterations)):
        num_in_batch = np.random.choice([5, 15, 25])  # random number of measurements in a batch
        dt_sim_arr = np.random.randint(10., 200., num_in_batch) ### randomely generate timesteps
        # zs_batch = []
        for dt_sim in dt_sim_arr:
            time_duration += dt_sim
            sate_pos = np.concatenate(satellite_pos_true.update(dt_sim)).reshape(1,-1).flatten()
            # print(sate_pos)
            xs_true.append(sate_pos)
            ### get radar measurement
            z = np.array(radar.noisy_reading(satellite_pos=sate_pos[:3], time=time_duration))
            # print(z)
            # zs_batch.append(z)
            zs.append(z)
            ts.append(z[3])
            # print(np.array(ts), np.array(ts).shape)
            if lat_long_height(sate_pos[0], sate_pos[1], sate_pos[2])[2] <=0: break

        # print("iteration=", iter, "\tbatch length=", num_in_batch, "\tts=", ts)
        # print("iteration=", iter, "\tbatch length=", num_in_batch, "\nvarying dt:", np.array(ts)[-num_in_batch:] - np.array(ts)[-num_in_batch-1:-1])
        # print("iteration=", iter, "\tbatch length=", num_in_batch, "time=", np.array(ts),
        #       "\nvarying dt:", np.array(ts)[-num_in_batch:] - np.array(ts)[-num_in_batch-1:-1],
        #       "\nbatch:",  np.array(zs)[-num_in_batch:])

        ### UKF with batch processing (each batch has varying timesteps) =======================================
        # (x, cov) = ukf.batch_filter(zs=np.array(zs)[-num_in_batch:, :3],
        #                                 dts=np.array(ts)[-num_in_batch:] - np.array(ts)[-num_in_batch-1:-1])
        # print(np.array(ts)[-num_in_batch:])
        # print(np.array(ts)[-num_in_batch-1:-1])
        for i, (z, dt) in enumerate(zip(np.array(zs)[-num_in_batch:, :],
                                        np.array(ts)[-num_in_batch:] - np.array(ts)[-num_in_batch-1:-1])):
            ukf.predict(dt=dt)
            xs_prior.append(ukf.x_prior)
            ukf.update(z[:3])
            x_post = ukf.x
            xs.append(x_post)
            x_cov = ukf.P
            Ps.append(x_cov)
            # print(np.random.multivariate_normal(mean=x_post, cov=x_cov, size=5))

            """Predict landing ====================================================================="""
            altitude_val = lat_long_height(sate_pos[0], sate_pos[1], sate_pos[2])[2]

            print("altitude=", altitude_val)

            if altitude_val <= 0:   # if state position is less than ? meters away from the earth
                break
            elif altitude_val <= 200e3:
                print("============== predict landing ================")
                ### sample from the updated state distribution and predict landing position
                state_samples = np.random.multivariate_normal(mean=x_post, cov=x_cov, size=5)

                ### record the landing position (inertia coord), and landing time
                predicted_landing_ECI = []
                predicted_landing_latlon = []
                predicted_landing_time = []

                for state0 in state_samples:
                    # t_eval_arr = np.linspace(ts[-1], ts[-1]+100000, 1000)
                    stop_condition.terminal = True
                    stop_condition.direction = -1
                    start = ts[-1]
                    end = start + 1000000
                    landing = solve_ivp(fun=ode, t_span=[start, end], y0=state0, method='RK45', t_eval=[end], events= stop_condition)
                    while (landing.success and len(landing.y)> 0):
                        end += 1000000
                        landing = solve_ivp(fun=ode, t_span=[start, end], y0=state0, method='RK45', t_eval=[end], events= stop_condition)
                    if landing.success and len(landing.y)==0:
                        # print(landing.y_events, np.linalg.norm(landing.y_events[0][0][:3])-R_EARTH)
                        # print(landing.y_events[0][0][:3])
                        landing_time = landing.t_events[0][0]
                        landing_position = landing.y_events[0][0][:3]
                        landing_position_latlon = lat_long_height(landing_position[0], landing_position[1], landing_position[2])
                        predicted_landing_ECI.append(landing_position)
                        predicted_landing_latlon.append(landing_position_latlon)
                        predicted_landing_time.append(landing_time)


    xs_true = np.array(xs_true)
    xs_prior = np.array(xs_prior)
    zs = np.array(zs)
    ts = np.array(ts)
    # xs = np.concatenate(xs, axis=0)
    # Ps = np.concatenate(Ps, axis=0)
    xs = np.array(xs)
    Ps = np.array(Ps)
