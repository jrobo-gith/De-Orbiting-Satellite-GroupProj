import numpy as np
from PyQt5 import QtCore
from datetime import datetime, timedelta
import os
import time

class Helper(QtCore.QObject):
    changedSignal = QtCore.pyqtSignal(str, tuple)

from visualiser.V3.partials.constants import G, M_EARTH, R_EARTH_POLES, R_EARTH_EQUATOR, CD
from visualiser.V3.debug import debug_print
from visualiser.V3.partials.coordinate_conversions import eci2ecef_matrix, lla2ecef, ecef2enu, enu2ecef, ecef2lla, get_gmst

script_dir = os.path.dirname(os.path.abspath(__file__))

def get_sat_data():
    sat_pos_x = []
    sat_pos_y = []
    sat_pos_z = []
    sat_vx = []
    sat_vy = []
    sat_vz = []
    t_vals = []

    sat_file = os.path.join(script_dir, 'sat_traj.dat')

    try:
        with open(sat_file, 'r') as file:
            for line_num, line in enumerate(file, 1):
                values = line.strip().split()
                sat_pos_x.append(float(values[0]))
                sat_pos_y.append(float(values[1]))
                sat_pos_z.append(float(values[2]))
                sat_vx.append(float(values[3]))
                sat_vy.append(float(values[4]))
                sat_vz.append(float(values[5]))
                t_vals.append(float(values[-1]))

    except FileNotFoundError:
        debug_print("simulator", "Error: File 'sat_traj.dat' not found.")
    except ValueError as e:
        debug_print("simulator", f"Error: Could not convert value to float on line {line_num}: {e}")

    sat_pos_x = np.array(sat_pos_x)
    sat_pos_y = np.array(sat_pos_y)
    sat_pos_z = np.array(sat_pos_z)
    sat_vx = np.array(sat_vx)
    sat_vy = np.array(sat_vy)
    sat_vz = np.array(sat_vz)
    t_vals = np.array(t_vals)
    sat_pos_eci = np.column_stack((sat_pos_x, sat_pos_y, sat_pos_z))
    sat_vel = np.column_stack((sat_vx, sat_vy, sat_vz))

    nt = t_vals.shape[0]
    dt = t_vals[1] - t_vals[0]

    sim_start_time = datetime(2025, 5, 4, 12, 0, 0)
    sim_times = np.array([sim_start_time + timedelta(seconds=i * dt) for i in range(nt)])

    # Compute GMST angles for simulation times
    gmst_angles = np.array([get_gmst(t) for t in sim_times])

    # Convert ECI to ECEF
    sat_pos_ecef = np.zeros_like(sat_pos_eci)
    for i, (pos, theta) in enumerate(zip(sat_pos_eci, gmst_angles)):
        Rot_eci2ecef = eci2ecef_matrix(theta)
        sat_pos_ecef[i] = Rot_eci2ecef.dot(pos)
    return sat_pos_ecef, sat_vel, t_vals

# Radar class - contains radar specifications and functions
class Radar:
    # Contains radar specific information
    __range_uncertainty = 100
    __theta_uncertainty = 0.0001
    __phi_uncertainty = 0.0001
    __vel_uncertainty = 0.05

    def __init__(self, longlat, azfov=360, elfov=180):
        # Inputs - An array containing lat-long-alt of radar
        #        - Azimuth field of view in degrees
        #        - Elevation field of view in degrees
        self.lon = longlat[0]  # Longitude
        self.lat = longlat[1]  # Latitude
        self.alt = longlat[2]  # Altitude
        self.pos_lla = longlat
        self.azfov = azfov  # Azimuth angle field of view
        self.elfov = elfov  # Elevation angle field of view
        self.pos_ecef = lla2ecef(np.array([self.lon, self.lat, self.alt]))

    def __add_noise(self, radM):
        # Inputs  - An array containing radar measurements
        # Outputs - An array containing noisy radar measurements
        rho, theta, phi, vx, vy, vz = radM
        rho += np.random.normal(0, self.__range_uncertainty)
        theta += np.random.normal(0, self.__theta_uncertainty)
        phi += np.random.normal(0, self.__phi_uncertainty)
        vx += np.random.normal(0, self.__vel_uncertainty)
        vy += np.random.normal(0, self.__vel_uncertainty)
        vz += np.random.normal(0, self.__vel_uncertainty)
        return np.array([rho, theta, phi, vx, vy, vz])

    # This function is effectively converting ENU to its equivalent spherical coordinates
    def radar_measurements(self, sat_enu_coords, sat_vel, noise=True):
        # Inputs  - ENU coordinates of the object
        #           Boolean to check whether to add noise to the measurements
        # Outputs - An array containing radar measurements (range, azimuth, elevation)
        e, n, u = sat_enu_coords
        vx, vy, vz = sat_vel

        rho = np.sqrt(e ** 2 + n ** 2 + u ** 2)  # Range
        theta = np.arctan2(e, n)  # Azimuth
        horizontal = np.sqrt(e ** 2 + n ** 2)
        phi = np.arctan2(u, horizontal)  # Elevation
        # Add noise
        if (noise == True):
            radM = self.__add_noise((rho, theta, phi, vx, vy, vz))
        else:
            radM = np.array([rho, theta, phi, vx, vy, vz])
        return radM

    # Convert radar measurements to ENU coordinates
    def radM2enu(self, radM_pos):
        # Inputs  - An array containing radar measurements
        # Outputs - ENU coordinates of the object
        rho, theta, phi = radM_pos  # Range, azimuth, elevation
        # Do conversion
        e = rho * np.cos(phi) * np.sin(theta)
        n = rho * np.cos(phi) * np.cos(theta)
        u = rho * np.sin(phi)
        return np.array([e, n, u])

    def check_fov(self, radM_pos):
        # Inputs  - An array containing radar measurements
        # Outputs - Boolean True if object is in radar's field of view
        #           Boolean False otherwise
        rho, theta, phi = radM_pos

        theta_deg = np.degrees(theta)
        phi_deg = np.degrees(phi)

        if ((rho < 10)
                or (theta_deg > self.azfov)
                or (phi < 0) or (phi_deg > self.elfov)):
            return False
        return True

# Simulate radar measurements
def get_radar_measurements(radars, graph_helper, earth_helper, predictor_helper):
    # Inputs  - An array of satellite positions in ECEF coordinates
    #         - A dictionary of radar objects
    # Outputs - A dictionary of measurements taken by each radar
    sat_pos_ecef, sat_vel, t_vals = get_sat_data()
    measurements = {key: [] for key in radars.keys()}

    nt = t_vals.shape[0]
    dt = t_vals[1] - t_vals[0]

    sim_start_time = datetime(2025, 5, 4, 12, 0, 0)
    sim_times = np.array([sim_start_time + timedelta(seconds=i * dt) for i in range(nt)])

    # Check every t seconds
    for i in range(t_vals.shape[0]):
        curr_sat_pos = sat_pos_ecef[i]
        closest_radar_distance = np.inf
        seen_satellite = False
        for rname, radobj in radars.items():
            # Compute relative position of satellite from radar and range
            rel_pos = curr_sat_pos - radobj.pos_ecef

            # Convert to ENU coordinates to compute azimuth and elevation
            rel_pos_enu = ecef2enu(rel_pos, radobj.pos_lla)

            radM = radobj.radar_measurements(rel_pos_enu, sat_vel[i],
                                             noise=True)  # Set it back to noise = TRUE later !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            radM_no_noise = radobj.radar_measurements(rel_pos_enu, sat_vel[i], noise=False)  # NO NOISE

            # Check if the satellite is in field of view of the radar
            if (radobj.check_fov(radM[:3])):
                seen_satellite = True
                # IN FOV
                radar_dist = radM[0]
                if radar_dist < closest_radar_distance:
                    closest_radar_distance = radar_dist
                    measurement = (*radM,)
                    true_pos = (*curr_sat_pos, *sat_vel[i], CD)
                    info = {"name": rname, "obs-time": t_vals[i], "stime": sim_times[i], 'radobj': radobj,
                            'rdist': radM[0],
                            "state_no_noise": true_pos}

        if seen_satellite:
            predictor_helper.changedSignal.emit(info, measurement)
        else:
            info = {"name": "no radar", "obs-time": t_vals[i], "stime": sim_times[i], 'radobj': radobj, 'rdist': "none"}
            predictor_helper.changedSignal.emit(info, (0, 0, 0))

        radM_enu_nn = radobj.radM2enu(radM_no_noise[:3])
        radM_ecef_nn = enu2ecef(radM_enu_nn, radobj.pos_lla)
        lat, lon, _ = ecef2lla(radM_ecef_nn)

        earth_helper.changedSignal.emit(info, (lat, lon))
        graph_helper.changedSignal.emit(info, (radM_ecef_nn,))

        time.sleep(.2)
    for key, vals in measurements.items():
        measurements[key] = np.array(vals)
    return measurements

radars = {}

def initialise_radars(latlon:list):
    for i, pos in enumerate(latlon):
        radars[f"radar{i}"] = Radar(pos, azfov=60, elfov=60)

    return radars