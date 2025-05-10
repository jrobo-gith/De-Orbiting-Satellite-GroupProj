import threading

import numpy as np
import matplotlib.pyplot as plt
from PyQt5 import QtCore
from pyproj import CRS, Transformer, Proj
from datetime import datetime, timedelta
import os
import time
from debug import debug_print

class Helper(QtCore.QObject):
    changedSignal = QtCore.pyqtSignal(str, tuple)

# Constants
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
M_EARTH = 5.972e24  # Mass of Earth (kg)
R_EARTH_EQUATOR = 6378e3  # Radius of Earth near equator (m)
R_EARTH_POLES = 6357e3 # Radius of Earth near poles (m)

script_dir = os.path.dirname(os.path.abspath(__file__))

# Compute Greenwich Mean Sidereal Time for the given time
def get_gmst(t):
    # Inputs  - A datetime object
    # Outputs - GMST angle in radians
    epoch_J2000 = datetime(2000, 1, 1, 12, 0, 0)
    days_since_J2000 = (t - epoch_J2000).total_seconds() / 86400.0
    gmst = 2 * np.pi * (0.779057 + 1.002738 * days_since_J2000)
    gmst %= 360
    return np.radians(gmst)

# ECI to ECEF rotation matrix
def eci2ecef_matrix(theta):
    # Inputs  - GMST angle in radians
    # Outputs - Rotation matrix to convert from ECI to ECEF
    return np.array([[np.cos(theta), np.sin(theta), 0],
                     [-np.sin(theta), np.cos(theta), 0],
                     [0,0,1]])


def get_sat_data():
    sat_pos_x = []
    sat_pos_y = []
    sat_pos_z = []
    t_vals = []

    sat_file = os.path.join(script_dir, 'sat_traj.dat')

    try:
        with open(sat_file, 'r') as file:
            for line_num, line in enumerate(file, 1):
                values = line.strip().split()
                sat_pos_x.append(float(values[0]))
                sat_pos_y.append(float(values[1]))
                sat_pos_z.append(float(values[2]))
                t_vals.append(float(values[-1]))
    except FileNotFoundError:
        debug_print("simulator", "Error: File 'sat_traj.dat' not found.")
    except ValueError as e:
        debug_print("simulator", f"Error: Could not convert value to float on line {line_num}: {e}")

    sat_pos_x = np.array(sat_pos_x)
    sat_pos_y = np.array(sat_pos_y)
    sat_pos_z = np.array(sat_pos_z)
    t_vals = np.array(t_vals)
    sat_pos_eci = np.column_stack((sat_pos_x, sat_pos_y, sat_pos_z))

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
    return sat_pos_ecef, t_vals

# Radar class - contains radar specifications and functions
class Radar:
    # Contains radar specific information
    __range_uncertainty = 100
    __theta_uncertainty = 0.01
    __phi_uncertainty = 0.01

    def __init__(self, longlat, azfov=360, elfov=180):
        # Inputs - An array containing lat-long-alt of radar
        #        - Azimuth field of view in degrees
        #        - Elevation field of view in degrees
        self.lon = longlat[0] # Longitude
        self.lat = longlat[1] # Latitude
        self.alt = longlat[2] # Altitude
        self.pos_lla = longlat
        self.azfov = azfov # Azimuth angle field of view
        self.elfov = elfov # Elevation angle field of view
        self.pos_ecef = lla2ecef(np.array([self.lon, self.lat, self.alt]))

    def __add_noise(self, radM):
        # Inputs  - An array containing radar measurements
        # Outputs - An array containing noisy radar measurements
        rho, theta, phi = radM
        rho += np.random.normal(0, self.__range_uncertainty)
        theta += np.random.normal(0, self.__theta_uncertainty)
        phi += np.random.normal(0, self.__phi_uncertainty)
        return np.array([rho, theta, phi])

    # This function is effectively converting ENU to its equivalent spherical coordinates
    def radar_measurements(self, sat_enu_coords, noise=True):
        # Inputs  - ENU coordinates of the object
        #           Boolean to check whether to add noise to the measurements
        # Outputs - An array containing radar measurements (range, azimuth, elevation)
        e, n, u = sat_enu_coords
        rho = np.sqrt(e**2 + n**2 + u**2) # Range
        theta = np.arctan2(e, n) # Azimuth
        horizontal = np.sqrt(e**2 + n**2)
        phi = np.arctan2(u, horizontal) # Elevation
        # Add noise
        if (noise == True):
            radM = self.__add_noise((rho, theta, phi))
        else:
            radM = np.array([rho, theta, phi])
        return radM
    
    # Convert radar measurements to ENU coordinates
    def radM2enu(self, radar_vals):
        # Inputs  - An array containing radar measurements
        # Outputs - ENU coordinates of the object      
        rho, theta, phi = radar_vals # Range, azimuth, elevation
        # Do conversion
        e = rho * np.cos(phi) * np.sin(theta)
        n = rho * np.cos(phi) * np.cos(theta)
        u = rho * np.sin(phi)
        return np.array([e, n, u])
    
    def check_fov(self, radM):
        # Inputs  - An array containing radar measurements
        # Outputs - Boolean True if object is in radar's field of view
        #           Boolean False otherwise
        rho, theta, phi = radM
        
        theta_deg = np.degrees(theta)
        phi_deg = np.degrees(phi)

        if ((rho < 10) 
            or (theta_deg > self.azfov) 
            or (phi < 0) or (phi_deg > self.elfov)):
            return False
        return True

# Conversions between ECEF and Geodetic (latitude-longitude-altitude) coordinate systems
def ecef2lla(ecef_coords):
    # Inputs  - ECEF coordinates of the object
    # Outputs - Lat-long coordinates of the object
    transformer = Transformer.from_crs(
        "EPSG:4978",
        "EPSG:4326",
        always_xy = True
    )
    lon, lat, alt = transformer.transform(*ecef_coords)
    return np.array([lat, lon, alt])

def lla2ecef(lla_coords):
    # Inputs  - Lat-long coordinates of the object
    # Outputs - ECEF coordinates of the object
    transformer = Transformer.from_crs(
        "EPSG:4326",
        "EPSG:4978",
        always_xy = True
    )
    x, y, z = transformer.transform(*lla_coords) 
    return np.array([x, y, z])

# Conversions between ENU and ECEF coordinate systems
def enu2ecef(enu_coords, ref_lla_coords):
    # Inputs  - ENU coordinates of the object
    #           Reference coordinates in lat-long
    # Outputs - ECEF coordinates of the object
    ecef_coords = lla2ecef(ref_lla_coords) # ECEF coordinates of the reference point
    lon, lat, alt = ref_lla_coords
    Xform_matrix = np.array([[-np.sin(lon), -np.sin(lat)*np.cos(lon), np.cos(lat)*np.cos(lon)],
                             [np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)*np.sin(lon)],
                             [0, np.cos(lat), np.sin(lat)]])
    ecef_coords += Xform_matrix.dot(enu_coords)
    return ecef_coords

def ecef2enu(ecef_coords, ref_lla_coords):
    # Inputs  - ECEF coordinates of the object
    #           Reference coordinates in lat-long
    # Outputs - ENU coordinates of the object
    lon, lat, alt = ref_lla_coords
    Xform_matrix = np.array([[-np.sin(lon), np.cos(lon), 0],
                             [-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)],
                             [np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)]])
    enu_coords = Xform_matrix.dot(ecef_coords)
    return enu_coords

# Conversion between geodetic and ENU coordinate systems
def lla2enu(lla_coords, ref_lla_coords):
    # Inputs  - Lat-long coordinates of the object
    #           Reference coordinates in lat-long
    # Outputs - ENU coordinates of the object
    ecef_coords = lla2ecef(lla_coords)
    ref_ecef_coords = lla2ecef(ref_lla_coords)
    enu_coords = ecef2enu(ecef_coords - ref_ecef_coords, ref_lla_coords)
    return enu_coords

def enu2lla(enu_coords, ref_lla_coords):
    # Inputs  - ENU coordinates of the object
    #           Reference coordinates in lat-long
    # Outputs - Lat-long coordinates of the object
    ecef_coords = enu2ecef(enu_coords, ref_lla_coords)
    lla_coords = ecef2lla(ecef_coords)
    return lla_coords

# Simulate radar measurements
def get_radar_measurements(radars, earth_helper, predictor_helper):
    # Inputs  - An array of satellite positions in ECEF coordinates
    #         - A dictionary of radar objects
    # Outputs - A dictionary of measurements taken by each radar
    sat_pos_ecef, t_vals = get_sat_data()
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
        for rname,radobj in radars.items():
            # Compute relative position of satellite from radar and range
            rel_pos = curr_sat_pos - radobj.pos_ecef

            # Convert to ENU coordinates to compute azimuth and elevation
            rel_pos_enu = ecef2enu(rel_pos, radobj.pos_lla)
            radM = radobj.radar_measurements(rel_pos_enu)

            radM_no_noise = radobj.radar_measurements(rel_pos_enu, noise=False)  # NO NOISE

            radM_enu = radobj.radM2enu(radM_no_noise) #CHANGE TO RADM
            radM_ecef = enu2ecef(radM_enu, radobj.pos_lla)
            gmst_angle = get_gmst(sim_times[i])
            Rot_eci2ecef = eci2ecef_matrix(gmst_angle).T
            pos_x, pos_y, pos_z = Rot_eci2ecef.dot(radM_ecef)

            # Check if the satellite is in field of view of the radar
            if (radobj.check_fov(radM)):
                seen_satellite = True
                # IN FOV
                radar_dist = radM[0]
                if radar_dist < closest_radar_distance:
                    closest_radar_distance = radar_dist
                    measurement = (pos_x, pos_y, pos_z)
                    info = {"name": rname, "obs-time": t_vals[i], "stime": sim_times[i], 'radobj': radobj, 'rdist': radM[0]}

        if seen_satellite:
            predictor_helper.changedSignal.emit(info, measurement)
        else:
            info = {"name": "no radar", "obs-time": t_vals[i], "stime": sim_times[i], 'radobj': radobj, 'rdist': "none"}
            predictor_helper.changedSignal.emit(info, (0, 0, 0))

        radM_enu_nn = radobj.radM2enu(radM_no_noise)
        radM_ecef_nn = enu2ecef(radM_enu_nn, radobj.pos_lla)
        lat, lon, _ = ecef2lla(radM_ecef_nn)

        earth_helper.changedSignal.emit(info, (lat, lon))

        time.sleep(.2)
    for key, vals in measurements.items():
        measurements[key] = np.array(vals)
    return measurements

# Convert satellite's position to lat-long given its ECEF coordinates
def sat_ecef2lla(sat_pos_ecef):
    # Inputs  - An array of satellite positions in ECEF coordinates
    # Outputs - An array of satellite positions in LLA coordinates    
    sat_pos_lla = []
    for i in range(sat_pos_ecef.shape[0]):
        sat_pos_lla.append(ecef2lla(sat_pos_ecef[i]))
    sat_pos_lla = np.array(sat_pos_lla)
    return sat_pos_lla

# Function to convert satellite position in ECI to radar's local spherical coordinates
def do_conversions(eci_coords, stime, radar):
    # Inputs  - ECI coordinates of the satellite
    #         - A datetime object corresponding to the satellite position
    #         - Name of the radar
    # Outputs - An array containing range, azimuth and elevation

    sim_start_time = datetime(2025, 5, 4, 12, 0, 0)


    # radar = radars[radar_name]
    # debug_print("simulator", radar)
    # Convert satellite position from ECI to ECEF
    gmst_angle = get_gmst(stime)
    Rot_eci2ecef = eci2ecef_matrix(gmst_angle)
    pos_ecef = Rot_eci2ecef.dot(eci_coords)

    # Convert the relative position between satellite and radar to ENU coordinates
    rel_pos = pos_ecef - radar.pos_ecef
    rel_pos_enu = ecef2enu(rel_pos, radar.pos_lla)
    # Get spherical coordinate values
    radM = radar.radar_measurements(rel_pos_enu, noise=False)

    return radM

radars = {}

def initialise_radars(latlon:list):
    for i, pos in enumerate(latlon):
        radars[f"radar{i}"] = Radar(pos, azfov=60, elfov=60)

    return radars