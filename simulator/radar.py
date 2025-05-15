import numpy as np
from PyQt5 import QtCore
from datetime import datetime, timedelta
import os
import time

class Helper(QtCore.QObject):
    changedSignal = QtCore.pyqtSignal(str, tuple)

from debug import debug_print, dev_mode
from partials.constants import CD, toHrs, toKM
from partials.coordinate_conversions import eci2ecef_matrix, lla2ecef, ecef2enu, enu2ecef, ecef2lla, get_gmst
from simulator.simulator import lat_long_height

script_dir = os.path.dirname(os.path.abspath(__file__))

def get_sat_data():
    """ 
    Reads satellite trajectory data from a file and converts it to ECEF coordinates.
    
    Returns:
        sat_pos_ecef (np.ndarray): Satellite positions in ECEF coordinates.
        sat_pos_eci (np.ndarray): Satellite positions in ECI coordinates.
        sat_vel (np.ndarray): Satellite velocities.
        t_vals (np.ndarray): Time values corresponding to the satellite positions.
    """
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
    return sat_pos_ecef, sat_pos_eci, sat_vel, t_vals

# Radar class - contains radar specifications and functions
class Radar:
    """
    Radar class to simulate radar measurements and check field of view.
    
    Attributes:
        lon (float): Longitude of the radar.
        lat (float): Latitude of the radar.
        alt (float): Altitude of the radar.
        pos_lla (list): Latitude, longitude, and altitude of the radar.
        azfov (float): Azimuth field of view in degrees.
        elfov (float): Elevation field of view in degrees.
        azcen (float): Azimuth center in degrees.
        maxr (float): Maximum range of the radar in meters.
        minr (float): Minimum range of the radar in meters.
        minel (float): Minimum elevation angle in degrees.
    """
    # Contains radar specific information
    __range_uncertainty = 100
    __theta_uncertainty = 0.001
    __phi_uncertainty = 0.001
    __vel_uncertainty = 0.05

    def __init__(self, longlat, azfov=360, elfov=180, azcen=0, maxr=5e6, minr=10, minel=2):
        """
        Initializes the Radar object with given parameters.
        
        Args:
            longlat (list): Latitude, longitude, and altitude of the radar.
            azfov (float): Azimuth field of view in degrees.
            elfov (float): Elevation field of view in degrees.
            azcen (float): Azimuth center in degrees.
            maxr (float): Maximum range of the radar in meters.
            minr (float): Minimum range of the radar in meters.
            minel (float): Minimum elevation angle in degrees.
        """
        # Inputs - An array containing lat-long-alt of radar
        #        - Azimuth field of view in degrees
        #        - Elevation field of view in degrees
        self.lat = longlat[0]  # Latitude
        self.lon = longlat[1]  # Longitude
        self.alt = longlat[2]  # Altitude
        longlat = [longlat[1], longlat[0], longlat[2]]
        self.pos_lla = longlat
        self.azfov = azfov  # Azimuth angle field of view
        self.elfov = elfov  # Elevation angle field of view
        self.azcen = azcen # Azimuth centre
        self.maxr = maxr # Maximum range
        self.minr = minr # Minimum range
        self.minel = minel # Minimum elevation
        self.pos_ecef = lla2ecef(np.array([self.lon, self.lat, self.alt]))

    def __add_noise(self, radM):
        """
        Adds noise to radar measurements.
        
        Args:
            radM (tuple): Radar measurements (range, azimuth, elevation, velocity).
        
        Returns:
            np.ndarray: Noisy radar measurements.
        """
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
        """
        Computes radar measurements from ENU coordinates.
        
        Args:
            sat_enu_coords (np.ndarray): ENU coordinates of the object.
            sat_vel (np.ndarray): Velocity of the object.
            noise (bool): Whether to add noise to the measurements.
        
        Returns:
            np.ndarray: Radar measurements (range, azimuth, elevation).
        """
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
        """
        Converts radar measurements to ENU coordinates.
        
        Args:
            radM_pos (np.ndarray): Radar measurements (range, azimuth, elevation).
        
        Returns:
            np.ndarray: ENU coordinates of the object.
        """
        # Inputs  - An array containing radar measurements
        # Outputs - ENU coordinates of the object
        rho, theta, phi = radM_pos  # Range, azimuth, elevation
        # Do conversion
        e = rho * np.cos(phi) * np.sin(theta)
        n = rho * np.cos(phi) * np.cos(theta)
        u = rho * np.sin(phi)
        return np.array([e, n, u])

    def check_fov(self, radM_pos):
        """
        Checks if the object is within the radar's field of view.
        
        Args:
            radM_pos (np.ndarray): Radar measurements (range, azimuth, elevation).
        
        Returns:
            bool: True if the object is within the field of view, False otherwise.
        """
        # Inputs  - An array containing radar measurements
        # Outputs - Boolean True if object is in radar's field of view
        #           Boolean False otherwise
        rho, theta, phi = radM_pos

        theta_deg = np.degrees(theta) % 360
        phi_deg = np.degrees(phi)

        azcen = np.random.randint(360)
        diff = ((theta_deg - azcen) + 180) % 360 - 180 # wrap to [-180, 180]

        if ((self.minr <= rho <= self.maxr)
            and (abs(diff) <= self.azfov/2)
            and (self.minel <= phi_deg <= self.elfov)):
            debug_print("simulator", f"{phi_deg}, {diff}, {rho}")
            return True
        return False

# Simulate radar measurements
def get_radar_measurements(radars, graph_helper, earth_helper, predictor_helper):
    """
    Simulates radar measurements for a given set of radars and satellite positions.
    
    Args:
        radars (dict): Dictionary of radar objects.
        graph_helper (Helper): Helper object for graph-related operations.
        earth_helper (Helper): Helper object for Earth-related operations.
        predictor_helper (Helper): Helper object for prediction-related operations.
    
    Returns:
        bool: True if the simulation was successful.
    """
    # Inputs  - An array of satellite positions in ECEF coordinates
    #         - A dictionary of radar objects
    # Outputs - A dictionary of measurements taken by each radar
    sat_pos_ecef, sat_pos_eci, sat_vel, t_vals = get_sat_data()

    nt = t_vals.shape[0]
    dt = t_vals[1] - t_vals[0]

    sim_start_time = datetime(2025, 5, 4, 12, 0, 0)
    sim_times = np.array([sim_start_time + timedelta(seconds=i * dt) for i in range(nt)])

    if dev_mode:
        rad_file = os.path.join(script_dir, "rad_measurements.dat")
        fp = open(rad_file, "w")

    # Check every t seconds
    for i in range(t_vals.shape[0]):
        curr_sat_pos = sat_pos_ecef[i]
        curr_sat_pos_eci = sat_pos_eci[i]
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

            # Convert radM to ecef for plotting
            radM_enu_noise = radobj.radM2enu(radM[:3])
            radM_ecef_noise = enu2ecef(radM_enu_noise, radobj.pos_lla)

            # Check if the satellite is in field of view of the radar
            if ((radobj.check_fov(radM_no_noise[:3])) or t_vals[i] == 0):
                seen_satellite = True
                # IN FOV
                radar_dist = radM[0]
                if radar_dist < closest_radar_distance:
                    closest_radar_distance = radar_dist
                    measurement = (*radM,)
                    true_pos = (*curr_sat_pos_eci, *sat_vel[i],CD)
                    info = {"name": rname, "obs-time": t_vals[i], "stime": sim_times[i],
                            'rdist': radM[0],
                            "state_no_noise": true_pos, 'state_noise': radM_ecef_noise,}

        if seen_satellite:
            predictor_helper.changedSignal.emit(info, measurement)
            if dev_mode:
                fp.write(f"{info}; {measurement}\n")
        else:
            info = {"name": "no radar", "obs-time": t_vals[i], "stime": sim_times[i], 'rdist': "none"}
            measurement = (0,0,0,0,0,0)
            predictor_helper.changedSignal.emit(info, measurement)
            if dev_mode:
                fp.write(f"{info}; {measurement}\n")

        radM_enu_nn = radobj.radM2enu(radM_no_noise[:3])
        radM_ecef_nn = enu2ecef(radM_enu_nn, radobj.pos_lla)
        lat, lon, _ = ecef2lla(radM_ecef_nn)

        altitude_sat = lat_long_height(radM_ecef_nn[0], radM_ecef_nn[1], radM_ecef_nn[2])[2]
        altitude_x = [t_vals[i]*toHrs, t_vals[i]*toHrs]
        altitude_y = [altitude_sat*toKM, 140e3*toKM]
        radar_alt = [altitude_x, altitude_y]

        radar_name = info['name']
        earth_helper.changedSignal.emit(info, (radM_ecef_nn,))
        graph_helper.changedSignal.emit(info, (radM_ecef_nn*toKM, radar_alt, radar_name))

        time.sleep(0.1)
    if dev_mode:
        fp.close()
    return True

radars = {}

def initialise_radars(lonlat:list):
    """
    Initializes radar objects with given latitude and longitude.
    
    Args:
        lonlat (list): List of latitude and longitude coordinates for the radars.
    
    Returns:
        dict: Dictionary of radar objects.
    """
    for i, pos in enumerate(lonlat):
        radars[f"radar{i}"] = Radar(pos, azfov=120, elfov=80, azcen=0, maxr=5e6)

    return radars