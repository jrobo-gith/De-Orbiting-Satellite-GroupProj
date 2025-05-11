import numpy as np
from pyproj import CRS, Transformer, Proj
from datetime import datetime, timedelta

# ECI to ECEF rotation matrix
def eci2ecef_matrix(theta):
    # Inputs  - GMST angle in radians
    # Outputs - Rotation matrix to convert from ECI to ECEF
    return np.array([[np.cos(theta), np.sin(theta), 0],
                     [-np.sin(theta), np.cos(theta), 0],
                     [0,0,1]])

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

# Convert satellite's position to lat-long given its ECEF coordinates
def sat_ecef2lla(sat_pos_ecef):
    # Inputs  - An array of satellite positions in ECEF coordinates
    # Outputs - An array of satellite positions in LLA coordinates
    sat_pos_lla = []
    for i in range(sat_pos_ecef.shape[0]):
        sat_pos_lla.append(ecef2lla(sat_pos_ecef[i]))
    sat_pos_lla = np.array(sat_pos_lla)
    return sat_pos_lla

def radM2eci(radM, stime, radar):
    radM_enu = radar.radM2enu(radM[:3]) #CHANGE TO RADM
    radM_ecef = enu2ecef(radM_enu, radar.pos_lla)
    gmst_angle = get_gmst(stime)
    Rot_eci2ecef = eci2ecef_matrix(gmst_angle).T
    pos_eci = Rot_eci2ecef.dot(radM_ecef)
    return pos_eci

def get_gmst(t):
    # Inputs  - A datetime object
    # Outputs - GMST angle in radians
    epoch_J2000 = datetime(2000, 1, 1, 12, 0, 0)
    days_since_J2000 = (t - epoch_J2000).total_seconds() / 86400.0
    gmst = 2 * np.pi * (0.779057 + 1.002738 * days_since_J2000)
    gmst %= 360
    return np.radians(gmst)

# Function to convert satellite position in ECI to radar's local spherical coordinates
def do_conversions(radar_z_ECI, stime, radar):
    # Inputs  - ECI coordinates of the satellite
    #         - A datetime object corresponding to the satellite position
    #         - Name of the radar
    # Outputs - An array containing range, azimuth and elevation

    eci_coords = radar_z_ECI[:3]
    eci_vel = radar_z_ECI[3:6]

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
    radM = radar.radar_measurements(rel_pos_enu, eci_vel, noise=False)

    return radM