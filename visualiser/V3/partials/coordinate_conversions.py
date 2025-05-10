import numpy as np
from pyproj import CRS, Transformer, Proj

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