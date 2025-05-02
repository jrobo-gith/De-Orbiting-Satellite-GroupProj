def run_full_simulation(init_pos:list, init_vel:list):
    """
    This is the simulator (RK45) that will run the simulation and return latitudes and longitudes.
    list:init_pos - 1X3 for x, y, z
    list:init_vel - 1X3 for vx, vy, vz
    """
    # FOR NOW WE USE PRE_MADE DATA
    sim_directory = 'widgets/graph_stuff/partials/sim_data_test.dat' # Test data for simulation (replaced by live simulation)
    lat = []
    lon = []
    height = []
    with open(sim_directory, 'r') as file:
        for line in file:
            details = line.strip().split()
            lat.append(float(details[0]))
            lon.append(float(details[1]))
            height.append(float(details[2]))

    return lat, lon, height