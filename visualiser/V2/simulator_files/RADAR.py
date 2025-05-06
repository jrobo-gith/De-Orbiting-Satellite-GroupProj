from visualiser.V2.simulator_files.FULL_SIM import run_full_simulation
import time
import numpy as np

def radar(helper, name, latlon_data): # Will most likely need two helpers, one to go to kalman filter, one to update the visualiser.
    """
    Gives predictor noisy simulation outputs, updates the predictor and GUI with data.
    """
    # Run full sim
    lat, lon = latlon_data
    lat = (180 / np.pi) * lat
    lon = (180 / np.pi) * lon

    x = np.linspace(0, 5400, lat.shape[0])

    for i in range(1, len(lat)):
        outgoing_x = [x[i]]
        outgoing_lat = [lat[i]]
        outgoing_lon = [lon[i]]
        helper.changedSignal.emit(name, (outgoing_x, outgoing_lat, outgoing_lon))
        time.sleep(1)

