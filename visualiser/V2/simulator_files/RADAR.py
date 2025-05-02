from visualiser.V2.simulator_files.FULL_SIM import run_full_simulation
import time

def radar(helper, name): # Will most likely need two helpers, one to go to kalman filter, one to update the visualiser.
    """
    Gives predictor noisy simulation outputs, updates the predictor and GUI with data.
    """
    # Run full sim
    lat, lon, height = run_full_simulation(init_pos=[0, 0, 0],
                                           init_vel=[0, 0, 0])

    for i in range(1, len(lat)):
        outgoing_lat = [lat[i]]
        outgoing_lon = [lon[i]]
        helper.changedSignal.emit(name, (outgoing_lat, outgoing_lon))
        time.sleep(1)

