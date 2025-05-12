from PyQt5 import QtCore
from PyQt5.QtWidgets import QWidget

import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise

from visualiser.V3.simulator.simulator import lat_long_height
from visualiser.V3.debug import debug_print
from visualiser.V3.partials.coordinate_conversions import do_conversions, radM2eci

import sys
import threading

np.random.seed(0)

class Helper(QtCore.QObject):
    changedSignal = QtCore.pyqtSignal(dict, tuple)

class Predictor(QWidget):
    def __init__(self, grapher, earth, state0, dt=50.0, Cd=2.0):
        super().__init__()
        self.grapher = grapher
        self.earth = earth

        """state0 :list. e.g. state0=[EARTH_SEMIMAJOR + 400e3, 1, -1, 1, 7700, 1] 
            """

        """ =============== Generate sigma points """
        ### initialise self.ukf
        self.sigmas_generator = MerweScaledSigmaPoints(n=7, alpha=0.1, beta=2., kappa=-4.)  # kappa = 3-n.
        self.ukf = UKF(dim_x=7, dim_z=6, fx=f_with_Cd, hx=None, dt=dt, points=self.sigmas_generator)

        """ ============== Define items in self.ukf """
        ### initial state values of (x,y,z,vx,vy,vz, Cd)
        state0.append(Cd)
        self.ukf.x = np.array(state0)  # set dummy initial state. this is the true starting position of the satellite
        print(f'=============== Satellite starts at: {self.ukf.x}')
        ### initial uncertainty of the state
        self.ukf.P = np.diag([50000 ** 2, 50000 ** 2, 50000 ** 2, 100 ** 2, 100 ** 2, 100 ** 2, 0.5 ** 2])
        ### uncertainty in the process model
        self.ukf.Q = ukf_Q_7dim(dim=7, dt=dt, var_=0.001, Cd_var=1e-6)

        ### uncertainty in the measurement
        range_std = 100  # meters.
        vel_std = 0.05
        self.ukf.R = np.diag([range_std ** 2, azim_std ** 2, elev_std ** 2,
                              vel_std ** 2, vel_std ** 2, vel_std ** 2])

        self.xs_prior = []
        self.zs = []
        self.xs = []
        self.Ps = []
        self.ts = [0.0]


        first_update = (plot_x, plot_y)

        self.grapher_helper = Helper()
        self.grapher_helper.changedSignal.connect(self.grapher.update_plots, QtCore.Qt.QueuedConnection)
        threading.Thread(target=send_to_graph, args=(self.grapher_helper, {'shape': (3, 3)}, first_update), daemon=True).start()  # Target will be GRAPHS

        self.earth_helper = Helper()
        self.earth_helper.changedSignal.connect(self.earth.update_prediction, QtCore.Qt.QueuedConnection)
        threading.Thread(target=send_to_graph, args=(self.earth_helper, {'predicting-landing': False, "time": 0.}, (0, 0)), daemon=True).start()

        self.Cd = Cd

    @QtCore.pyqtSlot(dict, tuple)
    def predictor_loop(self, info, update):

        debug_print("predictor", f"{info['obs-time']}, dt={self.dt}")
        self.ts.append(info['obs-time'])
        radar_z = list(update)
        debug_print("predictor", f'dimension of radar z:  {len(radar_z)} \nRadar measurement = {radar_z}')

        if info['obs-time'] == 0:
            init_pos = radM2eci(radM=update, stime=stime, radar=radobj)  # get the radar z of position in ECI
            debug_print("predictor", f'first radar z position (ECI): {init_pos}. type is: {type(init_pos)}')
            init_vel = radar_z[3:]
            debug_print("predictor", f'first radar z velocity (ECI): {init_vel}. type is: {type(init_vel)}')
            init_state = np.concatenate([init_pos, init_vel, [self.Cd]]).tolist()  # complete the initial state
            debug_print("predictor", f'first radar z (ECI): {init_state}. type is: {type(init_state)}')
            self.ukf.x = init_state
            debug_print('predictor', f'first state of ukf is: {self.ukf.x}')

        else:
            ### Start predicting ====================================================
            self.ukf.hx = lambda x: do_conversions(x[:6], stime, radobj)  # update hx according to radar pos
            self.ukf.predict(dt=self.dt)
            x_prior = self.ukf.x_prior.copy()
            debug_print("predictor", f'x_prior is: {x_prior}')

            ### Decide whether to update ============================================
            ###### if no radar measurement, don't update.
                debug_print("predictor", 'NO RADAR!!!!!!!!!!!!!!!!!!!!!')


                x_cov = self.ukf.P_post.copy()
                # debug_print("predictor", f'P_post is:, {x_cov}')

                self.dt_adjust = self.dt_adjust + self.dt
                debug_print("predictor", f'increased dt = {self.dt_adjust}')

                # self.ukf.Q = ukf_Q_7dim(dim=7, dt=self.dt_adjust, var_=0.001, Cd_var=1e-6)
                debug_print("predictor", f'Q = {self.ukf.Q}')

            ###### if has radar measurement, update.
            else:
                self.dt_adjust = self.dt  # reset self.dt_adjust
                self.latest_measurement_time = self.ts[-1]  #
                debug_print("predictor", f'RADAR Z: {radar_z}')

                ### Update
                self.ukf.update(radar_z)
                x_cov = self.ukf.P_post.copy()
                # debug_print("predictor", f'P_post is:, {x_cov}')
                # self.ukf.Q =ukf_Q_7dim(dim=7, dt=self.dt_adjust, var_=0.001, Cd_var=1e-6)
                debug_print("predictor", f'Q = {self.ukf.Q}')

        altitude_prior = lat_long_height(x_prior[0], x_prior[1], x_prior[2])[2]

            radar_z_pos_ECI = radM2eci(radM=update, stime=stime, radar=radobj)
            altitude_z = lat_long_height(radar_z_pos_ECI[0], radar_z_pos_ECI[1], radar_z_pos_ECI[2])[2]
            debug_print("predictor", f"Altitude of measurement: {altitude_z}")
                #
        if altitude_post < 0:
            sys.exit()

            ### sample from the updated state distribution and predict landing position
            start = self.latest_measurement_time
            end = start + 10000000000
            stop_condition.terminal = True
            stop_condition.direction = -1
            landing_latlon_arr = []
            landing_time_arr = []
            for sample in state_samples:
                landing = solve_ivp(fun=ode_with_Cd, t_span=[start, end], y0=sample, method='RK45', t_eval=[end],
                                    max_step=50, events=stop_condition)
                landing_position = landing.y_events[0][0][:3]
                landing_time = landing.t_events
                # print('landing_position sample = \n',landing_position)
                landing_latlon = lat_long_height(landing_position[0], landing_position[1], landing_position[2])[:2]
                # print('landing_position sample = \n',landing_latlon)
                landing_latlon_arr.append(landing_latlon)
                landing_time_arr.append(landing_time)
            # print('landing_position ALL = \n',landing_latlon_arr)
            landing_latlon_mean = np.mean(landing_latlon_arr, axis=0)
            landing_latlon_cov = np.cov(np.array(landing_latlon_arr).T)

            landing_time_mean = np.mean(landing_time_arr)
            # landing_position_latlon = lat_long_height(landing_position_mean[0], landing_position_mean[1],
            #                                         landing_position_mean[2])[:2]

            earth_update = (landing_latlon_mean[0], landing_latlon_mean[1])

            ### Should it be landing.t_events (ODE solved landing time) instead?
            # send_to_graph(self.earth_helper, {'predicting-landing': True, "time": self.latest_measurement_time}, earth_update)
            send_to_graph(self.earth_helper, {'predicting-landing': True, "time": landing_time_mean}, earth_update)







            pred_update = (plot_x, plot_y)
            send_to_graph(self.grapher_helper, {'shape': (3,3)}, pred_update)

def send_to_graph(helper, name:dict, update:tuple):
    helper.changedSignal.emit(name, update)