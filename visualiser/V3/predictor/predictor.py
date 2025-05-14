from PyQt5 import QtCore
from PyQt5.QtWidgets import QWidget

import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints

from visualiser.V3.simulator.simulator import lat_long_height
from visualiser.V3.simulator.radar import radars
from visualiser.V3.predictor.predctor_functions import stop_condition, solve_ivp, ukf_Q_7dim, ode_with_Cd, f_with_Cd
from visualiser.V3.partials.coordinate_conversions import do_conversions, radM2eci
from visualiser.V3.partials.constants import CD, toHrs, toKM

import sys
import threading

class Helper(QtCore.QObject):
    changedSignal = QtCore.pyqtSignal(dict, tuple)

class Predictor(QWidget):
    def __init__(self, grapher, earth, state0, dt=50.0, Cd=2.0):
        super().__init__()
        self.grapher = grapher
        self.earth = earth

        """state0 :list. e.g. state0=[EARTH_SEMIMAJOR + 400e3, 1, -1, 1, 7700, 1] """

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
        self.ukf.Q = ukf_Q_7dim(dim=7, dt=dt, var_=1e-6, Cd_var=1e-6)

        ### uncertainty in the measurement
        range_std = 100  # meters.
        azim_std = 0.001  # radians. (theta)
        elev_std = 0.001  # radians. (phi)
        vel_std = 0.05
        self.ukf.R = np.diag([range_std ** 2, azim_std ** 2, elev_std ** 2,
                              vel_std ** 2, vel_std ** 2, vel_std ** 2])

        self.xs_prior = []
        self.zs = []
        self.xs = []
        self.Ps = []
        self.ts = [0.0]

        position_x = [0, 0, 0, 0]
        position_y = [0, 0, 0, 0]

        velocity_x = [0, 0, 0]
        velocity_y = [0, 0, 0]

        residual_x = [0, 0, 0]
        residual_y = [0, 0, 0]

        cov_x = [0]
        cov_y = [0]

        drag_x = [0, 0]
        drag_y = [2, 2]

        alt_x = [0, 0]
        alt_y = [0, 0]

        plot_x = [position_x, velocity_x, cov_x, residual_x, drag_x, alt_x]
        plot_y = [position_y, velocity_y, cov_y, residual_y, drag_y, alt_y]

        first_update = (plot_x, plot_y)

        self.grapher_helper = Helper()
        self.grapher_helper.changedSignal.connect(self.grapher.update_plots, QtCore.Qt.QueuedConnection)
        threading.Thread(target=send_to_graph, args=(self.grapher_helper, {'shape': (3, 3)}, first_update),
                         daemon=True).start()  # Target will be GRAPHS

        self.x_cov = np.zeros((3, 3))

        self.earth_helper = Helper()
        self.earth_helper.changedSignal.connect(self.earth.update_prediction, QtCore.Qt.QueuedConnection)
        threading.Thread(target=send_to_graph,
                         args=(self.earth_helper, {'predicting-landing': False, "time": 0.}, (0, 0, self.x_cov)),
                         daemon=True).start()

        self.Cd = Cd
        self.dt = dt
        self.dt_adjust = dt

        self.x_prior = self.ukf.x
        self.x_post = self.ukf.x

    @QtCore.pyqtSlot(dict, tuple)
    def predictor_loop(self, info, update):

        self.update = update

        pred_alt = 140e3
        self.ts.append(info['obs-time'])
        stime, radar_name = info['stime'], info['name']
        radobj = radars[radar_name] if radar_name != "no radar" else None
        radar_z = list(update)

        ### at time 0, take the radar measurement as state0 for UKF. A radar measurement at t=0 must be provided!
        if info['obs-time'] == 0:
            init_pos = radM2eci(radM=update, stime=stime, radar=radobj)  # get the radar z of position in ECI
            init_vel = radar_z[3:]
            init_state = np.concatenate([init_pos, init_vel, [self.Cd]]).tolist()  # complete the initial state
            self.ukf.x = init_state

            self.x_post = self.ukf.x.copy()
        ### at time > 0, start UKF process
        else:
            ### Start predicting ====================================================
            self.ukf.hx = lambda x: do_conversions(x[:6], stime, radobj)  # update hx according to radar pos
            self.ukf.predict(dt=self.dt)
            self.x_prior = self.ukf.x_prior.copy()

            ### Decide whether to update ============================================
            ###### if no radar measurement, don't update.
            if update == (0, 0, 0, 0, 0, 0):

                self.x_post = self.ukf.x_post.copy()
                self.x_cov = self.ukf.P_post.copy()
                self.dt_adjust = self.dt_adjust + self.dt
            ###### if has radar measurement, update.
            else:
                self.dt_adjust = self.dt  # reset self.dt_adjust
                self.latest_measurement_time = self.ts[-1]  #

                ### Update
                self.ukf.update(radar_z)
                self.x_post = self.ukf.x_post
                self.x_cov = self.ukf.P_post.copy()

        altitude_post = lat_long_height(self.x_post[0], self.x_post[1], self.x_post[2])[2]
        radar_z_pos_ECI = (0, 0, 0, 0, 0, 0)

        if update != (0, 0, 0, 0, 0, 0):
            radar_z_pos_ECI = radM2eci(radM=update, stime=stime, radar=radobj)
            altitude_z = lat_long_height(radar_z_pos_ECI[0], radar_z_pos_ECI[1], radar_z_pos_ECI[2])[2]

        if altitude_post < 0:
            sys.exit()

        # Predict landing ====================================================================="""

        ### start predict landing if altitude of x_prior is below threshold.
        ### this is in case we don't receive radar measurement around threshold altitude.
        if altitude_post <= pred_alt and update != (0, 0, 0, 0, 0, 0):
            ### sample from the updated state distribution and predict landing position
            state_samples = np.random.multivariate_normal(mean=self.x_post, cov=self.x_cov, size=20)
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
                landing_latlon = lat_long_height(landing_position[0], landing_position[1], landing_position[2])[:2]
                landing_latlon_arr.append(landing_latlon)
                landing_time_arr.append(landing_time)

            landing_latlon_mean = np.mean(landing_latlon_arr, axis=0)
            landing_latlon_cov = np.cov(np.array(landing_latlon_arr).T)
            landing_time_mean = np.mean(landing_time_arr)
            earth_update = (landing_latlon_mean[0], landing_latlon_mean[1], landing_latlon_cov)

            ### Should it be landing.t_events (ODE solved landing time) instead?
            send_to_graph(self.earth_helper, {'predicting-landing': True, "time": landing_time_mean}, earth_update)

        if info['name'] != 'no radar':
            time_hrs = self.ts[-1] * toHrs

            position_x = [info['state_no_noise'][0] * toKM, info['state_noise'][0] * toKM, self.x_prior[0] * toKM,
                          self.x_post[0] * toKM]
            position_y = [info['state_no_noise'][1] * toKM, info['state_noise'][1] * toKM, self.x_prior[1] * toKM,
                          self.x_post[1] * toKM]

            velocity_x = [info['state_no_noise'][3] * toKM, self.x_prior[3] * toKM, self.x_post[3] * toKM]
            velocity_y = [info['state_no_noise'][4] * toKM, self.x_prior[4] * toKM, self.x_post[4] * toKM]

            covariance_trace = self.x_cov[0, 0] + self.x_cov[1, 1] + self.x_cov[2, 2]

            cov_x = [time_hrs]
            cov_y = [covariance_trace]

            prior_residual = np.linalg.norm([self.x_prior[0], self.x_prior[1], self.x_prior[2]]) - np.linalg.norm(
                [info['state_no_noise'][0], info['state_no_noise'][1], info['state_no_noise'][2]])

            post_residual = np.linalg.norm([self.x_post[0], self.x_post[1], self.x_post[2]]) - np.linalg.norm(
                [info['state_no_noise'][0], info['state_no_noise'][1], info['state_no_noise'][2]])

            residual_x = [time_hrs, time_hrs, time_hrs]
            residual_y = [prior_residual, post_residual, 0]

            drag_x = [time_hrs, time_hrs]
            drag_y = [self.x_post[6], CD]

            alt_x = [time_hrs, time_hrs]
            alt_y = [altitude_post * toKM, 140e3 * toKM]

            plot_x = [position_x, velocity_x, cov_x, residual_x, alt_x, drag_x]
            plot_y = [position_y, velocity_y, cov_y, residual_y, alt_y, drag_y]

            pred_update = (plot_x, plot_y)
            send_to_graph(self.grapher_helper, {'shape': (3, 3)}, pred_update)

    @QtCore.pyqtSlot(str, tuple)
    def send_prediction(self, redund_name, redund_tuple):
        state_samples = np.random.multivariate_normal(mean=self.x_post, cov=self.x_cov, size=5)
        start = self.ts[-1]
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
            landing_latlon = lat_long_height(landing_position[0], landing_position[1], landing_position[2])[:2]
            landing_latlon_arr.append(landing_latlon)
            landing_time_arr.append(landing_time)
        landing_latlon_mean = np.mean(landing_latlon_arr, axis=0)
        landing_latlon_cov = np.cov(np.array(landing_latlon_arr).T)

        earth_update = (landing_latlon_mean[0], landing_latlon_mean[1], landing_latlon_cov)
        send_to_graph(self.earth_helper, {'predicting-landing': True, "time": self.ts[-1]}, earth_update)

def send_to_graph(helper, name: dict, update: tuple):
    helper.changedSignal.emit(name, update)