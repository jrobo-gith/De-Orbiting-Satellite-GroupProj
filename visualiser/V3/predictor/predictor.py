from PyQt5 import QtCore
from PyQt5.QtWidgets import QWidget

import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise

from visualiser.V3.simulator.simulator import lat_long_height
from visualiser.V3.predictor.predctor_functions import ode, stop_condition, solve_ivp, f, h_radar, ukf_Q
from visualiser.V3.debug import debug_print
from visualiser.V3.partials.coordinate_conversions import do_conversions, radM2eci

import sys
import threading

np.random.seed(0)

class Helper(QtCore.QObject):
    changedSignal = QtCore.pyqtSignal(dict, tuple)

class Predictor(QWidget):
    def __init__(self, grapher, earth, state0, dt=50.0):
        super().__init__()
        self.grapher = grapher
        self.earth = earth

        """state0 :list. e.g. state0=[EARTH_SEMIMAJOR + 400e3, 1, -1, 1, 7700, 1] 
            """

        """ =============== Generate sigma points """
        ### initialise self.ukf
        sigmas_generator = MerweScaledSigmaPoints(n=6, alpha=0.1, beta=2., kappa=-3.)  # kappa = -3.
        self.ukf = UKF(dim_x=6, dim_z=3, fx=f, hx=None, dt=dt, points=sigmas_generator)  # take f, h from Jai and Vijay
        # debug_print("predictor", self.ukf.Q)

        """ ============== Define items in self.ukf """
        ### initial state values of (x,y,z,vx,vy,vz)
        self.ukf.x = np.array(state0)  # set dummy initial state. this is the true starting position of the satellite
        print(f'satellite starts at: {self.ukf.x}')
        ### initial uncertainty of the state
        self.ukf.P = np.diag([50000 ** 2, 50000 ** 2, 5 ** 2, 1 ** 2, 1 ** 2, 1 ** 2])
        ### uncertainty in the process model
        self.ukf.Q = ukf_Q(6, dt=dt, var_=0.001)


        """### radar measurement noise (for the simple self.ukf only! change this!!!!!!!!!!!!!!!!!!!!!!)"""
        # x_std = 5  # meters.
        # y_std = 5  # meters.
        # z_std = 1  # meters.
        # self.ukf.R = np.diag([x_std**2, y_std**2, z_std**2])

        range_std = 100  # meters.
        azim_std = 0.01  # radians. (theta)
        elev_std = 0.01  # radians. (phi)
        vel_std = 0.05
        self.ukf.R = np.diag([range_std ** 2, azim_std ** 2, elev_std ** 2,
                              vel_std**2, vel_std**2, vel_std**2])

        self.xs_prior = []
        self.zs = []
        self.xs = []
        self.Ps = []
        self.ts = [0.0]

        position_x = [0, 0, 0]
        position_y = [0, 0, 0]

        velocity_x = [0, 0]
        velocity_y = [0, 0]

        alt_x = [0, 0, 0]
        alt_y = [0, 0, 0]

        plot_x = [position_x, velocity_x, alt_x]
        plot_y = [position_y, velocity_y, alt_y]

        first_update = (plot_x, plot_y)

        self.grapher_helper = Helper()
        self.grapher_helper.changedSignal.connect(self.grapher.update_plots, QtCore.Qt.QueuedConnection)
        threading.Thread(target=send_to_graph, args=(self.grapher_helper, {'shape': (3, 3)}, first_update), daemon=True).start()  # Target will be GRAPHS

        self.earth_helper = Helper()
        self.earth_helper.changedSignal.connect(self.earth.update_prediction, QtCore.Qt.QueuedConnection)
        threading.Thread(target=send_to_graph, args=(self.earth_helper, {'predicting-landing': False, "time": 0.}, (0, 0)), daemon=True).start()

        self.dt = 50
        self.dt_adjust = 50

    @QtCore.pyqtSlot(dict, tuple)
    def predictor_loop(self, info, update):

        debug_print("predictor", f"{info['obs-time']}, dt={self.dt}")
        self.ts.append(info['obs-time'])
        stime, radobj = info['stime'], info['radobj']

        radar_z = list(update)
        debug_print("predictor", f'dimension of radar z:  {len(radar_z)} \nRadar measurement = {radar_z}')

        ### at time 0, take the radar measurement as state0 for UKF
        if info['obs-time'] == 0:
            # init_pos = polar2ECI(radar_z, radar_name)
            init_pos = radM2eci(radM=update, stime=stime, radar=radobj)  # get the radar z of position in ECI
            debug_print("predictor", f'first radar z position (ECI): {init_pos}. type is: {type(init_pos)}')
            init_vel = radar_z[3:]
            debug_print("predictor", f'first radar z velocity (ECI): {init_vel}. type is: {type(init_vel)}')
            init_state = np.append(init_pos,
                                   radar_z[3:]).tolist()  # append the rest of radar z to complete the initial state
            debug_print("predictor", f'first radar z (ECI): {init_state}. type is: {type(init_state)}')
            self.ukf.x = init_state
            debug_print('predictor', f'first state of ukf is: {self.ukf.x}')

        ### at time > 0, start UKF process
        else:
            ### Start predicting ====================================================
            self.ukf.hx = lambda x: do_conversions(x[:6], stime, radobj)  # update hx according to radar pos
            self.ukf.predict(dt=self.dt)
            x_prior = self.ukf.x_prior.copy()
            debug_print("predictor", f'x_prior is: {x_prior}')

            ### Decide whether to update ============================================
            ###### if no radar measurement, don't update.
            if update == (0, 0, 0):
                debug_print("predictor", 'NO RADAR!!!!!!!!!!!!!!!!!!!!!')

                x_post = self.ukf.x_post.copy()
                debug_print("predictor", f'x_post is: {x_post}')

                x_cov = self.ukf.P_post.copy()
                # debug_print("predictor", f'P_post is:, {x_cov}')

                self.dt_adjust = self.dt_adjust + self.dt
                debug_print("predictor", f'increased dt = {self.dt_adjust}')

                self.ukf.Q = ukf_Q(6, dt=self.dt_adjust, var_=0.001)
                debug_print("predictor", f'Q = {self.ukf.Q}')

            ###### if has radar measurement, update.
            else:
                self.dt_adjust = self.dt  # reset self.dt_adjust
                self.latest_measurement_time = self.ts[-1]  #
                debug_print("predictor", f'RADAR Z: {radar_z}')

                self.ukf.update(radar_z)
                x_post = self.ukf.x_post
                debug_print("predictor", f'x_post is: {x_post}')
                x_cov = self.ukf.P_post.copy()
                # debug_print("predictor", f'P_post is:, {x_cov}')

        # """Predict landing ====================================================================="""
        altitude_post = lat_long_height(x_post[0], x_post[1], x_post[2])[2]

        # debug_print("predictor", f"radar height: {np.abs(np.linalg.norm(list(update)))-EARTH_SEMIMAJOR}")
        if update != (0, 0, 0):
            radar_z_pos_ECI = radM2eci(radM=update, stime=stime, radar=radobj)
            debug_print("predictor",
                        f"Altitude of measurement: {lat_long_height(radar_z_pos_ECI[0], radar_z_pos_ECI[1], radar_z_pos_ECI[2])[2]}")
        debug_print("predictor", f"Altitude of x_post: {altitude_post}\n")
                #
        if altitude_post < 0:
            sys.exit()


        if altitude_post <= 140e3:
            # debug_print("predictor", "============== predict landing ================")
            ### sample from the updated state distribution and predict landing position
            # state_samples = np.random.multivariate_normal(mean=x_post, cov=x_cov, size=5)
            start = self.latest_measurement_time
            end = start + 10000000000
            stop_condition.terminal = True
            stop_condition.direction = -1
            landing = solve_ivp(fun=ode, t_span=[start, end], y0=x_post, method='RK45', t_eval=[end],
                                max_step=50, events=stop_condition)
            landing_position = landing.y_events[0][0][:3]
            landing_position_latlon = lat_long_height(landing_position[0], landing_position[1],
                                                      landing_position[2])[:2]

            earth_update = (landing_position_latlon[0], landing_position_latlon[1])
            send_to_graph(self.earth_helper, {'predicting-landing': True, "time": self.latest_measurement_time}, earth_update)
            # for sample_state0 in state_samples:
            #     stop_condition.terminal = True
            #     stop_condition.direction = -1
                # landing = solve_ivp(fun=ode, t_span=[start, end], y0=sample_state0, method='RK45', t_eval=[end],
                #                     max_step=10, events=stop_condition)
                # debug_print("predictor", f'next step before landing: {landing.y}')

        #
        # #    ### record the landing position (inertia coord), and landing time
        #     predicted_landing_ECI = []
        #     predicted_landing_latlon = []
        #     predicted_landing_time = []
        #     predicted_landing_height = []
        # #
        #     debug_print("predictor", f"time start landing: {self.ts[-1]}')
        #     for sample_state0 in state_samples:
        #         # t_eval_arr = np.linspace(ts[-1], ts[-1]+100000, 1000)
        #         start = self.latest_measurement_time
        #         end = start + 100_000_000
        #         debug_print("predictor", "\nSOLVING ODE")
        #         landing = solve_ivp(fun=ode, t_span=[start, end], y0=sample_state0, method='RK45', t_eval=[end], max_step = 50,
        #                             events=stop_condition)
        #         debug_print("predictor", "FINISHED SOLVING ODE\n")
        #         # while (landing.success and len(landing.y) > 0):
        #         #     end += 10_000_000
        #         #     landing = solve_ivp(fun=ode, t_span=[start, end], y0=state0, method='RK45', t_eval=[end], max_step = dt,
        #         #                         events=stop_condition)
        #         if landing.success and len(landing.y) == 0:
        #             # debug_print("predictor", f'landing.y_events: {np.linalg.norm(landing.y_events[0][0][:3])-R_EARTH}')
        #             # debug_print("predictor", landing.y_events[0][0][:3])
        #             landing_time = landing.t_events[0][0]       #landing time
        #             landing_position = landing.y_events[0][0][:3]
        #             # landing_position_latlon = lat_long_height(landing_position[0], landing_position[1],
        #             #                                           landing_position[2])[:2]
        #             landing_position_latlon = ECI2latlon_earth_rotate(landing_position[0], landing_position[1],landing_position[2], time_duration=landing_time)
        #             # debug_print("predictor", f'landing position is: {landing_position}')
        #             landing_height = lat_long_height(landing_position[0], landing_position[1], landing_position[2])[2]
        #             predicted_landing_ECI.append(landing_position)
        #             predicted_landing_latlon.append(landing_position_latlon)
        #             predicted_landing_time.append(landing_time)
        #             predicted_landing_height.append(landing_height)



            # debug_print("predictor", f'landing positions are (lat lon degrees): {np.array(predicted_landing_latlon)}')
            # debug_print("predictor", f'landing heights are: {predicted_landing_height}')
            # debug_print("predictor", f'landing times are: {predicted_landing_time}')
            #
            #
            # mean_landing_latlon = np.mean(np.array(predicted_landing_latlon), axis=0)
            # cov_landing_latlon = np.cov(np.array(predicted_landing_latlon).T)
            # debug_print("predictor", f"mean_landing_latlon: {mean_landing_latlon}") # mean of the landing lat long
            # debug_print("predictor", f"cov_landing_latlon: {cov_landing_latlon}\n")  # covariance matrix of the landing lat long
            
            # debug_print("predictor", f"We are getting out: {x_post}")
            # debug_print("predictor", f"{self.ts}, {self.xs}")

        if info['rdist'] != 'none':
            position_x = [info['state_no_noise'][0], x_prior[0], x_post[0]]
            position_y = [info['state_no_noise'][1], x_prior[1], x_post[1]]

            velocity_x = [x_prior[3], x_post[3]]
            velocity_y = [x_prior[4], x_post[4]]

            alt_x = [self.ts[-1], self.ts[-1], self.ts[-1]]
            alt_y = [altitude_post, 50e3, 140e3]

            plot_x = [position_x, velocity_x, alt_x]
            plot_y = [position_y, velocity_y, alt_y]

            pred_update = (plot_x, plot_y)
            send_to_graph(self.grapher_helper, {'shape': (3,3)}, pred_update)

def send_to_graph(helper, name:dict, update:tuple):
    helper.changedSignal.emit(name, update)