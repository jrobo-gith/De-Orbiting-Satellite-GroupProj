from PyQt5 import QtCore
from PyQt5.QtWidgets import QWidget

import numpy as np
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.common import Q_discrete_white_noise

from visualiser.V2.simulator_files.Py_Simulation_Jai_Testing import lat_long_height, EARTH_SEMIMAJOR
from visualiser.V2.predictor_files.predictor_UseThisToIntegrateWithVisualiser_v1 import ode, stop_condition, solve_ivp, f, h_radar, ukf_Q

from visualiser.V2.simulator_files. sat_tracking import do_conversions

from datetime import datetime, timedelta
import sys
import threading
import matplotlib.pyplot as plt

class Helper(QtCore.QObject):
    changedSignal = QtCore.pyqtSignal(dict, tuple)

class Predictor(QWidget):
    def __init__(self, grapher, state0, dt=1.0):
        super().__init__()
        self.grapher = grapher

        """state0 :list. e.g. state0=[EARTH_SEMIMAJOR + 400e3, 1, -1, 1, 7700, 1] 
            """

        """ =============== Generate sigma points """
        ### initialise self.ukf
        sigmas_generator = MerweScaledSigmaPoints(n=6, alpha=0.1, beta=2., kappa=-3.)  # kappa = -3.
        self.ukf = UKF(dim_x=6, dim_z=3, fx=f, hx=h_radar, dt=dt, points=sigmas_generator)  # take f, h from Jai and Vijay
        # print(self.ukf.Q)

        """ ============== Define items in self.ukf """
        ### initial state values of (x,y,z,vx,vy,vz)
        self.ukf.x = np.array(state0)  # initial state
        ### initial uncertainty of the state
        self.ukf.P = np.diag([50 ** 2, 50 ** 2, 50 ** 2,
                         5 ** 2, 5 ** 2, 5 ** 2])  # experiment this
        ### uncertainty in the process model
        self.ukf.Q = ukf_Q(6, dt=dt, var_=0.01)
        # self.ukf.Q = np.zeros((6, 6))
        # self.ukf.Q[np.ix_([0, 3], [0, 3])] = Q_discrete_white_noise(dim=2, dt=dt,
        #                                                        var=0.01)  # Q matrix for how other noise affect x and vx
        # self.ukf.Q[np.ix_([1, 4], [1, 4])] = Q_discrete_white_noise(dim=2, dt=dt,
        #                                                        var=0.01)  # Q matrix for how other noise affect y and vy
        # self.ukf.Q[np.ix_([2, 5], [2, 5])] = Q_discrete_white_noise(dim=2, dt=dt,
        #                                                        var=0.01)  # Q matrix for how other noise affect z and vz

        # range_std = 10 # meters. change this!!!!!!!!!!!!!!!!!!!!!! (get from radar)
        # elev_std = math.radians(1)  # 1 degree in radians. change this!!!!!!!!!!!!!!!!!!!!!! (get from radar)
        # azim_std = math.radians(1)  # 1 degree in radians. change this!!!!!!!!!!!!!!!!!!!!!1 (get from radar)
        # self.ukf.R = np.diag([range_std**2, elev_std**2, azim_std**2])

        """### radar measurement noise (for the simple self.ukf only! change this!!!!!!!!!!!!!!!!!!!!!!)"""
        x_std = 500  # meters.
        y_std = 500  # meters.
        z_std = 10  # meters.
        self.ukf.R = np.diag([x_std**2, y_std**2, z_std**2])

        # range_std = 10  # meters.
        # azim_std = 0.002  # radians. (theta)
        # elev_std = 0.002  # radians. (phi)
        # self.ukf.R = np.diag([range_std ** 2, azim_std ** 2, elev_std ** 2])

        self.xs_prior = []
        self.zs = []
        self.xs = []
        self.Ps = []
        self.ts = [0.0]

        ## Setup thread to send to graphs
        self.grapher_helper = Helper()
        self.grapher_helper.changedSignal.connect(self.grapher.update_plots, QtCore.Qt.QueuedConnection)
        threading.Thread(target=send_to_graph, args=(self.grapher_helper, {'shape': (3,3)}, (np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]), np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]]))),
                         daemon=True).start()  # Target will be GRAPHS

        self.time = np.linspace(0, 10_000, 10_000)
        self.count = 0


    @QtCore.pyqtSlot(dict, tuple)
    def predictor_loop(self, info, update):
        # print(info['name'], " had distance ", info['rdist'], " from sat at ", info['obs-time'], 's.' '\n')

        # ## PROBABLY TEMPORARY DT
        dt = 50.
        time, stime, radobj = info['obs-time'], info['stime'], info['radobj']
        current_time = info['obs-time']

        self.ukf.Q = ukf_Q(dim=6, dt=dt, var_=0.01)
        self.ukf.predict(dt=dt)
        # # self.ukf.hx = lambda x: do_conversions(x[:3], stime, radobj)
        xs_prior = self.ukf.x_prior
        if update != (0, 0, 0):
            self.ukf.update(list(update))
        x_post = self.ukf.x
        x_cov = self.ukf.P

        plot1_x = np.array([[current_time, current_time, current_time],
                            [current_time, current_time, current_time],
                            [current_time, current_time, current_time]])

        plot1_y = np.array([update, xs_prior[:3], x_post[:3]])

        if info['rdist'] != 'none':
            pred_update = (plot1_x, plot1_y)
            send_to_graph(self.grapher_helper, {'shape': (3, 3)}, pred_update)
        self.count += 1


        # if info['rdist'] != 'none':
        #     inf = [info['rdist'], np.linalg.norm(list(update))]
        #     time = [self.time[self.count], self.time[self.count]]
        #     pred_update = ([time], [inf])
        #     # SEND DATA TO GRAPHER
        #     send_to_graph(self.grapher_helper, {'name': 'redundant-name'}, pred_update)

        # """Predict landing ====================================================================="""
        # altitude_val = lat_long_height(x_post[0], x_post[1], x_post[2])[2]

        # print("radar height: ", np.abs(np.linalg.norm(list(update)))-EARTH_SEMIMAJOR)
        # print("altitude=", altitude_val, "\n")
        #
        # if altitude_val < 0:
        #     sys.exit()
        #
        #
        # if altitude_val <= 200e3:
        #     # print("============== predict landing ================")
        #     ### sample from the updated state distribution and predict landing position
        #     state_samples = np.random.multivariate_normal(mean=x_post, cov=x_cov, size=5)
        #
        #     ### record the landing position (inertia coord), and landing time
        #     predicted_landing_ECI = []
        #     predicted_landing_latlon = []
        #     predicted_landing_time = []
        #
        #     for state0 in state_samples:
        #         # t_eval_arr = np.linspace(ts[-1], ts[-1]+100000, 1000)
        #         stop_condition.terminal = True
        #         stop_condition.direction = -1
        #         start = self.ts[-1]
        #         end = start + 1000000
        #         landing = solve_ivp(fun=ode, t_span=[start, end], y0=state0, method='RK45', t_eval=[end],
        #                             events=stop_condition)
        #         while (landing.success and len(landing.y) > 0):
        #             end += 1000000
        #             landing = solve_ivp(fun=ode, t_span=[start, end], y0=state0, method='RK45', t_eval=[end],
        #                                 events=stop_condition)
        #         if landing.success and len(landing.y) == 0:
        #             # print(landing.y_events, np.linalg.norm(landing.y_events[0][0][:3])-R_EARTH)
        #             # print(landing.y_events[0][0][:3])
        #             landing_time = landing.t_events[0][0]
        #             landing_position = landing.y_events[0][0][:3]
        #             landing_position_latlon = lat_long_height(landing_position[0], landing_position[1],
        #                                                       landing_position[2])
        #             predicted_landing_ECI.append(landing_position)
        #             predicted_landing_latlon.append(landing_position_latlon)
        #             predicted_landing_time.append(landing_time)


def send_to_graph(helper, name:dict, update:tuple):
    helper.changedSignal.emit(name, update)