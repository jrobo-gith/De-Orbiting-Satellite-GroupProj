import os
import sys
import json
import time
import numpy as np
import threading

root_dir = os.getcwd()
sys.path.insert(0, root_dir)

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QStackedWidget, QGridLayout, QLabel, QPushButton
from PyQt5 import QtCore
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

from visualiser.V2.simulator_files.Py_Simulation_Jai_Testing import system_solver, lat_long_height_plot
from visualiser.V2.widgets.graph_stuff.graph_script import Grapher
from visualiser.V2.widgets.graph_stuff.earth_script import Earth
from visualiser.V2.simulator_files.sat_tracking import initialise_radars
from visualiser.V2.simulator_files.sat_tracking import get_radar_measurements
from visualiser.V2.predictor_files.predictor import Predictor

# Import global settings
json_file = os.path.join(root_dir, "visualiser/V2/partials/global_settings.json")
with open(json_file) as f:
    glob_setting = json.load(f)

class Helper(QtCore.QObject):
    """Helper function used to emit data via multi-threading"""
    changedSignal = QtCore.pyqtSignal(dict, tuple)

# Time span
t_span = 10_000_000

class SimWidget(QWidget):
    """
    Window that contains the simulation. This is the first window that is loaded when the user clicks 'run simulation'
    and is a navigational window that connects the graph plots and the earth plots. This window also runs the full
    simulation, a solver taken from simulator_files/Py_Simulation_Jai_Testing, and parses it to earth script for
    plotting.

    This window's parent is the simulation menu in simulation_menu.py, navigable to using the simulation menu under the
    'run simulation' button in the middle.

    Functions:
    - __init__(self, stacked_widget, initial_conditions, radar_list)
    - click_graph_button(self)
    - click_earth_button(self)

    References:
    Tutorial followed for PyQt5 (GUI) can be found here - https://www.pythonguis.com/pyqt5-tutorial/

    Previous versions can be found in the Group GitHub - https://github.com/jrobo-gith/De-Orbiting-Satellite-GroupProj
    """
    def __init__(self, stacked_widget, initial_conditions, radar_list):
        """
        Initialises the SimWidget, runs the simulation based on the initial conditions parsed in through the parent -
        the simulation menu, or optionally, if in development mode, reads from the existing satellite trajectory data
        to make testing (and marking) faster.

        Creates a custom navbar containing the back-arrow to navigate back to the simulation menu, the title of the
        window, and the two buttons, graph_button and earth_button to switch between graph plots and earth plot.

        Establishes a connection to the python file 'sat_tracking.py' and the function 'get_radar_measurements' via
        multithreading to allow the radar to send observations to the predictor in 'predictor_files/predictor.py' and to
        the earth script to update the satellite's position.

        :param stacked_widget: QStackedWidget that contains the main windows such that the user can navigate back to the
                               main menu.
        :param initial_conditions: conditions specified in simulation menu by the user and used to run the full sim.
        :param radar_list: list of radar location in lat and lon and height above sea level, used to plot on earth
                           script.
        """
        super().__init__()
        self.stacked_widget = stacked_widget
        self.initial_conditions = initial_conditions
        self.radar_list = radar_list

        dev_mode = False

        if dev_mode:
            self.x_sim = []
            self.y_sim = []
            self.z_sim = []
            self.t = []
            existing_file = os.path.join(root_dir, "visualiser/V2/simulator_files/sat_traj.dat")
            with open(existing_file, 'r') as file:
                for line in file:
                    x, y, z, vx, vy, vz, t = line.split()
                    self.x_sim.append(float(x))
                    self.y_sim.append(float(y))
                    self.z_sim.append(float(z))
                    self.t.append(float(t))

            self.x_sim = np.array(self.x_sim)
            self.y_sim = np.array(self.y_sim)
            self.z_sim = np.array(self.z_sim)
            self.t = np.array(self.t)
        else:
            ## RUN SIMULATOR TO GET ENTIRE SIMULATION LAT LON DATA
            self.solution = system_solver(t_span, self.initial_conditions)
            ## Use self.solution to compute XYZ coordinates to lat lon NOT INCLUDING rotation of earth
            self.x_sim, self.y_sim, self.z_sim, self.t = self.solution.y[0], self.solution.y[1], self.solution.y[2], self.solution.t

        ## Convert x, y, z to lat lon
        self.lat, self.lon, self.height = lat_long_height_plot(self.x_sim, self.y_sim, self.z_sim)

        ## graph-earth stacked widget
        self.graph_earth_container = QStackedWidget()

        ## NAVBAR
        sim_window_navbar = QGridLayout()
        title = QLabel("Simulation")
        title.setFont(QFont(glob_setting['font-family'], 35))
        title.setStyleSheet(f"color: rgb{glob_setting['font-color']}; text-decoration: underline; background: {glob_setting['background-color']}")

        backarrow = QPushButton("<--")
        backarrow.setStyleSheet(f"color: rgb{glob_setting['font-color']}; background: {glob_setting['background-color']}; text-align: left")
        backarrow.setFont(QFont(glob_setting['font-family'], 35))
        backarrow.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))

        self.key_sim = QLabel("Simulation Data")
        self.key_sim.setFont(QFont(glob_setting['font-family'], 20))
        self.key_sim.setStyleSheet(f"color: rgb(0, 0, 255);")

        self.key_pred = QLabel("Prediction Data")
        self.key_pred.setFont(QFont(glob_setting['font-family'], 20))
        self.key_pred.setStyleSheet(f"color: rgb(0, 255, 0);")

        self.graph_button = QPushButton("Graph View")
        self.graph_button.setStyleSheet(f"color: rgb{glob_setting['font-color']}; text-decoration: underline; background: {glob_setting['background-color']}")
        self.graph_button.setFont(QFont(glob_setting['font-family'], 20))
        self.graph_button.clicked.connect(self.click_graph_button)

        self.earth_button = QPushButton("Earth View")
        self.earth_button.setStyleSheet(f"color: rgb{glob_setting['font-color']}; background: {glob_setting['background-color']}")
        self.earth_button.setFont(QFont(glob_setting['font-family'], 20))
        self.earth_button.clicked.connect(self.click_earth_button)

        filler = QLabel("FILLER")
        filler.setStyleSheet(f"color: {glob_setting['background-color']};")

        sim_window_navbar.addWidget(backarrow, 0, 0, Qt.AlignCenter)
        sim_window_navbar.addWidget(filler, 0, 1, Qt.AlignCenter)
        sim_window_navbar.addWidget(title, 0, 2, Qt.AlignCenter)
        sim_window_navbar.addWidget(self.graph_button, 0, 3, Qt.AlignCenter)
        sim_window_navbar.addWidget(self.earth_button, 0, 4, Qt.AlignCenter)

        sim_window_navbar.addWidget(self.key_sim, 1, 1, Qt.AlignCenter)
        sim_window_navbar.addWidget(self.key_pred, 1, 3, Qt.AlignCenter)

        ## Earth window
        self.earth = Earth(full_sim_data=(self.lat, self.lon, self.t), radar_list=self.radar_list)
        ## graph_script
        self.graph = Grapher()

        ## Predictor TESTING
        self.predictor = Predictor(grapher=self.graph, state0=initial_conditions)

        self.radars = initialise_radars(radar_list)

        self.graph_earth_container.addWidget(self.graph)
        self.graph_earth_container.addWidget(self.earth)

        self.graph_earth_container.setStyleSheet("background: transparent;")

        container = QVBoxLayout()
        container.addLayout(sim_window_navbar, stretch=1)
        container.addWidget(self.graph_earth_container, stretch=18)
        self.setLayout(container)

        # Predictor
        pred_helper = Helper()
        pred_helper.changedSignal.connect(self.predictor.predictor_loop, QtCore.Qt.QueuedConnection)

        # Earth
        earth_helper = Helper()
        earth_helper.changedSignal.connect(self.earth.update_satellite_position, QtCore.Qt.QueuedConnection)

        threading.Thread(target=get_radar_measurements, args=(self.radars, earth_helper, pred_helper), daemon=True).start()


    def click_graph_button(self):
        """
        Function that sits inside the 'self.graph_button' QPushButton and navigates to the graph window when the button
        is clicked. Also underlines the graph script to indicate the graph window is the active window.
        """
        self.graph_earth_container.setCurrentIndex(0)
        self.graph_button.setStyleSheet(f"color: rgb{glob_setting['font-color']}; text-decoration: underline; background: {glob_setting['background-color']}")
        self.earth_button.setStyleSheet(f"color: rgb{glob_setting['font-color']}; background: {glob_setting['background-color']}")
        self.key_sim.setStyleSheet(f"color: rgb(0, 0, 255);")
        self.key_pred.setStyleSheet(f"color: rgb(0, 255, 0);")

    def click_earth_button(self):
        """
        Function that sits inside the 'self.earth_button' QPushButton and navigates to the earth window when the button
        is clicked. Also underlines the earth script to indicate the earth window is the active window.
        """
        self.graph_earth_container.setCurrentIndex(1)
        self.earth_button.setStyleSheet(f"color: rgb{glob_setting['font-color']}; text-decoration: underline; background: {glob_setting['background-color']}")
        self.graph_button.setStyleSheet(f"color: rgb{glob_setting['font-color']}; background: {glob_setting['background-color']}")
        self.key_sim.setStyleSheet(f"color: rgba(0, 0, 255, 0);")
        self.key_pred.setStyleSheet(f"color: rgba(0, 255, 0, 0);")




