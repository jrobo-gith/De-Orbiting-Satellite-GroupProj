import os
import sys

root_dir = os.getcwd()
sys.path.insert(0, root_dir)

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QStackedWidget, QGridLayout, QLabel, QPushButton)
from PyQt5 import QtCore
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

from visualiser.V2.simulator_files.Py_Simulation_Jai_Testing import system_solver, lat_long_height_plot
from visualiser.V2.widgets.graph_stuff.graph_script import Grapher
from visualiser.V2.widgets.graph_stuff.partials import data_gen
from visualiser.V2.widgets.graph_stuff.earth_script import Earth
from visualiser.V2.simulator_files.sat_tracking import initialise_radars
from visualiser.V2.simulator_files.sat_tracking import get_radar_measurements
from visualiser.V2.predictor_files.predictor import Predictor

import json
import time
import numpy as np
import threading

json_file = os.path.join(root_dir, "visualiser/V2/partials/global_settings.json")
with open(json_file) as f:
    glob_setting = json.load(f)

class Helper(QtCore.QObject):
    changedSignal = QtCore.pyqtSignal(dict, tuple)

def create_data(helper, name):
    """MUST OUTPUT Matrix of size P X L where:
    P is the number of plots in the window and
    L is the number of lines in the plot.
    """
    additional_x = list(np.arange(3.0, 500.0, 0.1))

    for i in range(len(additional_x)):
        outgoing_x = [[additional_x[i]], [additional_x[i]], [additional_x[i]], [additional_x[i]]]
        outgoing_y = [data_gen.sinusoid(outgoing_x[0]), data_gen.tangent(outgoing_x[1]), data_gen.cosine(outgoing_x[2]), data_gen.cosine(outgoing_x[2])]
        helper.changedSignal.emit({'name': name}, (outgoing_x, outgoing_y))
        time.sleep(.1)

init_x = [list(np.linspace(-3.0, 3.0, 100))]
init_y = [data_gen.sinusoid(init_x[0])]

init_x.append(list(np.linspace(-3.0, 3.0, 100)))
init_y.append(data_gen.tangent(init_x[1]))

init_x.append(list(np.linspace(-3.0, 3.0, 100)))
init_y.append(data_gen.cosine(init_x[2]))

# Time span
t_span = (0, 50000)

class SimWidget(QWidget):
    def __init__(self, stacked_widget, initial_conditions, radar_list):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.initial_conditions = initial_conditions
        self.radar_list = radar_list

        ## RUN SIMULATOR TO GET ENTIRE SIMULATION LAT LON DATA
        self.solution = system_solver(t_span, self.initial_conditions, t_evals=1000)

        ## Use self.solution to compute XYZ coordinates to lat lon NOT INCLUDING rotation of earth
        self.x_sim, self.y_sim, self.z_sim = self.solution.y[0], self.solution.y[1], self.solution.y[2]

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
        self.earth = Earth(full_sim_data=(self.lat, self.lon, self.solution.t))
        ## graph_script
        self.graph = Grapher(init_x=init_x, init_y=init_y)

        ## Predictor TESTING
        self.predictor = Predictor(state0=[6378137.0  + 400e3, 1, -1, 1, 7700, 1] )

        self.radars = initialise_radars(radar_list)

        self.graph_earth_container.addWidget(self.graph)
        self.graph_earth_container.addWidget(self.earth)

        self.graph_earth_container.setStyleSheet("background: transparent;")

        container = QVBoxLayout()
        container.addLayout(sim_window_navbar, stretch=1)
        container.addWidget(self.graph_earth_container, stretch=18)
        self.setLayout(container)

        # Set threads up to feed data into graph and earth to
        graph_helper = Helper()
        graph_helper.changedSignal.connect(self.graph.update_plots, QtCore.Qt.QueuedConnection)
        threading.Thread(target=create_data, args=(graph_helper, "redundant_name"), daemon=True).start() # Target will be RADAR

        # Predictor
        pred_helper = Helper()
        pred_helper.changedSignal.connect(self.predictor.predictor_loop, QtCore.Qt.QueuedConnection)

        # Earth
        earth_helper = Helper()
        earth_helper.changedSignal.connect(self.earth.update_satellite_position, QtCore.Qt.QueuedConnection)

        threading.Thread(target=get_radar_measurements, args=(self.radars, earth_helper, pred_helper), daemon=True).start()


    def click_graph_button(self):
        self.graph_earth_container.setCurrentIndex(0)
        self.graph_button.setStyleSheet(f"color: rgb{glob_setting['font-color']}; text-decoration: underline; background: {glob_setting['background-color']}")
        self.earth_button.setStyleSheet(f"color: rgb{glob_setting['font-color']}; background: {glob_setting['background-color']}")
        self.key_sim.setStyleSheet(f"color: rgb(0, 0, 255);")
        self.key_pred.setStyleSheet(f"color: rgb(0, 255, 0);")

    def click_earth_button(self):
        self.graph_earth_container.setCurrentIndex(1)
        self.earth_button.setStyleSheet(f"color: rgb{glob_setting['font-color']}; text-decoration: underline; background: {glob_setting['background-color']}")
        self.graph_button.setStyleSheet(f"color: rgb{glob_setting['font-color']}; background: {glob_setting['background-color']}")
        self.key_sim.setStyleSheet(f"color: rgba(0, 0, 255, 0);")
        self.key_pred.setStyleSheet(f"color: rgba(0, 255, 0, 0);")





