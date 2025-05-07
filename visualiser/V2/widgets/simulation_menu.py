import os
import sys

root_dir = os.getcwd()
sys.path.insert(0, root_dir)

from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QStackedWidget, QLabel, QLineEdit, QPushButton, QHBoxLayout)
from PyQt5 import QtCore
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
import numpy as np

from visualiser.V2.widgets.graph_stuff.simulator_window import SimWidget
from visualiser.V2.simulator_files.Py_Simulation_Jai_Testing import EARTH_SEMIMAJOR

from visualiser.V2.partials.navbar import Navbar
import json
json_file = os.path.join(root_dir, "visualiser/V2/partials/global_settings.json")
with open(json_file) as f:
    glob_setting = json.load(f)


class SimulationMenu(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget

        ## Layout
        page_container = QVBoxLayout()
        navbar = Navbar("Simulation Menu", self.stacked_widget)

        # Add input position ability
        self.input_X_pos = QLineEdit()
        self.input_X_pos.setFont(QFont(glob_setting['font-family'], 22))
        self.input_X_pos.setText("X-pos")
        self.input_X_pos.setStyleSheet(f"color: rgb{glob_setting['font-color']}; background: {glob_setting['background-color']}; border-radius: 10%;")
        self.input_X_pos.setFixedWidth(90)
        self.input_X_pos.setAlignment(Qt.AlignHCenter)

        self.input_Y_pos = QLineEdit()
        self.input_Y_pos.setFont(QFont(glob_setting['font-family'], 22))
        self.input_Y_pos.setText("Y-pos")
        self.input_Y_pos.setStyleSheet(f"color: rgb{glob_setting['font-color']}; background: {glob_setting['background-color']}; border-radius: 10%;")
        self.input_Y_pos.setFixedWidth(90)
        self.input_Y_pos.setAlignment(Qt.AlignHCenter)

        self.input_Z_pos = QLineEdit()
        self.input_Z_pos.setFont(QFont(glob_setting['font-family'], 22))
        self.input_Z_pos.setText("Z-pos")
        self.input_Z_pos.setStyleSheet(
            f"color: rgb{glob_setting['font-color']}; background: {glob_setting['background-color']}; border-radius: 10%;")
        self.input_Z_pos.setFixedWidth(90)
        self.input_Z_pos.setAlignment(Qt.AlignHCenter)

        positions = QHBoxLayout()
        positions.setAlignment(Qt.AlignHCenter)
        positions.addWidget(self.input_X_pos)
        positions.addWidget(self.input_Y_pos)
        positions.addWidget(self.input_Z_pos)

        # Add input velocity ability

        self.input_X_vel = QLineEdit()
        self.input_X_vel.setFont(QFont(glob_setting['font-family'], 22))
        self.input_X_vel.setText("X-vel")
        self.input_X_vel.setStyleSheet(
            f"color: rgb{glob_setting['font-color']}; background: {glob_setting['background-color']}; border-radius: 10%;")
        self.input_X_vel.setFixedWidth(90)
        self.input_X_vel.setAlignment(Qt.AlignHCenter)

        self.input_Y_vel = QLineEdit()
        self.input_Y_vel.setFont(QFont(glob_setting['font-family'], 22))
        self.input_Y_vel.setText("Y-vel")
        self.input_Y_vel.setStyleSheet(
            f"color: rgb{glob_setting['font-color']}; background: {glob_setting['background-color']}; border-radius: 10%;")
        self.input_Y_vel.setFixedWidth(90)
        self.input_Y_vel.setAlignment(Qt.AlignHCenter)

        self.input_Z_vel = QLineEdit()
        self.input_Z_vel.setFont(QFont(glob_setting['font-family'], 22))
        self.input_Z_vel.setText("Z-vel")
        self.input_Z_vel.setStyleSheet(
            f"color: rgb{glob_setting['font-color']}; background: {glob_setting['background-color']}; border-radius: 10%;")
        self.input_Z_vel.setFixedWidth(90)
        self.input_Z_vel.setAlignment(Qt.AlignHCenter)

        velocities = QHBoxLayout()
        velocities.setAlignment(Qt.AlignHCenter)
        velocities.addWidget(self.input_X_vel)
        velocities.addWidget(self.input_Y_vel)
        velocities.addWidget(self.input_Z_vel)


        start_sim_btn = QPushButton("Start Simulation")
        start_sim_btn.setFont(QFont(glob_setting['font-family'], 27))
        start_sim_btn.setStyleSheet(f"color: rgb{glob_setting['font-color']}; background: {glob_setting['background-color']}; border-radius: 10%;")
        start_sim_btn.clicked.connect(self.loadSim)
        start_sim_btn.setFixedWidth(270)
        start_sim_layout = QHBoxLayout()
        start_sim_layout.addWidget(start_sim_btn)
        start_sim_layout.setAlignment(Qt.AlignCenter)

        page_layout = QVBoxLayout()
        page_layout.setAlignment(Qt.AlignCenter)
        page_layout.addLayout(positions)
        page_layout.addLayout(velocities)
        page_layout.addLayout(start_sim_layout)


        page_container.addWidget(navbar, stretch=1)
        page_container.addLayout(page_layout, stretch=19)
        self.setLayout(page_container)


    def loadSim(self):
        # init_x_p = int(self.input_X_pos.text()) + EARTH_SEMIMAJOR
        # init_y_p = int(self.input_Y_pos.text())
        # init_z_p = int(self.input_Z_pos.text())
        #
        # init_x_v = int(self.input_X_vel.text())
        # init_y_v = int(self.input_Y_vel.text()) / np.sqrt(2)
        # init_z_v = int(self.input_Z_vel.text()) / np.sqrt(2)
        #
        # initial_conditions = [init_x_p, init_y_p, init_z_p, init_x_v, init_y_v, init_z_v]

        stable_condition = [300e3 + EARTH_SEMIMAJOR, 0, 0 , 0, 7800/np.sqrt(2), 7800/np.sqrt(2)]

        radar_list = [[-50, -1.5, 15], [37, -1.3, 1650], [100, 0.8, 25], [0.55, 50, 70], [0, 90, 1000]]

        if self.stacked_widget.count() > 4: # Means self.sim_stacked_widget doesn't exist
            self.stacked_widget.removeWidget(self.sim_stacked_widget) # Remove old instance
            self.sim_stacked_widget = QStackedWidget() # Create new simulation
            self.sim_stacked_widget.addWidget(SimulationMenu(self.stacked_widget))
            self.sim_stacked_widget.addWidget(SimWidget(self.stacked_widget, stable_condition, radar_list))
            self.sim_stacked_widget.setCurrentIndex(1)
            self.stacked_widget.addWidget(self.sim_stacked_widget)
            self.stacked_widget.setCurrentIndex(4)
        else:
            # Create stacked widget (load in sim window)
            self.sim_stacked_widget = QStackedWidget()
            self.sim_stacked_widget.addWidget(SimulationMenu(self.stacked_widget))
            self.sim_stacked_widget.addWidget(SimWidget(self.stacked_widget, stable_condition, radar_list))
            self.sim_stacked_widget.setCurrentIndex(1)
            self.stacked_widget.addWidget(self.sim_stacked_widget)
            self.stacked_widget.setCurrentIndex(4)