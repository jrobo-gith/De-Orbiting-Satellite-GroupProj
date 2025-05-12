# Import standard files
import os
import sys
import numpy as np
import json

root_dir = os.getcwd()
sys.path.insert(0, root_dir)

# Import PyQt Widgets
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QStackedWidget, QLineEdit, QPushButton, QHBoxLayout
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

# Import from other files in the project
from visualiser.V3.windows.model.model_window.model_window import SimWidget
from visualiser.V3.partials.constants import EARTH_SEMIMAJOR, MU_EARTH
from visualiser.V3.partials.navbar import Navbar

# Import global settings
json_file = os.path.join(root_dir, "visualiser/V3/partials/global_settings.json")
with open(json_file) as f:
    glob_setting = json.load(f)

class ModelMenu(QWidget):
    """
    Window displaying the model menu to allow the user to input initial conditions and begin the simulation. This
    limits the model from running as soon as the GUI is loaded and gives a more professional feel.

    This window's parent is the MasterWindow in master.py, navigable to using the MainMenu under the 'simulation'
    button.

    Functions:
   -  __init__(self, stacked_widget)

    References:
        Tutorial followed for PyQt5 (GUI) can be found here - https://www.pythonguis.com/pyqt5-tutorial/

    Previous versions can be found in the Group GitHub - https://github.com/jrobo-gith/De-Orbiting-Satellite-GroupProj
    """
    def __init__(self, stacked_widget):
        """
        Initialises the model menu window, contains an instance of the 'navbar' class, also contains 'QLineEdits'
        that are used to allow the user to input initial conditions. If the user inputs anything other than floating
        point numbers or ints, the program will not continue. A collection of QLabels, QLineEdits and a QPushButton are
        stacked and layed out to display the simulation menu.

        :param stacked_widget: Widget containing pages of the GUI, used to navigate back to the main menu.
        """
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
        """
        Connected to the 'start_sim_btn' QPushButton. Runs when the 'start_sim_btn' button is clicked. This takes the
        input values from each QLineEdit and constructions a list of 'initial conditions' to be fed into
        'simulation_window.py'. It also takes the inputs of radar locations into a list and passes it into
        'simulation_window.py'.

        It then checks if an instance of the simulation already exists, if so, it removes the instance and adds it back,
        effectively restarting the simulation. If it does not exist, it creates a new instance of the Simwidget.
        """
        # init_x_p = int(self.input_X_pos.text()) + EARTH_SEMIMAJOR
        # init_y_p = int(self.input_Y_pos.text())
        # init_z_p = int(self.input_Z_pos.text())
        #
        # init_x_v = int(self.input_X_vel.text())
        # init_y_v = int(self.input_Y_vel.text()) / np.sqrt(2)
        # init_z_v = int(self.input_Z_vel.text()) / np.sqrt(2)
        #
        # initial_conditions = [init_x_p, init_y_p, init_z_p, init_x_v, init_y_v, init_z_v]

        # Stable non-equatorial
        stable_condition_none = [150e3 + EARTH_SEMIMAJOR, 0, 0 , 0, 7900/np.sqrt(2), 7800/np.sqrt(2)]
        # Stable equitorial
        stable_condition_e = [150e3 + EARTH_SEMIMAJOR, 0, 0 , 0, np.sqrt(MU_EARTH/(150e3 + EARTH_SEMIMAJOR)) * 1.002, 0]

        radar_list = give_random_radar_locations(500, equatorial=False)

        stable_condition = stable_condition_none

        # radar_list = [[-50, -1.5, 15], [37, -1.3, 1650], [100, 0.8, 25], [0.55, 50, 70], [0, 90, 1000]]

        if self.stacked_widget.count() > 4: # Means self.sim_stacked_widget doesn't exist
            self.stacked_widget.removeWidget(self.sim_stacked_widget) # Remove old instance
            self.sim_stacked_widget = QStackedWidget() # Create new simulation
            self.sim_stacked_widget.addWidget(ModelMenu(self.stacked_widget))
            self.sim_stacked_widget.addWidget(SimWidget(self.stacked_widget, stable_condition, radar_list))
            self.sim_stacked_widget.setCurrentIndex(1)
            self.stacked_widget.addWidget(self.sim_stacked_widget)
            self.stacked_widget.setCurrentIndex(4)
        else:
            # Create stacked widget (load in sim window)
            self.sim_stacked_widget = QStackedWidget()
            self.sim_stacked_widget.addWidget(ModelMenu(self.stacked_widget))
            self.sim_stacked_widget.addWidget(SimWidget(self.stacked_widget, stable_condition, radar_list))
            self.sim_stacked_widget.setCurrentIndex(1)
            self.stacked_widget.addWidget(self.sim_stacked_widget)
            self.stacked_widget.setCurrentIndex(4)

def give_random_radar_locations(num_radars, equatorial:bool=True):
    radar_list = []
    for i in range(num_radars):
        if equatorial:
            rand_lat = 0
        else:
            rand_lat = np.random.randint(-90, 90)
        rand_lon = np.random.randint(-180, 180)
        rand_height = np.random.randint(0, 1000)
        radar_details = [rand_lat, rand_lon, rand_height]
        radar_list.append(radar_details)
    return radar_list
