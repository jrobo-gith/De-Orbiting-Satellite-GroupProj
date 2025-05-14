# Import standard files
import os
import sys
import numpy as np
import json

root_dir = os.getcwd()
sys.path.insert(0, root_dir)

# Import PyQt Widgets
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QStackedWidget, QLineEdit, QPushButton, QHBoxLayout, QGridLayout, \
    QLabel
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

# Import from other files in the project
from windows.model.model_window.model_window import SimWidget
from partials.constants import EARTH_SEMIMAJOR, MU_EARTH
from partials.navbar import Navbar

# Import global settings
json_file = os.path.join(root_dir, "partials/global_settings.json")
with open(json_file) as f:
    glob_setting = json.load(f)

class ModelMenu(QWidget):
    """
    Window displaying the model menu to allow the user to input initial conditions and begin the simulation. This
    limits the model from running as soon as the GUI is loaded and gives a more professional feel.

    This window's parent is the MasterWindow in master.py, navigable to using the MainMenu under the 'Simulation'
    button.

    Functions:
   -  __init__(self, stacked_widget)
   - load_sim(self)
   - set_num_radars(self, btn_instance, n)
   - set_equatorial_bool(self, btn_instance, equatorial_bool: bool)

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
        self.num_radars = 10
        self.equatorial_bool = None

        ## Layout
        page_container = QVBoxLayout()
        navbar = Navbar("Simulation Menu", self.stacked_widget)

        # Add input position ability
        self.input_X_pos = QLineEdit()
        self.input_X_pos.setFont(QFont(glob_setting['font-family'], glob_setting['font-size']))
        self.input_X_pos.setText("X-pos (Km)")
        self.input_X_pos.setStyleSheet(f"color: rgb{glob_setting['font-color']}; background: {glob_setting['background-color']}; border-radius: 10%;")
        self.input_X_pos.setFixedWidth(200)
        self.input_X_pos.setAlignment(Qt.AlignHCenter)

        self.input_Y_pos = QLineEdit()
        self.input_Y_pos.setFont(QFont(glob_setting['font-family'], glob_setting['font-size']))
        self.input_Y_pos.setText("Y-pos (Km)")
        self.input_Y_pos.setStyleSheet(f"color: rgb{glob_setting['font-color']}; background: {glob_setting['background-color']}; border-radius: 10%;")
        self.input_Y_pos.setFixedWidth(200)
        self.input_Y_pos.setAlignment(Qt.AlignHCenter)

        self.input_Z_pos = QLineEdit()
        self.input_Z_pos.setFont(QFont(glob_setting['font-family'], glob_setting['font-size']))
        self.input_Z_pos.setText("Z-pos (Km)")
        self.input_Z_pos.setStyleSheet(
            f"color: rgb{glob_setting['font-color']}; background: {glob_setting['background-color']}; border-radius: 10%;")
        self.input_Z_pos.setFixedWidth(200)
        self.input_Z_pos.setAlignment(Qt.AlignHCenter)

        positions = QHBoxLayout()
        positions.setAlignment(Qt.AlignHCenter)
        positions.addWidget(self.input_X_pos)
        positions.addWidget(self.input_Y_pos)
        positions.addWidget(self.input_Z_pos)

        # Add input velocity ability
        self.input_X_vel = QLineEdit()
        self.input_X_vel.setFont(QFont(glob_setting['font-family'], glob_setting['font-size']))
        self.input_X_vel.setText("X-vel (Km/s)")
        self.input_X_vel.setStyleSheet(
            f"color: rgb{glob_setting['font-color']}; background: {glob_setting['background-color']}; border-radius: 10%;")
        self.input_X_vel.setFixedWidth(200)
        self.input_X_vel.setAlignment(Qt.AlignHCenter)

        self.input_Y_vel = QLineEdit()
        self.input_Y_vel.setFont(QFont(glob_setting['font-family'], glob_setting['font-size']))
        self.input_Y_vel.setText("Y-vel (Km/s)")
        self.input_Y_vel.setStyleSheet(
            f"color: rgb{glob_setting['font-color']}; background: {glob_setting['background-color']}; border-radius: 10%;")
        self.input_Y_vel.setFixedWidth(200)
        self.input_Y_vel.setAlignment(Qt.AlignHCenter)

        self.input_Z_vel = QLineEdit()
        self.input_Z_vel.setFont(QFont(glob_setting['font-family'], glob_setting['font-size']))
        self.input_Z_vel.setText("Z-vel (Km/s)")
        self.input_Z_vel.setStyleSheet(
            f"color: rgb{glob_setting['font-color']}; background: {glob_setting['background-color']}; border-radius: 10%;")
        self.input_Z_vel.setFixedWidth(200)
        self.input_Z_vel.setAlignment(Qt.AlignHCenter)

        velocities = QHBoxLayout()
        velocities.setAlignment(Qt.AlignHCenter)
        velocities.addWidget(self.input_X_vel)
        velocities.addWidget(self.input_Y_vel)
        velocities.addWidget(self.input_Z_vel)

        ## Add radar locations ability
        radar_10 = QPushButton("10 Radars")
        radar_10.setFont(QFont(glob_setting['font-family'], glob_setting['font-size']))
        radar_10.setStyleSheet(f"color: rgb{glob_setting['font-color']}; background: {glob_setting['background-color']}; border-radius: 10%;")
        radar_10.setFixedWidth(250)
        radar_10.clicked.connect(lambda: self.set_num_radars(radar_10, 10))

        radar_25 = QPushButton("25 Radars")
        radar_25.setFont(QFont(glob_setting['font-family'], glob_setting['font-size']))
        radar_25.setStyleSheet(f"color: rgb{glob_setting['font-color']}; background: {glob_setting['background-color']}; border-radius: 10%;")
        radar_25.setFixedWidth(250)
        radar_25.clicked.connect(lambda: self.set_num_radars(radar_25, 25))

        radar_50 = QPushButton("50 Radars")
        radar_50.setFont(QFont(glob_setting['font-family'], glob_setting['font-size']))
        radar_50.setStyleSheet(f"color: rgb{glob_setting['font-color']}; background: {glob_setting['background-color']}; border-radius: 10%;")
        radar_50.setFixedWidth(250)
        radar_50.clicked.connect(lambda: self.set_num_radars(radar_50, 50))

        radar_100 = QPushButton("100 Radars")
        radar_100.setFont(QFont(glob_setting['font-family'], glob_setting['font-size']))
        radar_100.setStyleSheet(f"color: rgb{glob_setting['font-color']}; background: {glob_setting['background-color']}; border-radius: 10%;")
        radar_100.setFixedWidth(250)
        radar_100.clicked.connect(lambda: self.set_num_radars(radar_100, 100))

        radar_150 = QPushButton("150 Radars")
        radar_150.setFont(QFont(glob_setting['font-family'], glob_setting['font-size']))
        radar_150.setStyleSheet(f"color: rgb{glob_setting['font-color']}; background: {glob_setting['background-color']}; border-radius: 10%;")
        radar_150.setFixedWidth(250)
        radar_150.clicked.connect(lambda: self.set_num_radars(radar_150, 150))

        radar_200 = QPushButton("200 Radars")
        radar_200.setFont(QFont(glob_setting['font-family'], glob_setting['font-size']))
        radar_200.setStyleSheet(
            f"color: rgb{glob_setting['font-color']}; background: {glob_setting['background-color']}; border-radius: 10%;")
        radar_200.setFixedWidth(250)
        radar_200.clicked.connect(lambda: self.set_num_radars(radar_200, 200))

        radar_250 = QPushButton("250 Radars")
        radar_250.setFont(QFont(glob_setting['font-family'], glob_setting['font-size']))
        radar_250.setStyleSheet(
            f"color: rgb{glob_setting['font-color']}; background: {glob_setting['background-color']}; border-radius: 10%;")
        radar_250.setFixedWidth(250)
        radar_250.clicked.connect(lambda: self.set_num_radars(radar_250, 250))

        radar_500 = QPushButton("500 Radars")
        radar_500.setFont(QFont(glob_setting['font-family'], glob_setting['font-size']))
        radar_500.setStyleSheet(
            f"color: rgb{glob_setting['font-color']}; background: {glob_setting['background-color']}; border-radius: 10%;")
        radar_500.setFixedWidth(250)
        radar_500.clicked.connect(lambda: self.set_num_radars(radar_500, 500))

        # Add equatorial, non equatorial
        equatorial_btn = QPushButton("Equatorial Layout")
        equatorial_btn.setFont(QFont(glob_setting['font-family'], glob_setting['font-size']))
        equatorial_btn.setStyleSheet(f"color: rgb{glob_setting['font-color']}; background: {glob_setting['background-color']}; border-radius: 10%;")
        equatorial_btn.setFixedWidth(400)
        equatorial_btn.clicked.connect(lambda: self.set_equatorial_bool(equatorial_btn, True))

        non_equatorial_btn = QPushButton("Non-Equatorial Layout")
        non_equatorial_btn.setFont(QFont(glob_setting['font-family'], glob_setting['font-size']))
        non_equatorial_btn.setStyleSheet(
            f"color: rgb{glob_setting['font-color']}; background: {glob_setting['background-color']}; border-radius: 10%;")
        non_equatorial_btn.setFixedWidth(400)
        non_equatorial_btn.clicked.connect(lambda: self.set_equatorial_bool(non_equatorial_btn, False))

        # Add all radar buttons to grid layout
        radar_buttons = QGridLayout()
        radar_buttons.setAlignment(Qt.AlignCenter)
        radar_buttons.addWidget(radar_10, 0, 0)
        radar_buttons.addWidget(radar_25, 0, 1)
        radar_buttons.addWidget(radar_50, 0, 2)
        radar_buttons.addWidget(radar_100, 0, 3)
        radar_buttons.addWidget(radar_150, 1, 0)
        radar_buttons.addWidget(radar_200, 1, 1)
        radar_buttons.addWidget(radar_250, 1, 2)
        radar_buttons.addWidget(radar_500, 1, 3)
        radar_buttons.addWidget(equatorial_btn, 2, 0, 1, 2)
        radar_buttons.addWidget(non_equatorial_btn, 2, 2, 1, 2)

        self.radar_buttons_list = []
        self.radar_buttons_list.append(radar_10)
        self.radar_buttons_list.append(radar_25)
        self.radar_buttons_list.append(radar_50)
        self.radar_buttons_list.append(radar_100)
        self.radar_buttons_list.append(radar_150)
        self.radar_buttons_list.append(radar_200)
        self.radar_buttons_list.append(radar_250)
        self.radar_buttons_list.append(radar_500)

        self.equatorial_btn_list = [equatorial_btn, non_equatorial_btn]

        start_sim_btn = QPushButton("Start Simulation")
        start_sim_btn.setFont(QFont(glob_setting['font-family'], 23))
        start_sim_btn.setStyleSheet(f"color: rgb{glob_setting['font-color']}; background: {glob_setting['background-color']}; border-radius: 10%;")
        start_sim_btn.clicked.connect(self.load_sim)
        start_sim_btn.setFixedWidth(250)
        start_sim_layout = QHBoxLayout()
        start_sim_layout.addWidget(start_sim_btn)
        start_sim_layout.setAlignment(Qt.AlignCenter)

        recommendations_layout = QGridLayout()

        # Short equatorial Crash
        list_ = []
        layout = QVBoxLayout()
        equatorial_orbit_short = QLabel("Short Equatorial Crash (~7 Orbits)")
        equatorial_orbit_position = QLabel("Position: [150, 0, 0]")
        equatorial_orbit_velocity = QLabel("Velocity: [0, 7.82953, 0]")
        equatorial_orbit_radars = QLabel("Recommended num radars: 150")
        layout.addWidget(equatorial_orbit_short)
        layout.addWidget(equatorial_orbit_position)
        layout.addWidget(equatorial_orbit_velocity)
        layout.addWidget(equatorial_orbit_radars)
        list_.append(equatorial_orbit_short)
        list_.append(equatorial_orbit_position)
        list_.append(equatorial_orbit_velocity)
        list_.append(equatorial_orbit_radars)
        [(obj.setStyleSheet(f"color: rgb(150, 150, 150); background: {glob_setting['background-color']}; border-radius: 10%;"),
          obj.setFixedWidth(350), obj.setAlignment(Qt.AlignCenter), obj.setFont(QFont(glob_setting['font-family'], 14))) for obj in list_]
        recommendations_layout.addLayout(layout, 0, 0)

        # Medium non-equatorial crash
        list_ = []
        layout = QVBoxLayout()
        non_equatorial_orbit_medium = QLabel("Medium non-Equatorial Crash (~10 Orbits)")
        non_equatorial_orbit_position = QLabel("Position: [150, 0, 0]")
        non_equatorial_orbit_velocity = QLabel("Velocity: [0, 5.586, 5.515]")
        non_equatorial_orbit_radars = QLabel("Recommended num radars: 500")
        layout.addWidget(non_equatorial_orbit_medium)
        layout.addWidget(non_equatorial_orbit_position)
        layout.addWidget(non_equatorial_orbit_velocity)
        layout.addWidget(non_equatorial_orbit_radars)
        list_.append(non_equatorial_orbit_medium)
        list_.append(non_equatorial_orbit_position)
        list_.append(non_equatorial_orbit_velocity)
        list_.append(non_equatorial_orbit_radars)
        [(obj.setStyleSheet(f"color: rgb(150, 150, 150); background: {glob_setting['background-color']}; border-radius: 10%;"),
          obj.setFixedWidth(350), obj.setAlignment(Qt.AlignCenter), obj.setFont(QFont(glob_setting['font-family'], 14))) for obj in list_]
        recommendations_layout.addLayout(layout, 0, 1)

        # Short non-equatorial crash
        list_ = []
        layout = QVBoxLayout()
        non_equatorial_orbit_short = QLabel("Short non-Equatorial Crash (~0 Orbits)")
        short_non_equatorial_orbit_position = QLabel("Position: [200, 0, 0]")
        short_non_equatorial_orbit_velocity = QLabel("Velocity: [0, 7.7, 1]")
        short_non_equatorial_orbit_radars = QLabel("Recommended num radars: 250")
        layout.addWidget(non_equatorial_orbit_short)
        layout.addWidget(short_non_equatorial_orbit_position)
        layout.addWidget(short_non_equatorial_orbit_velocity)
        layout.addWidget(short_non_equatorial_orbit_radars)
        list_.append(non_equatorial_orbit_short)
        list_.append(short_non_equatorial_orbit_position)
        list_.append(short_non_equatorial_orbit_velocity)
        list_.append(short_non_equatorial_orbit_radars)
        [(obj.setStyleSheet(f"color: rgb(150, 150, 150); background: {glob_setting['background-color']}; border-radius: 10%;"),
          obj.setFixedWidth(350), obj.setAlignment(Qt.AlignCenter), obj.setFont(QFont(glob_setting['font-family'], 14))) for obj in list_]
        recommendations_layout.addLayout(layout, 1, 0)


        recommendations_layout.addLayout(layout, 1, 0)
        # recommendations_layout.addLayout(layout, 1, 1)


        page_layout = QVBoxLayout()
        page_layout.setAlignment(Qt.AlignCenter)
        page_layout.addLayout(positions)
        page_layout.addLayout(velocities)
        page_layout.addLayout(radar_buttons)
        page_layout.addLayout(start_sim_layout)
        page_layout.addLayout(recommendations_layout)

        page_container.addWidget(navbar, stretch=1)
        page_container.addLayout(page_layout, stretch=19)
        self.setLayout(page_container)

    def load_sim(self):
        """
        Connected to the 'start_sim_btn' QPushButton. Runs when the 'start_sim_btn' button is clicked. This takes the
        input values from each QLineEdit and constructions a list of 'initial conditions' to be fed into
        'simulation_window.py'. It also takes the inputs of radar locations into a list and passes it into
        'simulation_window.py'.

        It then checks if an instance of the simulation already exists, if so, it removes the instance and adds it back,
        effectively restarting the simulation. If it does not exist, it creates a new instance of the Simwidget.
        """

        init_x_p = (float(self.input_X_pos.text()) * 1000)
        init_y_p = (float(self.input_Y_pos.text()) * 1000)
        init_z_p = (float(self.input_Z_pos.text()) * 1000)

        if init_x_p != 0:
            init_x_p += EARTH_SEMIMAJOR
        if init_y_p != 0:
            init_y_p += EARTH_SEMIMAJOR
        if init_z_p != 0:
            init_z_p += EARTH_SEMIMAJOR

        init_x_v = (float(self.input_X_vel.text()) * 1000)
        init_y_v = (float(self.input_Y_vel.text()) * 1000)
        init_z_v = (float(self.input_Z_vel.text()) * 1000)

        initial_conditions = [init_x_p, init_y_p, init_z_p, init_x_v, init_y_v, init_z_v]

        radar_list = give_random_radar_locations(self.num_radars, equatorial=self.equatorial_bool)
        print(f"INITIALISING RADARS, NUM RADARS = {self.num_radars}, and equatorial is {self.equatorial_bool}")

        if self.stacked_widget.count() > 4: # Means self.sim_stacked_widget doesn't exist
            self.stacked_widget.removeWidget(self.sim_stacked_widget) # Remove old instance
            self.sim_stacked_widget.destroy()
            self.sim_stacked_widget = QStackedWidget() # Create new simulation
            self.sim_stacked_widget.addWidget(ModelMenu(self.stacked_widget))
            self.sim_stacked_widget.addWidget(SimWidget(self.stacked_widget, initial_conditions, radar_list))
            self.sim_stacked_widget.setCurrentIndex(1)
            self.stacked_widget.addWidget(self.sim_stacked_widget)
            self.stacked_widget.setCurrentIndex(4)
        else:
            # Create stacked widget (load in sim window)
            self.sim_stacked_widget = QStackedWidget()
            self.sim_stacked_widget.addWidget(ModelMenu(self.stacked_widget))
            self.sim_stacked_widget.addWidget(SimWidget(self.stacked_widget, initial_conditions, radar_list))
            self.sim_stacked_widget.setCurrentIndex(1)
            self.stacked_widget.addWidget(self.sim_stacked_widget)
            self.stacked_widget.setCurrentIndex(4)

    def set_num_radars(self, btn_instance, n):
        """
        Function to set the number of radars to use when initialising the random list of radars, to be fed into instance
        of SimWidget. Also turns the rest of the buttons to green and the clicked button red to indicate the current
        number radars.

        :param btn_instance: Instance of button to set stylesheet to red
        :param n: number of radars
        """

        self.num_radars = n
        [oth_btn.setStyleSheet(f"color: rgb{glob_setting['font-color']}; background: {glob_setting['background-color']}; border-radius: 10%;") for oth_btn in self.radar_buttons_list]
        btn_instance.setStyleSheet(f"color: rgb(255, 0, 0); background: {glob_setting['background-color']}; border-radius: 10%;")

    def set_equatorial_bool(self, btn_instance, equatorial_bool: bool):
        """
        Function to set the preference of radars to be initialised along the equator or to be randomly distributed
        around the whole earth. This will be fed into the radar spawner 'give_random_radar_locations'. Also turns the
        other option green and the clicked button red to make the choice clear.

        :param btn_instance: Instance of button to set stylesheet to red
        :param equatorial_bool: if true, spawn radars along the equator, if false, spawn radars randomly over the earth.
        """

        [oth_btn.setStyleSheet(f"color: rgb{glob_setting['font-color']}; background: {glob_setting['background-color']}; border-radius: 10%;") for oth_btn in self.equatorial_btn_list]
        btn_instance.setStyleSheet(f"color: rgb(255, 0, 0); background: {glob_setting['background-color']}; border-radius: 10%;")

        if equatorial_bool:
            self.equatorial_bool = True
        else:
            self.equatorial_bool = False

def give_random_radar_locations(num_radars, equatorial:bool=True):
    """
    Generates a list of random [lat, lon, height] to specify radar locations placed on earth.

    :param num_radars: Taken from 'set_num_radars' to specify the number of radars to be spawned (added to the list).
    :param equatorial: Specifies whether to generate radars along the equator or not. Taken from 'set_equatorial_bool'
    :return: list[[lat, lon, height]] specifying details about each radar.
    """
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
