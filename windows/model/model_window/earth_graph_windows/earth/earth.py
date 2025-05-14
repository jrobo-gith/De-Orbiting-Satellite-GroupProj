import os
import sys
from PIL import Image
import json
import numpy as np
import threading
from partials.constants import EARTH_ROTATION_ANGLE
from simulator.simulator import lat_long_height
from windows.model.model_window.earth_graph_windows.graph.plots.single_plot import Plot

root_dir = os.getcwd()
sys.path.insert(0, root_dir)

# Import necessary PyQt5 components
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QCheckBox, QPushButton, QLabel, QGridLayout
from PyQt5.QtCore import Qt
import pyqtgraph as pg
from PyQt5 import QtCore
from PyQt5.QtGui import QFont

# Import Global settings
json_file = os.path.join(root_dir, "partials/global_settings.json")
with open(json_file) as f:
    glob_setting = json.load(f)

class Helper(QtCore.QObject):
    changedSignal = QtCore.pyqtSignal(str, tuple)

class Earth(pg.GraphicsLayoutWidget):
    """
    Window displaying the Earth Window in the simulation, used for visualising the satellite orbit trajectory,
    prediction of landing sites, and radar locations where each one can be dynamically switched off to gain a better
    view of the user's interest.
    This window's parent is the SimWidget in simulator_window.py, navigable to using the SimWidget under the 'EarthView'
    button at the top right.

    Functions:
    - __init__(self, stacked_widget)
    - update_satellite_position(self, name, update)
    - sim_overlay_switch(self)
    - radar_overlay_switch(self)
    - prediction_overlay_switch(self)

    References:
        Tutorial followed for PyQt5 (GUI) can be found here - https://www.pythonguis.com/pyqt5-tutorial/

    Previous versions can be found in the Group GitHub - https://github.com/jrobo-gith/De-Orbiting-Satellite-GroupProj
    """
    def __init__(self, predictor, full_sim_data, radar_list):
        """
        Initialises the Earth Window, converts a .jpg image of the world into a plottable ImageItem through a numpy
        array. Also takes the full simulation data, ran in simulator_window.py, and plots it on the 2D map of earth.

        Also lays out optional checkboxes which turn simulation data, radar locations or prediction info off and on at
        the user's convenience.

        :param full_sim_data: the full simulation data in lat lon, used to plot the full simulation on the 2d world map.
        :param radar_list: list of radar lat lon's, initialised in model_window.py, used to plot on the 2d world map.
        """
        super().__init__()

        self.predictor = predictor
        self.lat, self.lon, self.t = full_sim_data
        self.radar_list = radar_list

        # Keep track of number of predictions made
        self.prediction_count = 0

        if len(self.lat) > 10_000:
            self.lat = self.lat[0:len(self.lat):10]
            self.lon = self.lon[0:len(self.lon):10]
            self.adjusted_t = self.t[0:len(self.t):10]
        else:
            self.adjusted_t = self.t

        self.plot_widget = pg.PlotWidget()
        self.earth_graph = pg.GraphicsLayoutWidget()

        world_map = os.path.join(root_dir, "windows/model/model_window/earth_graph_windows/earth/world_map.jpg")
        world_map = Image.open(world_map).convert('RGB')
        world_map = np.transpose(np.array(world_map), (1, 0, 2))
        world_img = pg.ImageItem(world_map)

        ## Account for earth's rotation
        EARTH_ROTATION =  EARTH_ROTATION_ANGLE * self.adjusted_t
        self.lon -= EARTH_ROTATION
        ## Convert to Miller Coordinates
        self.lat = (5/4) * np.arcsinh(np.tan((4*self.lat)/5))

        self.lon *= (180 / np.pi)
        self.lat *= (180 / np.pi)

        self.lon = (self.lon + 180) % 360 - 180

        # Get final crash location (pre-accounting for earth's rotation)
        self.final_crash_vector = np.linalg.norm([self.lat[-1], self.lon[-1]])

        ## Switch x and y (lat and lon) to account for inverting world axis
        self.x = self.lat
        self.y = self.lon

        # Simulation Overlay
        full_sim_pixels = latlon2pixel(self.x, self.y)
        self.sim_plot = pg.ScatterPlotItem(x=full_sim_pixels[0][:-1], y=full_sim_pixels[1][:-1], size=5, brush=pg.mkBrush('yellow'))
        self.crash_site = pg.ScatterPlotItem(x=[full_sim_pixels[0][-1]], y=[full_sim_pixels[1][-1]], size=10, brush=pg.mkBrush('red'), pen=pg.mkPen('red'))
        self.crash_site_outline1 = pg.ScatterPlotItem(x=[full_sim_pixels[0][-1]], y=[full_sim_pixels[1][-1]], size=20,
                                             brush=pg.mkBrush('red'), pen=pg.mkPen('red'))
        self.crash_site_outline2 = pg.ScatterPlotItem(x=[full_sim_pixels[0][-1]], y=[full_sim_pixels[1][-1]], size=30,
                                             brush=pg.mkBrush('red'), pen=pg.mkPen('red'))
        self.satellite_start_position = pg.ScatterPlotItem(x=[full_sim_pixels[0][0]], y=[full_sim_pixels[1][0]], size=20, brush=pg.mkBrush('black'))
        # Keep array of live satellite position
        self.live_satellite_array_x = [full_sim_pixels[0][0]]
        self.live_satellite_array_y = [full_sim_pixels[0][1]]
        # Plot live satellite array
        self.live_satellite_trail = pg.ScatterPlotItem(x=self.live_satellite_array_x, y=self.live_satellite_array_y, size=5, brush=pg.mkBrush('red'), pen=pg.mkPen('red'))

        self.sim_plot.setOpacity(0.9)
        self.crash_site.setOpacity(0.9)
        self.crash_site_outline1.setOpacity(0.4)
        self.crash_site_outline2.setOpacity(0.1)
        self.satellite_start_position.setOpacity(0.9)

        # Prediction Overlay (Alternate variable "size" to show uncertainty)
        x, y = latlon2pixel(np.array([0]), np.array([0]))
        self.prediction_crash_point = pg.ScatterPlotItem(x=x, y=y, size=20, brush=pg.mkBrush('green'), pen=pg.mkPen('green'))
        self.prediction_crash_1std = pg.ScatterPlotItem(x=np.zeros(100), y=np.zeros(100), size=10, brush=pg.mkBrush(0, 255, 0))
        self.prediction_crash_point.setOpacity(0.9)
        self.prediction_crash_1std.setOpacity(0.75)

        # Temp covariance plot
        self.pred_cov = pg.ScatterPlotItem


        # Add Radars locations
        self.radar_plots = []
        self.radar_texts = []
        for i, radar in enumerate(self.radar_list):
            x, y = latlon2pixel(np.array([radar[0]]), np.array([radar[1]]))
            radar_plot = pg.ScatterPlotItem(x=x, y=y, size=20, brush=pg.mkBrush((173, 216, 230)))
            self.radar_plots.append(radar_plot)

            # Add Radar Labels
            self.txt = pg.TextItem(f'R{i}')
            self.txt.setPos(x[0]-1, y[0]+1)
            self.radar_texts.append(self.txt)

        # Add world image
        self.plot_widget.addItem(world_img)
        # Add simulation data
        self.plot_widget.addItem(self.sim_plot)
        self.plot_widget.addItem(self.crash_site)
        self.plot_widget.addItem(self.crash_site_outline1)
        self.plot_widget.addItem(self.crash_site_outline2)
        self.plot_widget.addItem(self.live_satellite_trail)
        self.plot_widget.addItem(self.satellite_start_position)
        # Add prediction data
        self.plot_widget.addItem(self.prediction_crash_point)
        self.plot_widget.addItem(self.prediction_crash_1std)
        # self.plot_widget.addItem(self.prediction_crash_2std)
        [self.plot_widget.addItem(radar_plot) for radar_plot in self.radar_plots]
        [self.plot_widget.addItem(radar_text) for radar_text in self.radar_texts]


        # Plotting details
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.hideAxis('left')
        self.plot_widget.hideAxis('bottom')
        self.plot_widget.invertY(True)
        self.viewbox = self.plot_widget.getViewBox()
        self.viewbox.setRange(xRange=(0, 5400), yRange=(2700, 0))
        self.sim_plot.getViewBox().invertY(True) # Keep consistent with world map

        self.simulation_checkbox = QCheckBox("Show Full Simulation")
        self.simulation_checkbox.setStyleSheet(f"background-color: rgba{glob_setting['background-color']}; color: rgb{glob_setting['font-color']}")
        self.simulation_checkbox.setChecked(True)
        self.simulation_checkbox.stateChanged.connect(self.sim_overlay_switch)

        self.live_simulation_checkbox = QCheckBox("Show Live Satellite")
        self.live_simulation_checkbox.setStyleSheet(
            f"background-color: rgba{glob_setting['background-color']}; color: rgb{glob_setting['font-color']}")
        self.live_simulation_checkbox.setChecked(True)
        self.live_simulation_checkbox.stateChanged.connect(self.live_satellite_switch)

        self.radar_checkbox = QCheckBox("Show Radars")
        self.radar_checkbox.setStyleSheet(f"background-color: rgba{glob_setting['background-color']}; color: rgb{glob_setting['font-color']}")
        self.radar_checkbox.setChecked(False)
        self.radar_overlay_switch()
        self.radar_checkbox.stateChanged.connect(self.radar_overlay_switch)

        self.prediction_checkbox = QCheckBox("Show Prediction")
        self.prediction_checkbox.setStyleSheet(f"background-color: rgba{glob_setting['background-color']}; color: rgb{glob_setting['font-color']}")
        self.prediction_checkbox.setChecked(True)
        self.prediction_checkbox.stateChanged.connect(self.prediction_overlay_switch)

        self.make_prediction_button = QPushButton("Make Prediction")
        self.make_prediction_button.setStyleSheet(f"background-color: {glob_setting['background-color']}; color: rgb{glob_setting['font-color']}")
        self.make_prediction_button.setFont(QFont(glob_setting['font-family'], 20))

        self.filler = QLabel("Fill")
        self.filler.setStyleSheet(f"background-color: {glob_setting['background-color']}; color: {glob_setting['background-color']}")

        # ## ADD EARTH RESIDUAL GRAPH
        crash_res = os.path.join(root_dir, "windows/model/model_window/earth_graph_windows/graph/profiles/crash_residual.json")
        with open(crash_res) as f:
            crash_res = json.load(f)
        self.res_x = [[0]]
        self.res_y = [[0]]
        self.residual_plot = self.earth_graph.addPlot()
        self.residual_plot = Plot(self.residual_plot,
                                   self.res_x,
                                   self.res_y,
                                   args=crash_res)

        # Filter details
        self.filter_layout = QGridLayout()
        self.filter_layout.addWidget(self.simulation_checkbox, 0, 0)
        self.filter_layout.addWidget(self.live_simulation_checkbox, 0, 1)
        self.filter_layout.addWidget(self.radar_checkbox, 0, 2)
        self.filter_layout.addWidget(self.prediction_checkbox, 0, 3)
        self.filter_layout.addWidget(self.filler, 0, 4)
        self.filter_layout.addWidget(self.make_prediction_button, 0, 5)
        self.filter_layout.setAlignment(Qt.AlignCenter)

        self.earth_checkboxes = QVBoxLayout()
        self.earth_checkboxes.addWidget(self.plot_widget, stretch=19)
        self.earth_checkboxes.addLayout(self.filter_layout, stretch=1)

        self.global_grid = QHBoxLayout()

        self.global_grid.addLayout(self.earth_checkboxes, stretch=15)
        self.global_grid.addWidget(self.earth_graph, stretch=5)
        self.setLayout(self.global_grid)


    @QtCore.pyqtSlot(dict, tuple)
    def update_satellite_position(self, info, update):
        """
        Function to update the latitude and longitude of the satellite as it travels around the earth. Gets information
        from simulator/radar via multi-threading and a 'helper' function, which emits the satellite's location at a
        given time interval. We receive the data as (dict, tuple), we use the dict (info) in this instance to collect
        the time to account for earth's rotation, and the update is a tuple containing the latitude and longitude of the
        satellite.

        We update the scatter plot 'self.satellite_start_position' using the function 'setData(x, y)'

        :param info: radar info, contains observed time
        :param update: contains one latitude and one longitude to update the satellite's position.
        """
        XYZ = update[0]

        ## Convert x, y, z to lat lon
        lat, lon, _ = lat_long_height(XYZ[0], XYZ[1], XYZ[2])

        # Account for earth's rotation and apply miller's coordinates.
        lat, lon = prepare_latlon_for_graph(time=info['obs-time'], lat=lat, lon=lon)

        x, y = latlon2pixel([lat], [lon])
        self.satellite_start_position.setData(x, y)

        if len(self.live_satellite_array_x) > 50:
            # Remove oldest x, y
            self.live_satellite_array_x = self.live_satellite_array_x[1:]
            self.live_satellite_array_y = self.live_satellite_array_y[1:]

        # Add new datapoint
        self.live_satellite_array_x.append(x[0])
        self.live_satellite_array_y.append(y[0])
        # Plot data
        self.live_satellite_trail.setData(x=self.live_satellite_array_x, y=self.live_satellite_array_y)

    @QtCore.pyqtSlot(dict, tuple)
    def update_prediction(self, info, update):
        """
        Function to update the prediction of the crash site as it travels around the earth. Gets called from
        predictor/predictor.py via multi-threading and a 'helper' function, which emits the latitude and longitude of
        a point at which the satellite crashes onto earth.

        :param info: contains info such as whether we are predicting landing or not.
        :param update: contains the latitude and longitude of the predicted crash site.
        """

        self.prediction_count += 1

        # Collect means for latitude, longitude and the covariance (radians)
        lat, lon, cov = update

        # Creates a list of latitudes and longitudes for the covariance plot (radians)
        lat_cov, lon_cov = create_covariance_plot(cov_mat=cov, mean_lat=lat, mean_lon=lon)

        # Converts covariance lat and lon's to include earth's rotation and plotting standards(outlined in the function)
        lat_cov, lon_cov = prepare_latlon_for_graph(time=info['time'], lat=lat_cov, lon=lon_cov)

        # Convert to pixel values
        x_cov, y_cov = latlon2pixel(lat_cov, lon_cov)

        # Account for earth's rotation, also currently in radians
        lat, lon = prepare_latlon_for_graph(time=info['time'], lat=lat, lon=lon)

        # Compute error between two lat lon vectors and update the graph
        prediction_residual = np.linalg.norm([lat, lon]) - self.final_crash_vector
        self.update_crash_residual(prediction_residual)

        # Update the data for the prediction
        x, y = latlon2pixel([lat], [lon])
        self.prediction_crash_point.setData(x, y)
        self.prediction_crash_1std.setData(x_cov, y_cov)


    def sim_overlay_switch(self):
        """
        Function that sits inside the 'self.simulation_checkbox' QCheckBox. When altered (checked or unchecked), the
        function asks if the checkbox is checked, if not, it checks it, if it is, it unchecks it.
        """

        if self.simulation_checkbox.isChecked():
            self.sim_plot.show()
            self.crash_site.show()
            self.crash_site_outline1.show()
            self.crash_site_outline2.show()
        else:
            self.sim_plot.hide()
            self.crash_site.hide()
            self.crash_site_outline1.hide()
            self.crash_site_outline2.hide()

    def live_satellite_switch(self):
        if self.live_simulation_checkbox.isChecked():
            self.satellite_start_position.show()
            self.live_satellite_trail.show()
        else:
            self.satellite_start_position.hide()
            self.live_satellite_trail.hide()

    def radar_overlay_switch(self):
        """
        Function that sits inside the 'self.radar_checkbox' QCheckBox. When altered (checked or unchecked), the
        function asks if the checkbox is checked, if not, it checks it, if it is, it unchecks it.
        """

        if self.radar_checkbox.isChecked():
            [radar_plot.show() for radar_plot in self.radar_plots]
            [radar_text.show() for radar_text in self.radar_texts]
        else:
            [radar_plot.hide() for radar_plot in self.radar_plots]
            [radar_text.hide() for radar_text in self.radar_texts]

    def prediction_overlay_switch(self):
        """
        Function that sits inside the 'self.prediction_checkbox' QCheckBox. When altered (checked or unchecked), the
        function asks if the checkbox is checked, if not, it checks it, if it is, it unchecks it.
        """
        if self.prediction_checkbox.isChecked():
            self.prediction_crash_point.show()
            self.prediction_crash_1std.show()
        else:
            self.prediction_crash_point.hide()
            self.prediction_crash_1std.hide()

    def request_prediction(self, helper):
        """
        Sends a signal to the predictor to request a prediction. Sits inside the 'make_prediction_button' and triggers
        the predictor to sample x_post and propagate those samples to a landing site, from which, we can compute the
        mean and covariance for the landing site which will be plotted on the earth via the function update_prediction.

        :param helper: helper to emit a signal to the function 'send_prediction' in prediction.py.
        """

        helper.changedSignal.emit("Requested prediction", (0, 0))

    def set_predictor(self, predictor):
        """
        Function to get around adding a problem in 'model_window.py' where instance of earth is added to the instance
        of predictor and instance of predictor is added to instance of earth. We now initiate earth with
        (predictor=None) and add the predictor after the predictor is instantiated using this function.

        This function also builds a helper that connects to the 'send_prediction' function in predictor.py to allow
        signals to be sent to make predictions.

        :param predictor: instance of the predictor
        """

        self.predictor = predictor

        # Establish connection with helper
        prediction_requester_helper = Helper()
        prediction_requester_helper.changedSignal.connect(predictor.send_prediction, QtCore.Qt.QueuedConnection)
        threading.Thread(target=self.request_prediction, args=(prediction_requester_helper,), daemon=True)

        #Connect button to function
        self.make_prediction_button.clicked.connect(lambda: self.request_prediction(prediction_requester_helper))

    def update_crash_residual(self, Y_data):
        """
        Updates the plot in earth that plots the crash residual on the right of 'earth view'. Appends the number of
        guesses to self.res_x using the prediction count and adds the new residual calculated in 'update_prediction'.

        :param Y_data: New residual to be added to the existing plot.
        """

        self.res_x[0].append(self.prediction_count)
        self.res_y[0].append(Y_data)

        self.residual_plot.update_plot(np.array([self.res_x[0][-1]]), np.array([self.res_y[0][-1]]))

def latlon2pixel(lat:np.array, lon:np.array, screen_w:int=5400, screen_h:int=2700) -> tuple:
    """
    Returns pixel values for lat and lon. Returns tuple (x, y)

    :param lat: latitude trajectory
    :param lon: longitude trajectory
    :param screen_w: screen width
    :param screen_h: screen height
    """

    x = []
    y = []
    for i in range(len(lat)):
        x.append(int((lon[i] + 180) * (screen_w / 360)))
        y.append(int((90 - lat[i]) * (screen_h / 180)))
    return x, y

def prepare_latlon_for_graph(time: float, lon, lat):
    """
    Function that applies the necessary augmentations to existing lat lon data to be plotted on a 2d graph of earth.
    Applies earth's rotation to the longitude, converts the longitude and latitude to 'Miller's' coordinates (longitude
    is just longitude) and converts lat and lon to degrees.

    - Must take lat and lon in radians.
    - Miller's coordinates is a coordinate system designed to allow plotting on a 2D map of earth. Reference, can be
    found here - https://en.wikipedia.org/wiki/Miller_cylindrical_projection#:~:text=The%20Miller%20cylindrical%20projection%20is,retain%20scale%20along%20the%20equator.

    :param time: used to account for earth's rotation.
    :param lon: longitude.
    :param lat: latitude.
    :return: lat, lon: augmented latitude and longitude.
    """

    # Account for earth's rotation
    earth_rotation = EARTH_ROTATION_ANGLE * time
    lon -= earth_rotation

    # Convert to miller coordinates
    lat = (5 / 4) * np.arcsinh(np.tan((4 * lat) / 5))

    # Convert to degrees
    lon *= (180 / np.pi)
    lat *= (180 / np.pi)

    # Keep between -180 and 180 degrees
    lon = (lon + 180) % 360 - 180
    return lat, lon

def create_covariance_plot(cov_mat, mean_lat, mean_lon):
    """
    Function to create a covariance ellipsis around the mean latitude and longitude.

    A reference for how to do this comes from - https://cookierobotics.com/007/

    :param cov_mat: covariance matrix (radians)
    :param mean_lat: mean latitude (radians)
    :param mean_lon: mean longitude (radians)
    :return: covariance ellipsis latitude and longitude.
    """

    assert cov_mat[0, 1] == cov_mat[1, 0], "off-diagonals of covariance matrix should be equal!"

    a = cov_mat[0, 0]
    b = cov_mat[0, 1]
    c = cov_mat[1, 1]

    root = np.sqrt((((a-c)/2)**2) + b**2)

    lambda_1 = ((a+c)/2) + root
    lambda_2 = ((a+c)/2) - root

    if b == 0 and a >= c:
        theta = 0
    elif b == 0 and a < c:
        theta = np.pi/2
    else:
        theta = np.arctan2(lambda_1-a, b)

    # Compute x and y in radians
    t = np.linspace(0, 2*np.pi, 100)
    lat_cov = np.sqrt(lambda_1) * np.cos(theta) * np.cos(t) - np.sqrt(lambda_2) * np.sin(theta) * np.sin(t) + mean_lat
    lon_cov = np.sqrt(lambda_1) * np.sin(theta) * np.cos(t) + np.sqrt(lambda_2) * np.cos(theta) * np.sin(t) + mean_lon

    return lat_cov, lon_cov




