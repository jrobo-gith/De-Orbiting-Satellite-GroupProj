import os
import sys
from PIL import Image
import json
import numpy as np
from visualiser.V3.debug import debug_print
from visualiser.V3.partials.constants import EARTH_ROTATION_ANGLE

root_dir = os.getcwd()
sys.path.insert(0, root_dir)

# Import necessary PyQt5 components
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QCheckBox
from PyQt5.QtCore import Qt
import pyqtgraph as pg
from PyQt5 import QtCore

# Import Global settings
json_file = os.path.join(root_dir, "visualiser/V3/partials/global_settings.json")
with open(json_file) as f:
    glob_setting = json.load(f)

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
    def __init__(self, full_sim_data, radar_list):
        """
        Initialises the Earth Window, converts a .jpg image of the world into a plottable ImageItem through a numpy
        array. Also takes the full simulation data, ran in simulator_window.py, and plots it on the 2D map of earth.

        Also lays out optional checkboxes which turn simulation data, radar locations or prediction info off and on at
        the user's convenience.

        :param full_sim_data: the full simulation data in lat lon, used to plot the full simulation on the 2d world map.
        :param radar_list: list of radar lat lons, initialised in simulator_window.py, used to plot on the 2d world map.
        """
        super().__init__()

        self.lat, self.lon, self.t = full_sim_data
        self.radar_list = radar_list

        if len(self.lat) > 10_000:
            self.lat = self.lat[0:len(self.lat):10]
            self.lon = self.lon[0:len(self.lon):10]
            self.adjusted_t = self.t[0:len(self.t):10]
        else:
            self.adjusted_t = self.t

        self.plot_widget = pg.PlotWidget()

        world_map = os.path.join(root_dir, "visualiser/V3/windows/model/model_window/earth_graph_windows/earth/world_map.jpg")
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

        self.lon = (self.lon + 180) % 360 - 180  # wrap to [-180, 180]

        ## Switch x and y (lat and lon) to account for inverting world axis
        self.x = self.lat
        self.y = self.lon

        # Simulation Overlay
        full_sim_pixels = latlon2pixel(self.x, self.y)
        self.sim_plot = pg.ScatterPlotItem(x=full_sim_pixels[0][:-1], y=full_sim_pixels[1][:-1], size=5, brush=pg.mkBrush('yellow'))
        self.crash_site = pg.ScatterPlotItem(x=[full_sim_pixels[0][-1]], y=[full_sim_pixels[1][-1]], size=10, brush=pg.mkBrush('red'))
        self.satellite_start_position = pg.ScatterPlotItem(x=[full_sim_pixels[0][0]], y=[full_sim_pixels[1][0]], size=20, brush=pg.mkBrush('black'))
        self.sim_plot.setOpacity(0.9)
        self.crash_site.setOpacity(0.9)
        self.satellite_start_position.setOpacity(0.9)

        # Prediction Overlay (Alternate variable "size" to show uncertainty)
        x, y = latlon2pixel(np.array([0]), np.array([0]))
        self.prediction_crash_point = pg.ScatterPlotItem(x=x, y=y, size=10, brush=pg.mkBrush('green'), pen=pg.mkPen('green'))
        self.prediction_crash_1std = pg.ScatterPlotItem(x=x, y=y, size=30, brush=pg.mkBrush(0, 255, 0))
        self.prediction_crash_2std = pg.ScatterPlotItem(x=x, y=y, size=60, brush=pg.mkBrush((0, 255, 0), pen=pg.mkPen('green')))
        self.prediction_crash_point.setOpacity(0.9)
        self.prediction_crash_1std.setOpacity(0.5)
        self.prediction_crash_2std.setOpacity(0.25)

        # Add Radars locations
        self.radar_plots = []
        for radar in self.radar_list:
            x, y = latlon2pixel(np.array([radar[1]]), np.array([radar[0]]))
            radar_plot = pg.ScatterPlotItem(x=x, y=y, size=10, brush=pg.mkBrush('blue'))
            self.radar_plots.append(radar_plot)

        # Add world image
        self.plot_widget.addItem(world_img)
        # Add simulation data
        self.plot_widget.addItem(self.sim_plot)
        self.plot_widget.addItem(self.crash_site)
        self.plot_widget.addItem(self.satellite_start_position)
        # Add prediction data
        self.plot_widget.addItem(self.prediction_crash_point)
        self.plot_widget.addItem(self.prediction_crash_1std)
        self.plot_widget.addItem(self.prediction_crash_2std)
        [self.plot_widget.addItem(radar_plot) for radar_plot in self.radar_plots]

        # Plotting details
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.hideAxis('left')
        self.plot_widget.hideAxis('bottom')
        self.plot_widget.invertY(True)
        self.viewbox = self.plot_widget.getViewBox()
        self.viewbox.setRange(xRange=(0, 5400), yRange=(2700, 0))
        self.sim_plot.getViewBox().invertY(True) # Keep consistent with world map

        # Filter details
        self.filter_layout = QHBoxLayout()

        self.simulation_checkbox = QCheckBox("Show Simulation")
        self.simulation_checkbox.setStyleSheet(f"background-color: rgba{glob_setting['background-color']}; color: rgb{glob_setting['font-color']}")
        self.simulation_checkbox.setChecked(True)
        self.simulation_checkbox.stateChanged.connect(self.sim_overlay_switch)

        self.radar_checkbox = QCheckBox("Show Radars")
        self.radar_checkbox.setStyleSheet(f"background-color: rgba{glob_setting['background-color']}; color: rgb{glob_setting['font-color']}")
        self.radar_checkbox.setChecked(True)
        self.radar_checkbox.stateChanged.connect(self.radar_overlay_switch)

        self.prediction_checkbox = QCheckBox("Show Prediction")
        self.prediction_checkbox.setStyleSheet(f"background-color: rgba{glob_setting['background-color']}; color: rgb{glob_setting['font-color']}")
        self.prediction_checkbox.setChecked(True)
        self.prediction_checkbox.stateChanged.connect(self.prediction_overlay_switch)

        self.filter_layout.addWidget(self.simulation_checkbox)
        self.filter_layout.addWidget(self.radar_checkbox)
        self.filter_layout.addWidget(self.prediction_checkbox)
        self.filter_layout.setAlignment(Qt.AlignCenter)

        self.earth_filter = QVBoxLayout()
        self.earth_filter.addWidget(self.plot_widget, stretch=19)
        self.earth_filter.addLayout(self.filter_layout, stretch=1)
        self.setLayout(self.earth_filter)


    @QtCore.pyqtSlot(dict, tuple)
    def update_satellite_position(self, name, update):
        """
        Function to update the latitude and longitude of the satellite as it travels around the earth. Gets information
        from simulator/radar via multi-threading and a 'helper' function, which emits the satellite's location at a
        given time interval. We receive the data as (dict, tuple), we do not use the dict (name) in this instance, and
        it is redundant, and the update is a tuple containing the latitude and longitude of the satellite.

        We update the scatter plot 'self.satellite_start_position' using the function 'setData(x, y)'

        :param name: redundant parameter
        :param update: contains one latitude and one longitude to update the satellite's position.
        """
        lat, lon = update
        x, y = latlon2pixel([lat], [lon])
        self.satellite_start_position.setData(x, y)

    @QtCore.pyqtSlot(dict, tuple)
    def update_prediction(self, info, update):
        """
        Function to update the prediction of the crash site as it travels around the earth. Gets information from
        predictor/predictor.py via multi-threading and a 'helper' function, which emits the latitude and longitude of
        a point at which the satellite crashes onto earth.

        :param info: contains info such as whether we are predicting landing or not
        :param update: contains the latitude and longitude of the predicted crash site
        """
        lat, lon = update

        # Account for earth's rotation
        earth_rotation = EARTH_ROTATION_ANGLE * info['time']
        lon -= earth_rotation

        # Convert to miller coordinates
        lat = (5/4) * np.arcsinh(np.tan((4*lat)/5))

        # Convert to degrees
        lon *= (180 / np.pi)
        lat *= (180 / np.pi)

        # Keep between -180 and 180 degrees
        lon = (lon + 180) % 360 - 180

        x, y = latlon2pixel([lat], [lon])
        self.prediction_crash_point.setData(x, y)
        self.prediction_crash_1std.setData(x, y)
        self.prediction_crash_2std.setData(x, y)

    def sim_overlay_switch(self):
        """
        Function that sits inside the 'self.simulation_checkbox' QCheckBox. When altered (checked or unchecked), the
        function asks if the checkbox is checked, if not, it checks it, if it is, it unchecks it.
        """
        if self.simulation_checkbox.isChecked():
            self.satellite_start_position.show()
            self.sim_plot.show()
            self.crash_site.show()
        else:
            self.satellite_start_position.hide()
            self.sim_plot.hide()
            self.crash_site.hide()

    def radar_overlay_switch(self):
        """
        Function that sits inside the 'self.radar_checkbox' QCheckBox. When altered (checked or unchecked), the
        function asks if the checkbox is checked, if not, it checks it, if it is, it unchecks it.
        """
        if self.radar_checkbox.isChecked():
            [radar_plot.show() for radar_plot in self.radar_plots]
        else:
            [radar_plot.hide() for radar_plot in self.radar_plots]

    def prediction_overlay_switch(self):
        """
        Function that sits inside the 'self.prediction_checkbox' QCheckBox. When altered (checked or unchecked), the
        function asks if the checkbox is checked, if not, it checks it, if it is, it unchecks it.
        """
        if self.prediction_checkbox.isChecked():
            self.prediction_crash_point.show()
            self.prediction_crash_1std.show()
            self.prediction_crash_2std.show()
        else:
            self.prediction_crash_point.hide()
            self.prediction_crash_1std.hide()
            self.prediction_crash_2std.hide()

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