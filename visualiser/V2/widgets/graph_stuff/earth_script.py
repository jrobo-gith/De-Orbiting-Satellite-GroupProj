from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPixmap
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui

from PIL import Image
import json
import numpy as np

with open('partials/global_settings.json') as f:
    glob_setting = json.load(f)

class Earth(pg.GraphicsLayoutWidget):
    def __init__(self, full_sim_data):
        super().__init__()

        self.lat, self.lon = full_sim_data

        self.plot_widget = pg.PlotWidget()

        world_map = "widgets/graph_stuff/images/world_map.jpg"
        world_map = Image.open(world_map).convert('RGB')
        world_map = np.transpose(np.array(world_map), (1, 0, 2))
        world_img = pg.ImageItem(world_map)

        # Simulation Overlay
        full_sim_pixels = latlon2pixel(self.lat, self.lon)
        self.sim_plot = pg.ScatterPlotItem(x=full_sim_pixels[0][:-2], y=full_sim_pixels[1][:-2], size=5, brush=pg.mkBrush('red'))
        self.crash_site = pg.ScatterPlotItem(x=[full_sim_pixels[0][-1]], y=[full_sim_pixels[1][-1]], size=10, brush=pg.mkBrush('blue'))
        self.satellite_start_position = pg.ScatterPlotItem(x=[full_sim_pixels[0][0]], y=[full_sim_pixels[1][0]], size=10, brush=pg.mkBrush('Black'))
        self.sim_plot.setOpacity(0.9)
        self.crash_site.setOpacity(0.9)
        self.satellite_start_position.setOpacity(0.9)

        # Prediction Overlay (Alternate variable "size" to show uncertainty)
        x, y = latlon2pixel([51.509865], [-0.118092])
        self.prediction_crash_point = pg.ScatterPlotItem(x=x, y=y, size=10, brush=pg.mkBrush('green'), pen=pg.mkPen('green'))
        self.prediction_crash_1std = pg.ScatterPlotItem(x=x, y=y, size=30, brush=pg.mkBrush(0, 255, 0))
        self.prediction_crash_2std = pg.ScatterPlotItem(x=x, y=y, size=60, brush=pg.mkBrush((0, 255, 0), pen=pg.mkPen('green')))
        self.prediction_crash_point.setOpacity(0.9)
        self.prediction_crash_1std.setOpacity(0.5)
        self.prediction_crash_2std.setOpacity(0.25)

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

        # Plotting details
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.hideAxis('left')
        self.plot_widget.hideAxis('bottom')
        self.plot_widget.invertY(True)
        self.viewbox = self.plot_widget.getViewBox()
        self.viewbox.setRange(xRange=(0, 5400), yRange=(2700, 0))

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


    @QtCore.pyqtSlot(str, tuple)
    def update_satellite_position(self, name, update):
        lat, lon = update
        x, y = latlon2pixel(lat, lon)
        self.satellite_start_position.setData(x, y)

    def sim_overlay_switch(self):
        if self.simulation_checkbox.isChecked():
            self.satellite_start_position.show()
            self.sim_plot.show()
            self.crash_site.show()
        else:
            self.satellite_start_position.hide()
            self.sim_plot.hide()
            self.crash_site.hide()

    def radar_overlay_switch(self):
        if self.radar_checkbox.isChecked():
            print("Radar Checked!")
        else:
            print("Radar Not Checked!")

    def prediction_overlay_switch(self):
        if self.prediction_checkbox.isChecked():
            self.prediction_crash_point.show()
            self.prediction_crash_1std.show()
            self.prediction_crash_2std.show()
        else:
            self.prediction_crash_point.hide()
            self.prediction_crash_1std.hide()
            self.prediction_crash_2std.hide()

def latlon2pixel(lat:list, lon:list, screen_w:int=5400, screen_h:int=2700) -> tuple:
    """Returns pixel values for lat and lon. Returns tuple (x, y)"""
    x = []
    y = []
    for i in range(len(lat)):
        x.append(int((lon[i] + 180) * (screen_w / 360)))
        y.append(int((90 - lat[i]) * (screen_h / 180)))
    return x, y