from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel
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

        # Convert LatLon to pixel
        full_sim_pixels = latlon2pixel(self.lat, self.lon)
        world_plot = pg.ScatterPlotItem(x=full_sim_pixels[0][:-2], y=full_sim_pixels[1][:-2], size=5, brush=pg.mkBrush('red'))
        crash_site = pg.ScatterPlotItem(x=[full_sim_pixels[0][-1]], y=[full_sim_pixels[1][-1]], size=10, brush=pg.mkBrush('blue'))
        self.satellite_start_position = pg.ScatterPlotItem(x=[full_sim_pixels[0][0]], y=[full_sim_pixels[1][0]], size=10, brush=pg.mkBrush('Black'))


        self.plot_widget.addItem(world_img)
        self.plot_widget.addItem(world_plot)
        self.plot_widget.addItem(crash_site)
        self.plot_widget.addItem(self.satellite_start_position)

        # Plotting details
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.hideAxis('left')
        self.plot_widget.hideAxis('bottom')
        self.plot_widget.invertY(True)


        self.layout = QVBoxLayout()
        self.layout.addWidget(self.plot_widget)
        self.setLayout(self.layout)

    @QtCore.pyqtSlot(str, tuple)
    def update_satellite_position(self, name, update):
        lat, lon = update
        self.plot_widget.removeItem(self.satellite_start_position)
        self.satellite_start_position.setData(lat, lon)
        self.plot_widget.addItem(self.satellite_start_position)
        self.setLayout(self.layout)

def latlon2pixel(lat:list, lon:list, screen_w:int=5400, screen_h:int=2700) -> tuple:
    """Returns pixel values for lat and lon. Returns tuple (x, y)"""
    x = []
    y = []
    for i in range(len(lat)):
        x.append(int((lon[i] + 180) * (screen_w / 360)))
        y.append(int((90 - lat[i]) * (screen_h / 180)))
    return x, y