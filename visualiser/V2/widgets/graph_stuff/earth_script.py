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
    def __init__(self):
        super().__init__()
        plot_widget = pg.PlotWidget()

        world_map = "widgets/graph_stuff/images/world_map.jpg"
        world_map = Image.open(world_map).convert('RGB')
        world_map = np.transpose(np.array(world_map), (1, 0, 2))
        world_img = pg.ImageItem(world_map)

        # TEST
        london_lat_lon = (51.509865, -0.118092)
        london_pixel_place = latlon2pixel(london_lat_lon[0], london_lat_lon[1])
        world_plot_TEST = pg.ScatterPlotItem(x=[london_pixel_place[0]], y=[london_pixel_place[1]], size=20, brush=pg.mkBrush('red'))



        plot_widget.addItem(world_img)
        plot_widget.addItem(world_plot_TEST)

        # Plotting details
        plot_widget.setAspectLocked(True)
        plot_widget.hideAxis('left')
        plot_widget.hideAxis('bottom')
        plot_widget.invertY(True)

        self.layout = QVBoxLayout()
        self.layout.addWidget(plot_widget)
        self.setLayout(self.layout)


def latlon2pixel(lat: float, lon:float, screen_w:int=5400, screen_h:int=2700) -> tuple:
    """Returns pixel values for lat and lon. Returns tuple (x, y)"""
    x = (lon + 180) * (screen_w / 360)
    y = (90 - lat) * (screen_h / 180)
    return int(x), int(y)