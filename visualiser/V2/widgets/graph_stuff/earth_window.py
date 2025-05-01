import json

import pyqtgraph as pg
from visualiser.V2.widgets.graph_stuff.single_plot import Plot
from visualiser.V2.widgets.graph_stuff import data_gen
from PyQt5 import QtCore
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout
from visualiser.V2.partials.navbar import Navbar


class Earth(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget

        navbar = Navbar("Landing Site Prediction", self.stacked_widget)
        switcher = QHBoxLayout()

        container = QVBoxLayout
        container.addWidget(navbar)

        self.setLayout(container)
