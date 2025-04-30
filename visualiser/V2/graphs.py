from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout,
    QLabel, QStackedWidget, QMainWindow
)
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QFont

from plot import Plot
import data_gen
from partials.navbar import Navbar

import time
import numpy as np
import json
import threading
with open('partials/global_settings.json') as f:
    glob_setting = json.load(f)


class Graphs(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget

        ## NAVBAR
        navbar = Navbar("Graphs", self.stacked_widget)

        container = QVBoxLayout()
        graph_box = QWidget()
        graph_box.setStyleSheet(f"background-color: rgb{glob_setting['background-color']};")

        container.addWidget(navbar, stretch=1)
        container.addWidget(graph_box, stretch=20)

        self.setLayout(container)