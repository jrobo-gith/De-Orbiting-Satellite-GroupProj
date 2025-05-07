import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
from PyQt5.QtWidgets import QWidget


class Credits(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()

        self.stacked_widget = stacked_widget