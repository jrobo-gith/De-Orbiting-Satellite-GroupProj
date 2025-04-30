from main_menu import MainMenu
from instructions import Instructions
from graphs import Graphs
from credits import Credits
import data_gen

from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout,
    QLabel, QStackedWidget, QMainWindow)
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import json
import time
import threading
import numpy as np

with open('partials/global_settings.json') as f:
    glob_setting = json.load(f)

class MasterWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('SNOE Group Project ~ De-Orbiting Satellite')
        self.resize(1080, 720)
        self.setStyleSheet(f"background-color: rgb{glob_setting['background-color']};")

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.stacked_widget.addWidget(MainMenu(self.stacked_widget))
        self.stacked_widget.addWidget(Graphs(self.stacked_widget))
        self.stacked_widget.addWidget(Instructions(self.stacked_widget))
        self.stacked_widget.addWidget(Credits(self.stacked_widget))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MasterWindow()
    window.show()
    sys.exit(app.exec_())
