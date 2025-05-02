from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QStackedWidget, QMainWindow, QScrollArea)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from visualiser.V2.partials.navbar import Navbar
import json
with open('partials/global_settings.json') as f:
    glob_setting = json.load(f)

class Instructions(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget

        ## NAVBAR
        navbar = Navbar("Instructions", self.stacked_widget)

        ## Scroll bar
        scroll_widget = QWidget()

        container = QVBoxLayout()
        container.addWidget(navbar, stretch=1)
        container.addWidget(scroll_widget, stretch=19)

        self.setLayout(container)