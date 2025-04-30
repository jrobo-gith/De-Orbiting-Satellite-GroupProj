from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout,
    QLabel, QStackedWidget, QMainWindow
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from partials.navbar import Navbar

import json
with open('partials/global_settings.json') as f:
    glob_setting = json.load(f)

class Credits(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget

        ## NAVBAR
        navbar = Navbar("Credits", self.stacked_widget)

        container = QVBoxLayout()
        instructions_box = QWidget()
        instructions_box.setStyleSheet(f"background-color: rgb{glob_setting['background-color']};")

        container.addWidget(navbar, stretch=1)
        container.addWidget(instructions_box, stretch=20)

        self.setLayout(container)