from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout,
    QLabel, QStackedWidget, QMainWindow, QGridLayout
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import json
with open('partials/global_settings.json') as f:
    glob_setting = json.load(f)

class Navbar(QWidget):
    def __init__(self, name:str, stacked_widget):
        super().__init__()

        self.name = name
        self.stacked_widget = stacked_widget

        navbar = QGridLayout()
        self.setLayout(navbar)

        title = QLabel(self.name)
        title.setFont(QFont(glob_setting['font-family'], 35))
        title.setStyleSheet(f"color: rgb{glob_setting['font-color']};")

        backarrow = QPushButton("<--")
        backarrow.setStyleSheet(f"color: rgb{glob_setting['font-color']};")
        backarrow.setFont(QFont(glob_setting['font-family'], 35))
        backarrow.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))


        filler = QLabel("FILLER")
        filler.setStyleSheet(f"color: {glob_setting['background-color']};")

        navbar.addWidget(title, 0, 1, Qt.AlignHCenter)
        navbar.addWidget(backarrow, 0, 0, Qt.AlignHCenter)
        navbar.addWidget(filler, 0, 2)



