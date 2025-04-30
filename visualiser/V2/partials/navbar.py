from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout,
    QLabel, QStackedWidget, QMainWindow
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import json
with open('partials/global_settings.json') as f:
    glob_setting = json.load(f)

class Navbar(QWidget):
    def __init__(self, title:str, stacked_widget):
        super().__init__()

        self.title = title
        self.stacked_widget = stacked_widget

        navbar = QHBoxLayout()
        title = QLabel(self.title)
        title.setAlignment(Qt.AlignHCenter)
        title.setFont(QFont(glob_setting['font-family'], 30))
        title.setStyleSheet(f"color: rgb{glob_setting['font-color']};")

        backarrow = QPushButton("<-")
        backarrow.setStyleSheet(f"color: rgb{glob_setting['font-color']}; text-align: left; position: fixed")
        backarrow.setFont(QFont(glob_setting['font-family'], 30))
        backarrow.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))

        navbar.addWidget(backarrow, stretch=0)
        navbar.addWidget(title, stretch=50)

        self.setLayout(navbar)


