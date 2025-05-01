from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
import pyqtgraph as pg


import json
with open('partials/global_settings.json') as f:
    glob_setting = json.load(f)

class Earth(pg.GraphicsLayoutWidget):
    def __init__(self):
        super().__init__()

        self.layout = QVBoxLayout()

        temp_message = QLabel("This is where the earth will be...")
        temp_message.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        temp_message.setFont(QFont(glob_setting['font-family'], 30))
        temp_message.setStyleSheet(f"color: rgb{glob_setting['font-color']};")


        self.layout.addWidget(temp_message)
        self.setLayout(self.layout)
