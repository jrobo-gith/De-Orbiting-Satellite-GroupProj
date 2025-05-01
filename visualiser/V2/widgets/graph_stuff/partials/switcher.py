from PyQt5.QtWidgets import QWidget, QHBoxLayout, QPushButton
from PyQt5.QtGui import QFont


import json
with open('partials/global_settings.json') as f:
    glob_setting = json.load(f)

class Switcher(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget

        self.layout = QHBoxLayout()

        graph_button = QPushButton("Graph Window")
        graph_button.setStyleSheet(f"color: rgb{glob_setting['font-color']};")
        graph_button.setFont(QFont(glob_setting['font-family'], 30))
        graph_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))


        earth_button = QPushButton("Earth Window")
        earth_button.setStyleSheet(f"color: rgb{glob_setting['font-color']};")
        earth_button.setFont(QFont(glob_setting['font-family'], 30))
        earth_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(1))

        self.layout.addWidget(graph_button)
        self.layout.addWidget(earth_button)

        self.setLayout(self.layout)