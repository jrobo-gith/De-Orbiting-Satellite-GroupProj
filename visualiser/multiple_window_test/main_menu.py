import pyqtgraph as pg
from PyQt5.QtCore import Qt
from pyqtgraph.Qt import QtWidgets, QtCore
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout,
    QLabel, QStackedWidget, QMainWindow
)

class MainMenu(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()

        self.stacked_widget = stacked_widget

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Main Menu"), alignment=Qt.AlignCenter)

        self.grapher_button = QPushButton("Begin Simulation")
        self.instructions_button = QPushButton("Instructions")
        self.credits_button = QPushButton("Credits")

        self.grapher_button.clicked.connect(lambda: stacked_widget.setCurrentIndex(2))
        self.instructions_button.clicked.connect(lambda: stacked_widget.setCurrentIndex(3))
        self.credits_button.clicked.connect(lambda: stacked_widget.setCurrentIndex(4))


        layout.addWidget(self.instructions_button)
        layout.addWidget(self.credits_button)

        self.setLayout(layout)