from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout,
    QLabel, QStackedWidget, QMainWindow)
from PyQt5.QtCore import Qt
import json
import sys

with open('partials/global_settings.json') as f:
    glob_setting = json.load(f)


class MainMenu(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget

        # Menu
        layout = QVBoxLayout()

        menu_title = QLabel("Main Menu")
        menu_title.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        menu_title.setFont(QFont(glob_setting['font-family'], 30))
        menu_title.setStyleSheet(f"color: rgb{glob_setting['font-color']};")

        menu_welcome_text = QLabel("Welcome to our group project! Here we use the __ Kalman Filter to predict the crash site of a \nDe-orbiting satellite.")
        menu_welcome_text.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        menu_welcome_text.setFont(QFont(glob_setting['font-family'], 20))
        menu_welcome_text.setStyleSheet(f"color: rgb{glob_setting['font-color']};")

        graph_button = QPushButton("Begin Simulation")
        graph_button.setFont(QFont(glob_setting['font-family'], 27))
        graph_button.setStyleSheet(f"color: rgb{glob_setting['font-color']};")
        graph_button.clicked.connect(lambda: stacked_widget.setCurrentIndex(1))

        instructions_button = QPushButton("Instructions")
        instructions_button .setFont(QFont(glob_setting['font-family'], 25))
        instructions_button .setStyleSheet(f"color: rgb{glob_setting['font-color']};")
        instructions_button .clicked.connect(lambda: stacked_widget.setCurrentIndex(2))

        credits_button = QPushButton("Credits")
        credits_button.setFont(QFont(glob_setting['font-family'], 23))
        credits_button.setStyleSheet(f"color: rgb{glob_setting['font-color']};")
        credits_button.clicked.connect(lambda: stacked_widget.setCurrentIndex(3))

        exit_button = QPushButton("Exit")
        exit_button.setFont(QFont(glob_setting['font-family'], 20))
        exit_button.setStyleSheet(f"color: rgb{glob_setting['font-color']};")
        exit_button.clicked.connect(sys.exit)

        space_taker1 = QWidget()
        space_taker2 = QWidget()


        layout.addWidget(menu_title, stretch=10)
        layout.addWidget(menu_welcome_text, stretch=10)
        layout.addWidget(space_taker1, stretch=40)
        layout.addWidget(graph_button, stretch=5)
        layout.addWidget(instructions_button, stretch=5)
        layout.addWidget(credits_button, stretch=5)
        layout.addWidget(exit_button, stretch=5)
        layout.addWidget(space_taker2, stretch=20)
<<<<<<< HEAD:visualiser/V2/widgets/main_menu.py
=======



>>>>>>> main:visualiser/V2/main_menu.py


        self.setLayout(layout)
