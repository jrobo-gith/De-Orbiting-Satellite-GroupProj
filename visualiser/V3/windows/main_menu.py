import sys
import os
import json

root_dir = os.getcwd()

from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel
from PyQt5.QtCore import Qt

# Import global settings
json_file = os.path.join(root_dir, "visualiser/V2/partials/global_settings.json")
with open(json_file) as f:
    glob_setting = json.load(f)

class MainMenu(QWidget):
    """
    Window displaying the main menu, where the user can navigate to any part of this project. At the top is a welcome
    message and underneath are buttons that, when pushed, take the user to the respective window.

    This window is the parent of model_menu.py, instructions.py and credits.py. It also has an 'exit' button which
    exits the program. Each button executes the function within the 'clicked.connect()' method. In this case, sets
    the live window in the stacked widget to whatever index that button is accessing.

    Functions:
    - __init__(self, stacked_widget):

    References:
        Tutorial followed for PyQt5 (GUI) can be found here - https://www.pythonguis.com/pyqt5-tutorial/

    Previous versions can be found in the Group GitHub - https://github.com/jrobo-gith/De-Orbiting-Satellite-GroupProj
    """
    def __init__(self, stacked_widget):
        """
        Initialises the main_menu window, also contains 'QLabel's and 'QPushButton's that are stacked on top of each
        other in a Vertical Box Layout (QVBoxLayout) to produce the main menu window. Each button is wrapped in a
        Horizontal Box Layout (QHBoxLayout) to keep the button in the middle, and not span the entire page.

        :param stacked_widget: Widget containing pages of the GUI, used to navigate back to the main menu.
        """
        super().__init__()
        self.stacked_widget = stacked_widget

        # Menu
        layout = QVBoxLayout()

        menu_title = QLabel("Main Menu")
        menu_title.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        menu_title.setFont(QFont(glob_setting['font-family'], 30))
        menu_title.setStyleSheet(f"color: rgb{glob_setting['font-color']}; background: transparent; text-decoration: underline")
        menu_title.setFixedWidth(1080)
        menu_layout = QHBoxLayout()
        menu_layout.addWidget(menu_title)

        menu_welcome_text = QLabel("Welcome to our group project! Here we use the __ Kalman Filter to predict the crash site of a \nDe-orbiting satellite.")
        menu_welcome_text.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        menu_welcome_text.setFont(QFont(glob_setting['font-family'], 20))
        menu_welcome_text.setStyleSheet(f"color: rgb{glob_setting['font-color']}; background: transparent;")
        menu_welcome_text.setFixedWidth(1080)
        menu_welcome_layout = QHBoxLayout()
        menu_welcome_layout.addWidget(menu_welcome_text)

        graph_button = QPushButton("Simulation")
        graph_button.setFont(QFont(glob_setting['font-family'], 27))
        graph_button.setStyleSheet(f"color: rgb{glob_setting['font-color']}; background: {glob_setting['background-color']}; border-radius: 10%;")
        graph_button.clicked.connect(lambda: stacked_widget.setCurrentIndex(1))
        graph_button.setFixedWidth(180)
        graph_layout = QHBoxLayout()
        graph_layout.addWidget(graph_button)

        instructions_button = QPushButton("Instructions")
        instructions_button.setFont(QFont(glob_setting['font-family'], 25))
        instructions_button.setStyleSheet(f"color: rgb{glob_setting['font-color']}; background: {glob_setting['background-color']}; border-radius: 10%;")
        instructions_button.clicked.connect(lambda: stacked_widget.setCurrentIndex(2))
        instructions_button.setFixedWidth(165)
        instructions_layout = QHBoxLayout()
        instructions_layout.addWidget(instructions_button)

        credits_button = QPushButton("Credits")
        credits_button.setFont(QFont(glob_setting['font-family'], 23))
        credits_button.setStyleSheet(f"color: rgb{glob_setting['font-color']}; background: {glob_setting['background-color']}; border-radius: 10%;")
        credits_button.clicked.connect(lambda: stacked_widget.setCurrentIndex(3))
        credits_button.setFixedWidth(100)
        credits_layout = QHBoxLayout()
        credits_layout.addWidget(credits_button)

        exit_button = QPushButton("Exit")
        exit_button.setFont(QFont(glob_setting['font-family'], 20))
        exit_button.setStyleSheet(f"color: rgb{glob_setting['font-color']}; background: {glob_setting['background-color']}; border-radius: 10%; ")
        exit_button.clicked.connect(sys.exit)
        exit_button.setFixedWidth(70)
        exit_layout = QHBoxLayout()
        exit_layout.addWidget(exit_button)


        space_taker1 = QLabel()
        space_taker1.setStyleSheet("background: transparent")
        space_taker2 = QLabel()
        space_taker2.setStyleSheet("background: transparent")

        layout.addLayout(menu_layout, stretch=10)
        layout.addLayout(menu_welcome_layout, stretch=10)
        layout.addWidget(space_taker1, stretch=40)
        layout.addLayout(graph_layout, stretch=5)
        layout.addLayout(instructions_layout, stretch=5)
        layout.addLayout(credits_layout, stretch=5)
        layout.addLayout(exit_layout, stretch=5)
        layout.addWidget(space_taker2, stretch=20)


        self.setLayout(layout)
