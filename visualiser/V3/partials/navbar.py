import os
import sys
import json

# Find the root to make accessible for mac and Windows OS
root_dir = os.getcwd()
sys.path.insert(0, root_dir)

# Import everything needed from PyQt5
from PyQt5.QtWidgets import (QWidget, QPushButton, QLabel, QGridLayout)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

# Import global settings for font-family, font-color, etc.
json_file = os.path.join(root_dir, "visualiser/V2/partials/global_settings.json")
with open(json_file) as f:
    glob_setting = json.load(f)

class Navbar(QWidget):
    """
    Universal Navbar used in model_menu.py, instructions.py, credits.py. Provides general navbar functions
    including a title, and a button that takes the user back to the main menu.

    Functions:
    - __init__(self, name, stacked_widget)
    """
    def __init__(self, name:str, stacked_widget):
        """
        Initializes the Navbar class, takes inputs:

        self : class instance,
        name : str : Name for the top of the navbar such as instructions or credits.
        stacked_widget: QStackedWidget : Used to provide functionality to the back-arrow. Contains layout from
        main_menu such that the user can navigate back to the main menu easily.
        """
        super().__init__()

        self.name = name
        self.stacked_widget = stacked_widget

        # Layout as a grid to get the title in the middle and the back arrow button on the left
        navbar = QGridLayout()
        self.setLayout(navbar)

        title = QLabel(self.name)
        title.setFont(QFont(glob_setting['font-family'], 35))
        title.setStyleSheet(f"color: rgb{glob_setting['font-color']}; text-decoration: underline;background: {glob_setting['background-color']}")

        backarrow = QPushButton("<--")
        backarrow.setStyleSheet(f"color: rgb{glob_setting['font-color']};background: {glob_setting['background-color']}")
        backarrow.setFont(QFont(glob_setting['font-family'], 35))
        backarrow.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))

        # Crude filler to get the title to stay in the middle and not appear off to the side.
        filler = QLabel("FILLER")
        filler.setStyleSheet(f"color: {glob_setting['background-color']};")

        # Add all widgets to the layout.
        navbar.addWidget(title, 0, 1, Qt.AlignHCenter)
        navbar.addWidget(backarrow, 0, 0, Qt.AlignHCenter)
        navbar.addWidget(filler, 0, 2)



