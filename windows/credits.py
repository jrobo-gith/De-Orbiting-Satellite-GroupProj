import os
import sys
import json

root_dir = os.getcwd()
sys.path.insert(0, root_dir)

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from partials.navbar import Navbar

# Import global settings
json_file = os.path.join(root_dir, "partials/global_settings.json")
with open(json_file) as f:
    glob_setting = json.load(f)

class Credits(QWidget):
    """
    Window displaying the credits to attribute credit to the primary contributors of each section. By the end, everyone
    worked on each part but the main contributors are noted in this window.
    This window's parent is the MasterWindow in master.py, navigable to using the MainMenu under the 'credits' button.

    Functions:
    - __init__(self, stacked_widget)
    References:
        Tutorial followed for PyQt5 (GUI) can be found here - https://www.pythonguis.com/pyqt5-tutorial/

    Previous versions can be found in the Group GitHub - https://github.com/jrobo-gith/De-Orbiting-Satellite-GroupProj
    """
    def __init__(self, stacked_widget):
        """
        Initialises the credits window, contains an instance of the 'navbar' class, also contains 'QLabels' that are
        stacked on top of each other in a Vertical Box Layout (QVBoxLayout) to produce the credits window.

        Args:
            stacked_widget: Widget containing pages of the GUI, used to navigate back to the main menu.
        """
        super().__init__()
        self.stacked_widget = stacked_widget

        ## NAVBAR
        navbar = Navbar("Credits", self.stacked_widget)

        ## Add credits box
        credits_box = QVBoxLayout()
        credits_list = []
        credits_titles_list = []

        ## Add Credits
        ### Predictor
        pred_title = QLabel("Predictor")
        bee_credit = QLabel("Biqing Wang")
        khad_credit = QLabel("Khadijah Mughrbil")
        ### Simulator
        sim_title = QLabel("Simulator")
        jai_credit = QLabel("Jai Sagoo")
        vijay_credit = QLabel("Vijay Dharmavel")
        ### Visualiser
        vis_title = QLabel("Visualiser/GUI")
        jack_credit = QLabel("Jack Roberts")

        # Add credits to box
        credits_box.addWidget(pred_title)
        credits_box.addWidget(bee_credit)
        credits_box.addWidget(khad_credit)
        credits_box.addWidget(sim_title)
        credits_box.addWidget(jai_credit)
        credits_box.addWidget(vijay_credit)
        credits_box.addWidget(vis_title)
        credits_box.addWidget(jack_credit)

        # Add credits to list for adding styles
        credits_list.append(pred_title)
        credits_list.append(bee_credit)
        credits_list.append(khad_credit)
        credits_list.append(sim_title)
        credits_list.append(jai_credit)
        credits_list.append(vijay_credit)
        credits_list.append(vis_title)
        credits_list.append(jack_credit)

        ## Add credits titles to increase font
        credits_titles_list.append(pred_title)
        credits_titles_list.append(sim_title)
        credits_titles_list.append(vis_title)

        for credit in credits_list:
            # For loop to set style sheets of each QLabel
            credit.setStyleSheet(f"color: rgb{glob_setting['font-color']}; background: rgba(0, 0, 0, 0.7); border-radius: 30px;")
            credit.setAlignment(Qt.AlignHCenter)
            credit.setFont(QFont(glob_setting['font-family'], 15))
        for title in credits_titles_list:
            # Loop to increase each title's font.
            title.setFont(QFont(glob_setting['font-family'], 20, QFont.Bold))

        container = QVBoxLayout()
        container.addWidget(navbar, stretch=1)
        container.addLayout(credits_box, stretch=20)

        self.setLayout(container)