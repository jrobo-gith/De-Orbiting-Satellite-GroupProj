from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout,
    QLabel, QStackedWidget, QMainWindow
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from visualiser.V2.partials.navbar import Navbar

import json
with open('partials/global_settings.json') as f:
    glob_setting = json.load(f)

class Credits(QWidget):
    def __init__(self, stacked_widget):
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
        vis_title = QLabel("Visualiser")
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
            credit.setStyleSheet(f"color: rgb{glob_setting['font-color']}; background: rgba(0, 0, 0, 0.7); border-radius: 30px;")
            credit.setAlignment(Qt.AlignHCenter)
            credit.setFont(QFont(glob_setting['font-family'], 15))
        for title in credits_titles_list:
            title.setFont(QFont(glob_setting['font-family'], 20, QFont.Bold))

        container = QVBoxLayout()
        container.addWidget(navbar, stretch=1)
        container.addLayout(credits_box, stretch=20)

        self.setLayout(container)