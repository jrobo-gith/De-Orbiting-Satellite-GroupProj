from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QStackedWidget, QLabel, QLineEdit, QPushButton, QHBoxLayout)
from PyQt5 import QtCore
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt

from visualiser.V2.widgets.graph_stuff.simulator_window import SimWidget

from visualiser.V2.partials.navbar import Navbar
import json
with open('partials/global_settings.json') as f:
    glob_setting = json.load(f)


class SimulationMenu(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget

        ## Layout
        page_container = QVBoxLayout()
        navbar = Navbar("Simulation Menu", self.stacked_widget)

        input_position = QLineEdit()

        start_sim_btn = QPushButton("Start Simulation")
        start_sim_btn.setFont(QFont(glob_setting['font-family'], 27))
        start_sim_btn.setStyleSheet(f"color: rgb{glob_setting['font-color']};")
        start_sim_btn.clicked.connect(self.loadSim)

        start_sim =QHBoxLayout()
        start_sim.addWidget()
        start_sim.addWidget(start_sim_btn)
        start_sim.setAlignment(Qt.AlignCenter)

        page_container.addWidget(navbar, stretch=1)
        page_container.addLayout(start_sim, stretch=19)
        self.setLayout(page_container)


    def loadSim(self):
        if self.stacked_widget.count() > 4: # Means self.sim_stacked_widget doesn't exist
            self.stacked_widget.removeWidget(self.sim_stacked_widget) # Remove old instance
            self.sim_stacked_widget = QStackedWidget() # Create new simulation
            self.sim_stacked_widget.addWidget(SimulationMenu(self.stacked_widget))
            self.sim_stacked_widget.addWidget(SimWidget(self.stacked_widget))
            self.sim_stacked_widget.setCurrentIndex(1)
            self.stacked_widget.addWidget(self.sim_stacked_widget)
            self.stacked_widget.setCurrentIndex(4)
        else:
            # Create stacked widget (load in sim window)
            self.sim_stacked_widget = QStackedWidget()
            self.sim_stacked_widget.addWidget(SimulationMenu(self.stacked_widget))
            self.sim_stacked_widget.addWidget(SimWidget(self.stacked_widget))
            self.sim_stacked_widget.setCurrentIndex(1)
            self.stacked_widget.addWidget(self.sim_stacked_widget)
            self.stacked_widget.setCurrentIndex(4)