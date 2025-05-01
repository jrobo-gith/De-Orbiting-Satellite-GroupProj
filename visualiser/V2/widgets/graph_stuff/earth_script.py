from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout
from visualiser.V2.partials.navbar import Navbar
from visualiser.V2.widgets.graph_stuff.partials.switcher import Switcher



class Earth(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget

        ## Navbar
        navbar = Navbar("Landing Site Prediction", self.stacked_widget)

        ## Switching between graph and earth
        switcher = Switcher(self.stacked_widget)

        # Earth Window
        earth_script = QWidget()


        container = QVBoxLayout()
        container.addWidget(navbar, stretch=1)
        container.addWidget(switcher, stretch=1)
        container.addWidget(earth_script, stretch=18)

        self.setLayout(container)
