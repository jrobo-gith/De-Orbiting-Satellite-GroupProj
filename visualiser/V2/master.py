from visualiser.V2.widgets.main_menu import MainMenu
from visualiser.V2.widgets.instructions import Instructions
from visualiser.V2.widgets.simulation_menu import SimulationMenu
from visualiser.V2.widgets.credits import Credits

from PyQt5.QtWidgets import (QApplication, QStackedWidget, QMainWindow, QLabel, QWidget, QStackedLayout)
import sys
import json

with open('partials/global_settings.json') as f:
    glob_setting = json.load(f)

class MasterWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('SNOE Group Project ~ De-Orbiting Satellite')
        self.resize(glob_setting['screen-height'], glob_setting['screen-width'])
        self.setStyleSheet(f"""background-color: rgba{glob_setting['background-color']};
        background-image: url(partials/images/backgroundimage.jpg);
        background-repeat: no-repeat;""")

        self.stacked_widget = QStackedWidget()
        self.stacked_widget.addWidget(MainMenu(self.stacked_widget))
        self.stacked_widget.addWidget(SimulationMenu(self.stacked_widget))
        self.stacked_widget.addWidget(Instructions(self.stacked_widget))
        self.stacked_widget.addWidget(Credits(self.stacked_widget))

        self.setCentralWidget(self.stacked_widget)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MasterWindow()
    window.show()
    sys.exit(app.exec_())