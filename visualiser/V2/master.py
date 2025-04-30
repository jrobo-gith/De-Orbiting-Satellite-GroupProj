from visualiser.V2.widgets.main_menu import MainMenu
from visualiser.V2.widgets.instructions import Instructions
from visualiser.V2.widgets.graph_stuff.graphs import Graphs
from visualiser.V2.widgets.credits import Credits

from PyQt5.QtWidgets import (
    QApplication, QStackedWidget, QMainWindow)
import sys
import json

with open('partials/global_settings.json') as f:
    glob_setting = json.load(f)

class MasterWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('SNOE Group Project ~ De-Orbiting Satellite')
        self.resize(1080, 720)
        self.setStyleSheet(f"background-color: rgb{glob_setting['background-color']};")

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.stacked_widget.addWidget(MainMenu(self.stacked_widget))
        self.stacked_widget.addWidget(Graphs(self.stacked_widget))
        self.stacked_widget.addWidget(Instructions(self.stacked_widget))
        self.stacked_widget.addWidget(Credits(self.stacked_widget))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MasterWindow()
    window.show()
    sys.exit(app.exec_())
