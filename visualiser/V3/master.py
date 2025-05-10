import sys
import json
import subprocess
import os 
from debug import debug_print

# Get root directory to run from mac and windows.
root_dir = os.getcwd()
sys.path.insert(0, root_dir)
req_file = os.path.join(root_dir, "visualiser/V2/requirements.txt")
debug_print("visualiser", os.path.exists(req_file))

# Install required files
cmd = [sys.executable, "-m", "pip", "install", "-r", req_file]
subprocess.run(cmd, check=True)

# Import windows from other files
from widgets.main_menu import MainMenu
from widgets.instructions import Instructions
from widgets.simulation_menu import SimulationMenu
from widgets.credits import Credits

# Import Necessary Widgets
from PyQt5.QtWidgets import QApplication, QStackedWidget, QMainWindow

json_file = os.path.join(root_dir, "visualiser/V2/partials/global_settings.json")
with open(json_file) as f:
    glob_setting = json.load(f)

# Root for background image
image_file = os.path.join(root_dir, "visualiser/V2/partials/images/backgroundimage.jpg")

class MasterWindow(QMainWindow):
    """
    Master window that collects all windows and loads them sequentially in a 'stacked widget'. It is not a window
    itself, but a program that displays them.
    Imports .json file 'global_settings' containing the essential info for the entire window to provide easy access to
    changes if they need to be made.

    'global_settings' includes:
    "font-family":
    "background-color":
    "matrix-color":
    "font-color":
    "screen-height":
    "screen-width":

    Functions:
    - __init__(self):

    References:
        Tutorial followed for PyQt5 (GUI) can be found here - https://www.pythonguis.com/pyqt5-tutorial/

    Previous versions can be found in the Group GitHub - https://github.com/jrobo-gith/De-Orbiting-Satellite-GroupProj
    """
    def __init__(self):
        """
        Sets primary details for the GUI such as the window title, window size, background etc. Also builds a stacked
        widget which takes all windows, displaying the main menu first.
        """
        super().__init__()

        self.setWindowTitle('SNOE Group Project ~ De-Orbiting Satellite')
        self.resize(glob_setting['screen-height'], glob_setting['screen-width'])
        self.setStyleSheet(f"""background-color: rgba{glob_setting['background-color']};
        background-image: url(visualiser/V2/partials/images/backgroundimage.jpg);
        background-repeat: no-repeat;""")

        self.stacked_widget = QStackedWidget()
        self.stacked_widget.addWidget(MainMenu(self.stacked_widget))
        self.stacked_widget.addWidget(SimulationMenu(self.stacked_widget))
        self.stacked_widget.addWidget(Instructions(self.stacked_widget))
        self.stacked_widget.addWidget(Credits(self.stacked_widget))

        self.setCentralWidget(self.stacked_widget)

# Runs the application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MasterWindow()
    window.show()
    sys.exit(app.exec_())