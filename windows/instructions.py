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

class Instructions(QWidget):
    """
    Window displaying instructions on how to navigate the GUI and run simulations.
    This window's parent is the MasterWindow in master.py, navigable to using the MainMenu under the 'instructions'
    button.

    Functions:
    - __init__(self, stacked_widget)

     References:
        Tutorial followed for PyQt5 (GUI) can be found here - https://www.pythonguis.com/pyqt5-tutorial/

    Previous versions can be found in the Group GitHub - https://github.com/jrobo-gith/De-Orbiting-Satellite-GroupProj

    """
    def __init__(self, stacked_widget):
        """
        Initialises the instructions window, contains an instance of the 'navbar' class, contains many QLabels
        outlining the instructions and structured using a Vertical Box Layout (QVBoxLayout) to produce the instructions
        window.

        Args:
            stacked_widget: Widget containing pages of the GUI, used to navigate back to the main menu.
        """
        super().__init__()
        self.stacked_widget = stacked_widget

        ## NAVBAR
        navbar = Navbar("Instructions", self.stacked_widget)

        ## Instructions box
        instructions_box = QVBoxLayout()
        instructions = []
        instruction_titles = [] # For increasing font
        ## Add all instructions
        UI_title = QLabel("UI")
        UI = QLabel("""This UI is designed to make navigation easy. With only a few clicks, you can input your 
        initial values and begin the simulation.""")
        screens_title = QLabel("Main Menu")
        screens = QLabel("""When the program has loaded, the main menu window loads. From the main menu, there are four 
        buttons, 'Simulation', 'Instructions', 'Credits' and 'exit'. 'Instructions' takes you to this page, 'Credits' 
        takes you to the credits page so people can see what part of the team worked on what, and 'exit' is to exit the 
        program.""")
        sim_explanation_title = QLabel("The Simulation Menu")
        sim_explanation = QLabel("""The 'Simulation' button, when clicked, takes the user to a simulation menu, 
        whereupon the user may input the initial values for the satellite and begin the simulation. The team has 
        suggested values which best demonstrates a slow, decaying orbit but the user is encouraged to experiment with 
        whichever inputs they like.""")
        graph_title = QLabel("The Graph Window")
        graph_explanation = QLabel("""Once initiated, the user will be taken to the simulation, where they are met 
        with detailed graphs, each showing important information about the satellite trajectory, velocity, etc. There 
        are two windows within this area, one window displays graphical information, the other is the Earth window.""")
        earth_title = QLabel("The Earth Window")
        earth_explination = QLabel("""The 'Earth Window' button, when clicked, takes the user to a 2d static 
        model of Earth where the simulated trajectory of the satellite (the true trajectory), radar locations, 
        and predicted landing site are all displayed and can be turned on and off using checkboxes to the right of the 
        world map.""")
        restart_simulation_title = QLabel("Restarting the Simulation")
        restart_simulation = QLabel("""Once inside the graph-earth window, one can navigate between the two for the 
        duration of the simulation. To restart the simulation, the user simply clicks the '<-' button at the top 
        left of the screen and will be taken back to the main menu, where they can navigate back into the simulation 
        menu to input new initial values for the satellite.""")

        # Add instructions to instructions box
        instructions_box.addWidget(UI_title)
        instructions_box.addWidget(UI)
        instructions_box.addWidget(screens_title)
        instructions_box.addWidget(screens)
        instructions_box.addWidget(sim_explanation_title)
        instructions_box.addWidget(sim_explanation)
        instructions_box.addWidget(graph_title)
        instructions_box.addWidget(graph_explanation)
        instructions_box.addWidget(earth_title)
        instructions_box.addWidget(earth_explination)
        instructions_box.addWidget(restart_simulation_title)
        instructions_box.addWidget(restart_simulation)

        # Add instructions to list to loop through and add styling
        instructions.append(UI_title)
        instructions.append(UI)
        instructions.append(screens_title)
        instructions.append(screens)
        instructions.append(sim_explanation_title)
        instructions.append(sim_explanation)
        instructions.append(graph_title)
        instructions.append(graph_explanation)
        instructions.append(earth_title)
        instructions.append(earth_explination)
        instructions.append(restart_simulation_title)
        instructions.append(restart_simulation)

        instruction_titles.append(UI_title)
        instruction_titles.append(screens_title)
        instruction_titles.append(sim_explanation_title)
        instruction_titles.append(graph_title)
        instruction_titles.append(earth_title)
        instruction_titles.append(restart_simulation_title)

        for i in range(len(instructions)):
            instructions[i].setStyleSheet(f"color: rgb{glob_setting['font-color']}; background: rgba(0, 0, 0, 0.7); border-radius: 30px;")
            instructions[i].setFont(QFont(glob_setting['font-family'], 11))
            instructions[i].setAlignment(Qt.AlignHCenter)

        for title in instruction_titles:
            title.setFont(QFont(glob_setting['font-family'], 16, QFont.Bold))
            title.setStyleSheet(f"color: rgb{glob_setting['font-color']};")

        container = QVBoxLayout()
        container.addWidget(navbar, stretch=1)
        container.addLayout(instructions_box, stretch=19)

        self.setLayout(container)