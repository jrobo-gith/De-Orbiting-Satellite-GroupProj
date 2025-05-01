from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QStackedWidget)
from PyQt5 import QtCore

from visualiser.V2.partials.navbar import Navbar
from visualiser.V2.widgets.graph_stuff.partials.switcher import Switcher
from visualiser.V2.widgets.graph_stuff.graph_script import Grapher
from visualiser.V2.widgets.graph_stuff.partials import data_gen
from visualiser.V2.widgets.graph_stuff.earth_window import Earth

import json
import time
import numpy as np
import threading

with open('partials/global_settings.json') as f:
    glob_setting = json.load(f)

class Helper(QtCore.QObject):
    changedSignal = QtCore.pyqtSignal(str, tuple)

def create_data(helper, name):
    """MUST OUTPUT Matrix of size P X L where:
    P is the number of plots in the window and
    L is the number of lines in the plot.
    """
    additional_x = list(np.arange(3.0, 500.0, 0.1))

    for i in range(len(additional_x)):
        outgoing_x = [[additional_x[i]], [additional_x[i]], [additional_x[i]]]
        outgoing_y = [data_gen.sinusoid(outgoing_x[0]), data_gen.tangent(outgoing_x[1]), data_gen.cosine(outgoing_x[2])]
        helper.changedSignal.emit(name, (outgoing_x, outgoing_y))
        time.sleep(.1)

init_x = [list(np.linspace(-3.0, 3.0, 100))]
init_y = [data_gen.sinusoid(init_x[0])]

init_x.append(list(np.linspace(-3.0, 3.0, 100)))
init_y.append(data_gen.tangent(init_x[1]))

init_x.append(list(np.linspace(-3.0, 3.0, 100)))
init_y.append(data_gen.cosine(init_x[2]))

class Graphs(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget

        ## graph-earth stacked widget
        graph_earth_container = QStackedWidget()
        ## NAVBAR
        navbar = Navbar("Graphs", self.stacked_widget)
        ## SWITCHER
        switcher = Switcher(graph_earth_container)
        ## Earth window
        earth = Earth(graph_earth_container)
        ## graph_script
        graph = Grapher(init_x=init_x, init_y=init_y)

        graph_earth_container.addWidget(graph)
        graph_earth_container.addWidget(earth)

        container = QVBoxLayout()
        container.addWidget(navbar, stretch=1)
        container.addWidget(switcher, stretch=1)
        container.addWidget(graph_earth_container, stretch=18)
        self.setLayout(container)

        helper = Helper()
        helper.changedSignal.connect(graph.update_plots, QtCore.Qt.QueuedConnection)
        threading.Thread(target=create_data, args=(helper, "redundant_name"), daemon=True).start()