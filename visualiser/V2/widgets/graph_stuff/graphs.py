from PyQt5.QtWidgets import (QWidget, QVBoxLayout)
from PyQt5 import QtCore, QtGui
import pyqtgraph as pg

from visualiser.V2.partials.navbar import Navbar
from visualiser.V2.widgets.graph_stuff.graph_script import Grapher
from visualiser.V2.widgets.graph_stuff import data_gen

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

        ## NAVBAR
        navbar = Navbar("Graphs", self.stacked_widget)

        container = QVBoxLayout()
        graph = Grapher(init_x=init_x, init_y=init_y)

        container.addWidget(navbar, stretch=1)
        container.addWidget(graph, stretch=20)
        self.setLayout(container)

        helper = Helper()
        helper.changedSignal.connect(graph.update_plots, QtCore.Qt.QueuedConnection)
        threading.Thread(target=create_data, args=(helper, "redundant_name"), daemon=True).start()