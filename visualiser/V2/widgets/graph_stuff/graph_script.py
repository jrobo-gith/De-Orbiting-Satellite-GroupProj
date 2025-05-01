import json

import pyqtgraph as pg
from visualiser.V2.widgets.graph_stuff.single_plot import Plot
from visualiser.V2.widgets.graph_stuff.partials import data_gen
from PyQt5 import QtCore


class Grapher(pg.GraphicsLayoutWidget):
    def __init__(self, init_x, init_y):
        super().__init__()
        self.init_x = init_x
        self.init_y = init_y

        # Import profiles
        with open('profiles/example.json') as f:
            example = json.load(f)
        with open('profiles/velocity.json') as f:
            velocity = json.load(f)
        with open('profiles/position.json') as f:
            position = json.load(f)

        self.plot_list = []

        self.plot1 = self.addPlot(row=0, col=0, colspan=2)
        self.plot1 = Plot(self.plot1,
                          [self.init_x[0]],
                          [self.init_y[0]],
                          args=example,
                          data_func=data_gen.sinusoid)
        self.plot_list.append(self.plot1)

        self.plot2 = self.addPlot(row=1, col=0)
        self.plot2 = Plot(self.plot2,
                          [self.init_x[1]],
                          [self.init_y[1]],
                          args=position,
                          data_func=data_gen.tangent)
        self.plot_list.append(self.plot2)

        self.plot3 = self.addPlot(row=1, col=1)
        self.plot3 = Plot(self.plot3,
                          [self.init_x[2]],
                          [self.init_y[2]],
                          args=velocity,
                          data_func=data_gen.cosine)

        self.plot_list.append(self.plot3)

    @QtCore.pyqtSlot(str, tuple)
    def update_plots(self, name, update):
        """MUST take in matrix of P X L, then MUST update each plot with a vector of 1 X L where:
        L is the number of lines needing updates.
        """
        x, y = update
        for i, plot in enumerate(self.plot_list):
            plot.update_plot(x[i], y[i])