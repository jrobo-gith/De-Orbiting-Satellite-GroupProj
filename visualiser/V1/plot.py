import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import numpy as np
import sys

class Plot():
    """Class to plot and update plots at a time"""
    def __init__(self, plot_allocation, init_x, init_y, args):
        """Initialise plot"""

        assert type(init_x) == np.ndarray, print("Initial x must be a numpy array")
        assert type(init_y) == np.ndarray, print("Initial y must be a numpy array")
        assert type(args) == dict, print("Arguments must be a dictionary")

        self.plot_allocation = plot_allocation
        self.args = args
        self.init_x = init_x
        self.init_y = init_y

        self.num_lines = init_x.shape[0]
        print(self.num_lines)

        if args["legend"]:
            self.plot_allocation.addLegend()
        if args["grid"]:
            self.plot_allocation.showGrid(x=True, y=True)

        self.plot_allocation.setTitle(args["title"])
        self.plot_allocation.setLabel("bottom", args["label_title_x"])
        self.plot_allocation.setLabel("left", args["label_title_y"])

        ## Plot initial values
        self.lines = []
        for i in range(self.num_lines):
            self.line = self.plot_line(x=init_x[i],
                           y=init_y[i],
                           line_name=args["line_names"][i],
                           symbol=args["symbols"][i],
                           pen=args["pens"][i],)
            self.lines.append(self.line)

        # self.timer = QtCore.QTimer()
        # self.timer.setInterval(300)
        # self.timer.timeout.connect(self.update_plot)
        # self.timer.start()

    def plot_line(self, x:np.array, y:np.array, line_name:str, pen, symbol:list):
        line = self.plot_allocation.plot(x, y, name=line_name, symbol=symbol[0], symbolSize=symbol[1], pen=pen)
        return line

    def update_plot(self, new_data_X, new_data_Y):
        """Update plot"""
        self.init_x = self.init_x[1:]
        self.init_y = self.init_y[1:]

        self.init_x.append(new_data_X)
        self.init_y.append(new_data_Y)

        for self.line in self.lines:
            self.line.setData(self.init_x, self.init_y)
