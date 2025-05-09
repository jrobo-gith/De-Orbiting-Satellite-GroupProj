import pyqtgraph as pg
from PyQt5 import QtGui
from pyqtgraph.Qt import QtWidgets, QtCore
import numpy as np

class Plot(pg.PlotWidget):
    """Class to plot and update plots at a time"""
    def __init__(self, plot_allocation, init_x, init_y, args):
        super().__init__()
        """
        Initialise the plot

        plot_allocation:    The subplot we're plotting to
        init_x:             Initial x values
        init_y:             Initial y values
        interval:           How long between each update (ms)
        args:               Cosmetic details about the graph
        """

        assert type(init_x) == list, print("Initial x must be a list")
        assert type(init_y) == list, print("Initial y must be a list")
        assert type(args) == dict, print("Arguments must be a dictionary")

        self.plot_allocation = plot_allocation
        self.args = args
        self.init_x = init_x
        self.init_y = init_y

        self.num_lines = len(init_x)
        print("NUMLINES: ", self.num_lines)

        if args["legend"]:
            self.plot_allocation.addLegend()
        if args["grid"]:
            self.plot_allocation.showGrid(x=True, y=True)

        self.plot_allocation.setTitle(args["title"])
        self.plot_allocation.setLabel("bottom", args["label_title_x"])
        self.plot_allocation.setLabel("left", args["label_title_y"])

        if args['x-lim'] != "None":
            self.plot_allocation.setXRange(args["x-lim"][0], args["x-lim"][1])
        if args['y-lim'] != "None":
            self.plot_allocation.setYRange(args["y-lim"][0], args["y-lim"][1])

        # Plot initial values
        self.lines = []
        for i in range(self.num_lines):
            if args['plot-type'][i] == 'line':
                self.line = self.plot_line(x=init_x[i], y=init_y[i], line_name=args["line_names"][i],
                                           symbol=args["symbols"][i], pen=pg.mkPen(color=args["pens"][i]['color'],
                                                                                   width=args["pens"][i]['width']))
            elif args['plot-type'][i] == 'scatter':
                self.line = self.plot_scatter(x=init_x[i], y=init_y[i], line_name=args["line_names"][i],
                                              symbol=args["symbols"][i],pen=pg.mkPen(color=args["pens"][i]['color'],
                                                                                   width=args["pens"][i]['width']),
                                              brush=pg.mkBrush(args["brushes"][i]['color']))
            self.lines.append(self.line)

    def plot_line(self, x:list, y:list, line_name:str, pen, symbol:list):
        line = self.plot_allocation.plot(x=x, y=y, name=line_name, symbol=symbol[0], symbolSize=symbol[1], pen=pen)
        return line
    def plot_scatter(self, x:list, y:list, line_name:str, symbol:list, pen, brush):
        scatter = pg.ScatterPlotItem(x=x, y=y, name=line_name, symbol=symbol[0], symbolSize=symbol[1],
                                     pen=pen, brush=brush)
        self.plot_allocation.addItem(scatter)
        return scatter

    def update_plot(self, new_data_X:np.array, new_data_Y:np.array):
        """
        MUST take in list vectors of size 1 X L.
        """
        assert type(new_data_X) == np.ndarray, print(f"New X must be a numpy array. {type(new_data_X)}")
        assert type(new_data_Y) == np.ndarray, print(f"New Y must be a numpy array. {type(new_data_Y)}")

        assert new_data_X.shape == new_data_Y.shape, print(f"New X must be same size as new Y. {new_data_X.shape[0]} != {new_data_Y.shape[0]}")

        # assert new_data_X.shape == np.array(self.init_x[0]).shape, print(f"New X must be same size as initial X. {new_data_X.shape} != {np.array(self.init_x[0]).shape}")
        # assert new_data_Y.shape == np.array(self.init_y[0]).shape, print("New Y must be same size as initial Y.")

        for i, self.line in enumerate(self.lines):
            if len(self.init_x[i]) > 50: # If the length is larger than 100
                # Remove oldest datapoint
                self.init_x[i] = self.init_x[i][1:]
                self.init_y[i] = self.init_y[i][1:]

            # Add new data point
            self.init_x[i].append(new_data_X[i])
            self.init_y[i].append(new_data_Y[i])

            # Update Line
            self.line.setData(self.init_x[i], self.init_y[i])