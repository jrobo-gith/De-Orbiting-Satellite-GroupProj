import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore

class Plot:
    """Class to plot and update plots at a time"""
    def __init__(self, plot_allocation, init_x, init_y, interval, args, data_func):
        """
        Initialise the plot

        plot_allocation:    The subplot we're plotting to
        init_x:             Initial x values
        init_y:             Initial y values
        interval:           How long between each update (ms)
        args:               Cosmetic details about the graph
        data_func:          (TEMPORARY) Simulating data
        """

        assert type(init_x) == list, print("Initial x must be a list")
        assert type(init_y) == list, print("Initial y must be a list")
        assert type(args) == dict, print("Arguments must be a dictionary")

        self.plot_allocation = plot_allocation
        self.args = args
        self.init_x = init_x
        self.init_y = init_y
        self.interval = interval

        self.num_lines = len(init_x)
        print(f"Number of lines: {self.num_lines}")
        self.data_gen = data_func

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
                           pen=pg.mkPen(color=args["pens"][i]['color'], width=args["pens"][i]['width']),)
            self.lines.append(self.line)

        self.timer = QtCore.QTimer()
        self.timer.setInterval(self.interval)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()

    def plot_line(self, x:list, y:list, line_name:str, pen, symbol:list):
        line = self.plot_allocation.plot(x=x, y=y, name=line_name, symbol=symbol[0], symbolSize=symbol[1], pen=pen)
        return line

    def update_plot(self):
        """Update plot"""
        for i, self.line in enumerate(self.lines):
            self.init_x[i] = self.init_x[i][1:]
            self.init_y[i] = self.init_y[i][1:]

            self.init_x[i].append(self.init_x[i][-1] + 1)
            self.init_y[i].append(self.data_gen(self.init_x[i][-1]))
            self.line.setData(self.init_x[i], self.init_y[i])
