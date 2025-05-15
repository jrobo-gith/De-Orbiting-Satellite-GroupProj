import pyqtgraph as pg
import numpy as np


class Plot(pg.PlotWidget):
    """
    Instance of a single plot. Inside this class is a single plot, responsible for the styling of the plot and also the
    contents of the plot, the initial conditions and the styling.

    This window's parent is the graph_script.py, where each instance of this class is instantiated there.

    Functions:
    - __init__(self, plot_allocation, init_x, init_y, args)
    - plot_line(self, x, y, line_name, pen, symbol)
    - plot_scatter(self, x, y, line_name, symbol, pen, brush)
    - update_plot(self, new_data_X, new_data_Y)

    References:
        Tutorial followed for PyQt5 (GUI) can be found here - https://www.pythonguis.com/pyqt5-tutorial/

    Previous versions can be found in the Group GitHub - https://github.com/jrobo-gith/De-Orbiting-Satellite-GroupProj
    """
    def __init__(self, plot_allocation, init_x, init_y, args):
        """
        Initialises the plot. Mostly specifies the styling of the plot, which relies heavily on the 'args' parameter.
        Args is a json file specified as a profile, more on profiles can be found in profiles/profile_READ_ME and the
        example.json. This also plots the initial conditions of the graph, which are likely 0's and then will be updated
        later.

        Args:
            plot_allocation: Provides access to the plot instance in pyqtgraph for styling and plotting.
            init_x: Initial x values
            init_y: Initial y values
            args: Profile of the plot, used for styling.
        """

        super().__init__()

        # Check basic things
        assert type(init_x) == list, "Initial x must be a list"
        assert type(init_y) == list, "Initial y must be a list"
        assert type(args) == dict, "Arguments must be a dictionary"

        self.plot_allocation = plot_allocation
        self.args = args
        self.init_x = init_x
        self.init_y = init_y
        self.num_lines = len(init_x)

        # Plotting legend and grid
        if args["legend"]:
            self.plot_allocation.addLegend()
        if args["grid"]:
            self.plot_allocation.showGrid(x=True, y=True)

        # Specifying title and label titles
        self.plot_allocation.setTitle(args["title"])
        self.plot_allocation.setLabel("bottom", args["label_title_x"])
        self.plot_allocation.setLabel("left", args["label_title_y"])

        # If there is an x-lim or y-lim, add it to the plot.
        if args['x-lim'] != "None":
            self.plot_allocation.setXRange(args["x-lim"][0], args["x-lim"][1])
        if args['y-lim'] != "None":
            self.plot_allocation.setYRange(args["y-lim"][0], args["y-lim"][1])

        if args['x-lim'] != "None":
            self.plot_allocation.setXRange(args["x-lim"][0], args["x-lim"][1])
        if args['y-lim'] != "None":
            self.plot_allocation.setYRange(args["y-lim"][0], args["y-lim"][1])

        # Plot initial values for each line
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
        """
        Plots a line using pyqtgraph.

        Args:
            x: X value
            y: Y value
            line_name: name for legend
            pen: line styling
            symbol: datapoint styling
        """
        line = self.plot_allocation.plot(x=x, y=y, name=line_name, symbol=symbol[0], symbolSize=symbol[1], pen=pen)
        return line
    def plot_scatter(self, x:list, y:list, line_name:str, symbol:list, pen, brush):
        """
        Plots a scatter plot using pyqtgraph.

        Args:
            x: X value
            y: Y value
            line_name: name for legend
            symbol: datapoint styling
            pen: line styling
            brush: scatter point styling
        """
        scatter = pg.ScatterPlotItem(x=x, y=y, name=line_name, symbol=symbol[0], symbolSize=symbol[1],
                                     pen=pen, brush=brush)
        self.plot_allocation.addItem(scatter)
        return scatter

    def update_plot(self, new_data_X:np.array, new_data_Y:np.array):
        """
        Function called in graph_script.py in the update_plots function, inputs a 1xL vector where L is the number of
        lines needing to be plotted.

        Setup so that the length of the initial values self.init_x/y grow until a specified length, then, once reached,
        begins removing the oldest value then adds the new one, keeping the length of the list constant.

        Args:
            new_data_X: New x values
            new_data_Y: New y values
        """
        assert type(new_data_X) == np.ndarray, f"New X must be a numpy array. {type(new_data_X)}"
        assert type(new_data_Y) == np.ndarray, f"New Y must be a numpy array. {type(new_data_Y)}"
        assert new_data_X.shape == new_data_Y.shape, f"New X must be same size as new Y. {new_data_X.shape[0]} != {new_data_Y.shape[0]}"


        for i, self.line in enumerate(self.lines):
            if self.args['array-limit'] != "None":
                if len(self.init_x[i]) > self.args['array-limit']: # If the length is larger than the array limit in args
                    # Remove oldest datapoint
                    self.init_x[i] = self.init_x[i][1:]
                    self.init_y[i] = self.init_y[i][1:]

            # Add new data point
            self.init_x[i].append(new_data_X[i])
            self.init_y[i].append(new_data_Y[i])

            # Update Line
            self.line.setData(self.init_x[i], self.init_y[i])