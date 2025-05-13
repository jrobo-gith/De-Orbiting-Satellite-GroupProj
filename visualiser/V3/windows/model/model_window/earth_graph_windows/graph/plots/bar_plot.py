import pyqtgraph as pg
import numpy as np
from visualiser.V3.debug import debug_print

class BarPlot(pg.PlotWidget):
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
    def __init__(self, plot_allocation, radar_list, init_y, args):
        """
        Initialises the plot. Mostly specifies the styling of the plot, which relies heavily on the 'args' parameter.
        Args is a json file specified as a profile, more on profiles can be found in profiles/profile_READ_ME and the
        example.json. This also plots the initial conditions of the graph, which are likely 0's and then will be updated
        later.

        :param plot_allocation: Provides access to the plot instance in pyqtgraph for styling and plotting.
        :param init_x: Initial x values
        :param init_y: Initial y values
        :param args: Profile of the plot, used for styling.
        """
        super().__init__()

        # Check basic things
        assert type(init_y) == np.ndarray, debug_print("visualiser", "Initial y must be an array")
        assert type(args) == dict, debug_print("visualiser", "Arguments must be a dictionary")

        self.plot_allocation = plot_allocation
        self.args = args
        self.radar_list = radar_list
        self.init_y = init_y
        self.num_radars = len(radar_list)
        self.radar_hot_variable = np.zeros(self.num_radars)

        self.x_axis = []
        self.radar_ticks = []
        for i in range(self.num_radars):
            self.x_axis.append(i)
            self.radar_ticks.append((i, f"R: {i+1}"))
        self.x_axis = np.array(self.x_axis)

        self.bar_plot = pg.BarGraphItem(x=self.x_axis[:len(self.init_y)], height=self.init_y, width=0.6)
        self.plot_allocation.addItem(self.bar_plot)

        self.plot_allocation.getAxis('bottom').setTicks([self.radar_ticks])

        self.plot_allocation.setLabel('left', 'Frequency of Observations')
        self.plot_allocation.setTitle('Top 10 Radars by Most Observations')

    def update_plot(self, radar_name:str):
        """
        Function called in graph_script.py in the update_plots function, inputs a 1xL vector where L is the number of
        lines needing to be plotted.

        Setup so that the length of the initial values self.init_x/y grow until a specified length, then, once reached,
        begins removing the oldest value then adds the new one, keeping the length of the list constant.

        :param radar_name
        """
        if radar_name != 'no radar':
            index = int(radar_name[5:8])

            self.radar_hot_variable[index] += 1

            radar_hot_variable_copy = self.radar_hot_variable.copy()
            sorted_index_list = np.argsort(radar_hot_variable_copy)
            radar_hot_variable_copy = radar_hot_variable_copy[sorted_index_list][::-1]
            new_radar_ticks = []
            for i, index in enumerate(sorted_index_list[:len(self.init_y)]):
                new_radar_ticks.append((i, f"R{index+1}"))

            self.plot_allocation.removeItem(self.bar_plot)
            self.new_bar_plot = pg.BarGraphItem(x=self.x_axis[:len(self.init_y)], height=radar_hot_variable_copy[:len(self.init_y)], width=0.6)
            self.plot_allocation.addItem(self.new_bar_plot)
            self.plot_allocation.getAxis('bottom').setTicks([new_radar_ticks])

