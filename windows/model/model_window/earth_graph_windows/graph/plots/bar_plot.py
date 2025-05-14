import pyqtgraph as pg
import numpy as np


class BarPlot(pg.PlotWidget):
    """
    Instance of a bar plot, very similar to the file 'single_plot.py' but applied to bar graph. Not as well-developed
    as there is only one plot that needs this graph so it updates manually instead of from a list.

    This window's parent is the graph_script.py, where each instance of this class is instantiated there.

    Functions:
    - __init__(self, plot_allocation, init_x, init_y, args)
    - update_plot(self, radar_name:str)

    References:
        Tutorial followed for PyQt5 (GUI) can be found here - https://www.pythonguis.com/pyqt5-tutorial/

    Previous versions can be found in the Group GitHub - https://github.com/jrobo-gith/De-Orbiting-Satellite-GroupProj
    """
    def __init__(self, plot_allocation, radar_list, init_y):
        """
        Initialises the bar plot. Applies some simple styling to the plot.

        :param plot_allocation: Provides access to the plot instance in pyqtgraph for styling and plotting.
        :param radar_list: list of radars to use in the plot
        :param init_y: Initial y value for the plot, just list of zeros where init_y.shape == radar_list.shape
        """

        super().__init__()

        # Check basic things
        assert type(init_y) == np.ndarray, "Initial y must be an array"

        # Initialise variables
        self.plot_allocation = plot_allocation
        self.radar_list = radar_list
        self.init_y = init_y
        self.num_radars = len(radar_list)
        self.radar_hot_variable = np.zeros(self.num_radars)

        # Initialise x-axis
        self.x_axis = []
        self.radar_ticks = []
        for i in range(self.num_radars):
            self.x_axis.append(i)
            self.radar_ticks.append((i, f"R: {i+1}"))
        self.x_axis = np.array(self.x_axis)

        # Initialise bar plot and add to the allocation of the plot
        self.bar_plot = pg.BarGraphItem(x=self.x_axis[:len(self.init_y)], height=self.init_y, width=0.6)
        self.plot_allocation.addItem(self.bar_plot)

        # Apply styling to plot
        self.plot_allocation.getAxis('bottom').setTicks([self.radar_ticks])
        self.plot_allocation.setLabel('left', 'Frequency of Observations')
        self.plot_allocation.setTitle('Top 8 Radars by Most Observations')

    def update_plot(self, radar_name:str):
        """
        Function called in graph.py which updates the bar plot. This is a very specific function which finds which radar
        made the latest observation, and adds 1 to their index in radar_hot_variable. A copy of this variable is then
        made, sorted from highest to lowest. And the top ten radars with the most observations are plotted.
        """

        if radar_name != 'no radar':
            index = int(radar_name[5:8]) # Accounts for radars with single, double, or triple digits.

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
