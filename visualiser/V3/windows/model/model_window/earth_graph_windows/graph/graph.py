import os
import sys
import json
import numpy as np

root_dir = os.getcwd()
sys.path.insert(0, root_dir)

import pyqtgraph as pg
from PyQt5.QtWidgets import QWidget, QGridLayout
from PyQt5 import QtCore

from visualiser.V3.debug import debug_print
from visualiser.V3.windows.model.model_window.earth_graph_windows.graph.plots.single_plot import Plot

class Grapher(QWidget):
    """
    Class containing the plots for the graph window. This is the control hub for what the GUI plots in its plotting
    window. Each plot is contained in a GraphicsLayoutWidget to separate plots that have access to simulations and
    ones that don't. Each plot is an instance of the 'Plot' class which can be found in single_plot.py.

    This window's parent is the SimWidget in simulator_window.py, navigable to using the SimWidget under the 'EarthView'
    button at the top right.

    Functions:
    - __init__(self)
    - update_plots(self, name, update)

    References:
    Tutorial followed for PyQt5 (GUI) can be found here - https://www.pythonguis.com/pyqt5-tutorial/

    Previous versions can be found in the Group GitHub - https://github.com/jrobo-gith/De-Orbiting-Satellite-GroupProj
    """
    def __init__(self):
        """
        Initialises the plotting window. Opens the necessary profiles for styling each plot (see
         profiles/profile_READ_ME) for full details. Creates two graphics layout widgets and assigns different plots
         to each. Each instance of the Plot class must take initial_x and initial_y which must have the same dimensions
         as the update matrices. That is 1xL, where L is the number of lines.
        """
        super().__init__()
        pg.setConfigOption('foreground', 'white')

        # Import profiles
        position = os.path.join(root_dir, "visualiser/V3/windows/model/model_window/earth_graph_windows/graph/profiles/position.json")
        with open(position) as f:
            position = json.load(f)

        velocity = os.path.join(root_dir, "visualiser/V3/windows/model/model_window/earth_graph_windows/graph/profiles/velocities.json")
        with open(velocity) as f:
            velocity = json.load(f)

        alt = os.path.join(root_dir, "visualiser/V3/windows/model/model_window/earth_graph_windows/graph/profiles/altitude.json")
        with open(alt) as f:
            altitude = json.load(f)

        post_covariance = os.path.join(root_dir,
                           "visualiser/V3/windows/model/model_window/earth_graph_windows/graph/profiles/post_covariance.json")
        with open(post_covariance) as f:
            post_covariance = json.load(f)

        prior_post_res = os.path.join(root_dir,
                           "visualiser/V3/windows/model/model_window/earth_graph_windows/graph/profiles/prior_post_res.json")
        with open(prior_post_res) as f:
            prior_post_res = json.load(f)

        satellite_pos_live = os.path.join(root_dir,
                                      "visualiser/V3/windows/model/model_window/earth_graph_windows/graph/profiles/sat_pos_live.json")
        with open(satellite_pos_live) as f:
            live_position = json.load(f)

        ## Create two graphics layout widgets
        self.column0_graphs = pg.GraphicsLayoutWidget()
        self.column1_graphs = pg.GraphicsLayoutWidget()

        # Create layout and add layout widgets
        self.layout = QGridLayout()
        self.layout.addWidget(self.column0_graphs, 0, 0)
        self.layout.addWidget(self.column1_graphs, 0, 2)
        self.setLayout(self.layout)

        self.plot_list = []
        self.live_plot_list = []
        # Add column0 graphs
        self.position_graph = self.column0_graphs.addPlot(row=0, col=0)
        self.position_graph = Plot(self.position_graph,
                          [[0], [0], [0]],
                          [[0], [0], [0]],
                          args=position)
        self.plot_list.append(self.position_graph)

        self.velocity_graph = self.column0_graphs.addPlot(row=0, col=1)
        self.velocity_graph = Plot(self.velocity_graph,
                          [[0], [0]],
                          [[0], [0]],
                          args=velocity)
        self.plot_list.append(self.velocity_graph)

        self.post_covariance_graph = self.column0_graphs.addPlot(row=1, col=0)
        self.post_covariance_graph = Plot(self.post_covariance_graph,
                          [[0]],
                          [[0]],
                          args=post_covariance)
        self.plot_list.append(self.post_covariance_graph)

        self.prior_post_residual = self.column0_graphs.addPlot(row=1, col=1)
        self.prior_post_residual = Plot(self.prior_post_residual,
                          [[0], [0], [0]],
                          [[0], [0], [0]],
                          args=prior_post_res)
        self.plot_list.append(self.prior_post_residual)

        # Add column1 graphs
        self.altitude_graph = self.column1_graphs.addPlot(row=0, col=0)
        self.altitude_graph = Plot(self.altitude_graph,
                          [[0], [0], [0]],
                          [[0], [0], [0]],
                          args=altitude)
        self.plot_list.append(self.altitude_graph)

        self.satellite_pos_live = self.column1_graphs.addPlot(row=1, col=0)
        self.satellite_pos_live = Plot(self.satellite_pos_live,
                                       [[0]],
                                       [[0]],
                                       args=live_position)
        self.live_plot_list.append(self.satellite_pos_live)


    @QtCore.pyqtSlot(dict, tuple)
    def update_plots(self, name, update):
        """
        Updates each plot in the plot list given a matrix of P X L, where P is the number of plots and L is the number
        of lines in the plot. Then it update each plot with a vector of 1 X L where L is the number of lines needing
        updates.

        It receives data as a (dict, tuple) from the predictor_files/predictor.py via multi-threading and a helper
        function that emits data to this function.

        The tuple contains the x and y values of the update, the dimensions of which must be what is described above.

        :param name: redundant parameter
        :param update: contains x and y to update the plots.
        """
        assert type(name) == dict, debug_print("visualiser", "Name must be a dictionary")
        assert type(update) == tuple, debug_print("visualiser", "Update must be a tuple")
        assert type(update[0]) == type(update[1]) == list, debug_print("visualiser", f"""All of update must be a list, update[0]: {type(update[0])}, update[1]: {type(update[1])}""")

        x, y = update

        for i, plot in enumerate(self.plot_list):
            x_vals = np.array(x[i])
            y_vals = np.array(y[i])
            plot.update_plot(x_vals, y_vals)

    @QtCore.pyqtSlot(dict, tuple)
    def update_plot_no_radar(self, name, update):
        XYZ = update[0]
        x, y, z = [[XYZ[0]]], [[XYZ[1]]], [[XYZ[2]]]


        for i, plot in enumerate(self.live_plot_list):
            x_vals = np.array(x[i])
            y_vals = np.array(y[i])
            plot.update_plot(x_vals, y_vals)