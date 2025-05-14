import os
import sys
import json
import numpy as np

root_dir = os.getcwd()
sys.path.insert(0, root_dir)

import pyqtgraph as pg
from PyQt5.QtWidgets import QWidget, QGridLayout
from PyQt5 import QtCore

from windows.model.model_window.earth_graph_windows.graph.plots.single_plot import Plot
from windows.model.model_window.earth_graph_windows.graph.plots.bar_plot import BarPlot
from partials.constants import toHrs

class Grapher(QWidget):
    """
    Class containing the plots for the graph window. This is the control hub for what the GUI plots in its plotting
    window. Each plot is contained in a GraphicsLayoutWidget. Each plot is an instance of the 'Plot' class which can be
    found in single_plot.py.

    This window's parent is the SimWidget in simulator_window.py, navigable to using the SimWidget under the 'EarthView'
    button at the top right.

    Functions:
    - __init__(self)
    - update_plots(self, name, update)
    - update_plot_no_radar(self, name, update)

    References:
    Tutorial followed for PyQt5 (GUI) can be found here - https://www.pythonguis.com/pyqt5-tutorial/

    Previous versions can be found in the Group GitHub - https://github.com/jrobo-gith/De-Orbiting-Satellite-GroupProj
    """
    def __init__(self, radar_list):
        """
        Initialises the plotting window. Opens the necessary profiles for styling each plot (see
         profiles/profile_READ_ME) for full details. Creates two graphics layout widgets and assigns different plots
         to each. Each instance of the Plot class must take initial_x and initial_y which must have the same dimensions
         as the update matrices. That is 1xL, where L is the number of lines.
        """
        super().__init__()
        self.radar_list = radar_list
        self.uptime = 0
        self.downtime = 0

        pg.setConfigOption('foreground', 'white')

        # Import profiles
        position = os.path.join(root_dir,
        "windows/model/model_window/earth_graph_windows/graph/profiles/position.json")
        with open(position) as f:
            position = json.load(f)

        velocity = os.path.join(root_dir,
        "windows/model/model_window/earth_graph_windows/graph/profiles/velocities.json")
        with open(velocity) as f:
            velocity = json.load(f)

        alt = os.path.join(root_dir,
        "windows/model/model_window/earth_graph_windows/graph/profiles/altitude.json")
        with open(alt) as f:
            altitude = json.load(f)

        live_alt = os.path.join(root_dir,
        "windows/model/model_window/earth_graph_windows/graph/profiles/live_altitude.json")
        with open(live_alt) as f:
            live_alt = json.load(f)

        post_covariance = os.path.join(root_dir,
        "windows/model/model_window/earth_graph_windows/graph/profiles/post_covariance.json")
        with open(post_covariance) as f:
            post_covariance = json.load(f)

        prior_post_res = os.path.join(root_dir,
        "windows/model/model_window/earth_graph_windows/graph/profiles/prior_post_res.json")
        with open(prior_post_res) as f:
            prior_post_res = json.load(f)

        satellite_pos_live = os.path.join(root_dir,
        "windows/model/model_window/earth_graph_windows/graph/profiles/sat_pos_live.json")
        with open(satellite_pos_live) as f:
            live_position = json.load(f)

        drag = os.path.join(root_dir,
        "windows/model/model_window/earth_graph_windows/graph/profiles/drag.json")
        with open(drag) as f:
            drag = json.load(f)

        percent_uptime = os.path.join(root_dir,
        "windows/model/model_window/earth_graph_windows/graph/profiles/percent_uptime.json")
        with open(percent_uptime) as f:
            percent_uptime = json.load(f)

        ## Create one graphics layout widget
        self.column0_graphs = pg.GraphicsLayoutWidget()
        self.column1_graphs = pg.GraphicsLayoutWidget()

        # Create layout and add layout widgets
        self.layout = QGridLayout()
        self.layout.addWidget(self.column0_graphs, 0,0)
        self.layout.addWidget(self.column1_graphs, 0,1)
        self.setLayout(self.layout)

        self.plot_list = []
        # Add column0 graphs
        self.position_graph = self.column0_graphs.addPlot(row=0, col=0)
        self.position_graph = Plot(self.position_graph,
                          [[0], [0], [0], [0]],
                          [[0], [0], [0], [0]],
                          args=position)
        self.plot_list.append(self.position_graph)

        self.velocity_graph = self.column0_graphs.addPlot(row=0, col=1)
        self.velocity_graph = Plot(self.velocity_graph,
                          [[0], [0], [0]],
                          [[0], [0], [0]],
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
                          [[0], [0]],
                          [[0], [0]],
                          args=altitude)
        self.plot_list.append(self.altitude_graph)

        self.satellite_altitude_live = self.column1_graphs.addPlot(row=0, col=1)
        self.satellite_altitude_live = Plot(self.satellite_altitude_live,
                          [[0], [0]],
                          [[0], [0]],
                          args=live_alt)

        self.drag_plot = self.column1_graphs.addPlot(row=1, col=0)
        self.drag_plot = Plot(self.drag_plot,
                                        [[0], [0]],
                                        [[2], [2]],
                                        args=drag)
        self.plot_list.append(self.drag_plot)

        self.satellite_pos_live = self.column1_graphs.addPlot(row=1, col=1)
        self.satellite_pos_live = Plot(self.satellite_pos_live,
                                       [[0]],
                                       [[0]],
                                       args=live_position)


        self.radar_bar_graph = self.column1_graphs.addPlot(row=2, col=0)
        self.radar_bar_graph = BarPlot(self.radar_bar_graph,
                                       self.radar_list,
                                       np.zeros(8))

        self.percent_uptime = self.column1_graphs.addPlot(row=2, col=1,)
        self.percent_uptime.setYRange(0, 100)
        self.percent_uptime = Plot(self.percent_uptime,
                                   [[0]],
                                   [[0]],
                                   args=percent_uptime)


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
        assert type(name) == dict, "Name must be a dictionary"
        assert type(update) == tuple, "Update must be a tuple"
        assert type(update[0]) == type(update[1]) == list, f"""All of update must be a list, update[0]: 
                                                            {type(update[0])}, update[1]: {type(update[1])}"""

        x, y = update

        for i, plot in enumerate(self.plot_list):
            x_vals = np.array(x[i])
            y_vals = np.array(y[i])
            plot.update_plot(x_vals, y_vals)

    @QtCore.pyqtSlot(dict, tuple)
    def update_plot_no_radar(self, info, update):
        """
        Updates the few plots in the graph view on the right that continue even if no radar sees the satellite, giving
        us the simulated 'live' view of where the satellite actually is. Updates the live altitude plot, the bar graph,
        and the percentage uptime plot.

        :param info: Contains info on the update, like whether there is a radar observation or not.
        :param update:
        :return:
        """

        # Take ECEF coordinates from update
        XYZ = update[0]
        x, y, z = [XYZ[0]], [XYZ[1]], [XYZ[2]]

        # Take radar altitude from update
        radar_alt = update[1]
        alt_x, alt_y = np.array(radar_alt[0]), np.array(radar_alt[1])

        x_vals = np.array(x)
        y_vals = np.array(y)

        # Set red if there is no radar, else set green
        if info['name'] == 'no radar':
            self.satellite_pos_live.line.setPen(color=[255, 0, 0], width=5)
        else:
            self.satellite_pos_live.line.setPen(color=[0, 255, 0], width=5)

        # Update bar graph
        radar_name = update[2]
        self.radar_bar_graph.update_plot(radar_name)

        # Update downtime or uptime based on whether we see the satellite or not
        if info['name'] == 'no radar':
            self.downtime += 1
        else:
            self.uptime += 1

        # Compute current uptime percentage and add onto the existing array.
        current_time = np.array([info['obs-time']*toHrs])
        percent_uptime = np.array([(self.uptime / (self.uptime + self.downtime)) * 100])

        # Update plots
        self.percent_uptime.update_plot(current_time, percent_uptime)
        self.satellite_altitude_live.update_plot(alt_x, alt_y)
        self.satellite_pos_live.update_plot(x_vals, y_vals)


