import os
import sys

root_dir = os.getcwd()
sys.path.insert(0, root_dir)

import json
import numpy as np

import pyqtgraph as pg
from PyQt5.QtGui import QColor, QBrush
from PyQt5.QtWidgets import QWidget, QGridLayout, QGraphicsScene, QGraphicsRectItem, QGraphicsView
from PyQt5.QtCore import Qt

from visualiser.V2.widgets.graph_stuff.single_plot import Plot
from visualiser.V2.widgets.graph_stuff.single_3D_plot import ThreeDPlot

from visualiser.V2.widgets.graph_stuff.partials import data_gen
from PyQt5 import QtCore, QtGui, QtWidgets


class Grapher(QWidget):
    def __init__(self, ):
        super().__init__()
        pg.setConfigOption('foreground', 'white')

        # Import profiles
        position = os.path.join(root_dir, "visualiser/V2/profiles/prior_post.json")
        with open(position) as f:
            position = json.load(f)

        velocity = os.path.join(root_dir, "visualiser/V2/profiles/velocities.json")
        with open(velocity) as f:
            velocity = json.load(f)

        alt = os.path.join(root_dir, "visualiser/V2/profiles/altitude.json")
        with open(alt) as f:
            altitude = json.load(f)


        ## Create two graphics layout widgets
        self.simulator_graphs = pg.GraphicsLayoutWidget()
        self.predictor_graphs = pg.GraphicsLayoutWidget()
        # self.combined_graphs = pg.GraphicsLayoutWidget()


        # Add graphics scenes
        sim_scene = self.simulator_graphs.scene()
        sim_rect = QGraphicsRectItem(0, 0, 1000, 1000)
        sim_brush = QBrush(QColor(0, 0, 255, 35))
        sim_rect.setBrush(sim_brush)
        sim_rect.setZValue(-1)
        sim_scene.addItem(sim_rect)

        pred_scene = self.predictor_graphs.scene()
        pred_rect = QGraphicsRectItem(0, 0, 1000, 1000)
        pred_brush = QBrush(QColor(0, 255, 0, 35))
        pred_rect.setBrush(pred_brush)
        pred_rect.setZValue(-1)
        pred_scene.addItem(pred_rect)

        # combined_scene = self.combined_graphs.scene()
        # combined_rect = QGraphicsRectItem(0, 0, 1000, 1000)
        # combined_brush = QBrush(QColor(0, 255, 255, 35))
        # combined_rect.setBrush(combined_brush)
        # combined_rect.setZValue(-1)
        # combined_scene.addItem(combined_rect)


        # Create layout and add layout widgets
        self.layout = QGridLayout()
        self.layout.addWidget(self.simulator_graphs, 0, 0)
        # self.layout.addWidget(self.combined_graphs, 0, 1)
        self.layout.addWidget(self.predictor_graphs, 0, 2)
        self.setLayout(self.layout)


        self.plot_list = []
        # Add simulator plots
        self.sim_1 = self.simulator_graphs.addPlot(row=0, col=0)
        self.sim_1 = Plot(self.sim_1,
                          [[0], [0], [0]],
                          [[0], [0], [0]],
                          args=position)

        self.plot_list.append(self.sim_1)

        self.sim_2 = self.simulator_graphs.addPlot(row=1, col=0)
        self.sim_2 = Plot(self.sim_2,
                          [[0], [0]],
                          [[0], [0]],
                          args=velocity)

        self.plot_list.append(self.sim_2)

        # self.sim_2 = self.simulator_graphs.addPlot(row=1, col=0)
        # self.sim_2 = Plot(self.sim_2,
        #                   [self.init_x[1]],
        #                   [self.init_y[1]],
        #                   args=position,
        #                   data_func=data_gen.tangent)
        # self.plot_list.append(self.sim_2)
        #
        # # Add combined plots
        # self.comb_1 = self.combined_graphs.addPlot(row=0, col=0)
        # self.comb_2 = self.combined_graphs.addPlot(row=1, col=0)
        #
        # Add predictor plots
        self.pred_1 = self.predictor_graphs.addPlot(row=0, col=0)
        self.pred_1 = Plot(self.pred_1,
                          [[0], [0], [0]],
                          [[0], [0], [0]],
                          args=altitude)

        self.plot_list.append(self.pred_1)
        #
        # self.pred_2 = self.predictor_graphs.addPlot(row=1, col=0)
        # self.pred_2 = Plot(self.pred_2,
        #                    [self.init_x[2]],
        #                    [self.init_y[2]],
        #                    args=velocity,
        #                    data_func=data_gen.cosine)
        #
        # self.plot_list.append(self.pred_2)


    @QtCore.pyqtSlot(dict, tuple)
    def update_plots(self, name, update):
        """MUST take in matrix of P X L, then MUST update each plot with a vector of 1 X L where:
        L is the number of lines needing updates.
        """
        assert type(name) == dict, print("Name must be a dictionary")
        assert type(update) == tuple, print("Update must be a tuple")
        assert type(update[0]) == type(update[1]) == list, print(f"""All of update must be a list, update[0]: {type(update[0])}, update[1]: {type(update[1])}""")

        x, y = update

        for i, plot in enumerate(self.plot_list):
            x_vals = np.array(x[i])
            y_vals = np.array(y[i])
            plot.update_plot(x_vals, y_vals)