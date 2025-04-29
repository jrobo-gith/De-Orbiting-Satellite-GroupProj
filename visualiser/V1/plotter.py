import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import numpy as np
import sys
from plot import Plot
import data_gen
import matplotlib.pyplot as plt
import json


x1 = list(np.linspace(0, 50, 50))
y1 = list(data_gen.sinusoid(x1))
y2 = list(data_gen.tangent(x1))

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        with open('profiles/example.json') as f:
            example = json.load(f)
        with open('profiles/velocity.json') as f:
            velocity = json.load(f)
        with open('profiles/position.json') as f:
            position = json.load(f)

        self.win = pg.GraphicsLayoutWidget(show=True, title="Visualiser")
        self.win.resize(1080, 720)

        self.plot1 = self.win.addPlot(row=0, col=0, colspan=2)
        self.plo1 = Plot(self.plot1,
                            [x1],
                            [y1],
                            interval=300,
                            args=example,
                            data_func=data_gen.sinusoid)

        self.plot2 = self.win.addPlot(row=1, col=0)
        self.plot2 = Plot(self.plot2,
                            [x1],
                            [y2],
                            interval=100,
                            args=position,
                            data_func=data_gen.tangent)

        self.plot3 = self.win.addPlot(row=1, col=1)
        self.plot3 = Plot(self.plot3,
                          [x1],
                          [y2],
                          interval=100,
                          args=velocity,
                          data_func=data_gen.tangent)


app = QtWidgets.QApplication(sys.argv)
main = MainWindow()
app.exec_()