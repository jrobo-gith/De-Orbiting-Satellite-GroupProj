import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import numpy as np
import sys

class RealTimePlotter:
    def __init__(self):
        self.app = QtWidgets.QApplication(sys.argv)
        self.win = pg.GraphicsLayoutWidget(show=True, title="Live Multi-Plot Dashboard")
        self.win.resize(800, 400)

        # Create two plots
        self.plot1 = self.win.addPlot(title="Signal A")
        self.curve1 = self.plot1.plot(pen='y')

        self.win.nextRow()
        self.plot2 = self.win.addPlot(title="Signal B")
        self.curve2 = self.plot2.plot(pen='c')

        # Initialize data
        self.data1 = np.zeros(300)
        self.data2 = np.zeros(300)

        # Setup timer for updating plots
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(50)  # update every 50 ms

        self.ptr = 0

    def update(self):
        # Simulate new data points
        self.data1[:-1] = self.data1[1:]
        self.data1[-1] = np.sin(0.1 * self.ptr)

        self.data2[:-1] = self.data2[1:]
        self.data2[-1] = np.cos(0.1 * self.ptr)

        # Update plots
        self.curve1.setData(self.data1)
        self.curve2.setData(self.data2)

        self.ptr += 1

    def run(self):
        self.app.exec_()

# Run the app
if __name__ == '__main__':
    plotter = RealTimePlotter()
    plotter.run()
