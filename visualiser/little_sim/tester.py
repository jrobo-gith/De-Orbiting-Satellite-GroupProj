import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore, QtGui
import numpy as np
import sys

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        styles = {"color": "red", "font-size": "18px"}

        self.plot_graph = pg.PlotWidget()
        self.plot_graph.addLegend()
        self.plot_graph.setBackground("w")
        self.setCentralWidget(self.plot_graph)
        pen = pg.mkPen(color=(255, 0, 0), width=5, style=QtCore.Qt.DashLine)
        self.plot_graph.setTitle("Plot showing random data", size="20pt")
        self.plot_graph.showGrid(x=True, y=True)
        self.plot_graph.setLabel("left", "Temperature", **styles)
        self.plot_graph.setLabel("bottom", "Time", **styles)

        self.plot_graph.setYRange(20, 40)
        self.time = list(range(10))
        self.temperature = [np.random.randint(20, 40) for _ in range(10)]

        self.line = self.plot_graph.plot(self.time,
                                         self.temperature,
                                         name="Temp",
                                         pen=pen,
                                         symbol="o",
                                         symbolSize=10,)

        self.timer = QtCore.QTimer()
        self.timer.setInterval(300)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()

    def update_plot(self):
        self.time = self.time[1:]
        self.time.append(self.time[-1] + 1)
        self.temperature = self.temperature[1:]
        self.temperature.append(np.random.randint(20, 40))
        self.line.setData(self.time, self.temperature)


app = QtWidgets.QApplication(sys.argv)
main = MainWindow()
main.show()
app.exec_()