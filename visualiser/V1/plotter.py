import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore
import numpy as np
import sys
from plot import Plot


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        # Universal Styling
        uni_label = {"color": "black", "font-size": "15px"}
        uni_title = {"color": "black", "font-size": "20px"}

        example_plot_arg = {
            ## Example arguments for each plot, some arguments are encased
            # in a list to allow for multiple lines in one plot

            # Titles
            "title": "This title",
            "label_title_x": "X-Axis",
            "label_title_y": "Y-Axis",
            "line_names": ["Bias", "Variance"],
            # Plot Add-ons
            "legend": True,
            "grid": True,

            # Line options
            "pens": [pg.mkPen(color=(255, 0, 0), width=5, style=QtCore.Qt.DashLine)],
            "symbols": [["+", 0]],

            # Styles
            "label_style": [uni_label],
            "title_style": [uni_title],
        }

        self.win = pg.GraphicsLayoutWidget(show=True, title="Visualiser")
        self.win.resize(1080, 720)

        self.plot1 = self.win.addPlot(title="A")
        self.plo1 = Plot(self.plot1, np.array([np.linspace(0, 10, 100)]),
                         np.array([np.linspace(0, 10, 100), ]),
                         args=example_plot_arg)

        self.plot2 = self.win.addPlot(title="B")
        self.plot2 = Plot(self.plot2, np.array([np.linspace(0, 10, 100)]),
                         np.array([np.linspace(0, 10, 100)]),
                         args=example_plot_arg)





app = QtWidgets.QApplication(sys.argv)
main = MainWindow()
app.exec_()