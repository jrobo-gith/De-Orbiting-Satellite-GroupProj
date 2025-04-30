from mainwindow import MainWindow
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, QtCore, QtGui
import sys
import time
import threading
import data_gen
import matplotlib.pyplot as plt

class Helper(QtCore.QObject):
    changedSignal = QtCore.pyqtSignal(str, tuple)

def create_data(helper, name):
    """MUST OUTPUT Matrix of size P X L where:
    P is the number of plots in the window and
    L is the number of lines in the plot.
    """
    additional_x = list(np.arange(3.0, 500.0, 0.1))

    for i in range(len(additional_x)):
        outgoing_x = [[additional_x[i]], [additional_x[i]], [additional_x[i]]]
        outgoing_y = [data_gen.sinusoid(outgoing_x[0]), data_gen.tangent(outgoing_x[1]), data_gen.cosine(outgoing_x[2])]
        helper.changedSignal.emit(name, (outgoing_x, outgoing_y))
        time.sleep(.1)



init_x = [list(np.linspace(-3.0, 3.0, 100))]
init_y = [data_gen.sinusoid(init_x[0])]

init_x.append(list(np.linspace(-3.0, 3.0, 100)))
init_y.append(data_gen.tangent(init_x[1]))

init_x.append(list(np.linspace(-3.0, 3.0, 100)))
init_y.append(data_gen.cosine(init_x[2]))

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    helper = Helper()
    main = MainWindow(init_x, init_y)
    helper.changedSignal.connect(main.update_plots, QtCore.Qt.QueuedConnection)
    threading.Thread(target=create_data, args=(helper, "redundant_name"), daemon=True).start()
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtWidgets.QApplication.instance().exec_()