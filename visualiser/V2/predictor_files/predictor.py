from PyQt5 import QtCore
from PyQt5.QtWidgets import QWidget

class Predictor(QWidget):
    def __init__(self):
        super().__init__()

    @QtCore.pyqtSlot(str, tuple)
    def print_pred(self, name, update):
        print("PREDICTION: ", update)