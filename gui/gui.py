from PyQt6 import uic, QtWidgets

from pathlib import Path
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

from core.core import Plotter


class Canvas(FigureCanvasQTAgg):
    def __init__(self, figure) -> None:
        super().__init__(figure)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, title):
        super().__init__()

        uic.loadUi(Path(__file__).parent.resolve().joinpath('ui.ui'), self)
        self.setWindowTitle(title)

        self.plotter = Plotter()
        self.canvas = Canvas(self.plotter.figure)

        layout = self.placeholder.layout()
        layout.addWidget(self.canvas)

    def plot(self):
        self.plotter.params = self.get_params()
        self.plotter.plot_stack.append(self.plotter.plot_hist)
        self.plotter.plot()

    def get_params(self):
        return {
            'N': int(self.N.text()),
            'a': int(self.a.text()),
            'b': int(self.b.text()),
            'm': int(self.m.text()),
            'r': int(self.r.text()),
            'bins': int(self.bins.text()),
        }
