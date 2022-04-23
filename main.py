from PyQt6.QtWidgets import QApplication
import sys
from gui import MainWindow


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow(title='Лабораторная 2')
    window.show()

    sys.exit(app.exec())
