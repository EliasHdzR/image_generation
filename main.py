from PyQt6.QtWidgets import QApplication
from UI import Window

if __name__ == '__main__':
    import sys

    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec())
