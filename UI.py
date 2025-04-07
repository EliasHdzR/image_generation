from PyQt6.QtCore import QSize, pyqtSlot as Slot
from PyQt6.QtWidgets import QPushButton, QVBoxLayout, QWidget, QFileDialog, QHBoxLayout, QLabel
from PyQt6 import QtGui
from PyQt6.QtGui import QPixmap, QImage, QIcon

class ImageLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(360, 360)
        self.setMaximumSize(360, 360)
        self.setScaledContents(True)
        self.setStyleSheet("border: 1px solid black;")

class ControlButton(QPushButton):
    def __init__(self, icon):
        super().__init__('')
        self.setMinimumSize(QSize(50, 50))
        self.clicked.connect(self.on_click)
        self.setIcon(QtGui.QIcon('resources/'+ icon +'.png'))
        self.setIconSize(QSize(24, 24))

    @Slot()
    def on_click(self):
        print(f"{self.text()} button clicked.")

class Window(QWidget):
    def __init__(self):
        # variables
        self.videoProcessor = None
        self.video_path = None

        super().__init__()
        self.setWindowTitle("Image Generation")

        # LAYOUT BOTONES
        self.buttonOpen = QPushButton("Subir Imagen")
        BUTTON_SIZE = QSize(200, 50)
        self.buttonOpen.setMinimumSize(BUTTON_SIZE)

        btnLayout = QHBoxLayout()
        btnLayout.addWidget(self.buttonOpen)

        # # sub layout controles
        self.buttonReset = ControlButton("reset")
        self.buttonPause = ControlButton("pause")
        self.buttonPlay = ControlButton("play")
        self.buttonStep = ControlButton("step")

        btnSubLayout = QHBoxLayout()
        btnSubLayout.addWidget(self.buttonReset)
        btnSubLayout.addWidget(self.buttonPause)
        btnSubLayout.addWidget(self.buttonPlay)
        btnSubLayout.addWidget(self.buttonStep)
        btnLayout.addLayout(btnSubLayout)

        # LAYOUT IMAGENES
        self.algorithmImage = ImageLabel()
        self.bestImage = ImageLabel()
        self.originalImage = ImageLabel()

        imagesLayout = QHBoxLayout()
        imagesLayout.addWidget(self.algorithmImage)
        imagesLayout.addWidget(self.bestImage)
        imagesLayout.addWidget(self.originalImage)

        # MAIN LAYOUT
        layout = QVBoxLayout(self)
        layout.addLayout(btnLayout)
        layout.addLayout(imagesLayout)

    @Slot(QImage)
    def setImage(self, image):
        """
        Función que recibe una imagen de un hilo y la muestra en un WebCamLabel.
        :param image: Imagen a mostrar.
        :return:
        """
        pixmap = QPixmap.fromImage(image)
        self.videoPlayer.setPixmap(pixmap)