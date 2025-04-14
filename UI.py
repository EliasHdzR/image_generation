from PyQt6.QtCore import QSize
from PyQt6.QtWidgets import QPushButton, QVBoxLayout, QWidget, QFileDialog, QHBoxLayout, QLabel, QSlider, QComboBox
from PyQt6 import QtGui
from PyQt6.QtGui import QPixmap, QImage
import cv2
from PyQt6 import QtCore
from utils.ImageUtils import cvimage_to_qimage, np_arr_to_cvimage
from utils.GeneticAlgorithm import init_population, simulate
import numpy as np

class ImageLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(360, 360)
        self.setMaximumSize(360, 360)
        self.setScaledContents(True)
        self.setStyleSheet("border: 1px solid black;")

    def set_image(self, image):
        pixmap = QPixmap.fromImage(image)
        self.setPixmap(pixmap)

class ControlButton(QPushButton):
    def __init__(self, icon):
        super().__init__('')
        self.setMinimumSize(QSize(50, 50))
        self.setIcon(QtGui.QIcon('resources/'+ icon +'.png'))
        self.setIconSize(QSize(24, 24))

class Window(QWidget):
    def __init__(self):
        # variables
        self.population = []
        self.target_pixels = []

        super().__init__()
        self.setWindowTitle("Image Generation")
        self.setGeometry(100, 100, 1400, 500)

        # LAYOUT BOTONES
        self.buttonOpen = QPushButton("Subir Imagen")
        BUTTON_SIZE = QSize(200, 50)
        self.buttonOpen.setMinimumSize(BUTTON_SIZE)
        self.buttonOpen.clicked.connect(self.HandleOpen)

        btnLayout = QHBoxLayout()
        btnLayout.addWidget(self.buttonOpen)

        # # sub layout controles
        self.buttonReset = ControlButton("reset")
        self.buttonPause = ControlButton("pause")
        self.buttonPlay = ControlButton("play")
        self.buttonPlay.clicked.connect(self.run)
        self.buttonStep = ControlButton("step")

        btnSubLayout = QHBoxLayout()
        btnSubLayout.addWidget(self.buttonReset)
        btnSubLayout.addWidget(self.buttonPause)
        btnSubLayout.addWidget(self.buttonPlay)
        btnSubLayout.addWidget(self.buttonStep)
        btnLayout.addLayout(btnSubLayout)

        # LAYOUT IMAGENES
        imagesLayout = QHBoxLayout()

        self.algorithmImage = ImageLabel()
        tempLayout = QVBoxLayout()
        tempLayout.addWidget(QLabel("Solución del Algoritmo Genético"))
        tempLayout.addWidget(self.algorithmImage)
        imagesLayout.addLayout(tempLayout)

        self.bestImage = ImageLabel()
        tempLayout = QVBoxLayout()
        tempLayout.addWidget(QLabel("Mejor Solución Posible"))
        tempLayout.addWidget(self.bestImage)
        imagesLayout.addLayout(tempLayout)

        self.originalImage = ImageLabel()
        tempLayout = QVBoxLayout()
        tempLayout.addWidget(QLabel("Imagen Original"))
        tempLayout.addWidget(self.originalImage)
        imagesLayout.addLayout(tempLayout)

        # LAYOUT MENU CONTROLES
        menuLayout = QVBoxLayout()
        menuLayout.addStretch(1)

        menuLayout.addWidget(QLabel("Limite de Generaciones"))
        self.generationsLimitLabel = QLabel("50")
        menuLayout.addWidget(self.generationsLimitLabel)
        self.generationsLimitSlider = QSlider(QtCore.Qt.Orientation.Horizontal)
        self.generationsLimitSlider.setMinimum(10)
        self.generationsLimitSlider.setMaximum(250)
        self.generationsLimitSlider.setValue(50)
        self.generationsLimitSlider.valueChanged.connect(self.GenerationsValue)
        menuLayout.addWidget(self.generationsLimitSlider)

        menuLayout.addWidget(QLabel("Tipo de Selección"))
        self.selectionTypeCombo = QComboBox()
        self.selectionTypeCombo.addItem("Tournament")
        self.selectionTypeCombo.addItem("Elite")
        self.selectionTypeCombo.addItem("Roulette")
        menuLayout.addWidget(self.selectionTypeCombo)

        menuLayout.addWidget(QLabel("Tipo de Cruzamiento"))
        self.crossoverTypeCombo = QComboBox()
        self.crossoverTypeCombo.addItem("Uniform")
        self.crossoverTypeCombo.addItem("Single Point")
        self.crossoverTypeCombo.addItem("Two Point")
        menuLayout.addWidget(self.crossoverTypeCombo)

        menuLayout.addWidget(QLabel("Tipo de Mutación"))
        self.mutationTypeCombo = QComboBox()
        self.mutationTypeCombo.addItem("Uniform")
        self.mutationTypeCombo.addItem("Inversion")
        self.mutationTypeCombo.addItem("Swap")
        self.mutationTypeCombo.addItem("Scramble")
        menuLayout.addWidget(self.mutationTypeCombo)

        menuLayout.addWidget(QLabel("Probabilidad de Mutación"))
        self.mutationProbabilityLabel = QLabel("1%")
        menuLayout.addWidget(self.mutationProbabilityLabel)
        self.mutationProbabilitySlider = QSlider(QtCore.Qt.Orientation.Horizontal)
        self.mutationProbabilitySlider.setMinimum(0)
        self.mutationProbabilitySlider.setMaximum(100)
        self.mutationProbabilitySlider.setValue(1)
        self.mutationProbabilitySlider.setSingleStep(1)
        self.mutationProbabilitySlider.valueChanged.connect(self.MutationValue)
        menuLayout.addWidget(self.mutationProbabilitySlider)
        menuLayout.addStretch(1)

        imagesLayout.addLayout(menuLayout)

        # MAIN LAYOUT
        layout = QVBoxLayout(self)
        layout.addLayout(btnLayout)
        layout.addLayout(imagesLayout)

    def HandleOpen(self):
        path = QFileDialog.getOpenFileName(self, "Choose File", "", "Images(*.png *.jpg *.jpeg)")[0]
        if path:
            self.target_pixels = []

            # Cargar la imagen original y downscaling
            self.originalImage.set_image(QImage(path))
            downscaled_image = cv2.imread(path)
            downscaled_image = cv2.resize(downscaled_image, (6, 6), interpolation=cv2.INTER_AREA)

            self.target_pixels = np.array(downscaled_image)
            print(self.target_pixels)
            print("-----------------------")
            shape = downscaled_image.shape

            downscaled_image = cv2.resize(downscaled_image, (512, 512), interpolation=cv2.INTER_NEAREST)
            downscaled_image = cvimage_to_qimage(downscaled_image)
            self.bestImage.set_image(downscaled_image)

            # generar poblacion
            self.population = init_population(50, shape)
            print(self.population[0])
            temp = cv2.resize(self.population[0], (512, 512), interpolation=cv2.INTER_NEAREST)
            temp = cvimage_to_qimage(temp)
            self.algorithmImage.set_image(temp)

    def GenerationsValue(self):
        self.generationsLimitLabel.setText(str(self.generationsLimitSlider.value()))

    def MutationValue(self):
        self.mutationProbabilityLabel.setText(str(self.mutationProbabilitySlider.value()) + "%")

    def run(self):
        best_image = simulate(self.target_pixels, self.generationsLimitSlider.value(), self.mutationProbabilitySlider.value() / 100, 0.1, self.population)
        print(best_image)
        best_image = cv2.resize(best_image, (512, 512), interpolation=cv2.INTER_NEAREST)

        # 2. Obtener dimensiones
        height, width, channel = best_image.shape
        bytes_per_line = channel * width

        # 3. Convertir a QImage
        q_image = QImage(best_image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        self.algorithmImage.set_image(q_image)