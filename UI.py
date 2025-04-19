from PyQt6.QtCore import QSize
from PyQt6.QtWidgets import QPushButton, QVBoxLayout, QWidget, QFileDialog, QHBoxLayout, QLabel, QSlider, QComboBox
from PyQt6 import QtGui
from PyQt6.QtGui import QPixmap, QImage
import cv2
from PyQt6 import QtCore
from utils.ImageUtils import cvimage_to_qimage
from utils.GeneticAlgorithm import GeneticAlgorithm, init_population
import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)


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
        self.geneticAlgorithm = None


        super().__init__()
        self.setWindowTitle("Image Generation")
        self.setGeometry(100, 100, 1400, 900)

        # LAYOUT BOTONES
        self.buttonOpen = QPushButton("Subir Imagen")
        BUTTON_SIZE = QSize(200, 50)
        self.buttonOpen.setMinimumSize(BUTTON_SIZE)
        self.buttonOpen.clicked.connect(self.HandleOpen)

        btnLayout = QHBoxLayout()
        btnLayout.addWidget(self.buttonOpen)

        # # sub layout controles
        self.buttonReset = ControlButton("reset")
        self.buttonReset.clicked.connect(self.reset)
        self.buttonPause = ControlButton("pause")
        self.buttonPause.clicked.connect(self.pause)
        self.buttonPause.setEnabled(False)
        self.buttonPlay = ControlButton("play")
        self.buttonPlay.clicked.connect(self.run)
        self.buttonStep = ControlButton("step")
        self.buttonStep.clicked.connect(self.step)

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
        self.menuLayout = QVBoxLayout()
        self.menuLayout.addStretch(1)

        self.menuLayout.addWidget(QLabel("Limite de Generaciones"))
        self.generationsLimitLabel = QLabel("50")
        self.menuLayout.addWidget(self.generationsLimitLabel)
        self.generationsLimitSlider = QSlider(QtCore.Qt.Orientation.Horizontal)
        self.generationsLimitSlider.setMinimum(10)
        self.generationsLimitSlider.setMaximum(250)
        self.generationsLimitSlider.setValue(50)
        self.generationsLimitSlider.valueChanged.connect(self.GenerationsValue)
        self.menuLayout.addWidget(self.generationsLimitSlider)

        self.menuLayout.addWidget(QLabel("Tipo de Selección"))
        self.selectionTypeCombo = QComboBox()
        self.selectionTypeCombo.addItem("Tournament")
        self.selectionTypeCombo.addItem("Elite")
        self.selectionTypeCombo.addItem("Roulette")
        self.menuLayout.addWidget(self.selectionTypeCombo)
        self.selectionTypeCombo.currentTextChanged.connect(self.onMenuChange)

        self.menuLayout.addWidget(QLabel("Tipo de Cruzamiento"))
        self.crossoverTypeCombo = QComboBox()
        self.crossoverTypeCombo.addItem("Uniform Crossover")
        self.crossoverTypeCombo.addItem("Single Point")
        self.crossoverTypeCombo.addItem("Two Point")
        self.menuLayout.addWidget(self.crossoverTypeCombo)
        self.crossoverTypeCombo.currentTextChanged.connect(self.onMenuChange)

        self.menuLayout.addWidget(QLabel("Tipo de Mutación"))
        self.mutationTypeCombo = QComboBox()
        self.mutationTypeCombo.addItem("Uniform")
        self.mutationTypeCombo.addItem("Inversion")
        self.mutationTypeCombo.addItem("Swap")
        self.mutationTypeCombo.addItem("Scramble")
        self.menuLayout.addWidget(self.mutationTypeCombo)
        self.mutationTypeCombo.currentTextChanged.connect(self.onMenuChange)

        self.menuLayout.addWidget(QLabel("Probabilidad de Mutación"))
        self.mutationProbabilityLabel = QLabel("1%")
        self.menuLayout.addWidget(self.mutationProbabilityLabel)
        self.mutationProbabilitySlider = QSlider(QtCore.Qt.Orientation.Horizontal)
        self.mutationProbabilitySlider.setMinimum(0)
        self.mutationProbabilitySlider.setMaximum(100)
        self.mutationProbabilitySlider.setValue(1)
        self.mutationProbabilitySlider.setSingleStep(1)
        self.mutationProbabilitySlider.valueChanged.connect(self.MutationValue)
        self.menuLayout.addWidget(self.mutationProbabilitySlider)
        self.menuLayout.addStretch(1)

        imagesLayout.addLayout(self.menuLayout)

        # MAIN LAYOUT
        layout = QVBoxLayout(self)
        layout.addLayout(btnLayout)
        self.labelGen = QLabel("Generación Número: 0")
        layout.addWidget(self.labelGen)
        layout.addLayout(imagesLayout)
        self.canvas = MplCanvas(self, width=8, height=6, dpi=100)
        layout.addWidget(self.canvas)

        self.disableMenu()
        self.buttonPlay.setEnabled(False)
        self.buttonStep.setEnabled(False)
        self.buttonPause.setEnabled(False)
        self.buttonReset.setEnabled(False)
        self.buttonOpen.setEnabled(True)

    def HandleOpen(self):
        path = QFileDialog.getOpenFileName(self, "Choose File", "", "Images(*.png *.jpg *.jpeg)")[0]

        if not path or path is None:
            return

        self.target_pixels = []

        # Cargar la imagen original y downscaling
        self.originalImage.set_image(QImage(path))
        downscaled_image = cv2.imread(path)
        downscaled_image = cv2.resize(downscaled_image, (6, 6), interpolation=cv2.INTER_AREA)

        self.target_pixels = np.array(downscaled_image)
        downscaled_image = cv2.resize(downscaled_image, (512, 512), interpolation=cv2.INTER_NEAREST)
        downscaled_image = cvimage_to_qimage(downscaled_image)
        self.bestImage.set_image(downscaled_image)

        # generar poblacion
        self.population = init_population([6, 6, 3], 50)
        temp = cv2.resize(self.population[0], (512, 512), interpolation=cv2.INTER_NEAREST)
        temp = cvimage_to_qimage(temp)
        self.algorithmImage.set_image(temp)
        self.enableMenu()
        self.buttonReset.setEnabled(True)
        self.buttonPlay.setEnabled(True)
        self.buttonStep.setEnabled(True)

    def GenerationsValue(self):
        self.generationsLimitLabel.setText(str(self.generationsLimitSlider.value()))
        self.onMenuChange()

    def MutationValue(self):
        self.mutationProbabilityLabel.setText(str(self.mutationProbabilitySlider.value()) + "%")
        self.onMenuChange()

    def onMenuChange(self):
        if self.target_pixels is None:
            return

        self.geneticAlgorithm = GeneticAlgorithm(self.target_pixels, 50,
                                                 self.generationsLimitSlider.value(),
                                                 self.mutationProbabilitySlider.value() / 100,
                                                 self.selectionTypeCombo.currentText(),
                                                 self.crossoverTypeCombo.currentText(),
                                                 self.mutationTypeCombo.currentText())

    def reset(self):
        if self.target_pixels is None or self.geneticAlgorithm is None:
            return

        self.geneticAlgorithm.stop()
        self.population = init_population([6,6,3], 50)
        temp = cv2.resize(self.population[0], (512, 512), interpolation=cv2.INTER_NEAREST)
        temp = cvimage_to_qimage(temp)
        self.algorithmImage.set_image(temp)

        self.geneticAlgorithm = None
        self.labelGen.setText("Generación Número: 0")
        self.buttonPlay.setEnabled(True)
        self.buttonStep.setEnabled(True)
        self.buttonPause.setEnabled(False)
        self.enableMenu()
        self.canvas.ax.clear()
        self.update_plot([])

    def run(self):
        if self.target_pixels is None:
            return

        if self.geneticAlgorithm is not None and self.geneticAlgorithm.is_paused() :
            self.geneticAlgorithm.resume()
        else:
            self.geneticAlgorithm = GeneticAlgorithm(self.target_pixels, 50,
                                                     self.generationsLimitSlider.value(),
                                                     self.mutationProbabilitySlider.value() / 100,
                                                     self.selectionTypeCombo.currentText(),
                                                     self.crossoverTypeCombo.currentText(),
                                                     self.mutationTypeCombo.currentText())
            self.geneticAlgorithm.start()
            self.geneticAlgorithm.frame_signal.connect(self.algorithmImage.set_image)
            self.geneticAlgorithm.gen_signal.connect(self.setLabelGen)
            self.geneticAlgorithm.plot_signal.connect(self.update_plot)

        self.buttonPlay.setEnabled(False)
        self.buttonStep.setEnabled(False)
        self.buttonPause.setEnabled(True)
        self.disableMenu()

    def pause(self):
        if self.geneticAlgorithm is not None:
            self.geneticAlgorithm.pause()
            self.buttonPlay.setEnabled(True)
            self.buttonStep.setEnabled(True)
            self.buttonPause.setEnabled(False)

    def step(self):
        if self.geneticAlgorithm is not None:
            self.geneticAlgorithm.step()

    def setLabelGen(self, gen):
        self.labelGen.setText("Generacion Número: " + str(gen))

    def finished(self):
        self.buttonPlay.setEnabled(False)
        self.buttonStep.setEnabled(False)
        self.buttonPause.setEnabled(False)

    def update_plot(self, best_fitnessess):
        """
        Actualiza el gráfico con los datos emitidos por el hilo.
        """
        generations = range(1, len(best_fitnessess) + 1)  # Eje X (generaciones)
        self.canvas.ax.clear()  # Limpiar el gráfico actual
        self.canvas.ax.plot(generations, best_fitnessess, marker='o', color='b', label="Mejor Fitness")
        self.canvas.ax.set_title("Evolución del Mejor Fitness por Generación (Menor fitness es mejor)")
        self.canvas.ax.set_xlabel("Generación")
        self.canvas.ax.set_ylabel("Mejor Fitness")
        self.canvas.ax.legend()
        self.canvas.ax.grid()
        self.canvas.draw()  # Redibujar el gráfico actualizado

    def disableMenu(self):
        self.buttonOpen.setEnabled(False)
        self.generationsLimitSlider.setEnabled(False)
        self.selectionTypeCombo.setEnabled(False)
        self.crossoverTypeCombo.setEnabled(False)
        self.mutationTypeCombo.setEnabled(False)
        self.mutationProbabilitySlider.setEnabled(False)

    def enableMenu(self):
        self.buttonOpen.setEnabled(True)
        self.generationsLimitSlider.setEnabled(True)
        self.selectionTypeCombo.setEnabled(True)
        self.crossoverTypeCombo.setEnabled(True)
        self.mutationTypeCombo.setEnabled(True)
        self.mutationProbabilitySlider.setEnabled(True)