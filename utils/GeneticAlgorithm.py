import random
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal as Signal
from PyQt6.QtGui import QImage
import time
import cv2
from utils.ImageUtils import cvimage_to_qimage
import matplotlib.pyplot as plt

def calc_fitness(chromosome, target_image):
    """
    Calculate the fitness of a chromosome (image) compared to the target image.
    Returns mean absolute pixel difference.
    """
    return np.mean(np.abs(chromosome - target_image))

def init_population(target_image_shape, pop_size):
    """
    Create a population of random cv_images with the given shape.
    """
    return [np.random.randint(0, 256, size=target_image_shape, dtype=np.uint8) for _ in range(pop_size)]

class GeneticAlgorithm(QThread):

    frame_signal = Signal(QImage)
    gen_signal = Signal(int)
    plot_signal = Signal(list)

    def __init__(self, target_image, pop_size, generations, mutation_rate, parent_selection, crossover, mutation):
        super().__init__()
        self.target_image = target_image
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.population = init_population(target_image.shape, pop_size)
        self.best_fitnessess = [] # para usar en el gráfico

        self.running = True
        self.step_requested = False
        self.paused = False

        self.parent_selection = parent_selection
        self.crossover = crossover
        self.mutation = mutation

    # Selection methods
    def tournament_selection(self, fitness_values):
        """
        Select two parents using tournament selection.
        """
        tournament_size = 5
        selected_indices = random.sample(range(len(self.population)), tournament_size)
        selected_fitness = [fitness_values[i] for i in selected_indices]
        parent1_index = selected_indices[np.argmin(selected_fitness)]
        selected_indices.remove(parent1_index)
        parent2_index = selected_indices[np.argmin([fitness_values[i] for i in selected_indices])]
        return self.population[parent1_index], self.population[parent2_index]

    def roulette_wheel_selection(self, fitness_values):
        """
        Select two parents using roulette-wheel selection.
        Los individuos con menor fitness (mejor aptitud) tienen mayor probabilidad de ser seleccionados.
        """
        # Encontrar el fitness máximo para invertir las probabilidades
        max_fitness = max(fitness_values)

        # Invertir los valores de fitness para que los menores valores tengan mayor probabilidad
        inverted_fitness = [max_fitness - f for f in fitness_values]

        # Manejar el caso donde todos los fitness son iguales
        total_fitness = sum(inverted_fitness)
        if total_fitness == 0:
            # Si todos los fitness son iguales, usar probabilidad uniforme
            probabilities = [1.0 / len(self.population)] * len(self.population)
        else:
            # Calcular probabilidades normalizadas
            probabilities = [f / total_fitness for f in inverted_fitness]

        # Seleccionar dos padres usando las probabilidades calculadas
        parent1_index = np.random.choice(range(len(self.population)), p=probabilities)
        parent2_index = np.random.choice(range(len(self.population)), p=probabilities)

        return self.population[parent1_index], self.population[parent2_index]

    def elite_selection(self, fitness_values):
        """
        Select the best individuals (elites) based on fitness.
        """
        sorted_indices = np.argsort(fitness_values)
        parent1 = self.population[sorted_indices[0]]
        parent2 = self.population[sorted_indices[1]]
        return parent1, parent2

    # Crossover methods
    def uniform_crossover(self, parent1, parent2):
        """
        Perform a uniform crossover between two parents.
        """
        mask = np.random.rand(*parent1.shape) < 0.5
        child = np.where(mask, parent1, parent2)
        return child

    def single_point_crossover(self, parent1, parent2):
        crossover_point = random.randint(1, len(parent1) - 1)
        child = parent1[:crossover_point] + parent2[crossover_point:]
        return child

    def two_point_crossover(self, parent1, parent2):
        """
        Perform a two-point crossover between two parents.
        """
        point1 = random.randint(0, len(parent1) - 1)
        point2 = random.randint(point1 + 1, len(parent1))
        child = np.concatenate((parent1[:point1], parent2[point1:point2], parent1[point2:]))
        return child

    # Mutation methods
    def uniform_mutation(self, chromosome):
        """
        Perform a uniform mutation on a chromosome.
        """
        mutation_mask = np.random.rand(*chromosome.shape) < self.mutation_rate
        chromosome[mutation_mask] = np.random.randint(0, 256, size=chromosome[mutation_mask].shape)
        return chromosome

    def inversion_mutation(self, chromosome):
        """
        Perform an inversion mutation on a chromosome.
        """
        point1 = random.randint(0, len(chromosome) - 1)
        point2 = random.randint(point1 + 1, len(chromosome))
        chromosome[point1:point2] = np.flip(chromosome[point1:point2])
        return chromosome

    def swap_mutation(self, chromosome):
        """
        Perform a swap mutation on a chromosome.
        """
        point1 = random.randint(0, len(chromosome) - 1)
        point2 = random.randint(0, len(chromosome) - 1)
        chromosome[point1], chromosome[point2] = chromosome[point2], chromosome[point1]
        return chromosome

    def scramble_mutation(self, chromosome):
        """
        Perform a scramble mutation on a chromosome.
        """
        point1 = random.randint(0, len(chromosome) - 1)
        point2 = random.randint(point1 + 1, len(chromosome))
        segment = chromosome[point1:point2]
        np.random.shuffle(segment)
        chromosome[point1:point2] = segment
        return chromosome

    def evolve_population(self, selection_method, crossover_method, mutation_method):
        """
        Evolve the current population using the selected selection, crossover, and mutation methods.
        Handles elite selection differently from other selection methods.
        """
        new_population = []

        while len(new_population) < self.pop_size:
            fitness_values = [calc_fitness(ind, self.target_image) for ind in self.population]
            parent1, parent2 = selection_method(fitness_values)

            # Aplicar crossover
            if crossover_method in [self.single_point_crossover, self.two_point_crossover]:
                parent1_flat = parent1.flatten().tolist()
                parent2_flat = parent2.flatten().tolist()
                child_flat = crossover_method(parent1_flat, parent2_flat)
                child = np.array(child_flat, dtype=np.uint8).reshape(self.target_image.shape)
            else:
                child = crossover_method(parent1, parent2)

            # Aplicar mutación
            child = mutation_method(child)
            new_population.append(child)

        # Asegurar que la población tenga el tamaño correcto
        if len(new_population) > self.pop_size:
            new_population = new_population[:self.pop_size]

        return new_population

    def run(self):
        selected_selection, selected_crossover, selected_mutation = self.selected_functions(
            self.parent_selection,
            self.crossover,
            self.mutation
        )

        gen_no = 1

        while gen_no <= self.generations and self.running:
            if self.paused and not self.step_requested:
                time.sleep(0.1)
                continue  # sigue esperando si está pausado y no se ha pedido un paso

            # Evolucionar población
            self.population = self.evolve_population(selected_selection, selected_crossover, selected_mutation)

            # Obtener el mejor individuo
            best = min(self.population, key=lambda c: calc_fitness(c, self.target_image))
            self.best_fitnessess.append(calc_fitness(best, self.target_image))

            # Actualizar el gráfico en tiempo real
            self.plot_signal.emit(self.best_fitnessess)

            best = cv2.resize(best, (512, 512), interpolation=cv2.INTER_NEAREST)
            best_image = cvimage_to_qimage(best)

            # Emitir señales
            self.frame_signal.emit(best_image)
            self.gen_signal.emit(gen_no)

            gen_no += 1
            time.sleep(0.08)

            # Si se pidió un solo paso, pausamos automáticamente
            if self.step_requested:
                self.step_requested = False
                self.paused = True

        plt.ioff()
        plt.show()

    def init_real_time_plot(self):
        """
        Inicializa el gráfico interactivo de matplotlib.
        """
        plt.ion()  # Activa el modo interactivo
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title("Evolución del Mejor Fitness por Generación")
        ax.set_xlabel("Generación")
        ax.set_ylabel("Mejor Fitness")
        ax.grid()
        line, = ax.plot([], [], marker='o', color='b', label="Mejor Fitness")  # Línea vacía
        ax.legend()
        return fig, ax, line

    def pause(self):
        self.paused = True
        print(self.paused)

    def resume(self):
        self.paused = False

    def is_paused(self):
        """
        Check if the genetic algorithm is running.
        """
        return self.paused

    def function_dictionary(self):
        """
        Return a dictionary of the functions used in the genetic algorithm.
        """
        return {
            "Tournament": self.tournament_selection,
            "Roulette": self.roulette_wheel_selection,
            "Elite": self.elite_selection,
            "Uniform Crossover": self.uniform_crossover,
            "Single Point": self.single_point_crossover,
            "Two Point": self.two_point_crossover,
            "Uniform": self.uniform_mutation,
            "Inversion": self.inversion_mutation,
            "Swap": self.swap_mutation,
            "Scramble": self.scramble_mutation,
        }

    def selected_functions(self, selection_type, crossover_type, mutation_type):
        """
        Return the selected functions based on the user's choice.
        """
        function_dict = self.function_dictionary()
        return function_dict[selection_type], function_dict[crossover_type], function_dict[mutation_type]

    def step(self):
        """
        Request a step in the genetic algorithm.
        """
        self.step_requested = True
        self.paused = False

    def stop(self):
        """
        Stop the genetic algorithm.
        """
        self.running = False
        self.quit()
        self.wait()