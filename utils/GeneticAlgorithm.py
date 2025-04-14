import random
import numpy as np
from utils.ImageUtils import np_arr_to_cvimage

def calc_fitness(chromosome, target_image):
    """
    Calculate the fitness of a chromosome (image) compared to the target image.
    Returns mean absolute pixel difference.
    """
    return np.mean(np.abs(chromosome - target_image))

def evolve_population(population, target_image, mutation_rate, elite_ratio):
    """
    Evolve a population of image chromosomes.
    """
    fitness_values = [calc_fitness(ind, target_image) for ind in population]

    # Select elites
    elite_count = max(1, int(len(population) * elite_ratio))
    elite_indices = np.argsort(fitness_values)[:elite_count]
    elites = [population[i] for i in elite_indices]

    # Create new population
    new_population = elites.copy()
    while len(new_population) < len(population):
        parent1, parent2 = random.choices(elites, k=2)

        # Crossover
        child = np.zeros_like(parent1)
        mask = np.random.rand(*parent1.shape) < 0.5
        child[mask] = parent1[mask]
        child[~mask] = parent2[~mask]

        # Mutation
        mutation_mask = np.random.rand(*child.shape) < mutation_rate
        child[mutation_mask] = np.random.randint(0, 256, size=child[mutation_mask].shape)

        new_population.append(child.astype(np.uint8))

    return new_population

def init_population(pop_size, image_shape):
    """
    Create a population of random cv_images with the given shape.
    """
    return [np.random.randint(0, 256, size=image_shape, dtype=np.uint8) for _ in range(pop_size)]

def simulate(target_image, generations, mutation_rate, elite_ratio, population):
    """
    Run the genetic algorithm to evolve images that match the target image.
    """

    best = None

    for i in range(generations):
        population = evolve_population(population, target_image, mutation_rate, elite_ratio)
        best = min(population, key=lambda c: calc_fitness(c, target_image))
        print(f"Generation {i}: Best fitness {calc_fitness(best, target_image):.2f}")

    return best
