"""
Lesson 1: Hand-write a genetic algorithm from scratch.

Goal: find x in [0, 10] that maximizes f(x) = x * sin(10 * pi * x) + 2.
Uses real-valued encoding (a chromosome is just the float x).
"""

import numpy as np
import matplotlib.pyplot as plt

# ----- Hyperparameters -----
POP_SIZE = 50          # number of individuals in the population
GENERATIONS = 100      # how many generations to evolve
MUTATION_RATE = 0.1    # probability / strength of mutation
TOURNAMENT_K = 3       # how many compete in tournament selection
X_MIN, X_MAX = 0.0, 10.0
SEED = 42


def fitness(x):
    """Return f(x) = x * sin(10 * pi * x) + 2 for a single x or an array."""
    # TODO: implement the fitness function
    pass


def init_population(size):
    """Return an array of `size` random x values in [X_MIN, X_MAX]."""
    # TODO: generate random floats in the range
    pass


def select(pop, fitnesses):
    """Tournament selection: pick TOURNAMENT_K random individuals, return the best one."""
    # TODO: pick K random indices, return the individual with the highest fitness
    pass


def crossover(parent1, parent2):
    """Combine two parents into one child (e.g. average or blend)."""
    # TODO: produce a child from the two parents
    pass


def mutate(x, rate):
    """Add small random noise to x, then clip back into [X_MIN, X_MAX]."""
    # TODO: perturb x and clip to bounds
    pass


def main():
    np.random.seed(SEED)

    # TODO: 1. initialize the population
    # TODO: 2. loop over GENERATIONS:
    #          - evaluate fitness of every individual
    #          - keep the best individual (elitism)
    #          - build the next generation via select -> crossover -> mutate
    #          - record the best fitness of this generation
    # TODO: 3. print the best x and best f(x) found
    # TODO: 4. plot best-fitness-per-generation and save as 01_result.png
    pass


if __name__ == "__main__":
    main()
