from genetic_algorithm import GeneticAlgorithm
from objective_function import hypersphere
from population import nbits, decode_individual

genetic_algorithm = GeneticAlgorithm(objective_function=hypersphere,
                                     population_size=100,
                                     n_generations=1000,
                                     bounds=(-5, 5),
                                     N=2,
                                     precision=6,
                                     selection_method='tournament',
                                     selection_ratio=0.3,
                                     tournament_size=3,
                                     mutation_method='boundary',
                                     p_mutation=0.7,
                                     n_mutation_points=1,
                                     cross_method='single',
                                     p_cross=0.5,
                                     p_inversion=0.2,
                                     n_elites=1)

best_solution, best_value, best_generation = genetic_algorithm.evolve()

bits = nbits(-5, 5, 6)
decoded = decode_individual(best_solution, 2, bits, -5, 5)

print(f"Najlepsze rozwiazanie: {best_solution}")
print(f"Wartosc najlepszego rozwiazania: {best_value}")
print(f"Wartosc najlepszego rozwiazania (zdekodowana): {decoded}")
print(f"Numer najlepszej generacji: {best_generation}")