from datetime import datetime
from pathlib import Path

from genetic_algorithm import GeneticAlgorithm
from objective_function import hypersphere
from population import nbits, decode_individual
from file_writer import save
import time

from plotter import *

genetic_algorithm = GeneticAlgorithm(objective_function=hypersphere,
                                     population_size=100,
                                     n_generations=100,
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
start = time.time()
result = genetic_algorithm.evolve()
end = time.time()

bits = nbits(-5, 5, 6)
decoded = decode_individual(result['best_solution'], 2, bits, -5, 5)
result['decoded_solution'] = decoded
result['time'] = end - start

dir_name = f"../results/{datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}"
Path(dir_name).mkdir(parents=True, exist_ok=True)

save(f"{dir_name}/wynik.txt", result)

plot_function_3d(hypersphere, f"{dir_name}/funkcja_celu.png")

for population in genetic_algorithm.population_history:
    plot_population_3d(population[1], population[2], f"Generacja nr. {population[0]}",f"{dir_name}/generacja-nr-{population[0]}.png")

plot_best(genetic_algorithm.best_values_history, "Wykres wartości najlepszego rozwiązania w zależności od generacji", f"{dir_name}/wykres.png")