from src.population import nbits, generate_population, evaluate_population


class GeneticAlgorithm:
    def __init__(self, objective_function, population_size, n_generations, bounds, N=2, precision=6,
                 selection_method='best', selection_ratio=0.3, tournament_size=3, mutation_method='n_points',
                 mutation_probability=0.7, n_mutation_points=1, n_elites=1, cross_method='single', p_cross=0.5,
                 p_inversion=0.0):
        self.objective_function = objective_function
        self.population_size = population_size
        self.n_generations = n_generations
        self.bounds = bounds
        self.N = N
        self.precision = precision
        self.selection_method = selection_method
        self.selection_ratio = selection_ratio
        self.tournament_size = tournament_size
        self.mutation_method = mutation_method
        self.mutation_probability = mutation_probability
        self.n_mutation_points = n_mutation_points
        self.n_elites = n_elites
        self.cross_method = cross_method
        self.p_cross = p_cross
        self.p_inversion = p_inversion
        self.bits = nbits(self.bounds[0], self.bounds[1], self.precision)
        self.population = generate_population(self.population_size, self.N, self.bits)
        self.evaluated_population = evaluate_population()
        best_generation = 0
        list_best = []
        list_best_generation = []
        list_mean = []

    def evolve(self):
        pass

