import math
import random
import numpy as np


def nbits(a, b, precision):
    return math.ceil(math.log2((b - a) * 10 ** precision) + math.log2(1))


def generate_population(pop_size: int, n_variables: int, n_bits: int) -> np.ndarray:
    pop = np.random.randint(2, size=(pop_size, n_variables * n_bits))
    return pop


def decode_individual(individual: np.ndarray, N: int, B: int, a: float, b: float) -> np.ndarray:
    reshaped_by_vars = individual.reshape(N, B)

    powers_of_two = 1 << np.arange(reshaped_by_vars.shape[-1] - 1, -1, -1)
    decimals = reshaped_by_vars.dot(powers_of_two)

    return a + decimals * (b - a) / (2 ** B - 1)

def decode_population(population: np.ndarray, N: int, B: int, a: float, b: float) -> list:
    return [decode_individual(individual, N, B, a, b) for individual in population]


def evaluate_population(func, pop, N, B, a, b):
    decode = lambda x: decode_individual(x, N, B, a, b)
    decoded_pop = np.apply_along_axis(decode, 1, pop)
    evaluated_pop = np.apply_along_axis(func, 1, decoded_pop)

    return evaluated_pop


def get_best(pop, evaluated_pop, max=True):
    best_index = np.argmax(evaluated_pop) if max else np.argmin(evaluated_pop)

    best_individual = pop[best_index]
    best_value = evaluated_pop[best_index]
    return best_individual, best_value


def get_elites(pop: np.ndarray, evaluated_pop: np.ndarray, n: int, maximize: bool) -> tuple:
    if n <= 0:
        return np.empty((0, pop.shape[1]), dtype=int), np.empty((0, 1), dtype=int)

    n = min(n, len(pop))
    indices_asc = np.argsort(evaluated_pop)
    best_indices = indices_asc[::-1][:n] if maximize else indices_asc[:n]

    elites = pop[best_indices]
    return elites, best_indices


def select(pop, pop_results, method='best', selection_ratio=0.3, tournament_size=3, max=True):
    n = math.ceil(selection_ratio * len(pop_results))

    if method == "best":
        return best(pop, pop_results, n, max)
    elif method == "roulette":
        return roulette(pop, pop_results, n, max)
    elif method == "tournament":
        return tournament(pop, pop_results, n, tournament_size)
    else:
        raise ValueError()


def best(pop, pop_results, n, max=True):
    return get_elites(pop, pop_results, n, False)[0]


def roulette(pop, pop_results, n_selected, max=True):
    evaluated_pop = pop_results if max else pop_results ** (-1)

    sum_all = np.sum(evaluated_pop)
    probabilities = evaluated_pop / sum_all

    assert np.isclose(np.sum(probabilities), 1), f"Sum of propabilities does not equal 1. Found: {sum(probabilities)}"

    wheel = np.zeros_like(evaluated_pop)
    wheel[0] = probabilities[0]
    for i in range(1, len(probabilities)):
        wheel[i] = wheel[i - 1] + probabilities[i]

    selected = np.zeros_like(pop)[:n_selected]
    for i in range(n_selected):
        rand_num = random.random()
        for j in range(len(wheel)):
            if rand_num <= wheel[j]:
                selected[i] = pop[j]
                break

    return selected


def tournament(pop, pop_results, num_selected, tournament_size, max=True):
    selected = []
    available_indices = list(range(len(pop)))  # Lista dostępnych indeksów

    for _ in range(num_selected):
        if not available_indices:
            break

        tournament_indices = random.sample(available_indices, min(tournament_size, len(available_indices)))
        tournament_contestants = pop[tournament_indices]
        tournament_scores = pop_results[tournament_indices]

        best_idx = np.argmax(tournament_scores) if max else np.argmin(tournament_scores)
        selected.append(tournament_contestants[best_idx])

        available_indices.remove(tournament_indices[best_idx])

    return np.array(selected)
