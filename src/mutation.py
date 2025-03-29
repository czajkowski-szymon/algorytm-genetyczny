import numpy as np
import copy


def mutate_n(obj: np.ndarray, n: int):
    mutate_indices = np.random.choice(len(obj), size=n, replace=False)
    obj[mutate_indices] ^= 1


def mutate_boundary(obj: np.ndarray):
    idx = 0 if np.random.rand() < 0.5 else -1
    obj[idx] ^= 1


def mutate(pop: np.ndarray, pm: float, method='n_points', n: int = 1):
    if pm <= 0:
        return pop

    new_pop = copy.deepcopy(pop)
    for obj in new_pop:
        if np.random.rand() <= pm:
            if method == 'n_points':
                mutate_n(obj, n)
            elif method == 'boundary':
                mutate_boundary(obj)

    return new_pop
