import numpy as np


def cross(parent1, parent2, method="single", p=0.5) -> np.ndarray:
    if method == "single":
        return single_point_cross(parent1, parent2)
    elif method == "double":
        return double_point_cross(parent1, parent2)
    elif method == "uniform":
        return uniform_cross(parent1, parent2, p)
    elif method == "discrete":
        return discrete_cross(parent1, parent2)
    else:
        raise ValueError()


def single_point_cross(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    cross_index = np.random.randint(1, len(parent1) - 1)  # <1: B-1>
    child1 = np.concatenate((parent1[:cross_index], parent2[cross_index:]))
    child2 = np.concatenate((parent2[:cross_index], parent1[cross_index:]))
    return child1, child2


def double_point_cross(parent1: np.ndarray, parent2: np.ndarray) -> np.ndarray:
    point1, point2 = sorted(np.random.choice(range(1, len(parent1) - 1), size=2, replace=False))
    child1 = np.concatenate((parent1[:point1], parent2[point1:point2], parent1[point2:]))
    child2 = np.concatenate((parent2[:point1], parent1[point1:point2], parent2[point2:]))
    return child1, child2


def uniform_cross(parent1: np.ndarray, parent2: np.ndarray, p: float = 0.5) -> np.ndarray:
    mask = np.random.rand(len(parent1)) < p
    child1 = np.where(mask, parent2, parent1)
    child2 = np.where(mask, parent1, parent2)
    return child1, child2


def discrete_cross(parent1: np.ndarray, parent2: np.ndarray):
    mask = np.random.rand(len(parent1)) < 0.5
    return np.where(mask, parent2, parent1)[np.newaxis, ...]
