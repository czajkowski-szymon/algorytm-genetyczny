def objective_function(x_vector: list) -> float:
  return sum(xi**2 for xi in x_vector)