import numpy as np
from numpy.typing import NDArray
from dco import LocalObjective, Optimizer


if __name__ == "__main__":
    # Create a simple graph
    node_names = ["1", "2", "3", "4"]

    # Set parameters for ridge regression
    dim = 10

    np.random.seed(0)

    rho = 0.01
    u = {i: np.random.uniform(-1, 1, dim) for i in node_names}
    x_tilde = {i: np.multiply(0.1 * (int(i) - 1), np.ones(dim)) for i in node_names}
    epsilon = {i: np.random.normal(0, 5) for i in node_names}
    v = {i: u[i] @ x_tilde[i] + epsilon[i] for i in node_names}

    # Distributed optimization
    stepsizes = {
        "EXTRA": 0.16,
        "NIDS": 0.21,
        "DIGing": 0.11,
        "AugDGM": 0.31,
        "WE": 0.17,
        "RGT": 0.11,
    }

    import sys
    from logging import basicConfig, INFO

    basicConfig(level=INFO)

    if len(sys.argv) > 1:
        node_name = "".join(sys.argv[1:])
    else:
        print("Usage: python node.py <node_name>")
        sys.exit(1)

    algorithm = "RGT"  # Default algorithm
    gamma = stepsizes[algorithm]
    max_iter = 2000

    def f(var: NDArray[np.float64]) -> NDArray[np.float64]:
        return (u[node_name] @ var - v[node_name]) ** 2 + rho * var @ var

    local_obj = LocalObjective(dim, f)
    optimizer = Optimizer.create(node_name, local_obj, gamma, algorithm=algorithm)
    optimizer.solve_sync(max_iter)
