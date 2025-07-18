import numpy as np
from numpy.typing import NDArray
from dco import LocalObjective, Optimizer


def dco_task(
    algorithm: str,
    u_i: NDArray[np.float64],
    v_i: NDArray[np.float64],
    name: str,
    dim_i: int,
    rho_i: float,
    alpha: int | float,
    gamma: int | float,
    max_iter: int,
) -> None:
    def f(var: NDArray[np.float64]) -> NDArray[np.float64]:
        return (u_i @ var - v_i) ** 2 + rho_i * var @ var

    local_obj = LocalObjective(dim_i, f)
    optimizer = Optimizer.create(algorithm, name, local_obj, alpha, gamma)
    optimizer.solve_sync(max_iter)


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
    common_params = {"dim_i": dim, "rho_i": rho, "max_iter": 2000}
    algorithm_configs = {
        "EXTRA": {"alpha": 0.2, "gamma": 0.16},
        "NIDS": {"alpha": 0.2, "gamma": 0.21},
        "DIGing": {"alpha": 0.2, "gamma": 0.11},
        "AugDGM": {"alpha": 0.2, "gamma": 0.31},
        "WE": {"alpha": 0.2, "gamma": 0.17},
        "RGT": {"alpha": 0.2, "gamma": 0.11},
    }

    import sys
    from logging import basicConfig, INFO

    basicConfig(level=INFO)

    if len(sys.argv) > 1:
        node_name = "".join(sys.argv[1:])
    else:
        print("Usage: python node.py <node_name>")
        sys.exit(1)

    alg = "RGT"  # Default algorithm
    params = common_params.copy()
    params.update(algorithm_configs.get(alg, {}))

    dco_task(alg, u[node_name], v[node_name], node_name, **params)
