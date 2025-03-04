import os
import numpy as np
import networkx as nx
import cvxpy as cp
import matplotlib.pyplot as plt
from multiprocessing import Process
from numpy.typing import NDArray
from numpy.linalg import norm
from typing import Dict, List
from functools import partial
from dco import Model, Solver
from gossip import create_gossip_network, Gossip


class NodeDCO(Process):
    def __init__(
        self,
        algorithm: str,
        model: Model,
        communicator: Gossip,
        alpha: int | float,
        gamma: int | float,
        max_iter: int,
    ):
        super().__init__()

        self.algorithm = algorithm
        self.model = model
        self.communicator = communicator
        self.alpha = alpha
        self.gamma = gamma
        self.max_iter = max_iter

    def run(self) -> None:
        solver = Solver(self.model, self.communicator)
        solver.solve(self.algorithm, self.alpha, self.gamma, self.max_iter)

        save_path = f"results/ridge_regression/{self.algorithm}"
        solver.save_results(save_path)


if __name__ == "__main__":
    """
    Graph:
    1 - 2 - 3
        |
        4
    """
    N = 4

    node_names = [f"{i}" for i in range(1, N + 1)]
    edge_pairs = [("1", "2"), ("2", "3"), ("2", "4")]

    G = nx.Graph()
    G.add_nodes_from(node_names)
    G.add_edges_from(edge_pairs)

    node_pos = {"1": (0, 1), "2": (0, 0), "3": (-0.8, -0.6), "4": (0.8, -0.6)}

    options = {
        "with_labels": True,
        "font_size": 20,
        "node_color": "white",
        "node_size": 1000,
        "edgecolors": "black",
        "linewidths": 1.5,
        "width": 1.5,
    }

    fig, ax = plt.subplots(1, 1)
    ax.set_aspect(1)
    nx.draw(G, pos=node_pos, ax=ax, **options)

    os.makedirs("figures/ridge_regression", exist_ok=True)
    fig.savefig("figures/ridge_regression/graph.png", dpi=300, bbox_inches="tight")

    """
    Ridge regression problem:
    """
    T = 2000
    dim = 10

    # Ridge regression
    np.random.seed(0)

    rho = 0.01
    u = {i: np.random.uniform(-1, 1, dim) for i in node_names}
    x_tilde = {i: 0.1 * (int(i) - 1) * np.ones(dim) for i in node_names}
    epsilon = {i: np.random.normal(0, 5) for i in node_names}
    v = {i: u[i] @ x_tilde[i] + epsilon[i] for i in node_names}

    def f(var, index):
        return (u[index] @ var - v[index]) ** 2 + rho * var @ var

    models = {
        i: Model(dim, partial(f, index=i), record_history=True) for i in node_names
    }

    """
    Centralized optimization:
    """
    x = cp.Variable(dim)

    F = cp.sum([(u[i] @ x - v[i]) ** 2 for i in node_names]) / N + rho * cp.sum_squares(
        x
    )

    prob = cp.Problem(cp.Minimize(F))
    prob.solve(cp.OSQP)

    x_star = x.value

    """
    Distributed optimization:
    """
    algorithms = ["EXTRA", "NIDS", "DIGing", "AugDGM", "WE", "RGT"]
    alphas = {
        "EXTRA": 0.2,
        "NIDS": 0.2,
        "DIGing": 0.2,
        "AugDGM": 0.2,
        "WE": 0.2,
        "RGT": 0.2,
    }
    gammas = {
        "EXTRA": 0.15,
        "NIDS": 0.15,
        "DIGing": 0.06,
        "AugDGM": 0.07,
        "WE": 0.06,
        "RGT": 0.06,
    }

    for alg in algorithms:
        gossip_network = create_gossip_network(node_names, edge_pairs, noise_scale=0.005)

        nodes: Dict[str, NodeDCO] = {
            i: NodeDCO(alg, models[i], gossip_network[i], alphas[alg], gammas[alg], T)
            for i in node_names
        }

        for node in nodes.values():
            node.start()

        for node in nodes.values():
            node.join()

    """
    Plotting:
    """
    iter_series = np.arange(T)

    fig1, ax1 = plt.subplots()
    ax1.set_xlim([0, T])
    ax1.set_xlabel("iterations k")
    ax1.set_ylabel("MSE")

    for alg in algorithms:
        results: List[NDArray[np.float64]] = [
            np.load(f"results/ridge_regression/{alg}/node_{i}.npy") for i in node_names
        ]
        mse = sum([norm(result - x_star, ord=2, axis=1) ** 2 for result in results]) / N

        ax1.semilogy(iter_series, mse, label=alg, linewidth=3, linestyle="--")

    ax1.legend(loc=(0.7, 0.2))
    ax1.grid(True, which="major", linestyle="-", linewidth=0.8)

    fig1.savefig("figures/ridge_regression/mse.png", dpi=300, bbox_inches="tight")
