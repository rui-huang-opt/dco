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

# Set up directories for figures and results
script_dir = os.path.dirname(os.path.abspath(__file__))
fig_dir = os.path.join(script_dir, "figures", "ridge_regression")
res_dir = os.path.join(script_dir, "results", "ridge_regression")

# Create directories if they do not exist
os.makedirs(fig_dir, exist_ok=True)
os.makedirs(res_dir, exist_ok=True)

# Create a simple graph
node_names = ["1", "2", "3", "4"]
edge_pairs = [("1", "2"), ("2", "3"), ("2", "4")]

n_nodes = len(node_names)

# Plot the graph
graph = nx.Graph()
graph.add_nodes_from(node_names)
graph.add_edges_from(edge_pairs)

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
nx.draw(graph, pos=node_pos, ax=ax, **options)

fig.savefig(os.path.join(fig_dir, "graph.png"), dpi=300, bbox_inches="tight")

# Set parameters for ridge regression
dim = 10

np.random.seed(0)

rho = 0.01
u = {i: np.random.uniform(-1, 1, dim) for i in node_names}
x_tilde = {i: 0.1 * (int(i) - 1) * np.ones(dim) for i in node_names}
epsilon = {i: np.random.normal(0, 5) for i in node_names}
v = {i: u[i] @ x_tilde[i] + epsilon[i] for i in node_names}


def f(var, index):
    return (u[index] @ var - v[index]) ** 2 + rho * var @ var


models = {i: Model(dim, partial(f, index=i)) for i in node_names}

# Centralized optimization
x = cp.Variable(dim)

loss = cp.sum([(u[i] @ x - v[i]) ** 2 for i in node_names]) / n_nodes
regularizer = rho * cp.sum_squares(x)

prob = cp.Problem(cp.Minimize(loss + regularizer))
prob.solve(cp.OSQP)

x_star = x.value

# Distributed optimization
parameters = {
    "EXTRA": {"alpha": 0.2, "gamma": 0.15, "max_iter": 2000},
    "NIDS": {"alpha": 0.2, "gamma": 0.15, "max_iter": 2000},
    "DIGing": {"alpha": 0.2, "gamma": 0.06, "max_iter": 2000},
    "AugDGM": {"alpha": 0.2, "gamma": 0.07, "max_iter": 2000},
    "WE": {"alpha": 0.2, "gamma": 0.06, "max_iter": 2000},
    "RGT": {"alpha": 0.2, "gamma": 0.06, "max_iter": 2000},
}


def dco_task(
    algorithm: str,
    model: Model,
    communicator: Gossip,
    alpha: int | float,
    gamma: int | float,
    max_iter: int,
) -> None:
    solver = Solver(model, communicator)
    solver.solve(algorithm, alpha, gamma, max_iter)

    save_path = os.path.join(res_dir, algorithm)
    solver.save_results(save_path)


for alg, params in parameters.items():
    processes: List[Process] = []
    gossip_network = create_gossip_network(node_names, edge_pairs, noise_scale=0.005)

    for i in node_names:
        model = models[i]
        communicator = gossip_network[i]
        process = Process(
            target=dco_task,
            args=(alg, model, communicator),
            kwargs=params,
        )
        processes.append(process)
        process.start()

    for process in processes:
        process.join()

# Plot results
fig1, ax1 = plt.subplots()
ax1.set_xlim([0, 2000])
ax1.set_xlabel("iterations k")
ax1.set_ylabel("MSE")

line_options = {"linewidth": 3, "linestyle": "--"}

for alg in parameters.keys():
    results = np.stack(
        [np.load(os.path.join(res_dir, alg, f"node_{i}.npy")) for i in node_names]
    )
    mse = np.mean(norm(results - x_star, axis=2) ** 2, axis=0)

    line, = ax1.semilogy(mse, label=alg, **line_options)

ax1.legend(loc=(0.7, 0.2))
ax1.grid(True, which="major", linestyle="-", linewidth=0.8)

fig1.savefig(os.path.join(fig_dir, "mse.png"), dpi=300, bbox_inches="tight")
