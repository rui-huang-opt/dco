import os
import numpy as np
import networkx as nx
import cvxpy as cp
import matplotlib.pyplot as plt
from multiprocessing import Process
from numpy.typing import NDArray
from numpy.linalg import norm
from typing import List
from dco import Model, Solver
from gossip import create_gossip_network, Gossip


def dco_task(
    algorithm: str,
    u_p: NDArray[np.float64],
    v_p: NDArray[np.float64],
    communicator: Gossip,
    dim_p: int,
    rho_p: float,
    r_dir: str,
    alpha: int | float,
    gamma: int | float,
    max_iter: int,
) -> None:
    def f(var):
        return (u_p @ var - v_p) ** 2 + rho_p * var @ var

    model = Model(dim_p, f)

    solver = Solver(model, communicator)
    solver.solve(algorithm, alpha, gamma, max_iter)

    save_path = os.path.join(r_dir, algorithm)
    solver.save_results(save_path)


if __name__ == "__main__":
    # Set up directories for figures and results
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(script_dir, "figures", "ridge_regression")
    res_dir = os.path.join(script_dir, "results", "ridge_regression")

    # Create directories if they do not exist
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    # Create a simple graph
    node_names = ["1", "2", "3", "4"]
    edge_pairs = [("1", "2"), ("2", "3"), ("3", "4"), ("4", "1")]

    # Set parameters for ridge regression
    dim = 10

    np.random.seed(0)

    rho = 0.01
    u = {i: np.random.uniform(-1, 1, dim) for i in node_names}
    x_tilde = {i: 0.1 * (int(i) - 1) * np.ones(dim) for i in node_names}
    epsilon = {i: np.random.normal(0, 5) for i in node_names}
    v = {i: u[i] @ x_tilde[i] + epsilon[i] for i in node_names}

    # Centralized optimization
    x = cp.Variable(dim)

    n_nodes = len(node_names)
    loss = cp.sum([(u[i] @ x - v[i]) ** 2 for i in node_names]) / n_nodes
    regularizer = rho * cp.sum_squares(x)

    prob = cp.Problem(cp.Minimize(loss + regularizer))
    prob.solve(cp.OSQP)

    x_star = x.value

    # Distributed optimization
    common_params = {"dim_p": dim, "rho_p": rho, "r_dir": res_dir, "max_iter": 2000}
    algorithm_configs = {
        "EXTRA": {"alpha": 0.2, "gamma": 0.16},
        "NIDS": {"alpha": 0.2, "gamma": 0.21},
        "DIGing": {"alpha": 0.2, "gamma": 0.11},
        "AugDGM": {"alpha": 0.2, "gamma": 0.31},
        "WE": {"alpha": 0.2, "gamma": 0.17},
        "RGT": {"alpha": 0.2, "gamma": 0.11},
    }

    for alg, params in algorithm_configs.items():
        params |= common_params

        processes: List[Process] = []
        gossip_network = create_gossip_network(
            node_names, edge_pairs, noise_scale=0.001
        )

        for i in node_names:
            process = Process(
                target=dco_task,
                args=(alg, u[i], v[i], gossip_network[i]),
                kwargs=params,
            )
            processes.append(process)
            process.start()

        for process in processes:
            process.join()

    # Plot results
    plt.rcParams["text.usetex"] = True  # 使用外部 LaTeX 编译器
    plt.rcParams["font.family"] = "serif"  # 设置字体为 LaTeX 的默认 serif 字体

    plt.rcParams.update(
        {
            "font.size": 14,  # 全局字体大小
            "axes.titlesize": 16,  # 坐标轴标题字体大小
            "axes.labelsize": 16,  # 坐标轴标签字体大小
            "xtick.labelsize": 16,  # x轴刻度标签字体大小
            "ytick.labelsize": 16,  # y轴刻度标签字体大小
            "legend.fontsize": 13,  # 图例字体大小
        }
    )

    fig1, ax1 = plt.subplots()
    ax1.set_xlim([0, 2000])
    ax1.set_xlabel("iterations k")
    ax1.set_ylabel("MSE")

    line_options = {"linewidth": 3, "linestyle": "--"}

    for alg in algorithm_configs.keys():
        results = np.stack(
            [np.load(os.path.join(res_dir, alg, f"node_{i}.npy")) for i in node_names]
        )
        mse = np.mean((results - x_star[np.newaxis, np.newaxis, :]) ** 2, axis=(0, 2))

        (line,) = ax1.semilogy(mse, label=alg, **line_options)

    ax1.legend(loc=(0.7, 0.28))
    ax1.grid(True, which="major", linestyle="-", linewidth=0.8)

    fig1.savefig(os.path.join(fig_dir, "mse.pdf"), format="pdf", bbox_inches="tight")
    fig1.savefig(os.path.join(fig_dir, "mse.png"), dpi=300, bbox_inches="tight")
