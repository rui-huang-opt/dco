import os
import time
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import multiprocessing.synchronize as mps
from multiprocessing import Process, Event, Barrier
from numpy.typing import NDArray
from typing import List
from dco import Model, solve_async, Logger
from gossip import Gossip, create_async_network, NodeHandle


def dco_task(
    algorithm: str,
    u_i: NDArray[np.float64],
    v_i: NDArray[np.float64],
    communicator: Gossip,
    dim_i: int,
    rho_i: float,
    alpha: int | float,
    gamma: int | float,
    stop_event: mps.Event,
    sync_barrier: mps.Barrier,
    logger: Logger,
) -> None:
    def f(var):
        return (u_i @ var - v_i) ** 2 + rho_i * var @ var

    model = Model(dim_i, f)

    solve_async(
        model,
        communicator,
        alpha,
        gamma,
        stop_event,
        algorithm,
        sync_barrier=sync_barrier,
        logger=logger,
    )


if __name__ == "__main__":
    # Set up directories for figures and results
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(script_dir, "figures", "async_regression")

    # Create directories if they do not exist
    os.makedirs(fig_dir, exist_ok=True)

    # Create a simple graph
    node_names = ["1", "2", "3", "4"]
    edge_pairs = [
        ("1", "2"),
        ("2", "3"),
        ("3", "4"),
        ("4", "1"),
        ("1", "3"),
        ("2", "4"),
    ]

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

    x_star = x.value if x.value is not None else np.zeros(dim)

    # Distributed optimization
    nh = NodeHandle()
    global_logger = Logger()
    barrier = Barrier(len(node_names) + 1)

    alg = "RAugDGM"
    params = {
        "alpha": 0.2,
        "gamma": 0.31,
        "dim_i": dim,
        "rho_i": rho,
        "stop_event": nh.stop_event,
        "sync_barrier": barrier,
        "logger": global_logger,
    }

    processes: List[Process] = []
    gossip_network = create_async_network(
        nh, node_names, edge_pairs, n_channels=2, maxsize=50
    )

    for i in node_names:
        process = Process(
            target=dco_task,
            args=(alg, u[i], v[i], gossip_network[i]),
            kwargs=params,
        )
        processes.append(process)
        process.start()

    barrier.wait()
    time.sleep(0.5)
    nh.stop()

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
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("MSE")

    line_options = {"linewidth": 3, "linestyle": "--"}

    results = global_logger.export_log()
    times = {
        i: results[f"node_{i}"]["timestamp"] - results[f"node_{i}"]["start_time"]
        for i in node_names
    }
    x_evo = {i: results[f"node_{i}"]["x_i"] for i in node_names}
    mse = {i: np.mean((x_evo[i] - x_star) ** 2, axis=1) for i in node_names}

    for i in node_names:
        ax1.semilogy(
            times[i],
            mse[i],
            label=f"node {i}",
            **line_options,
        )

    ax1.legend(loc="upper right")
    ax1.grid(True, which="major", linestyle="-", linewidth=0.8)

    fig1.savefig(
        os.path.join(fig_dir, "async_regression.pdf"), format="pdf", bbox_inches="tight"
    )
    fig1.savefig(
        os.path.join(fig_dir, "async_regression.png"), dpi=300, bbox_inches="tight"
    )
