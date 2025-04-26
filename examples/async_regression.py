import os
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import multiprocessing.synchronize as mps
from multiprocessing import Process, Event, Barrier
from numpy.typing import NDArray
from typing import List
from dco import Model, Solver
from gossip import Gossip, create_async_network


def dco_task(
    algorithm: str,
    u_i: NDArray[np.float64],
    v_i: NDArray[np.float64],
    communicator: Gossip,
    dim_i: int,
    rho_i: float,
    r_dir: str,
    alpha: int | float,
    gamma: int | float,
    stop_event: mps.Event,
    sync_barrier: mps.Barrier,
) -> None:
    def f(var):
        return (u_i @ var - v_i) ** 2 + rho_i * var @ var

    model = Model(dim_i, f)

    solver = Solver(model, communicator)
    solver.solve_async(
        algorithm,
        alpha,
        gamma,
        stop_event=stop_event,
        sync_barrier=sync_barrier,
        sleep_time=0.001,
    )

    save_path = os.path.join(r_dir, algorithm)
    solver.save_results(save_path)
    np.save(
        os.path.join(save_path, f"time_{communicator.name}.npy"),
        np.array(solver.time_list),
    )


if __name__ == "__main__":
    # Set up directories for figures and results
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(script_dir, "figures", "async_regression")
    res_dir = os.path.join(script_dir, "results", "async_regression")

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
    event = Event()
    barrier = Barrier(len(node_names))

    alg = "RAugDGM"
    params = {
        "alpha": 0.2,
        "gamma": 0.31,
        "dim_i": dim,
        "rho_i": rho,
        "r_dir": res_dir,
        "stop_event": event,
        "sync_barrier": barrier,
    }

    processes: List[Process] = []
    gossip_network = create_async_network(
        node_names, edge_pairs, n_channels=2, maxsize=100
    )

    for i in node_names:
        process = Process(
            target=dco_task,
            args=(alg, u[i], v[i], gossip_network[i]),
            kwargs=params,
        )
        processes.append(process)
        process.start()

    event.wait(0.5)
    event.set()

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

    times = {
        i: np.load(os.path.join(res_dir, alg, f"time_{i}.npy")) for i in node_names
    }
    results = {
        i: np.load(os.path.join(res_dir, alg, f"node_{i}.npy")) for i in node_names
    }
    mse = {
        i: np.mean((results[i] - x_star[np.newaxis, :]) ** 2, axis=1)
        for i in node_names
    }

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
