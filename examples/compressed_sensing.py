import os
import numpy as np
import cvxpy as cp
import networkx as nx
import matplotlib.pyplot as plt
from typing import List
from multiprocessing import Process
from numpy.typing import NDArray
from scipy.fftpack import fft, dct, idct
from dco import Model, solve_sync, Logger
from gossip import Gossip, create_sync_network


def dco_task(
    algorithm: str,
    theta_i: NDArray[np.float64],
    y_i: NDArray[np.float64],
    communicator: Gossip,
    dim_i: int,
    lam_i: int | float,
    alpha: int | float,
    gamma: int | float,
    max_iter: int,
    logger: Logger,
) -> None:
    from jax.numpy.linalg import norm

    def f(var: NDArray[np.float64]) -> np.float64:
        return norm(theta_i @ var - y_i) ** 2

    model = Model(dim_i, f, g_type="l1", lam=lam_i, backend="jax")

    x_i = solve_sync(
        model, communicator, alpha, gamma, algorithm, max_iter, logger=logger
    )

    if algorithm == "RAugDGM" and communicator.name == 3:
        logger.record_local(recovered_signal=x_i)
        logger.merge_local_to_global("node_3")


if __name__ == "__main__":
    # Set the script type: "cen", "dis", or "plot"
    script_type = "plot"

    # Set up directories for figures and results
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(script_dir, "figures", "compressed_sensing")
    res_dir = os.path.join(script_dir, "results", "compressed_sensing")

    # Create directories if they do not exist
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    # Sensors communication topology
    n_sens = 16

    sens_pos = {i: (i // 4, i % 4) for i in range(n_sens)}
    radius = 1

    graph = nx.random_geometric_graph(n_sens, radius, pos=sens_pos)

    sens_names = [i for i in graph.nodes]
    edge_pairs = [(i, j) for i, j in graph.edges]

    # Original signal
    n = 4096
    t = np.linspace(0, 1, n)
    x = np.cos(2 * 97 * np.pi * t) + np.cos(2 * 777 * np.pi * t)
    xt = fft(x)
    psd = (np.abs(xt) ** 2) / n

    # Signal sampling
    np.random.seed(3)
    p_total = 128
    p_sens = p_total // n_sens
    perm = {i: np.round(np.random.rand(p_sens) * n).astype(int) for i in sens_names}
    y = {i: x[perm[i]] for i in sens_names}

    # Sensing matrix and regularization parameter
    Psi = dct(np.eye(n), norm="ortho")
    Theta = {i: Psi[perm[i], :] for i in sens_names}
    lam = 0.01

    # Centralized optimization
    if script_type == "cen":
        s_hat = cp.Variable(n)
        cost = sum(
            [cp.norm2(Theta[i] @ s_hat - y[i]) ** 2 for i in sens_names]
        ) / n_sens + lam * cp.norm1(s_hat)
        prob = cp.Problem(cp.Minimize(cost))
        prob.solve(solver=cp.ECOS)

        s_hat_star = s_hat.value if s_hat.value is not None else np.zeros(n)

        np.save(os.path.join(res_dir, "s_hat_star.npy"), s_hat_star)

    # Distributed optimization
    elif script_type == "dis":
        import logging

        logging.basicConfig(level=logging.INFO)

        performance_logger = Logger()
        common_params = {
            "dim_i": n,
            "lam_i": lam,
            "max_iter": 7000,
            "logger": performance_logger,
        }
        algorithm_configs = {
            "WE": {"alpha": 0.1, "gamma": 0.893},
            "AtcWE": {"alpha": 0.1, "gamma": 1.123},
            "RGT": {"alpha": 0.1, "gamma": 0.656},
            "RAugDGM": {"alpha": 0.1, "gamma": 1.358},
        }

        processes: List[Process] = []
        gossip_network = create_sync_network(sens_names, edge_pairs, noise_scale=0.005)

        alg = "RAugDGM"
        params = algorithm_configs[alg] | common_params

        for i in sens_names:
            process = Process(
                target=dco_task,
                args=(alg, Theta[i], y[i], gossip_network[i]),
                kwargs=params,
            )
            processes.append(process)
            process.start()

        for process in processes:
            process.join()

        performance_logger.save(os.path.join(res_dir, alg))

    # Plot results
    elif script_type == "plot":
        s_hat_star = np.load(os.path.join(res_dir, "s_hat_star.npy"))

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

        sens_color = [
            "#1f77b4",  # 蓝色
            "#ff7f0e",  # 橙色
            "#2ca02c",  # 绿色
            "#d62728",  # 红色
            "#9467bd",  # 紫色
            "#8c564b",  # 棕色
            "#e377c2",  # 粉色
            "#7f7f7f",  # 灰色
            "#bcbd22",  # 黄绿色
            "#17becf",  # 青色
            "#393b79",  # 深蓝
            "#52519e",  # 蓝紫色
            "#6b4196",  # 紫罗兰
            "#85397e",  # 深紫红
            "#7f4172",  # 梅红色
            "#7f4e52",  # 深红棕
        ]

        fig1, ax1 = plt.subplots()
        ax1.set_xlim((0.25, 0.31))
        ax1.set_ylim((-2, 2))

        ax1.plot(t, x, "k")
        for i in sens_names:
            ax1.plot(t[perm[i]], y[i], "x", color=sens_color[i], markeredgewidth=3)

        ax1.set_xlabel("time (s)")

        fig1.savefig(
            os.path.join(fig_dir, "original_signal.pdf"),
            format="pdf",
            bbox_inches="tight",
        )
        fig1.savefig(
            os.path.join(fig_dir, "original_signal.png"),
            dpi=300,
            bbox_inches="tight",
        )

        fig2, ax2 = plt.subplots()
        ax2.set_xlim((0, n // 2))
        ax2.set_ylim((0, 1200))
        ax2.set_xlabel("Frequency (Hz)")

        ax2.plot(psd[: n // 2], "k")

        fig2.savefig(
            os.path.join(fig_dir, "origin_signal_freq.pdf"),
            format="pdf",
            bbox_inches="tight",
        )
        fig2.savefig(
            os.path.join(fig_dir, "origin_signal_freq.png"),
            dpi=300,
            bbox_inches="tight",
        )

        fig3, ax3 = plt.subplots()
        ax3.set_xlim((0, 7000))
        ax3.set_xlabel("iterations $k$")
        ax3.set_ylabel("MSE")

        line_options = {"linewidth": 3, "linestyle": "--"}
        algs = ["RAugDGM", "AtcWE", "WE", "RGT"]

        for alg in algs:
            results = np.stack(
                [
                    np.load(os.path.join(res_dir, alg, f"node_{i}.npz"))["x_i"]
                    for i in sens_names
                ]
            )
            mse = np.mean(
                (results - s_hat_star[np.newaxis, np.newaxis, :]) ** 2, axis=(0, 2)
            )

            (line,) = ax3.semilogy(mse, label=alg, **line_options)

        ax3.legend(loc="upper right")
        ax3.grid(True, which="major", linestyle="-", linewidth=0.8)

        fig3.savefig(
            os.path.join(fig_dir, "compressed_sensing_mse.pdf"),
            format="pdf",
            bbox_inches="tight",
        )
        fig3.savefig(
            os.path.join(fig_dir, "compressed_sensing_mse.png"),
            dpi=300,
            bbox_inches="tight",
        )

        fig4, ax4 = plt.subplots()

        recovered_signal_path = os.path.join(res_dir, "RAugDGM", "node_3.npz")
        recovered_signal = np.load(recovered_signal_path)["recovered_signal"]
        x_recon = idct(recovered_signal, norm="ortho").reshape(-1)

        ax4.plot(t, x_recon, color=sens_color[3])

        ax4.set_xlim((0.25, 0.31))
        ax4.set_ylim((-2, 2))
        ax4.set_xlabel("time (s)")

        fig4.savefig(
            os.path.join(fig_dir, "recovered_signal.pdf"),
            format="pdf",
            bbox_inches="tight",
        )
        fig4.savefig(
            os.path.join(fig_dir, "recovered_signal.png"),
            dpi=300,
            bbox_inches="tight",
        )

        fig5, ax5 = plt.subplots()

        x_recon_t = fft(x_recon)
        psd_recon = (np.abs(x_recon_t) ** 2) / n

        ax5.set_xlim((0, n // 2))
        ax5.set_ylim((0, 1200))
        ax5.set_xlabel("Frequency (Hz)")
        ax5.plot(psd_recon[: n // 2], color=sens_color[3])

        fig5.savefig(
            os.path.join(fig_dir, "recovered_signal_freq.pdf"),
            format="pdf",
            bbox_inches="tight",
        )
        fig5.savefig(
            os.path.join(fig_dir, "recovered_signal_freq.png"),
            dpi=300,
            bbox_inches="tight",
        )
