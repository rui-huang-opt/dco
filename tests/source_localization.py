import os
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process
from typing import List
from scipy.optimize import minimize, OptimizeResult
from numpy.typing import NDArray
from numpy.linalg import norm
from matplotlib.colors import BoundaryNorm
from gossip import create_gossip_network, Gossip


def dco_task(
    algorithm: str,
    sens_loc_p: NDArray[np.float64],
    meas_p: NDArray[np.float64],
    communicator: Gossip,
    dim_p: int,
    a_p: NDArray[np.float64],
    rho_p: int | float,
    r_dir: str,
    alpha: int | float,
    gamma: int | float,
    max_iter: int,
) -> None:
    import jax.numpy as jnp
    from dco import Model, Solver

    def f(var):
        return jnp.mean(
            (meas_p - a_p / jnp.linalg.norm(var - sens_loc_p)) ** 2
        ) + rho_p * jnp.sum(var**2)

    model = Model(dim_p, f, grad_backend="jax")

    solver = Solver(model, communicator)
    solver.solve(algorithm, alpha, gamma, max_iter)

    save_path = os.path.join(r_dir, algorithm)
    solver.save_results(save_path)


if __name__ == "__main__":
    # Set the script type: "centralized", "distributed", or "plot results"
    script_type = "plot results"

    # Set up directories for figures and results
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(script_dir, "figures", "source_localization")
    res_dir = os.path.join(script_dir, "results", "source_localization")

    # Create directories if they do not exist
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    # Set parameters for the source localization problem
    A = 100
    src_loc = np.array([10, 40])

    # Communication topology
    sens_names = [i for i in range(10)]
    edge_pairs = [
        [0, 3],
        [3, 9],
        [0, 2],
        [1, 2],
        [1, 7],
        [7, 8],
        [1, 5],
        [5, 6],
        [4, 6],
    ]

    # Sensor locations
    n_sens = len(sens_names)

    np.random.seed(0)
    sens_loc_x = np.random.uniform(-10, 30, n_sens)
    sens_loc_y = np.random.uniform(20, 60, n_sens)
    sens_loc = np.vstack((sens_loc_x, sens_loc_y))

    # Generate the measurements
    n_meas = 15
    meas_var = 1
    meas_normal = A / norm(src_loc[:, np.newaxis] - sens_loc, axis=0)
    meas = meas_normal[np.newaxis, :] + np.random.normal(0, meas_var, (n_meas, n_sens))

    # Regularization parameter
    rho = 0.0001

    # Centralized optimization
    if script_type == "centralized":

        def f_cen(theta: NDArray[np.float64]) -> float:
            res = 0

            for i in range(n_sens):
                for j in range(n_meas):
                    res = res + (meas[j, i] - A / norm(theta - sens_loc[:, i])) ** 2

            res = res / (n_sens * n_meas) + rho * np.sum(theta**2)

            return res

        result: OptimizeResult = minimize(f_cen, np.array([0, 0]))
        theta_star = result.get("x")

        print(f"Centralized solution: {theta_star}")

        np.save(os.path.join(res_dir, "theta_star.npy"), theta_star)

    # Distributed optimization
    elif script_type == "distributed":
        common_params = {
            "dim_p": 2,
            "a_p": A,
            "rho_p": rho,
            "r_dir": res_dir,
            "max_iter": 7000,
        }
        algorithm_configs = {
            "WE": {"alpha": 0.15, "gamma": 0.050},
            "AtcWE": {"alpha": 0.15, "gamma": 0.061},
            "RGT": {"alpha": 0.15, "gamma": 0.039},
            "RAugDGM": {"alpha": 0.15, "gamma": 0.085},
        }

        processes: List[Process] = []
        gossip_network = create_gossip_network(sens_names, edge_pairs, noise_scale=0.01)
        alg = "RAugDGM"
        params = algorithm_configs[alg] | common_params

        for i in sens_names:
            process = Process(
                target=dco_task,
                args=(alg, sens_loc[:, i], meas[:, i], gossip_network[i]),
                kwargs=params,
            )
            processes.append(process)
            process.start()

        for process in processes:
            process.join()

    # Plot results
    elif script_type == "plot results":
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

        fig1, ax1 = plt.subplots(figsize=(8, 6))
        ax1.set_aspect(1)
        ax1.set_xlim([-5, 35])
        ax1.set_ylim([19, 59])
        ax1.set_xlabel("$x$-axis", fontsize=14)
        ax1.set_ylabel("$y$-axis", fontsize=14)
        ax1.tick_params(axis="both", which="major", labelsize=12)

        xx = np.linspace(-5, 35, 1500)
        yy = np.linspace(19, 59, 1500)
        X, Y = np.meshgrid(xx, yy)
        Z = A / np.sqrt((src_loc[0] - X) ** 2 + (src_loc[1] - Y) ** 2)
        Z = np.clip(Z, 0, 1000)

        levels = np.logspace(np.log10(np.min(Z)), np.log10(np.max(Z)), 12)
        cmap = plt.get_cmap("YlGnBu")
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        contour1 = ax1.contourf(X, Y, Z, levels=levels, cmap=cmap, norm=norm, alpha=0.8)

        cbar1 = fig1.colorbar(contour1, ax=ax1, pad=0.02)
        cbar1.set_label("Signal Strength (Energy)", fontsize=12)
        cbar1.ax.tick_params(labelsize=10)

        for edge in edge_pairs:
            ax1.plot(
                *(sens_loc[:, edge]),
                linestyle="--",
                color="gray",
                linewidth=1.5,
                alpha=0.7,
                label="Links" if edge == edge_pairs[0] else None,
            )

        ax1.scatter(
            sens_loc[0, :],
            sens_loc[1, :],
            s=50,
            color="blue",
            label="Sensors",
            edgecolors="black",
            linewidth=0.5,
        )

        ax1.plot(
            src_loc[0],
            src_loc[1],
            "x",
            color="red",
            label="Source",
            markersize=10,
            markeredgewidth=2,
        )

        ax1.legend(loc="lower right", fontsize=12, frameon=True, framealpha=0.9)
        ax1.grid(visible=True, linestyle="--", alpha=0.5)

        fig1.savefig(
            os.path.join(fig_dir, "source_localization.pdf"),
            format="pdf",
            bbox_inches="tight",
        )
        fig1.savefig(
            os.path.join(fig_dir, "source_localization.png"),
            dpi=300,
            bbox_inches="tight",
        )

        fig2, ax2 = plt.subplots()
        ax2.set_xlim([0, 7000])
        ax2.set_xlabel("iterations k")
        ax2.set_ylabel("MSE")

        line_options = {"linewidth": 3, "linestyle": "--"}
        algs = ["RAugDGM", "AtcWE", "WE", "RGT"]

        theta_star = np.load(os.path.join(res_dir, "theta_star.npy"))

        for alg in algs:
            results = np.stack(
                [
                    np.load(os.path.join(res_dir, alg, f"node_{i}.npy"))
                    for i in sens_names
                ]
            )
            mse = np.mean(
                (results - theta_star[np.newaxis, np.newaxis, :]) ** 2, axis=(0, 2)
            )

            (line,) = ax2.semilogy(mse, label=alg, **line_options)

        ax2.legend(loc="upper right")
        ax2.grid(True, which="major", linestyle="-", linewidth=0.8)

        fig2.savefig(
            os.path.join(fig_dir, "source_localization_mse.pdf"),
            format="pdf",
            bbox_inches="tight",
        )
        fig2.savefig(
            os.path.join(fig_dir, "source_localization_mse.png"),
            dpi=300,
            bbox_inches="tight",
        )
