import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm

if __name__ == "__main__":
    # Set up directories for figures and results
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(script_dir, "figures", "precon_illustration")

    # Create directories if they do not exist
    os.makedirs(fig_dir, exist_ok=True)

    # Illustration of the function of preconditioned matrix B
    L = np.array([[1, -1], [-1, 1]])
    E = np.eye(2)

    B_2 = E - 0.25 * L
    B_3 = E - 0.499 * L

    sqrt_B_2 = sqrtm(B_2)
    sqrt_B_3 = sqrtm(B_3)

    Q = np.diag([0.1, 0.8])
    p = np.array([-0.45, -0.45])

    x = np.linspace(-0.6, 0.8, 2000)
    y = np.linspace(-0.6, 0.8, 2000)
    X, Y = np.meshgrid(x, y)

    Z = Q[0, 0] * X**2 + Q[1, 1] * Y**2 + p[0] * X + p[1] * Y

    T = 20
    gamma = 0.5

    A = E - 0.499 * L
    sqrt_C = 0.12 * L

    xx_ini = np.array([[-0.5, -0.5]])
    xx_1 = np.tile(xx_ini, (T, 1)).T
    xx_2 = np.tile(xx_ini, (T, 1)).T
    xx_3 = np.tile(xx_ini, (T, 1)).T

    yy_1 = np.zeros((2, T))
    yy_2 = np.zeros((2, T))
    yy_3 = np.zeros((2, T))

    for k in range(T - 1):
        xx_1[:, k + 1] = (
            A @ xx_1[:, k] - gamma * (2 * Q @ xx_1[:, k] + p) - sqrt_C @ yy_1[:, k]
        )
        xx_2[:, k + 1] = (
            A @ xx_2[:, k]
            - gamma * B_2 @ (2 * Q @ xx_2[:, k] + p)
            - sqrt_C @ yy_2[:, k]
        )
        xx_3[:, k + 1] = (
            A @ xx_3[:, k]
            - gamma * B_3 @ (2 * Q @ xx_3[:, k] + p)
            - sqrt_C @ yy_3[:, k]
        )

        yy_1[:, k + 1] = yy_1[:, k] + sqrt_C @ xx_1[:, k + 1]
        yy_2[:, k + 1] = yy_2[:, k] + sqrt_C @ xx_2[:, k + 1]
        yy_3[:, k + 1] = yy_3[:, k] + sqrt_C @ xx_3[:, k + 1]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect(1)
    ax.set_xlabel("$x_{1}$", fontsize=14)
    ax.set_ylabel("$x_{2}$", fontsize=14)
    ax.set_xlim([-0.6, 0.8])
    ax.set_ylim([-0.6, 0.8])
    ax.tick_params(axis="both", which="major", labelsize=12)

    ax.plot([-0.6, 0.8], [-0.6, 0.8], "k--", label="$x_{1}=x_{2}$", linewidth=1.5)
    contour = ax.contourf(X, Y, Z, levels=20, cmap="YlGnBu", alpha=0.8)

    ax.plot(xx_1[0, :], xx_1[1, :], "o-", label="$i=1$", markersize=4, linewidth=1.5)
    ax.plot(xx_2[0, :], xx_2[1, :], "s-", label="$i=2$", markersize=4, linewidth=1.5)
    ax.plot(xx_3[0, :], xx_3[1, :], "d-", label="$i=3$", markersize=4, linewidth=1.5)

    ax.plot(0.5, 0.5, "r*", markersize=12, label="Optimal Point (0.5, 0.5)")

    ax.legend(loc="lower right", fontsize=12, frameon=True, fancybox=True, shadow=True)

    fig.savefig(
        os.path.join(fig_dir, "precon_illustration.pdf"),
        format="pdf",
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(fig_dir, "precon_illustration.png"),
        dpi=300,
        bbox_inches="tight",
    )
