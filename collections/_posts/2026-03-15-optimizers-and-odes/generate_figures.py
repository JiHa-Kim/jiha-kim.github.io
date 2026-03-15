# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib"]
# ///
"""
Generate clean, uncluttered visualizations for the "Optimizers and ODEs" post.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUTDIR = Path(__file__).parent

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 180,
    "savefig.dpi": 180,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "axes.grid": True,
    "grid.alpha": 0.15,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

C = {
    "gd":       "#e74c3c",
    "implicit": "#2980b9",
    "lion":     "#27ae60",
    "tanh":     "#8e44ad",
    "eg":       "#16a085",
    "exact":    "#bdc3c7",
}


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Toy 1: Stiff quadratic — clean, minimal
# ═══════════════════════════════════════════════════════════════════════════════

def toy1():
    H = np.diag([100.0, 1.0])
    x0 = np.array([1.0, 1.0])

    # GD h=0.03 (explodes)
    h = 0.03
    gd = [x0.copy()]
    x = x0.copy()
    for _ in range(6):
        x = x - h * H @ x
        gd.append(x.copy())
    gd = np.array(gd)

    # Backward Euler h=0.03
    M = np.linalg.inv(np.eye(2) + h * H)
    be = [x0.copy()]
    x = x0.copy()
    for _ in range(40):
        x = M @ x
        be.append(x.copy())
    be = np.array(be)

    # Contour grid
    x1 = np.linspace(-2.5, 2.5, 200)
    x2 = np.linspace(-0.1, 1.15, 200)
    X1, X2 = np.meshgrid(x1, x2)
    Z = 0.5 * (100 * X1**2 + X2**2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for ax in (ax1, ax2):
        ax.contour(X1, X2, Z, levels=[1, 5, 15, 40, 80], colors="#d5d8dc",
                   linewidths=0.5)
        ax.plot(0, 0, "k*", markersize=10, zorder=5)
        ax.set_xlabel(r"$x_1$ (stiff)")

    # Left: GD explodes
    ax1.set_title("Forward Euler (GD) — unstable", color=C["gd"])
    ax1.plot(gd[:, 0], gd[:, 1], "o-", color=C["gd"], markersize=5,
             linewidth=1.5, zorder=4)
    ax1.set_ylabel(r"$x_2$ (soft)")
    ax1.set_xlim(-2.5, 2.5)

    # Right: backward Euler converges
    ax2.set_title("Backward Euler (Proximal) — stable", color=C["implicit"])
    ax2.plot(be[:, 0], be[:, 1], "o-", color=C["implicit"], markersize=3,
             linewidth=1.5, zorder=4, markevery=2)
    ax2.set_xlim(-0.1, 1.15)

    fig.tight_layout(w_pad=3)
    fig.savefig(OUTDIR / "toy1_stiff_quadratic.png")
    plt.close(fig)
    print("✓ toy1_stiff_quadratic.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Toy 2: Minmax — clean
# ═══════════════════════════════════════════════════════════════════════════════

def toy2():
    eta = 0.3
    x0 = np.array([1.0, 0.0])
    n = 40

    A_gda = np.array([[1, -eta], [eta, 1]])
    gda = [x0.copy()]
    x = x0.copy()
    for _ in range(n):
        x = A_gda @ x
        gda.append(x.copy())
    gda = np.array(gda)

    A_eg = np.array([[1 - eta**2, -eta], [eta, 1 - eta**2]])
    eg = [x0.copy()]
    x = x0.copy()
    for _ in range(n):
        x = A_eg @ x
        eg.append(x.copy())
    eg = np.array(eg)

    th = np.linspace(0, 2 * np.pi, 200)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    for ax in (ax1, ax2):
        ax.set_aspect("equal")
        ax.plot(np.cos(th), np.sin(th), "--", color=C["exact"], linewidth=1,
                alpha=0.4)
        ax.plot(0, 0, "k+", markersize=10, markeredgewidth=2, zorder=5)
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")

    # GDA diverges
    ax1.set_title("GDA — spirals outward", color=C["gd"])
    clip = 4.0
    mask = (np.abs(gda[:, 0]) < clip) & (np.abs(gda[:, 1]) < clip)
    last = np.where(mask)[0][-1] + 2
    ax1.plot(gda[:last, 0], gda[:last, 1], "o-", color=C["gd"],
             markersize=3, linewidth=1, zorder=3)
    ax1.plot(gda[0, 0], gda[0, 1], "o", color=C["gd"], markersize=7, zorder=5)
    ax1.set_xlim(-clip, clip)
    ax1.set_ylim(-clip, clip)

    # EG contracts
    ax2.set_title("Extragradient — contracts", color=C["eg"])
    ax2.plot(eg[:, 0], eg[:, 1], "o-", color=C["eg"],
             markersize=3, linewidth=1, zorder=3)
    ax2.plot(eg[0, 0], eg[0, 1], "o", color=C["eg"], markersize=7, zorder=5)
    ax2.set_xlim(-1.4, 1.4)
    ax2.set_ylim(-1.4, 1.4)

    fig.tight_layout(w_pad=3)
    fig.savefig(OUTDIR / "toy2_minmax.png")
    plt.close(fig)
    print("✓ toy2_minmax.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Extended Toy: clean 1×2 layout (x1 and x2 side by side)
# ═══════════════════════════════════════════════════════════════════════════════

def extended_toy():
    xstar = np.array([2.0, 2.0])
    H = np.diag([1000.0, 1.0])
    x0 = np.array([0.0, 0.0])
    n = 30
    eps, b1, b2, lam = 0.1, 0.9, 0.95, 1.0

    def grad(x):
        return H @ (x - xstar)

    # GD (stable)
    h_gd = 0.0019
    gd = [x0.copy()]
    x = x0.copy()
    for _ in range(n):
        x = x - h_gd * grad(x)
        gd.append(x.copy())
    gd = np.array(gd)

    # Implicit
    h_im = 0.1
    M = np.linalg.inv(np.eye(2) + h_im * H)
    impl = [x0.copy()]
    x = x0.copy()
    for _ in range(n):
        x = M @ (x + h_im * H @ xstar)
        impl.append(x.copy())
    impl = np.array(impl)

    # Lion (sign)
    lion = [x0.copy()]
    x, m = x0.copy(), np.zeros(2)
    for _ in range(n):
        g = grad(x)
        d = np.sign(b1 * m - (1 - b1) * g)
        x = x + eps * (d - lam * x)
        m = b2 * m - (1 - b2) * g
        lion.append(x.copy())
    lion = np.array(lion)

    # Lion (tanh)
    tanh_ = [x0.copy()]
    x, m = x0.copy(), np.zeros(2)
    for _ in range(n):
        g = grad(x)
        d = np.tanh(b1 * m - (1 - b1) * g)
        x = x + eps * (d - lam * x)
        m = b2 * m - (1 - b2) * g
        tanh_.append(x.copy())
    tanh_ = np.array(tanh_)

    steps = np.arange(n + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for ax, coord, label in [(ax1, 0, "$x_1$ (stiff, $\\lambda{=}1000$)"),
                              (ax2, 1, "$x_2$ (soft, $\\lambda{=}1$)")]:
        ax.plot(steps, gd[:, coord], "o-", color=C["gd"], markersize=2,
                linewidth=1.2, label="GD (stable)")
        ax.plot(steps, impl[:, coord], "s-", color=C["implicit"], markersize=2,
                linewidth=1.2, label="Implicit GD")
        ax.plot(steps, lion[:, coord], "^-", color=C["lion"], markersize=2.5,
                linewidth=1.2, label="Lion (sign)")
        ax.plot(steps, tanh_[:, coord], "D-", color=C["tanh"], markersize=2,
                linewidth=1.2, label="Lion (tanh)")
        ax.axhline(2.0, color="k", ls=":", lw=0.7, alpha=0.3)
        ax.axhline(1.0, color=C["lion"], ls=":", lw=0.7, alpha=0.4)
        ax.set_xlabel("Step $k$")
        ax.set_ylabel(label)
        ax.legend(loc="best", fontsize=8.5, framealpha=0.8)

    ax1.set_title("Stiff direction")
    ax2.set_title("Soft direction")

    fig.tight_layout(w_pad=3)
    fig.savefig(OUTDIR / "extended_toy_comparison.png")
    plt.close(fig)
    print("✓ extended_toy_comparison.png")


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Extended Toy 2D: clean trajectory overlay
# ═══════════════════════════════════════════════════════════════════════════════

def extended_toy_2d():
    xstar = np.array([2.0, 2.0])
    H = np.diag([1000.0, 1.0])
    x0 = np.array([0.0, 0.0])
    n = 30
    eps, b1, b2, lam = 0.1, 0.9, 0.95, 1.0

    def grad(x):
        return H @ (x - xstar)

    # Recompute
    impl = [x0.copy()]
    x = x0.copy()
    M = np.linalg.inv(np.eye(2) + 0.1 * H)
    for _ in range(n):
        x = M @ (x + 0.1 * H @ xstar)
        impl.append(x.copy())
    impl = np.array(impl)

    lion = [x0.copy()]
    x, m = x0.copy(), np.zeros(2)
    for _ in range(n):
        g = grad(x)
        d = np.sign(b1 * m - (1 - b1) * g)
        x = x + eps * (d - lam * x)
        m = b2 * m - (1 - b2) * g
        lion.append(x.copy())
    lion = np.array(lion)

    tanh_ = [x0.copy()]
    x, m = x0.copy(), np.zeros(2)
    for _ in range(n):
        g = grad(x)
        d = np.tanh(b1 * m - (1 - b1) * g)
        x = x + eps * (d - lam * x)
        m = b2 * m - (1 - b2) * g
        tanh_.append(x.copy())
    tanh_ = np.array(tanh_)

    fig, ax = plt.subplots(figsize=(7, 6.5))

    # Box constraint region
    box = plt.Rectangle((-1, -1), 2, 2, lw=1.5, edgecolor=C["lion"],
                         facecolor=C["lion"], alpha=0.05, ls="--")
    ax.add_patch(box)

    # Trajectories (skip GD — too noisy, distracts)
    ax.plot(impl[:, 0], impl[:, 1], "s-", color=C["implicit"], markersize=3.5,
            linewidth=1.5, label="Implicit GD", markevery=2, zorder=3)
    ax.plot(lion[:, 0], lion[:, 1], "^-", color=C["lion"], markersize=4,
            linewidth=1.5, label="Lion (sign)", markevery=2, zorder=3)
    ax.plot(tanh_[:, 0], tanh_[:, 1], "D-", color=C["tanh"], markersize=3,
            linewidth=1.5, label="Lion (tanh)", markevery=2, zorder=3)

    ax.plot(*x0, "ko", markersize=8, zorder=5, label="Start")
    ax.plot(*xstar, "k*", markersize=12, zorder=5, label="$x^\\star$")

    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("2-D Trajectories", fontsize=12)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.set_xlim(-0.2, 2.3)
    ax.set_ylim(-0.2, 2.3)

    fig.tight_layout()
    fig.savefig(OUTDIR / "extended_toy_2d.png")
    plt.close(fig)
    print("✓ extended_toy_2d.png")


if __name__ == "__main__":
    toy1()
    toy2()
    extended_toy()
    extended_toy_2d()
    print("\nDone.")
