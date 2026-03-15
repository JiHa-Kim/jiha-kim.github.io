# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "matplotlib", "pillow"]
# ///
"""
Generate an animated GIF of a 2-body orbit (Kepler problem) comparing
Forward Euler vs Velocity Verlet.

Euler leaks energy → orbit spirals outward.
Verlet conserves a modified Hamiltonian → orbit stays closed.

This is the classic demonstration that integrator choice matters.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from pathlib import Path
from PIL import Image
import io

OUTDIR = Path(__file__).parent

# ── Physics: Kepler orbit (gravity toward origin) ────────────────────────────

def accel(x):
    """Gravitational acceleration toward origin: a = -x / |x|^3"""
    r = np.linalg.norm(x)
    return -x / (r ** 3)


def integrate_euler(x0, v0, dt, nsteps):
    """Forward Euler integration."""
    xs = [x0.copy()]
    x, v = x0.copy(), v0.copy()
    for _ in range(nsteps):
        a = accel(x)
        x = x + dt * v
        v = v + dt * a
        xs.append(x.copy())
    return np.array(xs)


def integrate_verlet(x0, v0, dt, nsteps):
    """Velocity Verlet (symplectic) integration."""
    xs = [x0.copy()]
    x, v = x0.copy(), v0.copy()
    a = accel(x)
    for _ in range(nsteps):
        v_half = v + 0.5 * dt * a
        x = x + dt * v_half
        a_new = accel(x)
        v = v_half + 0.5 * dt * a_new
        a = a_new
        xs.append(x.copy())
    return np.array(xs)


def energy(xs, x0, v0, dt):
    """Compute total energy at each point (approximate v from finite diff)."""
    # For visualization only
    es = []
    for i in range(len(xs)):
        r = np.linalg.norm(xs[i])
        if i < len(xs) - 1:
            v_approx = (xs[min(i+1, len(xs)-1)] - xs[max(i-1, 0)]) / (2 * dt)
        else:
            v_approx = (xs[i] - xs[i-1]) / dt
        ke = 0.5 * np.dot(v_approx, v_approx)
        pe = -1.0 / r
        es.append(ke + pe)
    return np.array(es)


# ── Parameters ────────────────────────────────────────────────────────────────

# Circular orbit: v = 1/sqrt(r) for r=1
x0 = np.array([1.0, 0.0])
v0 = np.array([0.0, 1.0])  # circular orbit speed

dt = 0.08
nsteps = 500  # ~6.3 orbits worth of time

euler_traj = integrate_euler(x0, v0, dt, nsteps)
verlet_traj = integrate_verlet(x0, v0, dt, nsteps)

# Exact orbit (circle)
theta = np.linspace(0, 2 * np.pi, 200)
exact_x = np.cos(theta)
exact_y = np.sin(theta)


# ── Generate GIF frames ──────────────────────────────────────────────────────

COLORS = {
    "euler":  "#e74c3c",
    "verlet": "#2980b9",
    "exact":  "#bdc3c7",
    "sun":    "#f39c12",
}

frames = []
# Show every 2nd step to keep file size reasonable
frame_indices = list(range(0, nsteps + 1, 2))

for idx in frame_indices:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.set_aspect("equal")

    # Determine axis limits based on Euler trajectory extent so far
    euler_so_far = euler_traj[:idx+1]
    max_extent = max(2.0, np.max(np.abs(euler_so_far)) * 1.15)
    ax.set_xlim(-max_extent, max_extent)
    ax.set_ylim(-max_extent, max_extent)

    # Exact orbit
    ax.plot(exact_x, exact_y, "-", color=COLORS["exact"], linewidth=1.2,
            alpha=0.5, zorder=1)

    # Sun
    sun = Circle((0, 0), 0.06 * max_extent, color=COLORS["sun"],
                 zorder=5, alpha=0.9)
    ax.add_patch(sun)

    # Euler trail + current position
    trail_start = max(0, idx - 80)
    ax.plot(euler_traj[trail_start:idx+1, 0], euler_traj[trail_start:idx+1, 1],
            "-", color=COLORS["euler"], linewidth=1.5, alpha=0.6, zorder=2)
    ax.plot(euler_traj[idx, 0], euler_traj[idx, 1], "o",
            color=COLORS["euler"], markersize=7, zorder=4)

    # Verlet trail + current position
    ax.plot(verlet_traj[trail_start:idx+1, 0], verlet_traj[trail_start:idx+1, 1],
            "-", color=COLORS["verlet"], linewidth=1.5, alpha=0.6, zorder=2)
    ax.plot(verlet_traj[idx, 0], verlet_traj[idx, 1], "o",
            color=COLORS["verlet"], markersize=7, zorder=4)

    # Labels (minimal)
    ax.text(0.03, 0.97, "Forward Euler", transform=ax.transAxes,
            fontsize=11, fontweight="bold", color=COLORS["euler"],
            va="top", ha="left")
    ax.text(0.03, 0.91, "Velocity Verlet", transform=ax.transAxes,
            fontsize=11, fontweight="bold", color=COLORS["verlet"],
            va="top", ha="left")

    # Time indicator
    t_val = idx * dt
    ax.text(0.97, 0.03, f"t = {t_val:.1f}", transform=ax.transAxes,
            fontsize=9, color="#7f8c8d", va="bottom", ha="right",
            family="monospace")

    ax.set_xlabel("$x$", fontsize=11)
    ax.set_ylabel("$y$", fontsize=11)
    ax.grid(True, alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Title only on first frame
    if idx < 10:
        ax.set_title("Kepler orbit: same physics, different integrators",
                      fontsize=12, pad=10)

    fig.tight_layout()

    # Render to PIL image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    buf.seek(0)
    frames.append(Image.open(buf).copy())
    buf.close()
    plt.close(fig)

    if idx % 50 == 0:
        print(f"  frame {idx}/{nsteps}")

# Save GIF
print("Assembling GIF...")
# Hold last frame longer
frames_with_pause = frames + [frames[-1]] * 15

frames_with_pause[0].save(
    OUTDIR / "orbit_euler_vs_verlet.gif",
    save_all=True,
    append_images=frames_with_pause[1:],
    duration=40,  # ms per frame
    loop=0,
)
print(f"✓ orbit_euler_vs_verlet.gif ({len(frames)} frames)")


# ── Also generate a clean static summary frame ───────────────────────────────

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

for ax in (ax1, ax2):
    ax.set_aspect("equal")
    ax.plot(exact_x, exact_y, "-", color=COLORS["exact"], linewidth=1.2,
            alpha=0.5, label="Exact orbit")
    sun = Circle((0, 0), 0.05, color=COLORS["sun"], zorder=5, alpha=0.9)
    ax.add_patch(sun)
    ax.grid(True, alpha=0.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")

# Euler
ax1.set_title("Forward Euler — energy leaks", fontsize=12, color=COLORS["euler"])
ax1.plot(euler_traj[:, 0], euler_traj[:, 1], "-", color=COLORS["euler"],
         linewidth=0.8, alpha=0.7)
lim = np.max(np.abs(euler_traj)) * 1.1
ax1.set_xlim(-lim, lim)
ax1.set_ylim(-lim, lim)

# Verlet
ax2.set_title("Velocity Verlet — orbit stays closed", fontsize=12,
              color=COLORS["verlet"])
ax2.plot(verlet_traj[:, 0], verlet_traj[:, 1], "-", color=COLORS["verlet"],
         linewidth=0.8, alpha=0.7)
ax2.set_xlim(-1.5, 1.5)
ax2.set_ylim(-1.5, 1.5)

fig.suptitle("Same physics, different integrators", fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(OUTDIR / "orbit_static.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("✓ orbit_static.png")
