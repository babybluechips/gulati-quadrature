#!/usr/bin/env python3
"""Render the two-conductor coupling figure used by the paper.

This is a figure-only renderer.  It intentionally draws the actual object
described in the manuscript: two fixed-area conductor components moving from
unit disks to flattened facing plates/lobes.  It does not assemble or store a
dense Q matrix.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "q_autograd_shape_optimization" / "two_conductor_coupling_panels.png"


def circle(cx: float, cy: float, r: float, count: int = 240) -> tuple[list[float], list[float]]:
    xs: list[float] = []
    ys: list[float] = []
    for i in range(count + 1):
        t = 2.0 * math.pi * i / count
        xs.append(cx + r * math.cos(t))
        ys.append(cy + r * math.sin(t))
    return xs, ys


def rounded_plate(
    x_inner: float,
    side: int,
    *,
    width: float = 1.35,
    height: float = 2.40,
    radius: float = 0.28,
    count: int = 36,
) -> tuple[list[float], list[float]]:
    """Return a rounded rectangular conductor with its flat face at x_inner.

    side=-1 gives the left conductor extending left from x_inner; side=+1 gives
    the right conductor extending right from x_inner.
    """

    if side < 0:
        x0, x1 = x_inner - width, x_inner
    else:
        x0, x1 = x_inner, x_inner + width
    y0, y1 = -0.5 * height, 0.5 * height
    r = min(radius, 0.5 * width, 0.5 * height)
    centers = [(x1 - r, y1 - r), (x0 + r, y1 - r), (x0 + r, y0 + r), (x1 - r, y0 + r)]
    angle_spans = [(0.0, 0.5 * math.pi), (0.5 * math.pi, math.pi), (math.pi, 1.5 * math.pi), (1.5 * math.pi, 2.0 * math.pi)]
    xs: list[float] = []
    ys: list[float] = []
    for (cx, cy), (a0, a1) in zip(centers, angle_spans):
        for i in range(count + 1):
            t = a0 + (a1 - a0) * i / count
            xs.append(cx + r * math.cos(t))
            ys.append(cy + r * math.sin(t))
    xs.append(xs[0])
    ys.append(ys[0])
    return xs, ys


def draw_links(ax, x_left: float, x_right: float) -> None:
    for y in (-0.78, -0.39, 0.0, 0.39, 0.78):
        ax.plot([x_left, x_right], [y, y], color="0.68", linewidth=0.55, linestyle=(0, (2, 2)), zorder=0)


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "axes.edgecolor": "0.0",
            "axes.linewidth": 0.75,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.0), constrained_layout=True)

    # Initial unit disks: area exactly pi and gap exactly 1.
    ax = axes[0]
    for cx in (-1.5, 1.5):
        xs, ys = circle(cx, 0.0, 1.0)
        ax.fill(xs, ys, color="0.92", edgecolor="0.0", linewidth=1.45)
    ax.annotate("", xy=(-0.5, -1.22), xytext=(0.5, -1.22), arrowprops={"arrowstyle": "<->", "color": "0.2", "linewidth": 0.8})
    ax.text(0.0, -1.42, "gap = 1.00", ha="center", va="top", fontsize=8)
    ax.set_title("initial: two unit conductors", fontsize=11)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-2.8, 2.8)
    ax.set_ylim(-1.65, 1.65)
    ax.grid(True, color="0.88", linewidth=0.45)

    # Optimized pair: equal-area-normalized rounded plates/lobes with gap 0.363.
    ax = axes[1]
    gap = 0.363
    x_left = -0.5 * gap
    x_right = 0.5 * gap
    draw_links(ax, x_left, x_right)
    for x_inner, side in ((x_left, -1), (x_right, 1)):
        xs, ys = rounded_plate(x_inner, side)
        ax.fill(xs, ys, color="0.88", edgecolor="0.0", linewidth=1.45)
    ax.annotate("", xy=(x_left, -1.42), xytext=(x_right, -1.42), arrowprops={"arrowstyle": "<->", "color": "0.2", "linewidth": 0.8})
    ax.text(0.0, -1.60, "gap = 0.363", ha="center", va="top", fontsize=8)
    ax.text(0.0, 1.34, "inter-body Q energy links", ha="center", va="bottom", fontsize=8, color="0.25")
    ax.set_title("optimized: flattened facing plates", fontsize=11)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.85, 1.85)
    ax.set_ylim(-1.75, 1.75)
    ax.grid(True, color="0.88", linewidth=0.45)

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.savefig(OUT, dpi=220)
    plt.close(fig)
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
