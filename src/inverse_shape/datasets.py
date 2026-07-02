"""Deterministic synthetic shapes for tests and examples."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def complicated_polygon(n: int = 18) -> FloatArray:
    """Return a deterministic non-regular star-shaped polygon."""

    if n < 8:
        raise ValueError("n must be at least 8 for the complicated polygon")
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    theta = theta + 0.045 * np.sin(5.0 * theta) + 0.025 * np.cos(7.0 * theta)
    radius = (
        1.15
        + 0.22 * np.cos(3.0 * theta + 0.2)
        - 0.18 * np.sin(5.0 * theta - 0.4)
        + 0.09 * np.cos(8.0 * theta)
    )
    return np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])


def piecewise_curved_boundary(samples_per_segment: int = 28) -> FloatArray:
    """Return a closed piecewise-cubic curved boundary.

    The curve has lobes, a waist, and a lower point. It is intentionally not a
    single low-frequency Fourier mode, making it useful for visual regression
    tests of sampled Gulati reconstruction and low-mode approximation.
    """

    if samples_per_segment < 4:
        raise ValueError("samples_per_segment must be at least 4")

    anchors = np.array(
        [
            [0.00, 1.28],
            [0.52, 1.14],
            [1.05, 0.62],
            [0.92, 0.03],
            [0.56, -0.58],
            [0.00, -1.18],
            [-0.56, -0.58],
            [-0.92, 0.03],
            [-1.05, 0.62],
            [-0.52, 1.14],
        ],
        dtype=np.float64,
    )
    n = len(anchors)
    points: list[FloatArray] = []
    tension = 0.58
    for i in range(n):
        p0 = anchors[(i - 1) % n]
        p1 = anchors[i]
        p2 = anchors[(i + 1) % n]
        p3 = anchors[(i + 2) % n]
        m1 = tension * (p2 - p0) / 2.0
        m2 = tension * (p3 - p1) / 2.0
        u = np.linspace(0.0, 1.0, samples_per_segment, endpoint=False)
        h00 = 2.0 * u**3 - 3.0 * u**2 + 1.0
        h10 = u**3 - 2.0 * u**2 + u
        h01 = -2.0 * u**3 + 3.0 * u**2
        h11 = u**3 - u**2
        segment = (
            h00[:, None] * p1
            + h10[:, None] * m1
            + h01[:, None] * p2
            + h11[:, None] * m2
        )
        points.append(segment)
    return np.vstack(points)
