"""Shape reconstruction routines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

from inverse_shape.geometry import StarShapeModel, as_points, centroid, polygon_area
from inverse_shape.operators import (
    classical_mds,
    directed_inverse_square_moments,
    distance_matrix_from_gulati,
)

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class PolygonReconstructionResult:
    """Result from polygon reconstruction."""

    points: FloatArray
    residual_norm: float
    method: str


def _orient_counterclockwise(points: FloatArray) -> FloatArray:
    if polygon_area(points) >= 0:
        return points
    reflected = points.copy()
    reflected[:, 0] *= -1.0
    return reflected


def reconstruct_polygon_from_gulati(
    gulati_matrix: ArrayLike, *, orient: bool = True
) -> PolygonReconstructionResult:
    """Reconstruct polygon vertices from an inverse-square Gulati matrix.

    The output is determined up to Euclidean isometry, as expected from a
    distance-based reconstruction.
    """

    d = distance_matrix_from_gulati(gulati_matrix)
    pts = classical_mds(d, ndim=2)
    pts -= pts.mean(axis=0)
    if orient:
        pts = _orient_counterclockwise(pts)
    recon_d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
    residual = float(np.linalg.norm(recon_d - d) / max(np.linalg.norm(d), 1e-15))
    return PolygonReconstructionResult(points=pts, residual_norm=residual, method="classical_mds")


def reconstruct_polygon_from_moments(
    moments: ArrayLike,
    *,
    initial: ArrayLike | None = None,
    max_nfev: int = 20_000,
) -> PolygonReconstructionResult:
    """Recover labelled polygon vertices from directed inverse-square moments.

    This is a nonlinear numerical decoder for the algebraic map. It is intended
    for moderate ``n`` and good initialization; ``reconstruct_polygon_from_gulati``
    is preferred when a Gulati matrix is available.
    """

    sigma = np.asarray(moments, dtype=np.complex128)
    if sigma.ndim != 1 or len(sigma) < 3:
        raise ValueError("moments must be a one-dimensional complex array")
    n = len(sigma)
    if initial is None:
        radius = max(float(np.median(np.abs(sigma)) ** -0.5), 1e-3)
        theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        x0 = np.column_stack([radius * np.cos(theta), radius * np.sin(theta)])
    else:
        x0 = as_points(initial)
        if len(x0) != n:
            raise ValueError("initial point count must match moments")

    def unpack(v: FloatArray) -> FloatArray:
        pts = v.reshape(n, 2)
        return pts - pts.mean(axis=0)

    def residual(v: FloatArray) -> FloatArray:
        current = directed_inverse_square_moments(unpack(v))
        diff = current - sigma
        return np.concatenate([diff.real, diff.imag])

    result = least_squares(
        residual,
        x0.reshape(-1),
        max_nfev=max_nfev,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
    )
    pts = unpack(result.x)
    pts = _orient_counterclockwise(pts)
    return PolygonReconstructionResult(
        points=pts,
        residual_norm=float(np.linalg.norm(result.fun)),
        method="directed_moment_least_squares",
    )


def fit_star_shape_model(
    points: ArrayLike, modes: int, *, center: ArrayLike | None = None
) -> StarShapeModel:
    """Fit a radial Fourier model to boundary samples."""

    if modes < 0:
        raise ValueError("modes must be non-negative")
    pts = as_points(points)
    c = np.asarray(center, dtype=np.float64) if center is not None else centroid(pts)
    if c.shape != (2,):
        raise ValueError("center must have shape (2,)")
    rel = pts - c
    theta = np.mod(np.arctan2(rel[:, 1], rel[:, 0]), 2.0 * np.pi)
    radius = np.linalg.norm(rel, axis=1)
    order = np.argsort(theta)
    theta = theta[order]
    radius = radius[order]

    cols = [np.ones_like(theta)]
    for k in range(1, modes + 1):
        cols.append(np.cos(k * theta))
        cols.append(np.sin(k * theta))
    design = np.column_stack(cols)
    coeffs, *_ = np.linalg.lstsq(design, radius, rcond=None)
    cos = np.array([coeffs[2 * k - 1] for k in range(1, modes + 1)], dtype=np.float64)
    sin = np.array([coeffs[2 * k] for k in range(1, modes + 1)], dtype=np.float64)
    return StarShapeModel(center=c, base_radius=float(coeffs[0]), cos=cos, sin=sin)
