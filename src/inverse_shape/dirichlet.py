"""Finite-difference Dirichlet spectrum solvers for sampled planar domains."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import eigsh

from inverse_shape.geometry import as_points

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class RasterDomain:
    """Uniform square-grid rasterization of a planar boundary."""

    mask: BoolArray
    x: FloatArray
    y: FloatArray
    h: float

    @property
    def interior_count(self) -> int:
        return int(np.count_nonzero(self.mask))


def points_in_polygon(points: ArrayLike, polygon: ArrayLike) -> BoolArray:
    """Vectorized even-odd point-in-polygon test.

    Boundary grid nodes are not treated specially; for Dirichlet finite
    differences that convention is acceptable because the unknowns represent
    interior nodes only.
    """

    query = np.asarray(points, dtype=np.float64)
    if query.ndim != 2 or query.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    poly = as_points(polygon)
    px = poly[:, 0]
    py = poly[:, 1]
    x = query[:, 0]
    y = query[:, 1]
    inside = np.zeros(len(query), dtype=bool)
    j = len(poly) - 1
    for i in range(len(poly)):
        yi = py[i]
        yj = py[j]
        crosses = (yi > y) != (yj > y)
        x_intersect = (px[j] - px[i]) * (y - yi) / (yj - yi + 1e-300) + px[i]
        inside ^= crosses & (x < x_intersect)
        j = i
    return inside


def rasterize_boundary(
    boundary: ArrayLike,
    *,
    grid_size: int = 56,
    padding: float = 0.16,
) -> RasterDomain:
    """Rasterize a closed boundary onto a uniform square grid."""

    if grid_size < 8:
        raise ValueError("grid_size must be at least 8")
    if padding < 0:
        raise ValueError("padding must be non-negative")
    pts = as_points(boundary)
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    center = 0.5 * (lo + hi)
    span = float(np.max(hi - lo))
    if span <= 0:
        raise ValueError("boundary has zero diameter")
    half_width = 0.5 * span * (1.0 + 2.0 * padding)
    x = np.linspace(center[0] - half_width, center[0] + half_width, grid_size)
    y = np.linspace(center[1] - half_width, center[1] + half_width, grid_size)
    h = float(x[1] - x[0])
    xx, yy = np.meshgrid(x, y, indexing="xy")
    query = np.column_stack([xx.ravel(), yy.ravel()])
    mask = points_in_polygon(query, pts).reshape(grid_size, grid_size)
    return RasterDomain(mask=mask, x=x, y=y, h=h)


def dirichlet_laplacian(mask: ArrayLike, h: float) -> csr_matrix:
    """Build the positive five-point Dirichlet Laplacian for an interior mask."""

    interior = np.asarray(mask, dtype=bool)
    if interior.ndim != 2:
        raise ValueError("mask must be a two-dimensional boolean array")
    if h <= 0:
        raise ValueError("grid spacing h must be positive")
    index = -np.ones(interior.shape, dtype=np.int64)
    index[interior] = np.arange(np.count_nonzero(interior))
    n = int(index.max() + 1)
    if n == 0:
        raise ValueError("mask contains no interior nodes")

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    inv_h2 = 1.0 / (h * h)
    ny, nx = interior.shape
    for row in range(ny):
        for col in range(nx):
            current = int(index[row, col])
            if current < 0:
                continue
            rows.append(current)
            cols.append(current)
            data.append(4.0 * inv_h2)
            for drow, dcol in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr = row + drow
                nc = col + dcol
                if 0 <= nr < ny and 0 <= nc < nx and index[nr, nc] >= 0:
                    rows.append(current)
                    cols.append(int(index[nr, nc]))
                    data.append(-inv_h2)

    return coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()


def dirichlet_eigenvalues(
    boundary: ArrayLike,
    *,
    k: int = 8,
    grid_size: int = 56,
    padding: float = 0.16,
    tol: float = 1e-8,
    maxiter: int | None = None,
) -> FloatArray:
    """Approximate the first ``k`` Dirichlet eigenvalues of a planar domain."""

    if k < 1:
        raise ValueError("k must be positive")
    raster = rasterize_boundary(boundary, grid_size=grid_size, padding=padding)
    if raster.interior_count <= k + 1:
        raise ValueError(
            f"grid contains only {raster.interior_count} interior nodes, "
            f"which is too few for {k} eigenvalues"
        )
    laplacian = dirichlet_laplacian(raster.mask, raster.h)
    values = eigsh(laplacian, k=k, which="SM", return_eigenvectors=False, tol=tol, maxiter=maxiter)
    values = np.sort(np.asarray(values, dtype=np.float64))
    if not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise RuntimeError("Dirichlet eigensolver returned invalid eigenvalues")
    return values
