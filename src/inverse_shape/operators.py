"""Gulati operators and Hadamard Hessian extraction."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from inverse_shape.geometry import as_points

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def pairwise_squared_distances(points: ArrayLike) -> FloatArray:
    pts = as_points(points)
    diff = pts[:, None, :] - pts[None, :, :]
    return np.einsum("ijk,ijk->ij", diff, diff)


def gulati_laplacian(points: ArrayLike, *, diagonal: str = "row-sum") -> FloatArray:
    """Build the inverse-square Gulati Laplacian matrix.

    Off-diagonal entries are ``-|x_i-x_j|^-2`` and the default diagonal is the
    row-sum cancellation that makes constants lie in the kernel.
    """

    d2 = pairwise_squared_distances(points)
    n = len(d2)
    if np.any(d2[np.triu_indices(n, 1)] <= 0):
        raise ValueError("duplicate points produce singular Gulati weights")
    gu = np.zeros((n, n), dtype=np.float64)
    mask = ~np.eye(n, dtype=bool)
    gu[mask] = -1.0 / d2[mask]
    if diagonal == "row-sum":
        np.fill_diagonal(gu, -gu.sum(axis=1))
    elif diagonal == "zero":
        pass
    else:
        raise ValueError("diagonal must be 'row-sum' or 'zero'")
    return gu


def gulati_weight_adjacency(gulati_matrix: ArrayLike) -> FloatArray:
    """Return the zero-diagonal positive inverse-square weight adjacency."""

    mat = np.asarray(gulati_matrix, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("gulati_matrix must be a square matrix")
    mask = ~np.eye(len(mat), dtype=bool)
    if np.any(mat[mask] >= 0):
        raise ValueError("off-diagonal Gulati entries must be negative")
    weights = -mat.copy()
    np.fill_diagonal(weights, 0.0)
    return weights


def directed_inverse_square_moments(points: ArrayLike) -> ComplexArray:
    """Directed complex moments ``sigma_2^(i)=sum_{j!=i}(z_i-z_j)^-2``."""

    pts = as_points(points)
    z = pts[:, 0] + 1j * pts[:, 1]
    dz = z[:, None] - z[None, :]
    mask = ~np.eye(len(z), dtype=bool)
    inv2 = np.zeros_like(dz, dtype=np.complex128)
    inv2[mask] = dz[mask] ** -2
    out = np.zeros(len(z), dtype=np.complex128)
    out[:] = np.sum(inv2, axis=1)
    return out


def dressed_gulati_hessian(points: ArrayLike, flux: ArrayLike) -> FloatArray:
    """Leading Hadamard residual kernel for a flux-dressed Gulati Laplacian.

    The off-diagonal model is

    ``H_res(i,j) = 2/pi * p_i p_j / |x_i-x_j|^2``.

    Diagonal entries are left as zero because the continuum object is a
    finite-part distribution, not an ordinary pointwise kernel.
    """

    pts = as_points(points)
    p = np.asarray(flux, dtype=np.float64)
    if p.shape != (len(pts),):
        raise ValueError("flux must have shape (n,)")
    if np.any(p <= 0):
        raise ValueError("flux must be positive for Hopf-normalized extraction")
    return pressure_hessian_from_gulati(gulati_laplacian(pts), p)


def pressure_hessian_from_gulati(gulati_matrix: ArrayLike, pressure: ArrayLike) -> FloatArray:
    """Build the zero-diagonal pressure Hessian from a Gulati matrix.

    Off diagonal,

    ``H_ij = -(2/pi) p_i G_ij p_j = 2/pi * p_i p_j / |x_i-x_j|^2``.
    """

    weights = gulati_weight_adjacency(gulati_matrix)
    p = np.asarray(pressure, dtype=np.float64)
    if p.shape != (len(weights),):
        raise ValueError("pressure must have shape (n,)")
    if np.any(p <= 0):
        raise ValueError("pressure must be positive")
    return (2.0 / np.pi) * (p[:, None] * weights * p[None, :])


def apply_pressure_hessian_from_gulati(
    gulati_matrix: ArrayLike, pressure: ArrayLike, values: ArrayLike
) -> FloatArray:
    """Apply the zero-diagonal pressure Hessian without rebuilding distances."""

    weights = gulati_weight_adjacency(gulati_matrix)
    p = np.asarray(pressure, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    if p.shape != (len(weights),):
        raise ValueError("pressure must have shape (n,)")
    if v.shape != (len(weights),):
        raise ValueError("values must have shape (n,)")
    if np.any(p <= 0):
        raise ValueError("pressure must be positive")
    return (2.0 / np.pi) * p * (weights @ (p * v))


def pressure_gulati_energy_factor(points: ArrayLike, pressure: ArrayLike) -> FloatArray:
    """Return ``B_p`` with ``B_p.T @ B_p = (2/pi) D_p G D_p``.

    This factor gives the conservative pressure-dressed Gulati energy. Its
    off-diagonal entries are the negative of the zero-diagonal Hessian residual.
    """

    pts = as_points(points)
    p = np.asarray(pressure, dtype=np.float64)
    if p.shape != (len(pts),):
        raise ValueError("pressure must have shape (n,)")
    if np.any(p <= 0):
        raise ValueError("pressure must be positive")

    rows = len(pts) * (len(pts) - 1) // 2
    factor = np.zeros((rows, len(pts)), dtype=np.float64)
    scale = np.sqrt(2.0 / np.pi)
    row = 0
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            distance = np.linalg.norm(pts[i] - pts[j])
            if distance <= 0:
                raise ValueError("duplicate points produce singular Gulati weights")
            factor[row, i] = scale * p[i] / distance
            factor[row, j] = -scale * p[j] / distance
            row += 1
    return factor


def extract_flux_from_hessian(
    points: ArrayLike,
    h_res: ArrayLike,
    *,
    neighbor_window: int = 4,
    min_product: float = 1e-18,
) -> FloatArray:
    """Extract ground-state boundary flux from a sampled Hadamard residual.

    For nearby samples, the finite-part Laurent coefficient gives

    ``(pi/2) * H_res(i,j) * |x_i-x_j|^2 ~= p_i p_j``.

    We solve the overdetermined log-linear system

    ``log p_i + log p_j = log(product_ij)``

    using cyclic neighbor pairs. This is more stable than pointwise square
    roots and mirrors the continuum Laurent-coefficient extraction.
    """

    pts = as_points(points)
    h = np.asarray(h_res, dtype=np.float64)
    n = len(pts)
    if h.shape != (n, n):
        raise ValueError("h_res must have shape (n, n)")
    if neighbor_window < 1:
        raise ValueError("neighbor_window must be positive")

    d2 = pairwise_squared_distances(pts)
    rows: list[tuple[int, int]] = []
    values: list[float] = []
    for i in range(n):
        for offset in range(1, neighbor_window + 1):
            j = (i + offset) % n
            product = (np.pi / 2.0) * h[i, j] * d2[i, j]
            if product > min_product and np.isfinite(product):
                rows.append((i, j))
                values.append(float(np.log(product)))

    if len(rows) < n:
        raise ValueError("not enough positive neighbor products to recover flux")

    a = np.zeros((len(rows), n), dtype=np.float64)
    for row, (i, j) in enumerate(rows):
        a[row, i] = 1.0
        a[row, j] = 1.0
    b = np.asarray(values, dtype=np.float64)
    log_flux, *_ = np.linalg.lstsq(a, b, rcond=None)
    return np.exp(log_flux)


def extract_flux_from_gulati_hessian(
    gulati_matrix: ArrayLike,
    h_res: ArrayLike,
    *,
    neighbor_window: int = 4,
    min_product: float = 1e-18,
) -> FloatArray:
    """Extract pressure/flux using only the Gulati matrix and Hessian residual.

    For off-diagonal entries,

    ``p_i p_j = -(pi/2) H_ij / G_ij``.
    """

    gu = np.asarray(gulati_matrix, dtype=np.float64)
    h = np.asarray(h_res, dtype=np.float64)
    if gu.ndim != 2 or gu.shape[0] != gu.shape[1]:
        raise ValueError("gulati_matrix must be a square matrix")
    if h.shape != gu.shape:
        raise ValueError("h_res must match gulati_matrix shape")
    if neighbor_window < 1:
        raise ValueError("neighbor_window must be positive")
    n = len(gu)
    rows: list[tuple[int, int]] = []
    values: list[float] = []
    for i in range(n):
        for offset in range(1, neighbor_window + 1):
            j = (i + offset) % n
            if gu[i, j] >= 0:
                raise ValueError("off-diagonal Gulati entries must be negative")
            product = -(np.pi / 2.0) * h[i, j] / gu[i, j]
            if product > min_product and np.isfinite(product):
                rows.append((i, j))
                values.append(float(np.log(product)))

    if len(rows) < n:
        raise ValueError("not enough positive neighbor products to recover flux")

    a = np.zeros((len(rows), n), dtype=np.float64)
    for row, (i, j) in enumerate(rows):
        a[row, i] = 1.0
        a[row, j] = 1.0
    b = np.asarray(values, dtype=np.float64)
    log_flux, *_ = np.linalg.lstsq(a, b, rcond=None)
    return np.exp(log_flux)


def distance_matrix_from_gulati(gulati_matrix: ArrayLike) -> FloatArray:
    """Recover pairwise distances from a Gulati matrix."""

    mat = np.asarray(gulati_matrix, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("gulati_matrix must be a square matrix")
    off = mat.copy()
    np.fill_diagonal(off, np.nan)
    if np.any(off[~np.eye(len(mat), dtype=bool)] >= 0):
        raise ValueError("off-diagonal Gulati entries must be negative")
    d = np.sqrt(-1.0 / off)
    np.fill_diagonal(d, 0.0)
    return d


def classical_mds(distance_matrix: ArrayLike, ndim: int = 2) -> FloatArray:
    """Classical multidimensional scaling from pairwise distances."""

    d = np.asarray(distance_matrix, dtype=np.float64)
    if d.ndim != 2 or d.shape[0] != d.shape[1]:
        raise ValueError("distance_matrix must be square")
    n = len(d)
    j = np.eye(n) - np.ones((n, n)) / n
    b = -0.5 * j @ (d * d) @ j
    eigvals, eigvecs = np.linalg.eigh((b + b.T) / 2.0)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    positive = np.maximum(eigvals[:ndim], 0.0)
    return eigvecs[:, :ndim] * np.sqrt(positive)
