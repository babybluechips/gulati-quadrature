"""Finite-dimensional inverse reconstruction from Dirichlet eigenvalues only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

from inverse_shape.dirichlet import dirichlet_eigenvalues
from inverse_shape.geometry import BoundaryCurve, StarShapeModel

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SpectrumReconstructionResult:
    """Result of a finite-dimensional spectrum-only reconstruction."""

    coefficients: FloatArray
    boundary: FloatArray
    eigenvalues: FloatArray
    target_eigenvalues: FloatArray
    relative_residual: float
    initial_relative_residual: float
    success: bool
    nfev: int
    message: str


def star_boundary_from_spectral_coefficients(
    coefficients: ArrayLike,
    *,
    samples: int = 192,
    area: float = 1.0,
    first_harmonic: int = 2,
    include_sine: bool = True,
) -> FloatArray:
    """Build an area-normalized star boundary from paired cosine/sine coefficients.

    With ``include_sine=True`` the coefficient vector is ordered as
    ``[cos_m, ..., cos_M, sin_m, ..., sin_M]`` where ``m=first_harmonic``.
    With ``include_sine=False`` the vector contains only cosine coefficients.
    Harmonic one is skipped by default because it mostly represents translation,
    which the Dirichlet spectrum cannot determine.
    """

    coeffs = np.asarray(coefficients, dtype=np.float64)
    if coeffs.ndim != 1 or len(coeffs) == 0:
        raise ValueError("coefficients must be a non-empty vector")
    if first_harmonic < 1:
        raise ValueError("first_harmonic must be positive")
    if include_sine:
        if len(coeffs) % 2:
            raise ValueError("paired cosine/sine coefficients must have even length")
        modes = len(coeffs) // 2
        cos_coeffs = coeffs[:modes]
        sin_coeffs = coeffs[modes:]
    else:
        modes = len(coeffs)
        cos_coeffs = coeffs
        sin_coeffs = np.zeros(modes, dtype=np.float64)
    last_harmonic = first_harmonic + modes - 1
    cos = np.zeros(last_harmonic, dtype=np.float64)
    sin = np.zeros(last_harmonic, dtype=np.float64)
    cos[first_harmonic - 1 :] = cos_coeffs
    sin[first_harmonic - 1 :] = sin_coeffs
    model = StarShapeModel(
        center=np.array([0.0, 0.0]),
        base_radius=1.0,
        cos=cos,
        sin=sin,
    )
    return BoundaryCurve(model.boundary_points(samples)).normalized(area=area).points


def reconstruct_star_shape_from_spectrum(
    target_eigenvalues: ArrayLike,
    *,
    modes: int = 2,
    initial: ArrayLike | None = None,
    samples: int = 192,
    grid_size: int = 42,
    padding: float = 0.18,
    first_harmonic: int = 2,
    include_sine: bool = True,
    max_nfev: int = 220,
    coefficient_bound: float = 0.28,
    regularization: float = 2e-3,
) -> SpectrumReconstructionResult:
    """Fit a low-mode star-shaped domain using only Dirichlet eigenvalues.

    This is deliberately finite dimensional: finitely many eigenvalues do not
    determine arbitrary planar domains. The routine is intended as a numerical
    sanity check for constrained shape families.
    """

    target = np.asarray(target_eigenvalues, dtype=np.float64)
    if target.ndim != 1 or len(target) < 2:
        raise ValueError("target_eigenvalues must contain at least two values")
    if modes < 1:
        raise ValueError("modes must be positive")
    if np.any(target <= 0) or not np.all(np.isfinite(target)):
        raise ValueError("target_eigenvalues must be positive and finite")
    if initial is None:
        x0 = np.zeros((2 if include_sine else 1) * modes, dtype=np.float64)
    else:
        x0 = np.asarray(initial, dtype=np.float64)
        expected = (2 if include_sine else 1) * modes
        if x0.shape != (expected,):
            raise ValueError(f"initial must have shape ({expected},)")

    def values_for(coefficients: FloatArray) -> FloatArray:
        boundary = star_boundary_from_spectral_coefficients(
            coefficients,
            samples=samples,
            first_harmonic=first_harmonic,
            include_sine=include_sine,
        )
        return dirichlet_eigenvalues(
            boundary,
            k=len(target),
            grid_size=grid_size,
            padding=padding,
            tol=1e-7,
        )

    def objective(coefficients: FloatArray) -> float:
        try:
            values = values_for(coefficients)
        except (RuntimeError, ValueError):
            return 1e6
        residual = (values - target) / target
        penalty = (
            0.0
            if regularization <= 0
            else regularization * float(np.dot(coefficients, coefficients))
        )
        return float(np.dot(residual, residual) + penalty)

    initial_values = values_for(x0)
    initial_relative = float(np.linalg.norm((initial_values - target) / target))
    result = minimize(
        objective,
        x0,
        method="Powell",
        bounds=[(-coefficient_bound, coefficient_bound)] * len(x0),
        options={"maxfev": max_nfev, "xtol": 5e-4, "ftol": 5e-6},
    )
    coefficients = np.asarray(result.x, dtype=np.float64)
    boundary = star_boundary_from_spectral_coefficients(
        coefficients,
        samples=samples,
        first_harmonic=first_harmonic,
        include_sine=include_sine,
    )
    eigenvalues = dirichlet_eigenvalues(
        boundary,
        k=len(target),
        grid_size=grid_size,
        padding=padding,
        tol=1e-7,
    )
    relative = float(np.linalg.norm((eigenvalues - target) / target))
    return SpectrumReconstructionResult(
        coefficients=coefficients,
        boundary=boundary,
        eigenvalues=eigenvalues,
        target_eigenvalues=target,
        relative_residual=relative,
        initial_relative_residual=initial_relative,
        success=bool(result.success),
        nfev=int(result.nfev),
        message=str(result.message),
    )
