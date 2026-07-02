"""Spectral heat-trace fitting utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class HeatTraceFit:
    """Least-squares fit of a two-dimensional Dirichlet heat trace."""

    coefficients: FloatArray
    exponents: FloatArray
    residual_norm: float

    @property
    def area(self) -> float:
        return float(4.0 * np.pi * self.coefficients[0])

    @property
    def perimeter(self) -> float:
        return float(-8.0 * np.sqrt(np.pi) * self.coefficients[1])

    @property
    def berry_coefficients(self) -> FloatArray:
        return self.coefficients[2:].copy()


def heat_trace(eigenvalues: ArrayLike, t: ArrayLike) -> FloatArray:
    """Evaluate a finite heat trace ``sum exp(-lambda_k t)``."""

    lam = np.asarray(eigenvalues, dtype=np.float64)
    tv = np.asarray(t, dtype=np.float64)
    if np.any(lam <= 0) or np.any(~np.isfinite(lam)):
        raise ValueError("eigenvalues must be finite and positive")
    if np.any(tv <= 0):
        raise ValueError("t values must be positive")
    return np.exp(-tv[:, None] * lam[None, :]).sum(axis=1)


def fit_heat_trace_from_values(
    t: ArrayLike,
    values: ArrayLike,
    *,
    max_order: int = 4,
    weights: ArrayLike | None = None,
) -> HeatTraceFit:
    """Fit ``area/(4*pi*t) - perimeter/(8*sqrt(pi*t)) + sum b_j t^(j/2)``."""

    tv = np.asarray(t, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if tv.ndim != 1 or y.shape != tv.shape:
        raise ValueError("t and values must be one-dimensional arrays with matching shape")
    if max_order < 0:
        raise ValueError("max_order must be non-negative")
    exponents = np.array([-1.0, -0.5, *[j / 2.0 for j in range(max_order + 1)]])
    design = np.column_stack([tv**p for p in exponents])
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != tv.shape:
            raise ValueError("weights must match t shape")
        design = design * w[:, None]
        y = y * w
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = float(np.linalg.norm(design @ coeffs - y))
    return HeatTraceFit(coefficients=coeffs, exponents=exponents, residual_norm=residual)


def fit_heat_trace_coefficients(
    eigenvalues: ArrayLike,
    t: ArrayLike,
    *,
    max_order: int = 4,
    weights: ArrayLike | None = None,
) -> HeatTraceFit:
    """Fit heat-trace coefficients from a finite list of eigenvalues."""

    return fit_heat_trace_from_values(
        t,
        heat_trace(eigenvalues, t),
        max_order=max_order,
        weights=weights,
    )


def spectral_zeta(eigenvalues: ArrayLike, s: complex) -> complex:
    """Finite spectral zeta approximation ``sum lambda_k^-s``."""

    lam = np.asarray(eigenvalues, dtype=np.float64)
    if np.any(lam <= 0):
        raise ValueError("eigenvalues must be positive")
    return complex(np.sum(np.exp(-complex(s) * np.log(lam))))
