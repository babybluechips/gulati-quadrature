"""Tests for constrained spectrum-only reconstruction."""

from __future__ import annotations

import numpy as np

from inverse_shape.dirichlet import dirichlet_eigenvalues
from inverse_shape.metrics import rigid_rotation_hausdorff_align
from inverse_shape.spectrum_inverse import (
    reconstruct_star_shape_from_spectrum,
    star_boundary_from_spectral_coefficients,
)


def test_low_mode_star_shape_reconstructs_from_dirichlet_eigenvalues_only() -> None:
    true_coefficients = np.array([0.12, 0.05], dtype=np.float64)
    initial = np.array([0.04, 0.02], dtype=np.float64)
    target = star_boundary_from_spectral_coefficients(true_coefficients, samples=128)
    target_eigenvalues = dirichlet_eigenvalues(target, k=5, grid_size=34)

    result = reconstruct_star_shape_from_spectrum(
        target_eigenvalues,
        modes=1,
        initial=initial,
        samples=128,
        grid_size=34,
        max_nfev=110,
        regularization=0.0,
    )
    aligned = rigid_rotation_hausdorff_align(result.boundary, target, rotations=360)

    assert result.relative_residual < 0.02
    assert result.relative_residual < 0.12 * result.initial_relative_residual
    assert aligned.hausdorff_error < 0.02
