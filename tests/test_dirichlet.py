"""Tests for finite-difference Dirichlet spectra."""

from __future__ import annotations

import numpy as np

from inverse_shape.dirichlet import dirichlet_eigenvalues
from inverse_shape.geometry import BoundaryCurve, StarShapeModel


def test_dirichlet_eigenvalues_are_positive_and_sorted() -> None:
    model = StarShapeModel(
        center=np.array([0.0, 0.0]),
        base_radius=1.0,
        cos=np.zeros(1),
        sin=np.zeros(1),
    )
    boundary = BoundaryCurve(model.boundary_points(96)).normalized().points
    values = dirichlet_eigenvalues(boundary, k=5, grid_size=32)

    assert values.shape == (5,)
    assert np.all(values > 0)
    assert np.all(np.diff(values) >= 0)
