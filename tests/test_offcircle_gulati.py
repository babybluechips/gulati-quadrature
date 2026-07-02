import numpy as np

from inverse_shape.datasets import piecewise_curved_boundary
from inverse_shape.geometry import BoundaryCurve, StarShapeModel, resample_closed_curve
from inverse_shape.operators import gulati_laplacian
from inverse_shape.quadrature import (
    arclength_scaled_gulati_eigenvalues,
    gulati_coercivity_at_point,
    gulati_incidence_factor,
    gulati_weyl_pair_ratios,
    near_boundary_gulati_coercivity_table,
    outward_unit_normals,
    vertex_arclength_weights,
)


def _ellipse_points(n: int) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, 8 * n, endpoint=False)
    dense = np.column_stack([1.5 * np.cos(theta), 0.7 * np.sin(theta)])
    return resample_closed_curve(dense, n)


def _smooth_star_points(n: int) -> np.ndarray:
    model = StarShapeModel(
        center=np.array([0.0, 0.0]),
        base_radius=1.0,
        cos=np.array([0.18, -0.05, 0.03]),
        sin=np.array([0.04, 0.06, -0.02]),
    )
    return BoundaryCurve(model.boundary_points(8 * n)).resample(n).points


def _piecewise_points(n: int) -> np.ndarray:
    return BoundaryCurve(piecewise_curved_boundary(240)).resample(n).normalized().points


def test_offcircle_gulati_matrices_conserve_constants_and_are_psd() -> None:
    for points in (_ellipse_points(64), _smooth_star_points(64), _piecewise_points(64)):
        gu = gulati_laplacian(points)
        eigenvalues = np.linalg.eigvalsh((gu + gu.T) / 2.0)
        assert np.linalg.norm(gu @ np.ones(len(points)), ord=np.inf) < 1e-10
        assert eigenvalues[0] > -1e-10
        assert eigenvalues[1] > 1e-3


def test_offcircle_cauchy_gram_factorization() -> None:
    points = _smooth_star_points(32)
    gu = gulati_laplacian(points)
    factor = gulati_incidence_factor(points)
    assert np.allclose(factor.T @ factor, gu, atol=1e-11)


def test_vertex_weights_and_normals_are_geometric() -> None:
    points = _ellipse_points(128)
    weights = vertex_arclength_weights(points)
    normals = outward_unit_normals(points)
    assert np.all(weights > 0.0)
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0)
    assert normals[0, 0] > 0.99


def test_offcircle_coercivity_has_pi_over_delta_scaling() -> None:
    for points in (_ellipse_points(4096), _smooth_star_points(4096), _piecewise_points(4096)):
        rows = near_boundary_gulati_coercivity_table(
            points,
            sample_index=0,
            deltas=(4e-2, 2e-2, 1e-2),
        )
        ratios = [row["delta_times_coercivity_over_pi"] for row in rows]
        assert abs(ratios[-1] - 1.0) < 0.04
        assert abs(ratios[-1] - 1.0) < abs(ratios[0] - 1.0)


def test_offcircle_weighted_coercivity_scales_with_local_density() -> None:
    points = _ellipse_points(4096)
    theta = 2.0 * np.pi * np.arange(len(points)) / len(points)
    density = 1.5 + 0.5 * np.cos(theta)
    normal = outward_unit_normals(points)[0]
    target = points[0] + 1e-2 * normal
    value = gulati_coercivity_at_point(points, target, density_abs=density)
    assert abs(value * 1e-2 / np.pi - density[0]) < 0.07


def test_offcircle_scaled_spectrum_has_one_constant_mode() -> None:
    eigenvalues = arclength_scaled_gulati_eigenvalues(_ellipse_points(128))
    assert abs(eigenvalues[0]) < 1e-10
    assert eigenvalues[1] > 0.0
    assert np.all(np.diff(eigenvalues) >= -1e-10)


def test_offcircle_low_modes_follow_principal_weyl_slope() -> None:
    for points in (_ellipse_points(256), _smooth_star_points(256), _piecewise_points(256)):
        rows = gulati_weyl_pair_ratios(points, mode_start=8, mode_stop=16)
        ratios = [row["ratio"] for row in rows]
        assert min(ratios) > 0.90
        assert max(ratios) < 1.03
