import numpy as np

from inverse_shape.geometry import BoundaryCurve, StarShapeModel
from inverse_shape.operators import (
    apply_pressure_hessian_from_gulati,
    directed_inverse_square_moments,
    dressed_gulati_hessian,
    extract_flux_from_gulati_hessian,
    extract_flux_from_hessian,
    gulati_laplacian,
    gulati_weight_adjacency,
    pairwise_squared_distances,
    pressure_gulati_energy_factor,
    pressure_hessian_from_gulati,
)
from inverse_shape.reconstruction import reconstruct_polygon_from_gulati


def test_gulati_laplacian_has_constant_kernel() -> None:
    pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    gu = gulati_laplacian(pts)
    assert np.allclose(gu @ np.ones(4), 0.0)
    assert np.all(gu[np.triu_indices(4, 1)] < 0.0)


def test_gulati_reconstruction_preserves_distances() -> None:
    theta = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    pts = np.column_stack([np.cos(theta), 0.7 * np.sin(theta)])
    gu = gulati_laplacian(pts)
    result = reconstruct_polygon_from_gulati(gu)
    original_d2 = pairwise_squared_distances(pts)
    recon_d2 = pairwise_squared_distances(result.points)
    assert result.residual_norm < 1e-10
    assert np.allclose(original_d2, recon_d2, atol=1e-10)


def test_directed_moments_are_translation_invariant() -> None:
    pts = np.array([[0, 0], [1.2, 0], [1.0, 0.8], [-0.1, 0.7]], dtype=float)
    shifted = pts + np.array([3.0, -4.0])
    assert np.allclose(
        directed_inverse_square_moments(pts),
        directed_inverse_square_moments(shifted),
    )


def test_flux_extraction_from_dressed_hessian() -> None:
    model = StarShapeModel(
        center=np.array([0.0, 0.0]),
        base_radius=1.0,
        cos=np.array([0.05, -0.02]),
        sin=np.array([0.03, 0.01]),
    )
    curve = BoundaryCurve(model.boundary_points(160)).normalized()
    theta = np.linspace(0.0, 2.0 * np.pi, curve.n, endpoint=False)
    flux = 1.0 + 0.2 * np.cos(2.0 * theta) + 0.05 * np.sin(3.0 * theta)
    h_res = dressed_gulati_hessian(curve.points, flux)
    recovered = extract_flux_from_hessian(curve.points, h_res, neighbor_window=5)
    rel = np.linalg.norm(recovered - flux) / np.linalg.norm(flux)
    assert rel < 1e-10


def test_pressure_hessian_is_gulati_diagonal_dressing() -> None:
    model = StarShapeModel(
        center=np.array([0.0, 0.0]),
        base_radius=1.0,
        cos=np.array([0.05, -0.02]),
        sin=np.array([0.03, 0.01]),
    )
    curve = BoundaryCurve(model.boundary_points(48)).normalized()
    theta = np.linspace(0.0, 2.0 * np.pi, curve.n, endpoint=False)
    pressure = 1.0 + 0.2 * np.cos(2.0 * theta) + 0.05 * np.sin(3.0 * theta)
    values = np.cos(theta) - 0.3 * np.sin(4.0 * theta)

    gu = gulati_laplacian(curve.points)
    hessian = dressed_gulati_hessian(curve.points, pressure)
    from_gu = pressure_hessian_from_gulati(gu, pressure)
    applied = apply_pressure_hessian_from_gulati(gu, pressure, values)
    weights = gulati_weight_adjacency(gu)

    assert np.allclose(from_gu, hessian)
    assert np.allclose(applied, hessian @ values)
    expected = (2.0 / np.pi) * pressure[:, None] * weights * pressure[None, :]
    assert np.allclose(hessian, expected)


def test_pressure_gulati_factor_encodes_conservative_hessian() -> None:
    model = StarShapeModel(
        center=np.array([0.0, 0.0]),
        base_radius=1.0,
        cos=np.array([0.04, -0.03]),
        sin=np.array([0.02, 0.01]),
    )
    curve = BoundaryCurve(model.boundary_points(32)).normalized()
    theta = np.linspace(0.0, 2.0 * np.pi, curve.n, endpoint=False)
    pressure = 1.0 + 0.1 * np.cos(2.0 * theta)

    gu = gulati_laplacian(curve.points)
    hessian = pressure_hessian_from_gulati(gu, pressure)
    factor = pressure_gulati_energy_factor(curve.points, pressure)
    conservative = factor.T @ factor
    offdiag = ~np.eye(curve.n, dtype=bool)

    assert np.allclose(conservative[offdiag], -hessian[offdiag], atol=1e-12)
    assert np.all(np.linalg.eigvalsh(conservative) > -1e-10)


def test_flux_extraction_can_use_gulati_matrix_directly() -> None:
    model = StarShapeModel(
        center=np.array([0.0, 0.0]),
        base_radius=1.0,
        cos=np.array([0.03, -0.02]),
        sin=np.array([0.02, 0.04]),
    )
    curve = BoundaryCurve(model.boundary_points(96)).normalized()
    theta = np.linspace(0.0, 2.0 * np.pi, curve.n, endpoint=False)
    flux = 1.0 + 0.14 * np.cos(2.0 * theta) + 0.06 * np.sin(5.0 * theta)
    gu = gulati_laplacian(curve.points)
    hessian = pressure_hessian_from_gulati(gu, flux)

    recovered = extract_flux_from_gulati_hessian(gu, hessian, neighbor_window=5)
    assert np.linalg.norm(recovered - flux) / np.linalg.norm(flux) < 1e-10
