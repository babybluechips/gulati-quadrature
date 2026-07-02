import math

import numpy as np

from inverse_shape.fem import build_fem_boundary_dtn, relative_weighted_l2
from inverse_shape.q_dtn import (
    apply_continuum_repaid_dtn,
    apply_cycle_dtn,
    build_boundary_pullback_qjet,
    build_harmonic_moment_corrected_planar_qjet,
    build_helmholtz_moment_corrected_planar_qjet,
    build_planar_domain_qjet,
    circle_qjet_map,
    continuum_repaid_dtn_heat,
    continuum_repaid_dtn_helmholtz_resolvent,
    continuum_repaid_dtn_poisson_solve,
    continuum_repaid_dtn_wave,
    cycle_dtn_heat,
    cycle_dtn_helmholtz_resolvent,
    cycle_dtn_poisson_solve,
    cycle_dtn_wave,
    ellipse_weighted_dtn,
    ellipse_qjet_map,
    exact_disk_amplitude,
    harmonic_polynomial_trace,
    harmonic_polynomial_weak_flux,
    helmholtz_plane_wave_trace,
    helmholtz_plane_wave_weak_flux,
    q_disk_amplitude,
    radial_fourier_qjet_map,
    relative_error,
)


def cosine_mode(n: int, mode: int) -> list[float]:
    return [math.cos(2.0 * math.pi * mode * index / n) for index in range(n)]


def projected_amplitude(values, mode: int) -> complex:
    n = len(values)
    basis = cosine_mode(n, mode)
    numerator = sum(complex(values[index]) * basis[index] for index in range(n))
    denominator = sum(value * value for value in basis)
    return numerator / denominator


def test_q_dtn_laplace_mode_matches_disk_dtn() -> None:
    n = 2048
    mode = 7
    output = apply_cycle_dtn(cosine_mode(n, mode))
    assert abs(projected_amplitude(output, mode) - mode) / mode < 4.0e-3


def test_q_dtn_boundary_pde_modal_amplitudes_match_exact_disk_model() -> None:
    n = 4096
    mode = 9
    parameters = {
        "heat": {"time": 0.17},
        "poisson": {"mass": 0.35},
        "helmholtz": {"wavenumber": 3.7, "damping": 0.02},
        "wave": {"time": 0.8},
    }
    for problem in ("laplace_dtn", "heat", "poisson", "helmholtz", "wave"):
        q_value = q_disk_amplitude(problem, mode, n, **parameters.get(problem, {}))
        exact = exact_disk_amplitude(problem, mode, **parameters.get(problem, {}))
        assert relative_error(q_value, exact) < 1.0e-14


def test_repaid_q_dtn_operator_applications_match_exact_modal_amplitudes() -> None:
    n = 1024
    mode = 5
    values = cosine_mode(n, mode)
    cases = [
        (
            apply_continuum_repaid_dtn(values),
            exact_disk_amplitude("laplace_dtn", mode),
        ),
        (
            continuum_repaid_dtn_heat(values, 0.13),
            exact_disk_amplitude("heat", mode, time=0.13),
        ),
        (
            continuum_repaid_dtn_poisson_solve(values, mass=0.25),
            exact_disk_amplitude("poisson", mode, mass=0.25),
        ),
        (
            continuum_repaid_dtn_helmholtz_resolvent(values, 3.7, damping=0.02),
            exact_disk_amplitude("helmholtz", mode, wavenumber=3.7, damping=0.02),
        ),
        (
            continuum_repaid_dtn_wave(values, 0.8),
            exact_disk_amplitude("wave", mode, time=0.8),
        ),
    ]
    for output, exact in cases:
        assert np.isfinite(abs(projected_amplitude(output, mode)))
        assert relative_error(projected_amplitude(output, mode), exact) < 1.0e-12


def test_q_dtn_operator_applications_match_modal_amplitudes() -> None:
    n = 1024
    mode = 5
    values = cosine_mode(n, mode)
    cases = [
        (
            cycle_dtn_heat(values, 0.13),
            exact_disk_amplitude("heat", mode, time=0.13),
        ),
        (
            cycle_dtn_poisson_solve(values, mass=0.25),
            exact_disk_amplitude("poisson", mode, mass=0.25),
        ),
        (
            cycle_dtn_helmholtz_resolvent(values, 3.7, damping=0.02),
            exact_disk_amplitude("helmholtz", mode, wavenumber=3.7, damping=0.02),
        ),
        (
            cycle_dtn_wave(values, 0.8),
            exact_disk_amplitude("wave", mode, time=0.8),
        ),
    ]
    for output, exact in cases:
        assert np.isfinite(abs(projected_amplitude(output, mode)))
        assert relative_error(projected_amplitude(output, mode), exact) < 3.0e-2


def test_autodiff_pullback_repay_scales_circle_dtn() -> None:
    n = 512
    radius = 2.5
    mode = 6
    qjet = build_boundary_pullback_qjet(n, circle_qjet_map(radius))
    output = qjet.apply_dtn(cosine_mode(n, mode))
    generated = (mode * (n - mode) / n) / radius

    assert output.ledger.status == "borrowed_repaid"
    assert "matrix" not in vars(qjet)
    assert relative_error(projected_amplitude(output.values, mode), generated) < 1.0e-12


def test_autodiff_qjet_samples_ellipse_metric() -> None:
    n = 128
    a = 2.0
    b = 0.5
    qjet = build_boundary_pullback_qjet(n, ellipse_qjet_map(a, b))
    for index, speed in enumerate(qjet.speeds):
        theta = 2.0 * math.pi * index / n
        exact_speed = math.hypot(a * math.sin(theta), b * math.cos(theta))
        assert abs(speed - exact_speed) < 1.0e-12


def test_planar_domain_qjet_works_on_funky_sampled_domain() -> None:
    n = 96
    boundary = build_boundary_pullback_qjet(
        n,
        radial_fourier_qjet_map(
            1.0,
            cos_coefficients=(0.0, 0.22, 0.0, -0.08),
            sin_coefficients=(0.0, 0.0, 0.05),
        ),
    )
    qjet = build_planar_domain_qjet(boundary.points)
    mode = 5
    output = qjet.apply_dtn(cosine_mode(n, mode))

    assert output.ledger.status == "borrowed_repaid"
    assert "matrix" not in vars(qjet.qjet)
    assert len(output.values) == n
    assert output.stats["recommended_q"] == "multipole_zeta_q"
    assert np.isfinite(np.linalg.norm(output.values))


def test_planar_domain_boundary_pdes_are_matrix_free_on_polygon() -> None:
    points = [
        (math.cos(2.0 * math.pi * index / 32) * (1.0 + 0.18 * math.cos(4.0 * 2.0 * math.pi * index / 32)),
         math.sin(2.0 * math.pi * index / 32) * (1.0 + 0.18 * math.cos(4.0 * 2.0 * math.pi * index / 32)))
        for index in range(32)
    ]
    qjet = build_planar_domain_qjet(points)
    values = cosine_mode(32, 3)
    cases = [
        qjet.solve_boundary_problem("heat", values, time=0.02, max_steps=16),
        qjet.solve_boundary_problem("poisson", values, mass=0.5, iterations=24),
        qjet.solve_boundary_problem("helmholtz", values, wavenumber=1.0, damping=0.2, iterations=8),
        qjet.solve_boundary_problem("wave", values, time=0.05, max_steps=16),
    ]

    for result in cases:
        assert result.ledger.status == "borrowed_repaid"
        assert len(result.values) == len(values)
        assert np.all(np.isfinite([complex(value).real for value in result.values]))


def test_harmonic_moment_corrected_qjet_reproduces_linear_flux() -> None:
    n = 64
    boundary = build_boundary_pullback_qjet(n, ellipse_qjet_map(2.0, 0.7))
    qjet = build_harmonic_moment_corrected_planar_qjet(
        boundary.points,
        moment_degree=2,
        zeta_tail_degree=5,
    )
    values = harmonic_polynomial_trace(boundary.points, 1, "cos")
    weak_flux = harmonic_polynomial_weak_flux(boundary.points, 1, "cos")
    mass = [float(value) for value in qjet.harmonic_correction.mass]
    exact = [weak_flux[index] / mass[index] for index in range(n)]
    result = qjet.apply_dtn(values)

    assert result.ledger.status == "borrowed_repaid"
    assert result.stats["harmonic_moment_degree"] == 2
    assert result.stats["zeta_tail_rank"] > 0
    assert "matrix" not in vars(qjet.qjet)
    assert relative_weighted_l2(result.values, exact, mass) < 1.0e-10


def test_harmonic_moment_corrected_qjet_reproduces_quadratic_flux() -> None:
    n = 64
    boundary = build_boundary_pullback_qjet(
        n,
        radial_fourier_qjet_map(
            1.0,
            cos_coefficients=(0.0, 0.18, 0.0, -0.05),
            sin_coefficients=(0.0, 0.0, 0.04),
        ),
    )
    qjet = build_harmonic_moment_corrected_planar_qjet(
        boundary.points,
        moment_degree=2,
        zeta_tail_degree=6,
    )
    values = harmonic_polynomial_trace(boundary.points, 2, "sin")
    weak_flux = harmonic_polynomial_weak_flux(boundary.points, 2, "sin")
    mass = [float(value) for value in qjet.harmonic_correction.mass]
    exact = [weak_flux[index] / mass[index] for index in range(n)]
    result = qjet.apply_dtn(values)

    assert result.stats["correction_rank"] >= 5
    assert relative_weighted_l2(result.values, exact, mass) < 1.0e-10


def test_exact_weighted_ellipse_dtn_reproduces_affine_flux() -> None:
    n = 96
    a = 3.0
    b = 1.0
    mass = [
        math.hypot(
            a * math.sin(2.0 * math.pi * index / n),
            b * math.cos(2.0 * math.pi * index / n),
        )
        * 2.0
        * math.pi
        / n
        for index in range(n)
    ]
    x_values = [a * math.cos(2.0 * math.pi * index / n) for index in range(n)]
    y_values = [b * math.sin(2.0 * math.pi * index / n) for index in range(n)]
    exact_x = []
    exact_y = []
    for index in range(n):
        theta = 2.0 * math.pi * index / n
        raw_normal = (math.cos(theta) / a, math.sin(theta) / b)
        normal_scale = math.hypot(raw_normal[0], raw_normal[1])
        exact_x.append(raw_normal[0] / normal_scale)
        exact_y.append(raw_normal[1] / normal_scale)

    assert relative_weighted_l2(ellipse_weighted_dtn(x_values, a, b), exact_x, mass) < 1.0e-12
    assert relative_weighted_l2(ellipse_weighted_dtn(y_values, a, b), exact_y, mass) < 1.0e-12


def test_helmholtz_moment_corrected_qjet_reproduces_plane_wave_flux() -> None:
    n = 72
    k = 2.5
    angle = 0.37
    boundary = build_boundary_pullback_qjet(
        n,
        radial_fourier_qjet_map(
            1.0,
            cos_coefficients=(0.0, 0.16, 0.0, -0.04),
            sin_coefficients=(0.0, 0.0, 0.05),
        ),
    )
    qjet = build_helmholtz_moment_corrected_planar_qjet(
        boundary.points,
        k,
        plane_wave_directions=(angle,),
    )
    values = helmholtz_plane_wave_trace(boundary.points, k, angle)
    weak_flux = helmholtz_plane_wave_weak_flux(boundary.points, k, angle)
    mass = [float(value) for value in qjet.helmholtz_correction.mass]
    exact = [weak_flux[index] / mass[index] for index in range(n)]
    result = qjet.apply_helmholtz_dtn(values)

    assert result.ledger.status == "borrowed_repaid"
    assert result.stats["helmholtz_correction_rank"] == 1
    assert "matrix" not in vars(qjet.qjet)
    assert relative_weighted_l2(result.values, exact, mass) < 1.0e-10


def test_star_fan_fem_dtn_matches_circle_low_mode() -> None:
    n = 48
    mode = 5
    points = [(math.cos(2.0 * math.pi * index / n), math.sin(2.0 * math.pi * index / n)) for index in range(n)]
    fem = build_fem_boundary_dtn(points, radial_levels=16)
    values = cosine_mode(n, mode)
    flux = fem.apply_dtn(values)

    assert fem.mesh.node_count == 769
    assert fem.mesh.triangle_count == 1488
    assert relative_error(projected_amplitude(flux, mode), mode) < 3.5e-2


def test_star_fan_fem_heat_semigroup_matches_disk_mode() -> None:
    n = 48
    mode = 3
    time = 0.04
    points = [(math.cos(2.0 * math.pi * index / n), math.sin(2.0 * math.pi * index / n)) for index in range(n)]
    values = cosine_mode(n, mode)
    fem = build_fem_boundary_dtn(points, radial_levels=16)
    heat = fem.solve_boundary_problem("heat", values, time=time)
    exact = [math.exp(-time * mode) * value for value in values]

    assert relative_weighted_l2(heat, exact, fem.boundary_mass) < 2.5e-2
