import ast
from pathlib import Path

import pytest

from gulati_quadrature import (
    GOLDEN_MULTIPLIER,
    CylindricalTransparentDtN,
    TransparentTailBranchError,
    golden_tail_certificate,
    pde_spectral_shift,
    residue_class_sectors,
)
from inverse_shape.quadrature import PI, TAU, _cos, _log, _sin, _sqrt


def _trace(size):
    return tuple(
        _cos(TAU * index / size)
        + 0.21 * _sin(3.0 * TAU * index / size)
        - 0.07 * _cos(11.0 * TAU * index / size)
        for index in range(size)
    )


def _relative_error(reference, candidate):
    numerator = sum(
        abs(complex(left) - complex(right)) ** 2
        for left, right in zip(reference, candidate, strict=True)
    )
    denominator = sum(abs(complex(value)) ** 2 for value in reference)
    return (numerator / max(denominator, 1.0e-300)) ** 0.5


def test_transparent_tail_foundation_has_no_external_numerical_dependency() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "inverse_shape"
        / "transparent_tail.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert imported == ["inverse_shape.quadrature"]
    assert "numpy" not in source
    assert "scipy" not in source


def test_exact_cylinder_generator_identity_and_normalization() -> None:
    cap = CylindricalTransparentDtN(128)
    assert cap.cylinder_identity_residual() < 2.0e-14
    mode = cap.modes[7]
    sine = abs(_sin(PI * 7.0 / 128.0))
    expected = 2.0 * _log(sine + _sqrt(1.0 + sine * sine))
    assert abs(mode.generator.real - expected) < 2.0e-15
    assert abs(mode.pivot * mode.self_energy - 1.0) < 2.0e-15
    assert abs(mode.flux - (1.0 - mode.root)) < 2.0e-15
    assert mode.fixed_point_residual < 2.0e-15


def test_cross_ratio_is_an_exact_global_linearization() -> None:
    cap = CylindricalTransparentDtN(64, spectral_shift=0.35)
    for mode_index in (0, 1, 7, 19, 32):
        previous_error = None
        for depth in (1, 2, 5, 12, 30):
            certificate = cap.cross_ratio_certificate(mode_index, depth)
            assert certificate["linearization_residual"] < 2.0e-14
            assert certificate["pivot_reconstruction_residual"] < 2.0e-14
            assert certificate["pivot_error"] <= (
                certificate["pivot_error_bound"] + 3.0e-15
            )
            if previous_error is not None:
                assert certificate["pivot_error"] <= previous_error + 3.0e-15
            previous_error = certificate["pivot_error"]


def test_direct_shell_solve_matches_finite_riccati_symbol() -> None:
    cap = CylindricalTransparentDtN(64, spectral_shift=0.2)
    trace = _trace(64)
    for depth in (1, 3, 9, 31):
        direct = cap.solve_direct_dirichlet_shells(trace, depth)
        compiled = cap.apply_finite_dirichlet(trace, depth, quantity="flux")
        symbols = cap.finite_dirichlet_symbols(depth, quantity="flux")
        reused = cap.apply_mode_symbols(trace, symbols)
        assert _relative_error(direct, compiled) < 2.0e-14
        assert _relative_error(compiled, reused) < 2.0e-15


def test_fixed_point_cap_removes_tail_depth_error() -> None:
    cap = CylindricalTransparentDtN(64)
    trace = _trace(64)
    exact = cap.apply_boundary_dtn(trace)
    errors = []
    for depth in (2, 4, 8, 16, 32, 64):
        finite = cap.apply_finite_dirichlet(trace, depth, quantity="flux")
        errors.append(_relative_error(exact.values, finite))
    assert all(
        right < left
        for left, right in zip(errors[:-1], errors[1:], strict=True)
    )
    assert errors[-1] < 5.0e-6
    assert exact.ledger.status == "balanced"
    assert exact.stats["tail_depth_dependence"] == "none"
    assert exact.stats["dense_shell_matrix_stored"] is False


def test_laplace_zero_mode_has_exact_parabolic_finite_depth_law() -> None:
    cap = CylindricalTransparentDtN(32)
    assert cap.modes[0].marginal is True
    for depth in (1, 2, 7, 41):
        certificate = cap.cross_ratio_certificate(0, depth)
        assert certificate["linearization_residual"] < 3.0e-15
        assert abs(certificate["pivot_error"] - 1.0 / depth) < 3.0e-15
        assert abs(certificate["flux_error"] - 1.0 / (depth + 1.0)) < 3.0e-15
    assert cap.required_depth(0, 1.0e-3) == 999


def test_golden_fibonacci_checksum_and_multiplier() -> None:
    certificate = golden_tail_certificate(20)
    assert certificate["numerator"] == 267914296
    assert certificate["denominator"] == 102334155
    assert certificate["pivot_rational_residual"] < 5.0e-16
    assert certificate["error_law_residual"] < 5.0e-16
    assert abs(certificate["koenigs_multiplier"] - GOLDEN_MULTIPLIER) < 1.0e-16
    fourth = golden_tail_certificate(4)
    assert (fourth["numerator"], fourth["denominator"]) == (55, 21)


def test_pde_resolvent_factories_cover_supported_equations() -> None:
    assert pde_spectral_shift("laplace") == 0.0
    assert pde_spectral_shift("poisson") == 0.0
    assert pde_spectral_shift("screened_poisson", 0.5) == 0.25
    assert pde_spectral_shift("heat", 0.3) == 0.3
    assert abs(
        pde_spectral_shift("helmholtz", 0.7, damping=0.2)
        - complex(-0.49, 0.2)
    ) < 1.0e-15
    assert pde_spectral_shift("wave", 0.7, damping=0.2) == complex(0.2, 0.7) ** 2
    with pytest.raises(TransparentTailBranchError):
        pde_spectral_shift("helmholtz", 0.7)
    with pytest.raises(TransparentTailBranchError):
        CylindricalTransparentDtN(64, spectral_shift=-1.0)


@pytest.mark.parametrize(
    ("problem", "parameter", "damping"),
    (
        ("laplace", 0.0, 0.0),
        ("poisson", 0.0, 0.0),
        ("screened_poisson", 0.6, 0.0),
        ("heat_resolvent", 0.4, 0.0),
        ("helmholtz", 0.7, 0.15),
        ("wave_resolvent", 0.7, 0.2),
    ),
)
def test_all_pde_caps_apply_with_linear_storage(problem, parameter, damping) -> None:
    cap = CylindricalTransparentDtN.for_problem(
        64,
        problem,
        parameter=parameter,
        damping=damping,
    )
    result = cap.apply_boundary_dtn(_trace(64))
    assert len(result.values) == 64
    assert result.ledger.residual_norm < 3.0e-15
    assert result.stats["auxiliary_storage_big_o"] == "O(N_theta)"
    assert result.stats["apply_time_big_o"] == "O(N_theta log N_theta)"


def test_residue_class_sector_partition_is_exact() -> None:
    for bandwidth in (3, 5):
        sectors = residue_class_sectors(64, bandwidth)
        flattened = [index for sector in sectors for index in sector]
        assert sorted(flattened) == list(range(64))
        assert all(
            all(
                (index if index <= 32 else index - 64) % bandwidth == residue
                for index in sector
            )
            for residue, sector in enumerate(sectors)
        )


def test_perturbed_transition_has_a_certified_telescope_bound() -> None:
    cap = CylindricalTransparentDtN(64, spectral_shift=0.4)
    diagonal = tuple(0.02 * 0.5**shell for shell in range(12))
    coupling = tuple(-0.004 * 0.5**shell for shell in range(12))
    certificate = cap.perturbed_transition_certificate(
        3,
        diagonal,
        coupling,
    )
    assert certificate["terminal_cap_exact"] is True
    assert certificate["transition_shells"] == 12
    assert certificate["actual_pivot_error"] <= (
        certificate["certified_pivot_error_bound"] + 2.0e-15
    )
    assert certificate["maximum_local_lipschitz"] < 1.0
    assert len(certificate["local_rows"]) == 12

    smaller = cap.perturbed_transition_certificate(
        3,
        tuple(0.1 * value for value in diagonal),
        tuple(0.1 * value for value in coupling),
    )
    assert smaller["certified_pivot_error_bound"] < certificate[
        "certified_pivot_error_bound"
    ]
