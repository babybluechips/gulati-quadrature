import ast

import inverse_shape.scale_phase_cauchy as cauchy_module
import inverse_shape.quadrature as quadrature_module
from inverse_shape.scale_phase_cauchy import (
    ScalePhaseCauchyQJet,
    StaticTriangularCauchyPlan,
)
from inverse_shape.testing.reference_pairwise import (
    reference_scale_phase_mode,
    reference_scale_phase_spectral,
)


def _nonuniform_rhos(count):
    return tuple(
        -1.2
        + 2.4 * index / (count - 1)
        + 0.035
        * quadrature_module._sin(
            2.0 * cauchy_module.PI * index / (count - 1)
        )
        for index in range(count)
    )


def _relative_vector_error(left, right):
    numerator = max(
        abs(complex(value) - complex(reference))
        for value, reference in zip(left, right, strict=True)
    )
    denominator = max(1.0, *(abs(complex(value)) for value in right))
    return numerator / denominator


def _relative_grid_error(left, right):
    numerator = max(
        abs(complex(value) - complex(reference))
        for left_row, right_row in zip(left, right, strict=True)
        for value, reference in zip(left_row, right_row, strict=True)
    )
    denominator = max(
        1.0,
        *(
            abs(complex(value))
            for reference_row in right
            for value in reference_row
        ),
    )
    return numerator / denominator


def _field(rhos, n_theta):
    rows = []
    for scale, rho in enumerate(rhos):
        row = []
        for phase in range(n_theta):
            theta = cauchy_module.TAU * phase / n_theta
            row.append(
                0.4
                + 0.3 * rho
                + 0.2 * quadrature_module._cos(3.0 * theta)
                - 0.17 * quadrature_module._sin(5.0 * theta)
                + 0.03j * (scale + 1) * quadrature_module._cos(theta)
            )
        rows.append(tuple(row))
    return tuple(rows)


def test_scale_phase_cauchy_kernel_has_no_external_numerical_import() -> None:
    with open(cauchy_module.__file__, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert imported == ["inverse_shape.quadrature"]


def test_foundational_exponential_handles_subnormal_decay() -> None:
    value = quadrature_module._exp(-744.0)
    assert value > 0.0
    assert value < 1.0e-322


def test_static_plan_is_an_exact_pair_partition_with_bounded_blocks() -> None:
    rhos = _nonuniform_rhos(128)
    nodes = tuple(cauchy_module._exp(2.0 * rho) for rho in rhos)
    plan = StaticTriangularCauchyPlan(nodes)
    stats = plan.stats()
    assert plan.pair_partition_residual == 0
    assert stats["compressed_blocks"] > 0
    assert stats["compressed_pair_fraction"] > 0.5
    assert stats["stored_dense_matrix"] is False
    assert stats["stored_factor_entries"] == 0
    assert stats["stored_interaction_matrices"] == stats["compressed_blocks"]
    assert stats["stored_static_transform_entries"] < 10000 * len(nodes)
    assert stats["storage_complexity"] == "O(N) with fixed p"
    assert stats["compiled_block_count"] <= stats["static_block_budget"]
    assert stats["compiled_exact_pairs"] <= stats["exact_pair_budget"]
    assert stats["compile_pair_visits"] <= stats["compile_visit_budget"]
    assert stats["quadratic_fallback"] is False
    assert stats["reference_oracle_in_production_object"] is False
    assert not hasattr(plan, "direct_apply_exponential_mode")
    for block in plan.compressed_blocks:
        relative = plan.block_max_relative_error(block)
        assert relative <= max(8.0e-14, 1.05 * block.tail_bound)


def test_static_nonuniform_cauchy_modes_match_direct_reference() -> None:
    rhos = _nonuniform_rhos(96)
    nodes = tuple(cauchy_module._exp(2.0 * rho) for rho in rhos)
    values = tuple(
        quadrature_module._cos(0.31 * index)
        + 0.2j * quadrature_module._sin(0.17 * index)
        for index in range(len(nodes))
    )
    plan = StaticTriangularCauchyPlan(nodes)
    for mode in (0, 1, 4, 11):
        fast = plan.apply_exponential_mode(values, rhos, mode)
        direct = reference_scale_phase_mode(plan, values, rhos, mode)
        assert _relative_vector_error(fast, direct) < 2.5e-13


def test_nested_rho_basis_survives_extreme_scale_and_mode_ranges() -> None:
    count = 128
    rhos = tuple(-8.0 + 16.0 * index / (count - 1) for index in range(count))
    nodes = tuple(cauchy_module._exp(2.0 * rho) for rho in rhos)
    values = tuple(
        complex(1.0 / (index + 1), 0.1 if index % 2 else -0.1)
        for index in range(count)
    )
    plan = StaticTriangularCauchyPlan(nodes)
    assert plan.stats()["compressed_pair_fraction"] > 0.7
    for mode in (0, 1, 8, 64, 128):
        fast = plan.apply_exponential_mode(values, rhos, mode)
        direct = reference_scale_phase_mode(plan, values, rhos, mode)
        assert _relative_vector_error(fast, direct) < 3.0e-13


def test_full_scale_phase_qjet_matches_independent_spectral_sum() -> None:
    rhos = _nonuniform_rhos(48)
    n_theta = 16
    weights = tuple(
        cauchy_module._exp(2.0 * rho)
        * (1.0 + 0.07 * quadrature_module._cos(0.4 * index))
        for index, rho in enumerate(rhos)
    )
    qjet = ScalePhaseCauchyQJet(rhos, n_theta, weights)
    values = _field(rhos, n_theta)
    assert _relative_grid_error(
        qjet.apply(values),
        reference_scale_phase_spectral(qjet, values),
    ) < 3.0e-13
    assert qjet.constant_residual() < 3.0e-13
    stats = qjet.stats()
    assert stats["stored_dense_distance_matrix"] is False
    assert stats["stored_dense_operator_matrix"] is False
    assert stats["quadratic_fallback"] is False
    assert not hasattr(qjet, "direct_spectral_apply")


def test_scale_phase_operator_is_invariant_under_uniform_dilation() -> None:
    rhos = _nonuniform_rhos(40)
    n_theta = 16
    weights = tuple(
        cauchy_module._exp(2.0 * rho) * (1.0 + 0.01 * index)
        for index, rho in enumerate(rhos)
    )
    dilation = 3.7
    shift = quadrature_module._log(dilation)
    shifted_rhos = tuple(rho + shift for rho in rhos)
    shifted_weights = tuple(dilation * dilation * value for value in weights)
    base = ScalePhaseCauchyQJet(rhos, n_theta, weights)
    shifted = ScalePhaseCauchyQJet(
        shifted_rhos,
        n_theta,
        shifted_weights,
    )
    values = _field(rhos, n_theta)
    assert _relative_grid_error(
        base.apply(values),
        shifted.apply(values),
    ) < 4.0e-13
