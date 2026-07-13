import ast

import inverse_shape.golden_hyperbolic_atlas as atlas_module
import inverse_shape.quadrature as quadrature_module
from inverse_shape.golden_hyperbolic import (
    GOLDEN_MU,
    GOLDEN_TRANSLATION_LENGTH,
)
from inverse_shape.golden_hyperbolic_atlas import GoldenHyperbolicJetAtlas
from inverse_shape.testing.reference_pairwise import (
    reference_axisymmetric_physical,
)


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


def test_golden_atlas_has_only_project_numerical_imports() -> None:
    with open(atlas_module.__file__, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert imported == [
        "inverse_shape.axisymmetric_scale_phase",
        "inverse_shape.golden_hyperbolic",
        "inverse_shape.quadrature",
    ]


def test_trace_three_length_partitions_a_long_geodesic_into_two_patches() -> None:
    count = 17
    coordinates = tuple(
        -2.0 * GOLDEN_MU + 4.0 * GOLDEN_MU * index / (count - 1)
        for index in range(count)
    )
    radius_jets = []
    height_jets = []
    for coordinate in coordinates:
        radius = quadrature_module._exp(coordinate)
        radius_jets.append((radius, radius, radius, radius))
        height_jets.append((0.0, 0.0, 0.0, 0.0))
    atlas = GoldenHyperbolicJetAtlas(
        coordinates,
        radius_jets,
        height_jets,
    )
    assert atlas.stats()["quadratic_fallback"] is False
    assert len(atlas.patches) == 2
    assert abs(
        atlas.stats()["total_hyperbolic_length"]
        - 2.0 * GOLDEN_TRANSLATION_LENGTH
    ) < 2.0e-12
    for patch in atlas.patches:
        assert abs(patch.hyperbolic_span - GOLDEN_TRANSLATION_LENGTH) < 2.0e-12
        assert abs(patch.coordinates[0] + 1.0) < 2.0e-12
        assert abs(patch.coordinates[-1] - 1.0) < 2.0e-12
        qjet = patch.qjet(16, 8)
        field = tuple(
            tuple(
                tau
                + 0.2
                * quadrature_module._cos(
                    quadrature_module.TAU * phase / 8
                )
                for phase in range(8)
            )
            for tau in qjet.coordinates
        )
        assert _relative_grid_error(
            qjet.apply(field),
            reference_axisymmetric_physical(qjet, field),
        ) < 8.0e-13
        assert qjet.constant_residual() < 2.0e-13


def test_general_polynomial_meridian_round_trips_through_golden_jets() -> None:
    count = 17
    coordinates = tuple(-0.8 + 1.6 * index / (count - 1) for index in range(count))
    radius_jets = tuple(
        (
            1.2 + 0.1 * coordinate * coordinate,
            0.2 * coordinate,
            0.2,
            0.0,
        )
        for coordinate in coordinates
    )
    height_jets = tuple(
        (
            coordinate + 0.05 * coordinate**3,
            1.0 + 0.15 * coordinate * coordinate,
            0.3 * coordinate,
            0.3,
        )
        for coordinate in coordinates
    )
    atlas = GoldenHyperbolicJetAtlas(
        coordinates,
        radius_jets,
        height_jets,
    )
    assert atlas.stats()["stored_dense_matrix"] is False
    assert atlas.stats()["quadratic_fallback"] is False
    for patch in atlas.patches:
        for local, source_index in enumerate(range(patch.first, patch.last)):
            radius, height = patch.spline(patch.coordinates[local])
            assert abs(radius - radius_jets[source_index][0]) < 3.0e-13
            assert abs(height - height_jets[source_index][0]) < 3.0e-13
        generated = patch.uniform_coordinates(24)
        weights = patch.generated_meridional_weights(generated)
        assert all(value > 0.0 for value in weights)
