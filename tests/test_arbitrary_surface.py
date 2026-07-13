import ast

import inverse_shape.arbitrary_surface as surface_module
from inverse_shape.arbitrary_surface import (
    CertifiedArbitrarySurfaceQJet,
    GeneratedEuclideanDiscriminantQJet,
)
from inverse_shape.quadrature import PI, TAU, _cos, _sin
from inverse_shape.testing.reference_pairwise import reference_weighted_distance_graph


def _relative_error(reference, candidate):
    numerator = sum(
        abs(complex(left) - complex(right)) ** 2
        for left, right in zip(reference, candidate, strict=True)
    )
    denominator = sum(abs(complex(value)) ** 2 for value in reference)
    return (numerator / max(denominator, 1.0e-300)) ** 0.5


def _sphere(latitude_count=8, longitude_count=12):
    points = []
    weights = []
    for latitude in range(latitude_count):
        theta = PI * (latitude + 0.5) / latitude_count
        sine = _sin(theta)
        cosine = _cos(theta)
        for longitude in range(longitude_count):
            phase = TAU * longitude / longitude_count
            points.append(
                (sine * _cos(phase), sine * _sin(phase), cosine)
            )
            weights.append(
                sine * PI / latitude_count * TAU / longitude_count
            )
    return tuple(points), tuple(weights)


def _ellipsoid():
    points, weights = _sphere()
    return (
        tuple((1.8 * x, 0.7 * y, 1.2 * z) for x, y, z in points),
        weights,
    )


def _spherical_spiral(count=192):
    points = []
    for index in range(count):
        z = 1.0 - 2.0 * (index + 0.5) / count
        radius = max(1.0 - z * z, 0.0) ** 0.5
        phase = TAU * index / count
        points.append((radius * _cos(phase), radius * _sin(phase), z))
    return tuple(points), (4.0 * PI / count,) * count


def _torus(major_count=12, minor_count=8):
    points = []
    weights = []
    major_radius = 1.4
    minor_radius = 0.35
    for major in range(major_count):
        u = TAU * major / major_count
        for minor in range(minor_count):
            v = TAU * minor / minor_count
            radial = major_radius + minor_radius * _cos(v)
            points.append(
                (
                    radial * _cos(u),
                    radial * _sin(u),
                    minor_radius * _sin(v),
                )
            )
            weights.append(
                minor_radius
                * radial
                * TAU
                / major_count
                * TAU
                / minor_count
            )
    return tuple(points), tuple(weights)


def _folded_sheet(nx=12, ny=8):
    points = []
    weights = []
    for ix in range(nx):
        x = -1.0 + 2.0 * (ix + 0.5) / nx
        for iy in range(ny):
            y = -0.8 + 1.6 * (iy + 0.5) / ny
            z = 0.35 * _sin(1.7 * x) + 0.18 * _sin(2.3 * y + 0.4 * x)
            points.append((x, y, z))
            weights.append(3.2 / (nx * ny))
    return tuple(points), tuple(weights)


def _mobius(around=16, across=6):
    points = []
    weights = []
    half_width = 0.28
    for along in range(around):
        u = TAU * along / around
        for transverse in range(across):
            v = half_width * (2.0 * (transverse + 0.5) / across - 1.0)
            radial = 1.2 + v * _cos(0.5 * u)
            points.append(
                (
                    radial * _cos(u),
                    radial * _sin(u),
                    v * _sin(0.5 * u),
                )
            )
            weights.append(2.0 * half_width * TAU * 1.2 / (around * across))
    return tuple(points), tuple(weights)


def _star_surface():
    points, base_weights = _sphere()
    output = []
    for x, y, z in points:
        phase_radius = max(x * x + y * y, 0.0) ** 0.5
        modulation = 1.0 + 0.16 * (4.0 * z * z - 1.0) * phase_radius
        output.append((modulation * x, modulation * y, modulation * z))
    return tuple(output), base_weights


def _field(points):
    return tuple(
        x + 0.2 * y - 0.1 * z * z + 0.03j * (y + z)
        for x, y, z in points
    )


def _compiler(points, weights, kernel_power=2.0):
    return CertifiedArbitrarySurfaceQJet(
        points,
        weights,
        kernel_power=kernel_power,
        tolerance=2.0e-7,
        maximum_order=8,
        leaf_size=4,
    )


def test_arbitrary_surface_kernel_has_no_external_numerical_dependency() -> None:
    with open(surface_module.__file__, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert set(imported) == {
        "inverse_shape.quadrature",
        "inverse_shape.riesz_near_linear",
    }


def test_five_topologically_distinct_surfaces_match_exact_stream() -> None:
    for points, weights in (
        _sphere(),
        _ellipsoid(),
        _torus(),
        _folded_sheet(),
        _mobius(),
        _star_surface(),
    ):
        values = _field(points)
        qjet = _compiler(points, weights)
        candidate = qjet.apply(values)
        reference = reference_weighted_distance_graph(
            points,
            weights,
            values,
            2.0,
        )
        assert _relative_error(reference, candidate) < 3.0e-7
        assert qjet.constant_residual() == 0.0
        stats = qjet.stats()
        assert stats["pair_partition_residual"] == 0
        assert stats["temporary_pair_table_entries"] == 0
        assert stats["stored_dense_distance_matrix"] is False
        assert stats["stored_dense_operator_matrix"] is False
        assert stats["hard_no_quadratic_contract"] is True
        assert stats["quadratic_fallback"] is False


def test_generated_euclidean_log_discriminant_jet_matches_direct_q() -> None:
    points, weights = _mobius()
    values = _field(points)
    weight_laplacians = []
    field_laplacians = []
    for left, left_point in enumerate(points):
        weight_sum = 0.0 + 0.0j
        field_sum = 0.0 + 0.0j
        for right, right_point in enumerate(points):
            if left == right:
                continue
            distance_squared = sum(
                (left_point[axis] - right_point[axis]) ** 2
                for axis in range(3)
            )
            weight_sum += weights[right] / distance_squared
            field_sum += weights[right] * values[right] / distance_squared
        weight_laplacians.append(2.0 * weight_sum)
        field_laplacians.append(2.0 * field_sum)
    generated = GeneratedEuclideanDiscriminantQJet(len(points), dimension=3)
    candidate = generated.apply(
        values,
        weight_laplacians,
        field_laplacians,
    )
    reference = reference_weighted_distance_graph(
        points,
        weights,
        values,
        2.0,
    )
    assert _relative_error(reference, candidate) < 1.0e-14
    stats = generated.stats()
    assert stats["apply_complexity"] == "O(N) after peeled-jet generation"
    assert stats["generator_included"] is False
    assert stats["stored_dense_operator_matrix"] is False


def test_analytic_certificate_bounds_compression_error() -> None:
    points, weights = _spherical_spiral()
    values = _field(points)
    qjet = _compiler(points, weights)
    candidate = qjet.apply(values)
    reference = reference_weighted_distance_graph(
        points,
        weights,
        values,
        2.0,
    )
    actual_inf = max(
        abs(complex(left) - complex(right))
        for left, right in zip(reference, candidate, strict=True)
    )
    bound = qjet.compression_inf_bound(values)
    roundoff = 2.0e-14 * max(
        1.0,
        max(abs(complex(value)) for value in reference),
    )
    assert actual_inf <= bound + roundoff
    stats = qjet.stats()
    assert stats["analytic_blocks"] > 0
    assert stats["maximum_analytic_relative_tail"] <= 2.0e-7
    assert stats["adaptive_rank"] == 0
    assert stats["quadratic_fallback"] is False


def test_arbitrary_surface_graph_is_weighted_self_adjoint() -> None:
    points, weights = _torus()
    qjet = _compiler(points, weights)
    left = tuple(x - 0.1 * z for x, _y, z in points)
    right = tuple(y + 0.2 * z for _x, y, z in points)
    q_left = qjet.apply(left)
    q_right = qjet.apply(right)
    lhs = sum(
        weights[index] * left[index] * complex(q_right[index]).real
        for index in range(len(points))
    )
    rhs = sum(
        weights[index] * complex(q_left[index]).real * right[index]
        for index in range(len(points))
    )
    assert abs(lhs - rhs) / max(1.0, abs(lhs), abs(rhs)) < 2.0e-13


def test_surface_scale_covariance_for_inverse_square_and_cube() -> None:
    points, weights = _mobius()
    doubled_points = tuple(
        (2.0 * x, 2.0 * y, 2.0 * z) for x, y, z in points
    )
    doubled_weights = tuple(4.0 * value for value in weights)
    values = _field(points)
    for power, multiplier in ((2.0, 1.0), (3.0, 2.0)):
        unit = _compiler(points, weights, kernel_power=power).apply(values)
        doubled = _compiler(
            doubled_points,
            doubled_weights,
            kernel_power=power,
        ).apply(values)
        candidate = tuple(multiplier * complex(value) for value in doubled)
        assert _relative_error(unit, candidate) < 8.0e-7


def test_duplicate_surface_nodes_fail_closed() -> None:
    points, weights = _sphere(4, 6)
    duplicate = points + (points[0],)
    duplicate_weights = weights + (weights[0],)
    try:
        _compiler(duplicate, duplicate_weights)
    except ValueError as error:
        assert "coincide" in str(error)
    else:
        raise AssertionError("duplicate surface nodes must be rejected")


def test_triangle_mesh_entry_point_builds_lumped_boundary_weights() -> None:
    vertices = (
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
    )
    triangles = (
        (0, 2, 4),
        (2, 1, 4),
        (1, 3, 4),
        (3, 0, 4),
        (2, 0, 5),
        (1, 2, 5),
        (3, 1, 5),
        (0, 3, 5),
    )
    qjet = CertifiedArbitrarySurfaceQJet.from_triangle_mesh(
        vertices,
        triangles,
        tolerance=2.0e-13,
        maximum_order=12,
        leaf_size=2,
    )
    assert abs(sum(qjet.weights) - 4.0 * 3.0**0.5) < 2.0e-15
    values = tuple(x + 0.2 * y - 0.1 * z for x, y, z in vertices)
    reference = reference_weighted_distance_graph(
        vertices,
        qjet.weights,
        values,
        2.0,
    )
    assert _relative_error(reference, qjet.apply(values)) < 1.0e-13
