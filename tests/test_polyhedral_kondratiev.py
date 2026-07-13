import pytest

from inverse_shape.joukowski_endpoint import hurwitz_zeta_euler_maclaurin
from inverse_shape.polyhedral_kondratiev import (
    CertifiedPolyhedralSurfaceQJet,
    EdgeMellinPencil,
    MellinKondratievRepayment,
    MellinThreeJetChannel,
    PolyhedralMeshTopology,
    SparseSphericalDirichletPencil,
    VertexMellinPencil,
    mellin_midpoint_defect,
)
from inverse_shape.quadrature import PI


def _voxel_surface(cells):
    cells = frozenset(tuple(cell) for cell in cells)
    vertices = []
    lookup = {}
    triangles = []

    def vertex(point):
        if point not in lookup:
            lookup[point] = len(vertices)
            vertices.append(tuple(float(value) for value in point))
        return lookup[point]

    directions = (
        ((-1, 0, 0), lambda x, y, z: ((x, y, z), (x, y, z + 1), (x, y + 1, z + 1), (x, y + 1, z))),
        (
            (1, 0, 0),
            lambda x, y, z: (
                (x + 1, y, z),
                (x + 1, y + 1, z),
                (x + 1, y + 1, z + 1),
                (x + 1, y, z + 1),
            ),
        ),
        ((0, -1, 0), lambda x, y, z: ((x, y, z), (x + 1, y, z), (x + 1, y, z + 1), (x, y, z + 1))),
        (
            (0, 1, 0),
            lambda x, y, z: (
                (x, y + 1, z),
                (x, y + 1, z + 1),
                (x + 1, y + 1, z + 1),
                (x + 1, y + 1, z),
            ),
        ),
        ((0, 0, -1), lambda x, y, z: ((x, y, z), (x, y + 1, z), (x + 1, y + 1, z), (x + 1, y, z))),
        (
            (0, 0, 1),
            lambda x, y, z: (
                (x, y, z + 1),
                (x + 1, y, z + 1),
                (x + 1, y + 1, z + 1),
                (x, y + 1, z + 1),
            ),
        ),
    )
    for x, y, z in sorted(cells):
        for direction, quad_builder in directions:
            neighbor = (x + direction[0], y + direction[1], z + direction[2])
            if neighbor in cells:
                continue
            quad = tuple(vertex(point) for point in quad_builder(x, y, z))
            triangles.append((quad[0], quad[1], quad[2]))
            triangles.append((quad[0], quad[2], quad[3]))
    return tuple(vertices), tuple(triangles)


def _cube_surface():
    return _voxel_surface(((0, 0, 0),))


def _fichera_surface():
    return _voxel_surface(
        (x, y, z)
        for x in (-1, 0)
        for y in (-1, 0)
        for z in (-1, 0)
        if (x, y, z) != (0, 0, 0)
    )


def _spherical_octant_union(refinement, signs):
    n = int(refinement)
    points = []
    point_lookup = {}
    triangles = []

    def global_vertex(sign, i, j):
        k = n - i - j
        key = (
            0 if i == 0 else sign[0] * i,
            0 if j == 0 else sign[1] * j,
            0 if k == 0 else sign[2] * k,
        )
        if key not in point_lookup:
            norm = sum(value * value for value in key) ** 0.5
            point_lookup[key] = len(points)
            points.append(tuple(value / norm for value in key))
        return point_lookup[key]

    for sign in signs:
        local = {
            (i, j): global_vertex(sign, i, j)
            for i in range(n + 1)
            for j in range(n + 1 - i)
        }
        for i in range(n):
            for j in range(n - i):
                triangles.append(
                    (local[(i, j)], local[(i + 1, j)], local[(i, j + 1)])
                )
                if i + j <= n - 2:
                    triangles.append(
                        (
                            local[(i + 1, j)],
                            local[(i + 1, j + 1)],
                            local[(i, j + 1)],
                        )
                    )
    keys = {index: key for key, index in point_lookup.items()}
    return tuple(points), tuple(triangles), keys


def _octant_link(refinement):
    points, triangles, keys = _spherical_octant_union(
        refinement,
        ((1, 1, 1),),
    )
    boundary = tuple(
        index for index, key in keys.items() if any(value == 0 for value in key)
    )
    return SparseSphericalDirichletPencil(points, triangles, boundary)


def _fichera_link(refinement):
    signs = tuple(
        (sx, sy, sz)
        for sx in (-1, 1)
        for sy in (-1, 1)
        for sz in (-1, 1)
        if (sx, sy, sz) != (1, 1, 1)
    )
    points, triangles, keys = _spherical_octant_union(refinement, signs)
    boundary = tuple(
        index
        for index, key in keys.items()
        if all(value >= 0 for value in key) and any(value == 0 for value in key)
    )
    return SparseSphericalDirichletPencil(points, triangles, boundary)


def _midpoint(function, count):
    step = 1.0 / count
    return step * sum(function((index + 0.5) * step) for index in range(count))


def _beta_power_integral(exponent, order):
    value = 1.0
    for offset in range(order + 1):
        value *= exponent + offset
    factorial = 1
    for value_index in range(2, order + 1):
        factorial *= value_index
    return factorial / value


def test_polyhedral_topology_recovers_cube_and_fichera_openings() -> None:
    cube = PolyhedralMeshTopology(*_cube_surface())
    assert cube.closed is True
    assert abs(cube.signed_volume - 1.0) < 1.0e-14
    assert len(cube.edges) == 18  # twelve physical edges plus six face diagonals
    cube_physical = tuple(
        edge for edge in cube.edges if abs(edge.opening_angle - PI) > 1.0e-12
    )
    assert len(cube_physical) == 12
    assert max(abs(edge.opening_angle - 0.5 * PI) for edge in cube_physical) < 1.0e-14
    assert max(
        abs(EdgeMellinPencil.from_edge(edge).exponent - 2.0)
        for edge in cube_physical
    ) < 1.0e-14

    fichera = PolyhedralMeshTopology(*_fichera_surface())
    assert fichera.closed is True
    assert abs(fichera.signed_volume - 7.0) < 1.0e-13
    reentrant = tuple(edge for edge in fichera.edges if edge.is_reentrant)
    assert len(reentrant) == 3
    assert max(abs(edge.opening_angle - 1.5 * PI) for edge in reentrant) < 1.0e-14
    assert max(
        abs(EdgeMellinPencil.from_edge(edge).exponent - 2.0 / 3.0)
        for edge in reentrant
    ) < 2.0e-15


def test_topology_normalizes_components_and_rejects_ambiguous_open_wedges() -> None:
    points, faces = _voxel_surface(((0, 0, 0), (3, 0, 0)))
    mixed_faces = tuple(
        (face[0], face[2], face[1])
        if min(points[index][0] for index in face) >= 3.0
        else face
        for face in faces
    )
    topology = PolyhedralMeshTopology(points, mixed_faces)
    assert topology.connected_components == 2
    assert topology.orientation_reversed_components == 1
    assert abs(topology.signed_volume - 2.0) < 2.0e-14

    open_points = (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    open_topology = PolyhedralMeshTopology(
        open_points,
        ((0, 1, 2), (1, 0, 3)),
    )
    shared = next(
        edge for edge in open_topology.edges if len(edge.incident_faces) == 2
    )
    with pytest.raises(ValueError, match="closed mesh"):
        EdgeMellinPencil.from_edge(shared)


def test_sparse_spherical_octant_pencil_converges_to_lambda_three() -> None:
    coarse = VertexMellinPencil.from_spherical_link(
        _octant_link(8),
        tolerance=1.0e-9,
    )
    fine = VertexMellinPencil.from_spherical_link(
        _octant_link(16),
        tolerance=1.0e-9,
    )
    assert abs(fine.exponent - 3.0) < abs(coarse.exponent - 3.0)
    assert abs(fine.exponent - 3.0) < 0.035
    assert fine.spherical_link_eigenpair.residual < 2.0e-8


def test_sparse_fichera_vertex_pencil_approaches_published_exponent() -> None:
    reference = 0.45417371533061
    coarse = VertexMellinPencil.from_spherical_link(
        _fichera_link(4),
        tolerance=2.0e-9,
    )
    fine = VertexMellinPencil.from_spherical_link(
        _fichera_link(8),
        tolerance=2.0e-9,
    )
    assert abs(fine.exponent - reference) < abs(coarse.exponent - reference)
    assert abs(fine.exponent - reference) < 0.035
    assert fine.spherical_link_eigenpair.residual < 3.0e-8


def test_mellin_defect_value_and_log_derivative() -> None:
    exponent = 0.73
    step = 1.0 / 64.0
    value, estimate = mellin_midpoint_defect(exponent, step)
    zeta, _ = hurwitz_zeta_euler_maclaurin(1.0 - exponent, 0.5, 40, 7)
    reference = step**exponent * complex(zeta).real
    assert abs(value - reference) < 2.0e-14
    assert estimate < 1.0e-13

    derivative, _ = mellin_midpoint_defect(exponent, step, log_order=1)
    epsilon = 2.0e-5
    plus, _ = mellin_midpoint_defect(exponent + epsilon, step)
    minus, _ = mellin_midpoint_defect(exponent - epsilon, step)
    finite_difference = (plus - minus) / (2.0 * epsilon)
    assert abs(derivative - finite_difference) / max(1.0, abs(derivative)) < 2.0e-8

    second, _ = mellin_midpoint_defect(exponent, step, log_order=2)
    plus_first, _ = mellin_midpoint_defect(
        exponent + epsilon,
        step,
        log_order=1,
    )
    minus_first, _ = mellin_midpoint_defect(
        exponent - epsilon,
        step,
        log_order=1,
    )
    second_difference = (plus_first - minus_first) / (2.0 * epsilon)
    assert abs(second - second_difference) / max(1.0, abs(second)) < 3.0e-8

    third, _ = mellin_midpoint_defect(exponent, step, log_order=3)
    plus_second, _ = mellin_midpoint_defect(
        exponent + epsilon,
        step,
        log_order=2,
    )
    minus_second, _ = mellin_midpoint_defect(
        exponent - epsilon,
        step,
        log_order=2,
    )
    third_difference = (plus_second - minus_second) / (2.0 * epsilon)
    assert abs(third - third_difference) / max(1.0, abs(third)) < 5.0e-8


def test_edge_three_jet_restores_corner_convergence_without_refinement() -> None:
    exponent = 2.0 / 3.0
    coefficients = (1.0, -8.0, 28.0, -56.0)
    channel = MellinThreeJetChannel(
        EdgeMellinPencil(1.5 * PI),
        coefficients,
        label="reentrant_edge",
    )
    repayment = MellinKondratievRepayment((channel,))
    exact = _beta_power_integral(exponent, 8)
    rows = []
    for count in (16, 32, 64, 128, 256):
        raw = _midpoint(
            lambda radius: radius ** (exponent - 1.0) * (1.0 - radius) ** 8,
            count,
        )
        corrected = repayment.repay(raw, 1.0 / count).value
        rows.append((count, abs(raw - exact), abs(corrected - exact)))
    corrected_order = _log2(rows[-2][2] / rows[-1][2])
    raw_order = _log2(rows[-2][1] / rows[-1][1])
    assert 0.55 < raw_order < 0.8
    assert corrected_order > 4.2
    assert rows[-1][2] < 2.0e-12
    assert repayment.evaluate(1.0 / 64.0)["grid_refinement_iterations"] == 0
    assert repayment.evaluate(1.0 / 64.0)["stored_dense_matrix"] is False


def test_vertex_measure_shift_is_applied_exactly() -> None:
    exponent = 0.5
    pencil = VertexMellinPencil.from_exponent(exponent)
    assert abs(pencil.boundary_quadrature_exponent - 1.5) < 1.0e-15
    channel = MellinThreeJetChannel(pencil, (1.0, -8.0, 28.0, -56.0))
    exact = _beta_power_integral(exponent + 1.0, 8)
    errors = []
    for count in (16, 32, 64, 128, 256):
        raw = _midpoint(
            lambda radius: radius**exponent * (1.0 - radius) ** 8,
            count,
        )
        corrected = MellinKondratievRepayment((channel,)).repay(
            raw,
            1.0 / count,
        ).value
        errors.append(abs(corrected - exact))
    assert _log2(errors[-2] / errors[-1]) > 5.0
    assert errors[-1] < 3.0e-14


def test_polyhedral_wrapper_retains_topology_and_corner_ledger() -> None:
    vertices, triangles = _cube_surface()
    topology = PolyhedralMeshTopology(vertices, triangles)
    physical_edge = next(edge for edge in topology.edges if edge.is_geometric)
    channel = MellinThreeJetChannel(
        EdgeMellinPencil.from_edge(physical_edge),
        (1.0, -8.0, 28.0, -56.0),
        label="cube_edge",
    )
    surface = CertifiedPolyhedralSurfaceQJet(
        vertices,
        triangles,
        corner_channels=(channel,),
        tolerance=2.0e-13,
        maximum_order=12,
        leaf_size=4,
    )
    stats = surface.stats()
    assert stats["topology_preserved"] is True
    assert stats["geometric_edges"] == 12
    assert stats["coplanar_triangulation_seams"] == 6
    assert stats["corner_channel_count"] == 1
    assert stats["stored_dense_corner_matrix"] is False
    evaluation = surface.repay_corner_integral(0.0, 1.0 / 32.0)
    assert evaluation.stats["grid_refinement_iterations"] == 0
    assert evaluation.stats["stored_dense_matrix"] is False


def _log2(value):
    # The test targets a ratio near a small integer power and needs no stdlib math.
    from inverse_shape.quadrature import LN2, _log

    return _log(value) / LN2
