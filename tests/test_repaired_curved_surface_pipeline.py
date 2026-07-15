import math

import pytest

from gulati_quadrature import (
    CurvedPanelConfig,
    FeatureChannelConfig,
    ManifoldRepairConfig,
    PanelSingularRepaymentConfig,
    SurfaceQConfig,
    build_curved_panel_engine,
    build_curved_panel_surface,
    build_radial_quadric_panel_surface,
    repair_triangle_mesh,
)
from inverse_shape.curved_panels import PanelSingularRepayment3D
from inverse_shape.surface_feature_channels import (
    MellinKondratievPanelRepayment3D,
)


def cube():
    vertices = (
        (-1.0, -1.0, -1.0),
        (1.0, -1.0, -1.0),
        (1.0, 1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (1.0, -1.0, 1.0),
        (1.0, 1.0, 1.0),
        (-1.0, 1.0, 1.0),
    )
    faces = (
        (0, 2, 1),
        (0, 3, 2),
        (4, 5, 6),
        (4, 6, 7),
        (0, 1, 5),
        (0, 5, 4),
        (1, 2, 6),
        (1, 6, 5),
        (2, 3, 7),
        (2, 7, 6),
        (3, 0, 4),
        (3, 4, 7),
    )
    return vertices, faces


def projected_octahedron_once():
    vertices = [
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
    ]
    faces = (
        (0, 2, 4),
        (2, 1, 4),
        (1, 3, 4),
        (3, 0, 4),
        (2, 0, 5),
        (1, 2, 5),
        (3, 1, 5),
        (0, 3, 5),
    )
    cache = {}

    def midpoint(left, right):
        edge = (min(left, right), max(left, right))
        if edge not in cache:
            point = tuple(
                0.5 * (vertices[left][axis] + vertices[right][axis])
                for axis in range(3)
            )
            length = math.sqrt(sum(value * value for value in point))
            cache[edge] = len(vertices)
            vertices.append(tuple(value / length for value in point))
        return cache[edge]

    refined = []
    for first, second, third in faces:
        first_second = midpoint(first, second)
        second_third = midpoint(second, third)
        third_first = midpoint(third, first)
        refined.extend(
            (
                (first, first_second, third_first),
                (first_second, second, second_third),
                (third_first, second_third, third),
                (first_second, second_third, third_first),
            )
        )
    return tuple(vertices), tuple(refined)


def panel_edge_point(panel, edge, phase):
    barycentric = [0.0, 0.0, 0.0]
    barycentric[panel.vertex_indices.index(edge[0])] = 1.0 - phase
    barycentric[panel.vertex_indices.index(edge[1])] = phase
    return panel.evaluate(barycentric[1], barycentric[2]).position


def assert_panel_seams_close(surface, tolerance=2.0e-13):
    for edge in surface.topology.feature_edges:
        left = surface.panels[edge.incident_faces[0]]
        right = surface.panels[edge.incident_faces[1]]
        for phase in (0.2, 0.5, 0.8):
            first = panel_edge_point(left, edge.vertices, phase)
            second = panel_edge_point(right, edge.vertices, phase)
            assert max(abs(a - b) for a, b in zip(first, second, strict=True)) < (
                tolerance
            )


def assert_cubic_panel_jets(panel):
    u_value = 0.31
    v_value = 0.27
    step = 1.0e-4
    center = panel.evaluate(u_value, v_value)

    def position(u, v):
        return panel.evaluate(u, v).position

    plus_u = position(u_value + step, v_value)
    minus_u = position(u_value - step, v_value)
    plus_v = position(u_value, v_value + step)
    minus_v = position(u_value, v_value - step)
    plus_plus = position(u_value + step, v_value + step)
    plus_minus = position(u_value + step, v_value - step)
    minus_plus = position(u_value - step, v_value + step)
    minus_minus = position(u_value - step, v_value - step)
    for axis in range(3):
        derivative_u = (plus_u[axis] - minus_u[axis]) / (2.0 * step)
        derivative_v = (plus_v[axis] - minus_v[axis]) / (2.0 * step)
        derivative_uu = (
            plus_u[axis] - 2.0 * center.position[axis] + minus_u[axis]
        ) / step**2
        derivative_vv = (
            plus_v[axis] - 2.0 * center.position[axis] + minus_v[axis]
        ) / step**2
        derivative_uv = (
            plus_plus[axis]
            - plus_minus[axis]
            - minus_plus[axis]
            + minus_minus[axis]
        ) / (4.0 * step**2)
        assert abs(derivative_u - center.derivative_u[axis]) < 2.0e-8
        assert abs(derivative_v - center.derivative_v[axis]) < 2.0e-8
        assert abs(derivative_uu - center.derivative_uu[axis]) < 5.0e-7
        assert abs(derivative_uv - center.derivative_uv[axis]) < 5.0e-7
        assert abs(derivative_vv - center.derivative_vv[axis]) < 5.0e-7


def test_repair_welds_orients_fills_and_certifies_manifold() -> None:
    vertices, faces = cube()
    damaged_vertices = (*vertices, (1.0 + 1.0e-12, 1.0, 1.0))
    damaged_faces = list(faces)
    damaged_faces[7] = (1, 8, 5)
    damaged_faces.pop(3)
    damaged_faces[4] = tuple(reversed(damaged_faces[4]))
    damaged_faces.append(tuple(reversed(damaged_faces[0])))
    repaired = repair_triangle_mesh(damaged_vertices, damaged_faces)
    certificate = repaired.certificate
    assert certificate.welded_vertices == 1
    assert certificate.removed_duplicate_faces == 1
    assert certificate.boundary_edges_before_fill > 0
    assert certificate.boundary_loops_filled > 0
    assert certificate.boundary_edges == 0
    assert certificate.nonmanifold_edges == 0
    assert certificate.nonmanifold_vertices == 0
    assert certificate.inconsistent_oriented_edges == 0
    assert certificate.production_ready
    assert certificate.orientable_genus_sum == 0
    assert not certificate.self_intersection_certified


def test_repair_splits_two_closed_sheets_that_share_an_edge() -> None:
    vertices = (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, -1.0),
    )
    faces = (
        (0, 2, 1),
        (0, 1, 3),
        (1, 2, 3),
        (2, 0, 3),
        (0, 1, 4),
        (0, 5, 1),
        (1, 5, 4),
        (4, 5, 0),
    )
    repaired = repair_triangle_mesh(vertices, faces)
    assert repaired.certificate.nonmanifold_edges_before_split == 1
    assert repaired.certificate.split_sheet_vertices == 2
    assert repaired.certificate.connected_components == 2
    assert repaired.certificate.nonmanifold_vertices == 0
    assert repaired.certificate.production_ready


def test_repair_splits_a_pinched_vertex_and_can_fail_closed() -> None:
    vertices = (
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
        (-1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, -1.0),
    )
    faces = (
        (0, 2, 1),
        (0, 1, 3),
        (1, 2, 3),
        (2, 0, 3),
        (0, 4, 5),
        (0, 6, 4),
        (4, 6, 5),
        (5, 6, 0),
    )
    repaired = repair_triangle_mesh(vertices, faces)
    assert repaired.certificate.split_sheet_vertices == 1
    assert repaired.certificate.nonmanifold_vertices == 0
    assert repaired.certificate.connected_components == 2
    with pytest.raises(ValueError, match="closed oriented manifold"):
        repair_triangle_mesh(
            vertices,
            faces,
            config=ManifoldRepairConfig(split_nonmanifold_sheets=False),
        )


def test_curved_panel_api_rejects_unstable_repeated_laplacian_rungs() -> None:
    with pytest.raises(ValueError, match="zero or one"):
        PanelSingularRepaymentConfig(series_order=2)


def test_cubic_panels_improve_sphere_area_and_repay_every_smooth_node() -> None:
    vertices, faces = projected_octahedron_once()
    topology = repair_triangle_mesh(
        vertices,
        faces,
        config=ManifoldRepairConfig(sharp_angle_degrees=50.0),
    )
    surface = build_curved_panel_surface(
        topology,
        config=CurvedPanelConfig(quadrature_order=3),
    )
    exact_area = 4.0 * math.pi
    assert abs(sum(surface.weights) - exact_area) < abs(
        sum(topology.face_areas) - exact_area
    )
    assert all(surface.smooth_nodes)
    assert surface.maximum_panel_seam_gap < 2.0e-13
    assert_panel_seams_close(surface)
    assert surface.maximum_panel_seam_gap < 2.0e-13
    assert_cubic_panel_jets(surface.panels[0])
    repayment = PanelSingularRepayment3D(
        surface,
        config=PanelSingularRepaymentConfig(
            series_order=1,
        ),
    )
    constant_laplacian = repayment.laplacian((1.0,) * len(surface.points))
    assert max(abs(value) for value in constant_laplacian) < 5.0e-12
    stats = repayment.stats()
    assert stats["panel_singular_cells_repaid"] == len(surface.points)
    assert stats["stored_dense_panel_matrix"] is False
    assert stats["stored_pair_table"] is False


def test_sharp_panel_edges_are_shared_straight_curves() -> None:
    vertices, faces = cube()
    topology = repair_triangle_mesh(vertices, faces)
    surface = build_curved_panel_surface(
        topology,
        config=CurvedPanelConfig(quadrature_order=3),
    )
    assert_panel_seams_close(surface)
    for edge in topology.sharp_edges:
        panel = surface.panels[edge.incident_faces[0]]
        for phase in (0.2, 0.5, 0.8):
            actual = panel_edge_point(panel, edge.vertices, phase)
            expected = tuple(
                (1.0 - phase) * topology.vertices[edge.vertices[0]][axis]
                + phase * topology.vertices[edge.vertices[1]][axis]
                for axis in range(3)
            )
            assert max(
                abs(a - b) for a, b in zip(actual, expected, strict=True)
            ) < 2.0e-13


def test_radial_quadric_panels_are_an_independent_exact_sphere_geometry() -> None:
    vertices, faces = projected_octahedron_once()
    topology = repair_triangle_mesh(
        vertices,
        faces,
        config=ManifoldRepairConfig(sharp_angle_degrees=120.0),
    )
    surface = build_radial_quadric_panel_surface(
        topology,
        (1.0, 1.0, 1.0),
        config=CurvedPanelConfig(quadrature_order=4),
    )
    assert max(
        abs(sum(component * component for component in point) - 1.0)
        for point in surface.points
    ) < 8.0e-16
    assert max(abs(value - 1.0) for value in surface.gaussian_curvature) < 2.0e-14
    assert abs(sum(surface.weights) - 4.0 * math.pi) < 5.0e-5
    assert surface.stats["panel_geometry_kind"] == "RadialQuadricTriangle"


def test_edge_and_vertex_mellin_channels_reproduce_their_moments() -> None:
    vertices, faces = cube()
    topology = repair_triangle_mesh(vertices, faces)
    surface = build_curved_panel_surface(
        topology,
        config=CurvedPanelConfig(quadrature_order=3),
    )
    feature_config = FeatureChannelConfig(
        reference_quadrature_order=6,
        vertex_link_refinements=3,
    )
    repayment = MellinKondratievPanelRepayment3D(
        surface,
        config=feature_config,
    )
    stats = repayment.stats()
    assert stats["edge_mellin_channels"] == 12
    assert stats["vertex_mellin_channels"] == 8
    assert stats["vertex_link_pencil_failures"] == 0
    vertex_exponents = {
        round(channel.pencil_certificate["kondratiev_exponent"], 12)
        for channel in repayment.channels
        if channel.kind == "vertex"
    }
    assert len(vertex_exponents) == 1
    assert abs(next(iter(vertex_exponents)) - 3.0) < 0.05
    for channel in (
        next(row for row in repayment.channels if row.kind == "edge"),
        next(row for row in repayment.channels if row.kind == "vertex"),
    ):
        values = [0.0j for _ in surface.points]
        for offset, index in enumerate(channel.support):
            values[index] = channel.basis_values[0][offset]
        borrowed = sum(
            weight * value
            for weight, value in zip(surface.weights, values, strict=True)
        )
        result = repayment.repay_integral(
            values,
            borrowed_value=borrowed,
            channel_labels=(channel.label,),
        )
        assert abs(result.value - channel.exact_moments[0]) < 2.0e-12

    channel = next(row for row in repayment.channels if row.kind == "vertex")
    values = [0.0j for _ in surface.points]
    for offset, index in enumerate(channel.support):
        values[index] = channel.basis_values[0][offset]
    engine = build_curved_panel_engine(
        surface,
        feature_config=feature_config,
        config=SurfaceQConfig(
            harmonic_moment_degree=0,
            singular_cell_order=0,
        ),
    )
    public_result = engine.repay_feature_integral(
        values,
        channel_labels=(channel.label,),
    )
    assert abs(public_result.value - channel.exact_moments[0]) < 2.0e-12


def test_adaptive_moment_degree_uses_disjoint_next_degree_validation() -> None:
    vertices, faces = projected_octahedron_once()
    topology = repair_triangle_mesh(
        vertices,
        faces,
        config=ManifoldRepairConfig(sharp_angle_degrees=50.0),
    )
    surface = build_curved_panel_surface(
        topology,
        config=CurvedPanelConfig(quadrature_order=3),
    )
    engine = build_curved_panel_engine(
        surface,
        singular_config=PanelSingularRepaymentConfig(
            series_order=1,
        ),
        config=SurfaceQConfig(
            tolerance=1.0e-8,
            maximum_order=8,
            leaf_size=4,
            work_budget_factor=256,
            singular_cell_order=1,
            harmonic_moment_degree=2,
            adaptive_moment_degree=True,
            minimum_harmonic_moment_degree=1,
            moment_validation_tolerance=1.0,
        ),
    )
    result = engine.apply_dtn_principal(tuple(point[0] for point in engine.points))
    stats = result.stats
    assert stats["adaptive_selected_degree"] == 1
    assert stats["adaptive_validation_history"][0]["validation_degree"] == 2
    validation_modes = stats["adaptive_validation_history"][0]["validation_modes"]
    assert len(validation_modes) == 5
    assert all(name.startswith("H2_") for name in validation_modes)
    assert stats["adaptive_rank_growth_stored"] is False
    assert stats["independent_final_holdout_required"] is True
    engine_stats = engine.stats()
    assert engine_stats["dense_q_matrix_stored"] is False
    assert engine_stats["pair_table_stored"] is False
