import math

from gulati_quadrature import SurfaceQConfig, build_mesh_engine
from inverse_shape.surface_repayment import (
    MeshDifferentialGeometry,
    solid_harmonic_polynomials,
)


def projected_octahedron(refinements: int):
    vertices = [
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
    ]
    faces = [
        (0, 2, 4),
        (2, 1, 4),
        (1, 3, 4),
        (3, 0, 4),
        (2, 0, 5),
        (1, 2, 5),
        (3, 1, 5),
        (0, 3, 5),
    ]
    for _level in range(refinements):
        midpoint_cache = {}

        def midpoint(left: int, right: int) -> int:
            edge = (min(left, right), max(left, right))
            if edge not in midpoint_cache:
                point = tuple(
                    0.5 * (vertices[left][axis] + vertices[right][axis])
                    for axis in range(3)
                )
                length = math.sqrt(sum(value * value for value in point))
                midpoint_cache[edge] = len(vertices)
                vertices.append(tuple(value / length for value in point))
            return midpoint_cache[edge]

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
        faces = refined
    return tuple(vertices), tuple(faces)


def test_solid_harmonic_generator_has_exact_dimensions() -> None:
    modes = solid_harmonic_polynomials(4)
    assert [sum(mode.degree == degree for mode in modes) for degree in range(1, 5)] == [
        3,
        5,
        7,
        9,
    ]


def test_mesh_repayment_reproduces_compiled_solid_harmonics() -> None:
    points, faces = projected_octahedron(2)
    engine = build_mesh_engine(
        points,
        faces,
        config=SurfaceQConfig(
            tolerance=3.0e-13,
            maximum_order=16,
            leaf_size=4,
            work_budget_factor=192,
            harmonic_moment_degree=3,
        ),
    )
    geometry = engine._mesh_geometry
    assert isinstance(geometry, MeshDifferentialGeometry)

    maximum_error = 0.0
    for polynomial in solid_harmonic_polynomials(3):
        trace = []
        exact_flux = []
        for point, normal in zip(points, geometry.normals, strict=True):
            value, gradient = polynomial.value_gradient(point)
            trace.append(value)
            exact_flux.append(sum(left * right for left, right in zip(gradient, normal)))
        numerical = engine.apply_dtn_principal(trace)
        maximum_error = max(
            maximum_error,
            max(
                abs(complex(left) - right)
                for left, right in zip(numerical.values, exact_flux, strict=True)
            ),
        )
    assert maximum_error < 2.0e-11
    stats = engine.stats()
    assert stats["harmonic_repayment_rank"] == 15
    assert stats["stored_dense_harmonic_matrix"] is False
    assert stats["stored_dense_geometry_matrix"] is False
    assert stats["dense_q_matrix_stored"] is False


def test_mesh_repayment_keeps_the_constant_channel_exact() -> None:
    points, faces = projected_octahedron(2)
    engine = build_mesh_engine(points, faces)
    result = engine.apply_dtn_principal((2.5,) * len(points))
    assert result.values == (0.0j,) * len(points)

