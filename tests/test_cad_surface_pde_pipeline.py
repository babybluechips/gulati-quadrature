import math

from gulati_quadrature import (
    ExactMesh,
    MeshPart,
    SurfacePDEConfig,
    SurfaceQConfig,
    build_compiled_cad_engine,
    build_surface_pde_solver,
    compile_cad_surface,
)
from inverse_shape.quadrature import _cos, _sin


VERTICES = (
    (1.0, 0.0, 0.0),
    (-1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, 0.0, -1.0),
)
FACES = (
    (0, 2, 4),
    (2, 1, 4),
    (1, 3, 4),
    (3, 0, 4),
    (2, 0, 5),
    (1, 2, 5),
    (3, 1, 5),
    (0, 3, 5),
)


def relative_error(weights, expected, actual):
    numerator = sum(
        weight * abs(complex(left) - complex(right)) ** 2
        for weight, left, right in zip(weights, expected, actual, strict=True)
    )
    denominator = sum(
        weight * abs(complex(value)) ** 2
        for weight, value in zip(weights, expected, strict=True)
    )
    return math.sqrt(numerator / denominator)


def test_compiled_cad_surface_scans_every_face_and_preserves_measure() -> None:
    mesh = ExactMesh(
        vertices=VERTICES,
        faces=FACES,
        parts=(MeshPart("octahedron", 0, len(FACES), "test"),),
        scalar_bits=64,
        name="octahedron",
    )
    surface = compile_cad_surface(mesh, target_vertices=24)
    assert surface.processed_source_faces == len(FACES)
    assert abs(sum(surface.weights) / surface.source_area - 1.0) < 2.0e-15
    assert surface.stats["stored_dense_matrix"] is False
    assert surface.stats["stored_pair_table"] is False


def test_cad_harmonic_modal_inverse_and_helmholtz_flux() -> None:
    mesh = ExactMesh(
        vertices=VERTICES,
        faces=FACES,
        parts=(MeshPart("octahedron", 0, len(FACES), "test"),),
        scalar_bits=64,
        name="octahedron",
    )
    surface = compile_cad_surface(mesh, target_vertices=24)
    engine = build_compiled_cad_engine(
        surface,
        config=SurfaceQConfig(
            tolerance=1.0e-11,
            maximum_order=8,
            leaf_size=2,
            work_budget_factor=192,
            harmonic_moment_degree=1,
        ),
    )
    solver = build_surface_pde_solver(
        engine,
        config=SurfacePDEConfig(tolerance=1.0e-10, maximum_iterations=40),
    )
    direction = (1.0, 0.25, -0.1)
    trace = tuple(
        direction[0] * x + direction[1] * y + direction[2] * z
        for x, y, z in surface.vertices
    )
    mean = sum(
        weight * value
        for weight, value in zip(surface.weights, trace, strict=True)
    ) / sum(surface.weights)
    trace = tuple(value - mean for value in trace)
    flux = tuple(
        sum(left * right for left, right in zip(direction, normal, strict=True))
        for normal in surface.normals
    )
    dtn = solver.apply_laplace_dtn(trace)
    assert relative_error(surface.weights, flux, dtn.values) < 2.0e-12

    flux_mean = sum(
        weight * value
        for weight, value in zip(surface.weights, flux, strict=True)
    ) / sum(surface.weights)
    compatible = tuple(value - flux_mean for value in flux)
    poisson = solver.solve_poisson(compatible)
    assert relative_error(surface.weights, trace, poisson.values) < 2.0e-12
    assert poisson.stats["krylov_method"] == "fixed-rank harmonic modal solve"

    wave_number = 0.6
    plane = tuple(
        complex(_cos(wave_number * point[0]), _sin(wave_number * point[0]))
        for point in surface.vertices
    )
    exact_plane_flux = tuple(
        1j * wave_number * normal[0] * value
        for normal, value in zip(surface.normals, plane, strict=True)
    )
    helmholtz = solver.apply_helmholtz_dtn(plane, wavenumber=wave_number)
    assert relative_error(
        surface.weights,
        exact_plane_flux,
        helmholtz.values,
    ) < 2.0e-12
    assert engine.stats()["dense_q_matrix_stored"] is False

