import ast
from pathlib import Path

from gulati_quadrature import (
    SurfacePDEConfig,
    build_mesh_engine,
    build_surface_pde_solver,
)


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


def _solver():
    engine = build_mesh_engine(VERTICES, FACES)
    config = SurfacePDEConfig(
        tolerance=2.0e-11,
        maximum_iterations=160,
        heat_steps=1,
        wave_steps=4,
    )
    return engine, build_surface_pde_solver(engine, config=config)


def _maximum_error(left, right):
    return max(abs(complex(a) - complex(b)) for a, b in zip(left, right, strict=True))


def test_surface_poisson_recovers_manufactured_discrete_solution() -> None:
    _engine, solver = _solver()
    expected = tuple(x - 0.2 * y + 0.1 * z for x, y, z in VERTICES)
    rhs = solver.apply_laplace_dtn(expected).values
    result = solver.solve_poisson(rhs)

    assert result.converged
    assert result.relative_residual < 2.0e-11
    assert _maximum_error(result.values, expected) < 2.0e-11
    assert result.stats["dense_operator_stored"] is False
    assert result.stats["quadratic_fallback"] is False


def test_surface_screened_poisson_and_engine_dispatch() -> None:
    engine, solver = _solver()
    expected = tuple(0.3 + x + 0.2 * z for x, _y, z in VERTICES)
    applied = solver.apply_laplace_dtn(expected).values
    mass = 0.4
    rhs = tuple(
        value + mass * target
        for value, target in zip(applied, expected, strict=True)
    )
    result = engine.solve(
        "screened_poisson",
        rhs,
        mass=mass,
        config=SurfacePDEConfig(tolerance=2.0e-11),
    )

    assert result.converged
    assert _maximum_error(result.values, expected) < 2.0e-11
    assert result.problem == "screened_poisson"


def test_surface_helmholtz_recovers_manufactured_discrete_solution() -> None:
    _engine, solver = _solver()
    expected = tuple(x + 0.15j * y - 0.25 * z for x, y, z in VERTICES)
    first = solver.apply_laplace_dtn(expected).values
    second = solver.apply_laplace_dtn(first).values
    wave_number = 0.7
    damping = 0.3
    rhs = tuple(
        value - wave_number * wave_number * target + 1j * damping * target
        for value, target in zip(second, expected, strict=True)
    )
    result = solver.solve_helmholtz(
        rhs,
        wavenumber=wave_number,
        damping=damping,
    )

    assert result.converged
    assert result.relative_residual < 1.0e-10
    assert _maximum_error(result.values, expected) < 2.0e-10


def test_surface_heat_and_wave_preserve_constant_channel() -> None:
    _engine, solver = _solver()
    constant = (2.5 - 0.4j,) * len(VERTICES)
    heat = solver.solve_heat(constant, time=0.2)
    wave = solver.solve_wave(constant, time=0.2)

    assert heat.converged and wave.converged
    assert _maximum_error(heat.values, constant) < 2.0e-13
    assert _maximum_error(wave.values, constant) < 2.0e-13
    assert heat.stats["pde_scope"] == "boundary functional calculus; no arbitrary volume source"
    assert wave.stats["auxiliary_storage_big_o"] == "O(N)"


def test_surface_pde_facade_does_not_import_numpy_or_scipy() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "gulati_quadrature"
        / "surface_pde.py"
    ).read_text(encoding="utf-8")
    imports = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert not any(name == "numpy" or name.startswith("numpy.") for name in imports)
    assert not any(name == "scipy" or name.startswith("scipy.") for name in imports)
