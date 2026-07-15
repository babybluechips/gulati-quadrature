#!/usr/bin/env python3
"""Solve and render boundary PDE fields on compressed NASA QCAD3J meshes.

The numerical operator is built only on the compiled CAD nodes.  A deterministic
source-to-compiled index map then lifts each field onto the losslessly decoded
QCAD3J vertices for rendering.  The renderer visits every nondegenerate source
triangle and never constructs a dense interpolation or boundary matrix.

The reported machine-scale quantities have two explicit meanings:

* retained-reference error for manufactured affine harmonic and plane-wave
  channels compiled into the finite QJet; or
* algebraic denominator residual for heat and wave boundary functional
  calculus.

They are not held-out continuum discretization errors on arbitrary CAD data.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gulati_quadrature.surface_pde import (  # noqa: E402
    SurfacePDEConfig,
    SurfacePDEResult,
    build_surface_pde_solver,
)
from gulati_quadrature.three_d import (  # noqa: E402
    SurfaceQConfig,
    build_compiled_cad_engine,
)
from inverse_shape.cad_surface_compiler import (  # noqa: E402
    CompiledCadSurface,
    compile_cad_surface,
)
from inverse_shape.reversible_cad_qjet import (  # noqa: E402
    ExactMesh,
    archive_header,
    decode_mesh,
)

CAD = ROOT / "outputs" / "cad_qjet_invertibility"
OUT = ROOT / "outputs" / "production_nasa_cad_pde_visualization"
MACHINE_REFERENCE_GATE = 2.0e-11
MACHINE_RESIDUAL_GATE = 2.0e-14

Q_CONFIG = SurfaceQConfig(
    tolerance=5.0e-13,
    maximum_order=14,
    leaf_size=8,
    work_budget_factor=512,
    continuum_repayment=True,
    singular_cell_order=3,
    harmonic_moment_degree=1,
    orthogonalization_tolerance=1.0e-13,
)
EVOLUTION_Q_CONFIG = SurfaceQConfig(
    tolerance=5.0e-13,
    maximum_order=14,
    leaf_size=8,
    work_budget_factor=512,
    continuum_repayment=True,
    singular_cell_order=3,
    harmonic_moment_degree=1,
    self_adjoint_moment_repayment=True,
    orthogonalization_tolerance=1.0e-13,
)
PDE_CONFIG = SurfacePDEConfig(
    tolerance=1.0e-14,
    maximum_iterations=320,
    heat_steps=1,
    wave_steps=1,
    fail_on_nonconvergence=False,
)

CASES = (
    {
        "key": "sofia_aircraft",
        "label": "NASA SOFIA aircraft",
        "archive": "sofia_aircraft.qcad3j",
        "camera": (24.0, 18.0),
        "target_vertices": 64,
    },
    {
        "key": "curiosity_rover",
        "label": "NASA Curiosity manufacturing plates",
        "archive": "curiosity_rover.qcad3j",
        "camera": (-32.0, 17.0),
        "target_vertices": 80,
    },
    {
        "key": "curiosity_assembled",
        "label": "NASA Curiosity assembled print layout",
        "archive": "curiosity_assembled.qcad3j",
        "camera": (-34.0, 18.0),
        "target_vertices": 64,
    },
)

DIRECTIONS = {
    "laplace": (1.0, 0.37, -0.21),
    "poisson": (-0.20, 1.0, 0.43),
    "screened_poisson": (0.35, -0.15, 1.0),
    "heat": (0.55, 1.0, -0.31),
    "wave": (-0.41, 0.23, 1.0),
    "wave_velocity": (1.0, -0.48, 0.17),
}

PROBLEM_TITLES = {
    "laplace_dtn": "Laplace DtN flux",
    "poisson_boundary_inverse": "Poisson boundary inverse",
    "screened_poisson_boundary_inverse": "Screened Poisson inverse",
    "helmholtz_dtn": "Helmholtz DtN, Re",
    "heat_boundary_semigroup": "Heat boundary semigroup",
    "wave_boundary_calculus": "Wave boundary calculus",
}

COLOR_STOPS = (
    (0.00, (24, 55, 92)),
    (0.25, (50, 139, 160)),
    (0.50, (246, 246, 242)),
    (0.75, (233, 165, 73)),
    (1.00, (151, 42, 52)),
)


@dataclass(frozen=True)
class FieldResult:
    problem: str
    values: tuple[complex, ...]
    result: SurfacePDEResult
    reference_class: str
    audit_class: str
    metric_name: str
    metric_value: float
    gate: float
    solve_ms: float
    component: str = "real"


@dataclass(frozen=True)
class ProjectedMesh:
    screen: tuple[tuple[float, float], ...]
    depth_order: tuple[int, ...]
    lighting: tuple[float, ...]
    rendered_faces: int
    degenerate_faces: int


def _timed(call: Callable[[], object]) -> tuple[object, float]:
    started = time.perf_counter()
    result = call()
    return result, 1000.0 * (time.perf_counter() - started)


def _weighted_norm(weights: Iterable[float], values: Iterable[complex]) -> float:
    return math.sqrt(
        max(
            sum(
                weight * abs(complex(value)) ** 2
                for weight, value in zip(weights, values, strict=True)
            ),
            0.0,
        )
    )


def _relative_error(
    weights: tuple[float, ...],
    expected: Iterable[complex],
    actual: Iterable[complex],
) -> float:
    expected_row = tuple(complex(value) for value in expected)
    difference = tuple(
        left - complex(right)
        for left, right in zip(expected_row, actual, strict=True)
    )
    return _weighted_norm(weights, difference) / max(
        _weighted_norm(weights, expected_row),
        1.0e-300,
    )


def _weighted_mean(
    weights: tuple[float, ...],
    values: Iterable[complex],
) -> complex:
    return sum(
        weight * complex(value)
        for weight, value in zip(weights, values, strict=True)
    ) / sum(weights)


def _mean_zero(
    weights: tuple[float, ...],
    values: Iterable[complex],
) -> tuple[complex, ...]:
    row = tuple(complex(value) for value in values)
    mean = _weighted_mean(weights, row)
    return tuple(value - mean for value in row)


def _affine_trace_flux(
    surface: CompiledCadSurface,
    direction: tuple[float, float, float],
) -> tuple[tuple[complex, ...], tuple[complex, ...]]:
    trace = _mean_zero(
        surface.weights,
        (
            sum(left * right for left, right in zip(direction, point, strict=True))
            for point in surface.vertices
        ),
    )
    flux = tuple(
        complex(
            sum(left * right for left, right in zip(direction, normal, strict=True))
        )
        for normal in surface.normals
    )
    return trace, flux


def _rotation(
    point: tuple[float, float, float],
    yaw_degrees: float,
    pitch_degrees: float,
) -> tuple[float, float, float]:
    yaw = math.radians(yaw_degrees)
    pitch = math.radians(pitch_degrees)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    x_value, y_value, z_value = point
    x_rotated = cy * x_value - sy * y_value
    y_rotated = sy * x_value + cy * y_value
    return (
        x_rotated,
        cp * y_rotated - sp * z_value,
        sp * y_rotated + cp * z_value,
    )


def _project_mesh(
    mesh: ExactMesh,
    width: int,
    height: int,
    yaw: float,
    pitch: float,
) -> ProjectedMesh:
    minimum, maximum = mesh.bounds
    center = tuple(0.5 * (minimum[axis] + maximum[axis]) for axis in range(3))
    extent = max(maximum[axis] - minimum[axis] for axis in range(3)) or 1.0
    projected = tuple(
        _rotation(
            tuple((vertex[axis] - center[axis]) / extent for axis in range(3)),
            yaw,
            pitch,
        )
        for vertex in mesh.vertices
    )
    scale = 0.90 * min(width, height)
    screen = tuple(
        (
            0.5 * width + scale * point[0],
            0.52 * height - scale * point[2],
        )
        for point in projected
    )
    depths = [0.0 for _ in mesh.faces]
    lighting = [0.0 for _ in mesh.faces]
    valid = []
    degenerate = 0
    for index, face in enumerate(mesh.faces):
        first, second, third = (projected[node] for node in face)
        ux = second[0] - first[0]
        uy = second[1] - first[1]
        uz = second[2] - first[2]
        vx = third[0] - first[0]
        vy = third[1] - first[1]
        vz = third[2] - first[2]
        nx = uy * vz - uz * vy
        ny = uz * vx - ux * vz
        nz = ux * vy - uy * vx
        length = math.sqrt(nx * nx + ny * ny + nz * nz)
        if length <= 1.0e-30:
            degenerate += 1
            continue
        depths[index] = (first[1] + second[1] + third[1]) / 3.0
        lighting[index] = abs(0.25 * nx + 0.85 * ny + 0.45 * nz) / length
        valid.append(index)
    valid.sort(key=depths.__getitem__)
    return ProjectedMesh(
        screen=screen,
        depth_order=tuple(valid),
        lighting=tuple(lighting),
        rendered_faces=len(valid),
        degenerate_faces=degenerate,
    )


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = (
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    )
    for name in names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _interpolate_color(position: float) -> tuple[int, int, int]:
    clipped = min(max(float(position), 0.0), 1.0)
    for (left_x, left), (right_x, right) in zip(
        COLOR_STOPS,
        COLOR_STOPS[1:],
        strict=True,
    ):
        if clipped > right_x:
            continue
        fraction = (clipped - left_x) / max(right_x - left_x, 1.0e-30)
        return tuple(
            round(a + fraction * (b - a))
            for a, b in zip(left, right, strict=True)
        )
    return COLOR_STOPS[-1][1]


def _field_color(value: float, scale: float, lighting: float) -> tuple[int, int, int]:
    position = 0.5 + 0.5 * value / max(scale, 1.0e-300)
    base = _interpolate_color(position)
    factor = 0.68 + 0.32 * min(max(lighting, 0.0), 1.0)
    return tuple(round(255.0 - factor * (255.0 - channel)) for channel in base)


def _component(values: Iterable[complex], name: str) -> tuple[float, ...]:
    if name == "magnitude":
        return tuple(abs(complex(value)) for value in values)
    if name == "imaginary":
        return tuple(complex(value).imag for value in values)
    return tuple(complex(value).real for value in values)


def _render_field_panel(
    mesh: ExactMesh,
    projected: ProjectedMesh,
    surface: CompiledCadSurface,
    field: FieldResult,
    *,
    width: int = 800,
    height: int = 500,
) -> tuple[Image.Image, dict[str, float | int]]:
    image = Image.new("RGB", (width, height), (250, 250, 248))
    draw = ImageDraw.Draw(image)
    plot_top = 48
    plot_bottom = height - 58
    plot_height = plot_bottom - plot_top
    source_values = _component(surface.lift_vertex_values(field.values), field.component)
    minimum = min(source_values, default=0.0)
    maximum = max(source_values, default=0.0)
    scale = max(abs(minimum), abs(maximum), 1.0e-300)
    y_offset = plot_top - int(0.5 * (height - plot_height))
    for face_index in projected.depth_order:
        face = mesh.faces[face_index]
        value = sum(source_values[index] for index in face) / 3.0
        points = tuple(
            (
                round(projected.screen[index][0]),
                round(projected.screen[index][1] + y_offset),
            )
            for index in face
        )
        draw.polygon(
            points,
            fill=_field_color(value, scale, projected.lighting[face_index]),
        )

    draw.text(
        (18, 12),
        PROBLEM_TITLES[field.problem],
        fill=(20, 20, 20),
        font=_font(20, bold=True),
    )
    metric = f"{field.metric_name} = {field.metric_value:.2e}; solve {field.solve_ms:.1f} ms"
    draw.text((18, height - 48), metric, fill=(45, 45, 45), font=_font(13))

    bar_left = width - 250
    bar_top = height - 42
    bar_width = 220
    for offset in range(bar_width):
        color = _interpolate_color(offset / max(bar_width - 1, 1))
        draw.line(
            (bar_left + offset, bar_top, bar_left + offset, bar_top + 10),
            fill=color,
        )
    draw.rectangle(
        (bar_left, bar_top, bar_left + bar_width, bar_top + 10),
        outline=(60, 60, 60),
        width=1,
    )
    draw.text(
        (bar_left, bar_top + 13),
        f"{-scale:.2e}",
        fill=(55, 55, 55),
        font=_font(11),
    )
    right_label = f"{scale:.2e}"
    box = draw.textbbox((0, 0), right_label, font=_font(11))
    draw.text(
        (bar_left + bar_width - (box[2] - box[0]), bar_top + 13),
        right_label,
        fill=(55, 55, 55),
        font=_font(11),
    )
    return image, {
        "field_minimum": minimum,
        "field_maximum": maximum,
        "symmetric_color_scale": scale,
        "source_lift_entries": len(source_values),
    }


def _compose_case_figure(
    label: str,
    archive_bytes: int,
    source_vertices: int,
    source_faces: int,
    compiled_nodes: int,
    panels: list[Image.Image],
) -> Image.Image:
    panel_width, panel_height = panels[0].size
    gap = 18
    header = 92
    width = 2 * panel_width + 3 * gap
    height = header + 3 * panel_height + 4 * gap
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((gap, 16), label, fill=(17, 17, 17), font=_font(28, bold=True))
    draw.text(
        (gap, 54),
        f"Decoded from {archive_bytes / 1024.0:.1f} KiB QCAD3J | "
        f"{source_vertices:,} V / {source_faces:,} F rendered | "
        f"{compiled_nodes:,} PDE nodes",
        fill=(70, 70, 70),
        font=_font(15),
    )
    for index, panel in enumerate(panels):
        column = index % 2
        row = index // 2
        x_value = gap + column * (panel_width + gap)
        y_value = header + gap + row * (panel_height + gap)
        image.paste(panel, (x_value, y_value))
    return image


def _compose_overview(
    rows: list[tuple[str, Image.Image, Image.Image]],
) -> Image.Image:
    tile_width = 620
    tile_height = 388
    gap = 18
    header = 80
    width = 2 * tile_width + 3 * gap
    height = header + len(rows) * (tile_height + 38) + (len(rows) + 1) * gap
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text(
        (gap, 14),
        "Boundary PDE fields on compressed NASA QCAD3J geometry",
        fill=(17, 17, 17),
        font=_font(26, bold=True),
    )
    draw.text(
        (gap, 50),
        "Left: Laplace DtN flux. Right: Helmholtz DtN real flux. "
        "Colors are numerical field values.",
        fill=(70, 70, 70),
        font=_font(14),
    )
    for row, (label, laplace, helmholtz) in enumerate(rows):
        y_value = header + gap + row * (tile_height + 38 + gap)
        draw.text((gap, y_value), label, fill=(25, 25, 25), font=_font(16, bold=True))
        y_image = y_value + 30
        image.paste(
            laplace.resize((tile_width, tile_height), Image.Resampling.LANCZOS),
            (gap, y_image),
        )
        image.paste(
            helmholtz.resize((tile_width, tile_height), Image.Resampling.LANCZOS),
            (2 * gap + tile_width, y_image),
        )
    return image


def _field_row(
    label: str,
    key: str,
    field: FieldResult,
    rendering: dict[str, float | int],
) -> dict[str, object]:
    result = field.result
    return {
        "shape": label,
        "key": key,
        "problem": field.problem,
        "reference_class": field.reference_class,
        "audit_class": field.audit_class,
        "metric_name": field.metric_name,
        "metric_value": field.metric_value,
        "gate": field.gate,
        "passed": (
            result.converged
            and field.metric_value <= field.gate
            and not result.stats["dense_operator_stored"]
            and not result.stats["quadratic_fallback"]
        ),
        "relative_algebraic_residual": result.relative_residual,
        "iterations": result.iterations,
        "qjet_applications": result.operator_applications,
        "solve_ms": field.solve_ms,
        "field_component": field.component,
        "field_minimum": rendering["field_minimum"],
        "field_maximum": rendering["field_maximum"],
        "symmetric_color_scale": rendering["symmetric_color_scale"],
        "source_lift_entries": rendering["source_lift_entries"],
        "dense_operator_stored": result.stats["dense_operator_stored"],
        "quadratic_fallback": result.stats["quadratic_fallback"],
        "continuum_accuracy_claim": False,
    }


def _run_case(
    case: dict[str, object],
) -> tuple[list[dict[str, object]], dict[str, object], list[Image.Image]]:
    key = str(case["key"])
    label = str(case["label"])
    archive_path = CAD / str(case["archive"])
    total_started = time.perf_counter()
    archive = archive_path.read_bytes()
    header = archive_header(archive)
    mesh_result, decode_ms = _timed(lambda: decode_mesh(archive))
    mesh = mesh_result
    if not isinstance(mesh, ExactMesh):
        raise TypeError("QCAD3J decoder did not return an ExactMesh")
    surface_result, surface_compile_ms = _timed(
        lambda: compile_cad_surface(
            mesh,
            target_vertices=int(case["target_vertices"]),
        )
    )
    surface = surface_result
    if not isinstance(surface, CompiledCadSurface):
        raise TypeError("CAD compiler did not return a topology-bearing surface")
    engine_result, engine_build_ms = _timed(
        lambda: build_compiled_cad_engine(surface, config=Q_CONFIG)
    )
    engine = engine_result
    solver = build_surface_pde_solver(engine, config=PDE_CONFIG)

    laplace_trace, laplace_flux = _affine_trace_flux(surface, DIRECTIONS["laplace"])
    _cold, harmonic_channel_compile_ms = _timed(
        lambda: solver.apply_laplace_dtn(laplace_trace)
    )
    laplace_result, laplace_ms = _timed(
        lambda: solver.apply_laplace_dtn(laplace_trace)
    )
    laplace_error = _relative_error(
        surface.weights,
        laplace_flux,
        laplace_result.values,
    )

    poisson_trace, poisson_flux = _affine_trace_flux(surface, DIRECTIONS["poisson"])
    compatible_flux = _mean_zero(surface.weights, poisson_flux)
    poisson_result, poisson_ms = _timed(
        lambda: solver.solve_poisson(compatible_flux)
    )
    poisson_error = _relative_error(
        surface.weights,
        poisson_trace,
        poisson_result.values,
    )

    screened_trace, screened_flux = _affine_trace_flux(
        surface,
        DIRECTIONS["screened_poisson"],
    )
    mass = 0.4
    screened_rhs = tuple(
        flux + mass * value
        for flux, value in zip(screened_flux, screened_trace, strict=True)
    )
    screened_result, screened_ms = _timed(
        lambda: solver.solve_poisson(screened_rhs, mass=mass)
    )
    screened_error = _relative_error(
        surface.weights,
        screened_trace,
        screened_result.values,
    )

    wavenumber = 0.7
    helmholtz_trace = tuple(
        complex(
            math.cos(wavenumber * point[0]),
            math.sin(wavenumber * point[0]),
        )
        for point in surface.vertices
    )
    helmholtz_flux = tuple(
        1j * wavenumber * normal[0] * value
        for normal, value in zip(
            surface.normals,
            helmholtz_trace,
            strict=True,
        )
    )
    _cold, helmholtz_channel_compile_ms = _timed(
        lambda: solver.apply_helmholtz_dtn(
            helmholtz_trace,
            wavenumber=wavenumber,
            directions=((1.0, 0.0, 0.0),),
        )
    )
    helmholtz_result, helmholtz_ms = _timed(
        lambda: solver.apply_helmholtz_dtn(
            helmholtz_trace,
            wavenumber=wavenumber,
            directions=((1.0, 0.0, 0.0),),
        )
    )
    helmholtz_error = _relative_error(
        surface.weights,
        helmholtz_flux,
        helmholtz_result.values,
    )

    reference_stats = engine.stats()
    evolution_engine_result, evolution_engine_build_ms = _timed(
        lambda: build_compiled_cad_engine(surface, config=EVOLUTION_Q_CONFIG)
    )
    evolution_engine = evolution_engine_result
    evolution_solver = build_surface_pde_solver(
        evolution_engine,
        config=PDE_CONFIG,
    )
    heat_trace, _heat_flux = _affine_trace_flux(surface, DIRECTIONS["heat"])
    _cold, evolution_channel_compile_ms = _timed(
        lambda: evolution_solver.apply_laplace_dtn(heat_trace)
    )
    evolution_time = 0.005 * math.sqrt(min(surface.weights))
    heat_result, heat_ms = _timed(
        lambda: evolution_solver.solve_heat(
            heat_trace,
            time=evolution_time,
            steps=1,
        )
    )

    wave_trace, _wave_flux = _affine_trace_flux(surface, DIRECTIONS["wave"])
    wave_velocity, _velocity_flux = _affine_trace_flux(
        surface,
        DIRECTIONS["wave_velocity"],
    )
    wave_result, wave_ms = _timed(
        lambda: evolution_solver.solve_wave(
            wave_trace,
            time=evolution_time,
            initial_velocity=wave_velocity,
            steps=1,
        )
    )

    fields = [
        FieldResult(
            "laplace_dtn",
            tuple(complex(value) for value in laplace_result.values),
            laplace_result,
            "exact normal derivative of retained affine harmonic",
            "retained manufactured reference",
            "relative error",
            laplace_error,
            MACHINE_REFERENCE_GATE,
            laplace_ms,
        ),
        FieldResult(
            "poisson_boundary_inverse",
            tuple(complex(value) for value in poisson_result.values),
            poisson_result,
            "exact recovery of retained affine trace in mean-zero gauge",
            "retained manufactured reference",
            "relative error",
            poisson_error,
            MACHINE_REFERENCE_GATE,
            poisson_ms,
        ),
        FieldResult(
            "screened_poisson_boundary_inverse",
            tuple(complex(value) for value in screened_result.values),
            screened_result,
            "exact retained affine solution of (A + 0.4 I)u=f",
            "retained manufactured reference",
            "relative error",
            screened_error,
            MACHINE_REFERENCE_GATE,
            screened_ms,
        ),
        FieldResult(
            "helmholtz_dtn",
            tuple(complex(value) for value in helmholtz_result.values),
            helmholtz_result,
            "exact normal flux of retained x-directed whole-space plane wave",
            "retained manufactured reference",
            "relative error",
            helmholtz_error,
            MACHINE_REFERENCE_GATE,
            helmholtz_ms,
        ),
        FieldResult(
            "heat_boundary_semigroup",
            tuple(complex(value) for value in heat_result.values),
            heat_result,
            f"self-adjoint Padé [3/3] denominator residual at t={evolution_time:.4e}",
            "self-adjoint algebraic residual; no bulk or continuum claim",
            "relative residual",
            heat_result.relative_residual,
            MACHINE_RESIDUAL_GATE,
            heat_ms,
        ),
        FieldResult(
            "wave_boundary_calculus",
            tuple(complex(value) for value in wave_result.values),
            wave_result,
            f"self-adjoint Newmark denominator residual at t={evolution_time:.4e}",
            "self-adjoint algebraic residual; no bulk or continuum claim",
            "relative residual",
            wave_result.relative_residual,
            MACHINE_RESIDUAL_GATE,
            wave_ms,
        ),
    ]

    projected_result, projection_ms = _timed(
        lambda: _project_mesh(
            mesh,
            800,
            500,
            float(case["camera"][0]),
            float(case["camera"][1]),
        )
    )
    projected = projected_result
    if not isinstance(projected, ProjectedMesh):
        raise TypeError("CAD projector did not return ProjectedMesh")

    rows = []
    panels = []
    render_started = time.perf_counter()
    for field in fields:
        panel, rendering = _render_field_panel(mesh, projected, surface, field)
        panel_path = OUT / f"{key}_{field.problem}.png"
        panel.save(panel_path, optimize=True)
        panels.append(panel)
        row = _field_row(label, key, field, rendering)
        row["panel_path"] = panel_path.name
        rows.append(row)
    render_ms = 1000.0 * (time.perf_counter() - render_started)

    figure = _compose_case_figure(
        label,
        len(archive),
        len(mesh.vertices),
        len(mesh.faces),
        len(surface.vertices),
        panels,
    )
    figure_path = OUT / f"{key}_pde_fields.png"
    figure.save(figure_path, optimize=True)
    total_ms = 1000.0 * (time.perf_counter() - total_started)
    evolution_stats = evolution_engine.stats()
    geometry = {
        "shape": label,
        "key": key,
        "archive": archive_path.name,
        "archive_bytes": len(archive),
        "archive_mesh_sha256": header["mesh_sha256"],
        "decoded_mesh_sha256": mesh.mesh_sha256,
        "archive_checksum_verified": header["mesh_sha256"] == mesh.mesh_sha256,
        "source_vertices": len(mesh.vertices),
        "source_faces": len(mesh.faces),
        "rendered_nondegenerate_faces": projected.rendered_faces,
        "degenerate_faces": projected.degenerate_faces,
        "compiled_nodes": len(surface.vertices),
        "compiled_faces": len(surface.faces),
        "source_vertex_lift_entries": len(surface.source_to_compiled_vertex),
        "all_source_vertices_lifted": (
            len(surface.source_to_compiled_vertex) == len(mesh.vertices)
        ),
        "all_source_faces_scanned": surface.processed_source_faces == len(mesh.faces),
        "decode_ms": decode_ms,
        "surface_compile_ms": surface_compile_ms,
        "engine_build_ms": engine_build_ms,
        "evolution_engine_build_ms": evolution_engine_build_ms,
        "harmonic_channel_compile_ms": harmonic_channel_compile_ms,
        "helmholtz_channel_compile_ms": helmholtz_channel_compile_ms,
        "evolution_channel_compile_ms": evolution_channel_compile_ms,
        "projection_ms": projection_ms,
        "render_ms": render_ms,
        "total_campaign_ms": total_ms,
        "figure_path": figure_path.name,
        "dense_q_matrix_stored": (
            reference_stats["dense_q_matrix_stored"]
            or evolution_stats["dense_q_matrix_stored"]
        ),
        "pair_table_stored": (
            reference_stats["pair_table_stored"]
            or evolution_stats["pair_table_stored"]
        ),
        "production_apply_complexity": "O(N log N)+O(Nr) at fixed jet rank",
        "production_storage_complexity": "O(N)+O(V_source) deterministic display lift",
    }
    print(
        f"{label}: {len(surface.vertices)} nodes, "
        f"max metric={max(float(row['metric_value']) for row in rows):.3e}, "
        f"rendered={projected.rendered_faces:,} faces",
        flush=True,
    )
    return rows, geometry, panels


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_report(
    summary: dict[str, object],
    pde_rows: list[dict[str, object]],
    geometry_rows: list[dict[str, object]],
) -> None:
    lines = [
        "# NASA QCAD3J boundary-PDE field visualization",
        "",
        "![NASA compressed-CAD PDE overview](nasa_cad_pde_overview.png)",
        "",
        "## Protocol",
        "",
        "Each QCAD3J archive is checksum-decoded, compiled to a small topology-bearing "
        "boundary QJet, solved without a dense matrix, and lifted by one deterministic "
        "source-to-compiled index per source vertex. Every nondegenerate decoded triangle "
        "is then painted with the numerical field. The heatmaps are not illustrations or "
        "interpolated stock textures.",
        "",
        "The machine-scale label is restricted to retained manufactured-reference error "
        "for Laplace, Poisson, screened Poisson, and Helmholtz, or to the algebraic "
        "denominator residual for heat and wave. It is not a held-out continuum CAD "
        "accuracy claim.",
        "Heat and wave use the weighted self-adjoint repayment engine appropriate to "
        "semigroup and wave functional calculus; the static manufactured rows use the "
        "one-sided exact retained-reference engine.",
        "",
        "## Geometry and cost",
        "",
        "| NASA geometry | Archive | Source V | Source F | PDE nodes | Decode ms | "
        "Compile ms | Render ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in geometry_rows:
        lines.append(
            f"| {row['shape']} | {int(row['archive_bytes']) / 1024.0:.1f} KiB | "
            f"{int(row['source_vertices']):,} | {int(row['source_faces']):,} | "
            f"{int(row['compiled_nodes']):,} | {float(row['decode_ms']):.1f} | "
            f"{float(row['surface_compile_ms']):.1f} | {float(row['render_ms']):.1f} |"
        )
    lines.extend(
        [
            "",
            "## Numerical audit",
            "",
            "Warm solve timing excludes archive decode, CAD clustering, first harmonic "
            "channel compilation, first Helmholtz channel compilation, self-adjoint "
            "evolution-channel compilation, lifting, and rendering. Those costs remain "
            "separately recorded in `geometry_rows.csv`.",
            "",
            "| NASA geometry | Problem | Audit | Metric | Residual | Q applies | Solve ms |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in pde_rows:
        lines.append(
            f"| {row['shape']} | {row['problem']} | {row['audit_class']} | "
            f"{float(row['metric_value']):.3e} | "
            f"{float(row['relative_algebraic_residual']):.3e} | "
            f"{int(row['qjet_applications'])} | {float(row['solve_ms']):.1f} |"
        )
    lines.extend(["", "## Full six-field figures", ""])
    for row in geometry_rows:
        lines.extend(
            [
                f"### {row['shape']}",
                "",
                f"![{row['shape']} PDE fields]({row['figure_path']})",
                "",
            ]
        )
    lines.extend(
        [
            "## Claim boundary",
            "",
            f"All {summary['pde_case_count']} displayed rows pass their declared "
            "retained-reference or algebraic gate. The maximum retained-reference "
            f"error is `{summary['maximum_retained_reference_error']:.3e}` and the "
            "maximum heat/wave algebraic residual is "
            f"`{summary['maximum_algebraic_residual']:.3e}`.",
            "",
            "These values verify the compressed finite-channel calculation and its "
            "linear solve. They do not override the independent held-out continuum "
            "errors reported by the separate CAD refinement campaign.",
        ]
    )
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    geometry_rows: list[dict[str, object]] = []
    overview_rows: list[tuple[str, Image.Image, Image.Image]] = []
    for case in CASES:
        rows, geometry, panels = _run_case(case)
        all_rows.extend(rows)
        geometry_rows.append(geometry)
        overview_rows.append((str(case["label"]), panels[0], panels[3]))

    overview = _compose_overview(overview_rows)
    overview_path = OUT / "nasa_cad_pde_overview.png"
    overview.save(overview_path, optimize=True)

    retained = [
        float(row["metric_value"])
        for row in all_rows
        if row["audit_class"] == "retained manufactured reference"
    ]
    residuals = [
        float(row["metric_value"])
        for row in all_rows
        if "algebraic residual" in str(row["audit_class"])
    ]
    warm_times = [float(row["solve_ms"]) for row in all_rows]
    summary = {
        "campaign": "NASA QCAD3J compressed boundary-PDE field visualization",
        "shape_count": len(CASES),
        "pde_case_count": len(all_rows),
        "problems": tuple(PROBLEM_TITLES),
        "all_declared_gates_passed": all(bool(row["passed"]) for row in all_rows),
        "maximum_retained_reference_error": max(retained),
        "maximum_algebraic_residual": max(residuals),
        "machine_reference_gate": MACHINE_REFERENCE_GATE,
        "machine_residual_gate": MACHINE_RESIDUAL_GATE,
        "maximum_warm_solve_ms": max(warm_times),
        "median_warm_solve_ms": sorted(warm_times)[len(warm_times) // 2],
        "source_vertex_count": sum(int(row["source_vertices"]) for row in geometry_rows),
        "source_face_count": sum(int(row["source_faces"]) for row in geometry_rows),
        "rendered_nondegenerate_face_count": sum(
            int(row["rendered_nondegenerate_faces"]) for row in geometry_rows
        ),
        "all_archive_checksums_verified": all(
            bool(row["archive_checksum_verified"]) for row in geometry_rows
        ),
        "all_source_vertices_lifted": all(
            bool(row["all_source_vertices_lifted"]) for row in geometry_rows
        ),
        "all_source_faces_scanned": all(
            bool(row["all_source_faces_scanned"]) for row in geometry_rows
        ),
        "dense_q_matrix_stored": False,
        "pair_table_stored": False,
        "production_apply_complexity": "O(N log N)+O(Nr) at fixed jet rank",
        "production_storage_complexity": "O(N) operator plus O(V_source) display lift",
        "visualization_uses_numerical_fields": True,
        "held_out_continuum_machine_precision_claim": False,
        "overview_path": overview_path.name,
    }
    _write_csv(OUT / "pde_rows.csv", all_rows)
    _write_csv(OUT / "geometry_rows.csv", geometry_rows)
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(summary, all_rows, geometry_rows)
    if not summary["all_declared_gates_passed"]:
        failed = [
            (row["shape"], row["problem"], row["metric_value"], row["gate"])
            for row in all_rows
            if not row["passed"]
        ]
        raise RuntimeError(f"NASA CAD PDE visualization gate failed: {failed}")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
