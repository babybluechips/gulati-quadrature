#!/usr/bin/env python3
"""Helmholtz DtN benchmarks with analytic ground truth.

This separates true Helmholtz DtN tests from the older Steklov-resolvent
benchmarks.  Disk rows use exact Bessel/Hankel modal amplitudes.  Funky-domain
rows use manufactured Helmholtz fields whose boundary values and normal fluxes
are known pointwise.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable

import numpy as np
from scipy import special
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import splu

from inverse_shape.fem import (
    assemble_p1_stiffness,
    boundary_lumped_mass,
    build_star_fan_mesh,
    relative_weighted_l2,
)
from inverse_shape.geometry import as_points, polygon_area
from inverse_shape.q_dtn import (
    build_boundary_pullback_qjet,
    build_helmholtz_moment_corrected_planar_qjet,
    continuum_repaid_dtn_eigenvalue,
    cycle_dtn_eigenvalue,
    ellipse_qjet_map,
    radial_fourier_qjet_map,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "assets" / "q_dtn_helmholtz_ground_truth.json"
TAU = 2.0 * math.pi


ComplexField = Callable[[float, float], tuple[complex, tuple[complex, complex]]]


@dataclass(frozen=True)
class FEMHelmholtzDtN:
    """Dense Schur-complement Helmholtz DtN comparator.

    This is intentionally a baseline, not part of the matrix-free Q engine.
    It stores the boundary Schur complement for each tested wavenumber.
    """

    boundary_mass: np.ndarray
    schur: np.ndarray
    build_ms: float
    node_count: int
    triangle_count: int
    radial_levels: int

    def apply_dtn(self, values: Iterable[complex]) -> np.ndarray:
        vector = np.asarray(tuple(values), dtype=np.complex128)
        return (self.schur @ vector) / self.boundary_mass


def timed(fn):
    start = perf_counter()
    value = fn()
    return value, 1000.0 * (perf_counter() - start)


def assemble_p1_mass(mesh) -> csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for triangle in mesh.triangles:
        x = mesh.nodes[triangle, 0]
        y = mesh.nodes[triangle, 1]
        area = abs(float(0.5 * ((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))))
        if area <= 1.0e-30:
            raise ValueError("radial fan mesh has a degenerate triangle")
        for local_row in range(3):
            for local_col in range(3):
                rows.append(int(triangle[local_row]))
                cols.append(int(triangle[local_col]))
                data.append(area * (2.0 if local_row == local_col else 1.0) / 12.0)
    return coo_matrix((data, (rows, cols)), shape=(mesh.node_count, mesh.node_count)).tocsr()


def build_fem_helmholtz_dtn(
    boundary: Iterable[tuple[float, float]],
    k: float,
    *,
    radial_levels: int,
) -> FEMHelmholtzDtN:
    """Build true interior Helmholtz FEM DtN for ``Delta u + k^2 u = 0``."""

    def build():
        mesh = build_star_fan_mesh(boundary, radial_levels=radial_levels)
        stiffness = assemble_p1_stiffness(mesh)
        mass_matrix = assemble_p1_mass(mesh)
        system = (stiffness - (k * k) * mass_matrix).astype(np.complex128)
        boundary_nodes = mesh.boundary_nodes
        boundary_mask = np.zeros(mesh.node_count, dtype=bool)
        boundary_mask[boundary_nodes] = True
        interior_nodes = np.where(~boundary_mask)[0]
        interior_interior = system[interior_nodes][:, interior_nodes].tocsc()
        interior_boundary = system[interior_nodes][:, boundary_nodes]
        boundary_interior = system[boundary_nodes][:, interior_nodes]
        boundary_boundary = system[boundary_nodes][:, boundary_nodes]
        factor = splu(interior_interior)
        solved = factor.solve(interior_boundary.toarray())
        schur = boundary_boundary.toarray() - boundary_interior.toarray() @ solved
        schur = 0.5 * (schur + schur.T)
        return mesh, schur

    (mesh, schur), build_ms = timed(build)
    return FEMHelmholtzDtN(
        boundary_mass=boundary_lumped_mass(boundary),
        schur=np.asarray(schur, dtype=np.complex128),
        build_ms=build_ms,
        node_count=mesh.node_count,
        triangle_count=mesh.triangle_count,
        radial_levels=mesh.radial_levels,
    )


def disk_points(n: int, radius: float = 1.0) -> tuple[tuple[float, float], ...]:
    return tuple(
        (radius * math.cos(TAU * index / n), radius * math.sin(TAU * index / n))
        for index in range(n)
    )


def radial_points(n: int, base: float, cos_coefficients=(), sin_coefficients=()):
    return build_boundary_pullback_qjet(
        n,
        radial_fourier_qjet_map(
            base,
            cos_coefficients=cos_coefficients,
            sin_coefficients=sin_coefficients,
        ),
    ).points


def ellipse_points(n: int, a: float, b: float):
    return build_boundary_pullback_qjet(n, ellipse_qjet_map(a, b)).points


def square_points(n: int) -> tuple[tuple[float, float], ...]:
    out = []
    per_side = n // 4
    for side in range(4):
        for local in range(per_side):
            t = local / per_side
            if side == 0:
                out.append((1.0, -1.0 + 2.0 * t))
            elif side == 1:
                out.append((1.0 - 2.0 * t, 1.0))
            elif side == 2:
                out.append((-1.0, 1.0 - 2.0 * t))
            else:
                out.append((-1.0 + 2.0 * t, -1.0))
    return tuple(out[:n])


def cardioid_points(n: int) -> tuple[tuple[float, float], ...]:
    out = []
    for index in range(n):
        theta = TAU * index / n
        radius = 1.0 - 0.82 * math.cos(theta)
        out.append((radius * math.cos(theta), radius * math.sin(theta)))
    return tuple(out)


def rounded_square_points(n: int, exponent: float = 4.0) -> tuple[tuple[float, float], ...]:
    out = []
    for index in range(n):
        theta = TAU * index / n
        c = math.cos(theta)
        s = math.sin(theta)
        denom = (abs(c) ** exponent + abs(s) ** exponent) ** (1.0 / exponent)
        out.append((c / denom, s / denom))
    return tuple(out)


def shape_suite():
    return (
        ("ellipse_3_to_1", "smooth", lambda n: ellipse_points(n, 3.0, 1.0)),
        (
            "funky_flower_curve",
            "smooth_nonconvex",
            lambda n: radial_points(
                n,
                1.0,
                cos_coefficients=(0.0, 0.28, 0.0, -0.09),
                sin_coefficients=(0.0, 0.0, 0.06),
            ),
        ),
        ("rounded_square_superellipse", "smooth_high_curvature", rounded_square_points),
        ("square_polygon", "corner", square_points),
        ("cardioid_cusp", "cusp_endpoint", cardioid_points),
    )


def mode_values(n: int, mode: int) -> list[float]:
    if mode == 0:
        return [1.0 for _ in range(n)]
    return [math.cos(mode * TAU * index / n) for index in range(n)]


def project_cosine_amplitude(values: Iterable[complex], mode: int) -> complex:
    vector = tuple(complex(value) for value in values)
    n = len(vector)
    basis = mode_values(n, mode)
    numerator = sum(vector[index] * basis[index] for index in range(n))
    denominator = sum(value * value for value in basis)
    return numerator / denominator


def helmholtz_dtn_lambda(order: float, k: float, kind: str) -> complex:
    if kind == "interior":
        denominator = special.jv(order, k)
        return complex(k * special.jvp(order, k) / denominator)
    if kind == "exterior_outgoing":
        denominator = special.hankel1(order, k)
        return complex(k * special.h1vp(order, k) / denominator)
    raise ValueError(f"unknown disk Helmholtz kind: {kind}")


def helmholtz_denominator_abs(order: float, k: float, kind: str) -> float:
    if kind == "interior":
        return float(abs(special.jv(order, k)))
    if kind == "exterior_outgoing":
        return float(abs(special.hankel1(order, k)))
    raise ValueError(kind)


def q_cycle_helmholtz_dtn_lambda(n: int, mode: int, k: float, kind: str) -> complex:
    order = cycle_dtn_eigenvalue(mode, n)
    return helmholtz_dtn_lambda(order, k, kind)


def q_repaid_helmholtz_dtn_lambda(n: int, mode: int, k: float, kind: str) -> complex:
    order = continuum_repaid_dtn_eigenvalue(mode, n)
    return helmholtz_dtn_lambda(order, k, kind)


def relative_error(value: complex, reference: complex) -> float:
    return abs(value - reference) / max(abs(reference), 1.0e-14)


def complex_parts(value: complex) -> dict[str, float]:
    z_value = complex(value)
    return {"real": float(z_value.real), "imag": float(z_value.imag)}


def run_disk_modal_suite() -> list[dict[str, object]]:
    n_q = 8192
    n_boundary = 128
    radial_levels = 44
    modes = (0, 1, 2, 4, 8)
    rows: list[dict[str, object]] = []
    points = disk_points(n_boundary)
    mass = boundary_lumped_mass(points)
    first_zeros = {mode: float(special.jn_zeros(mode, 1)[0]) for mode in modes}
    case_specs: list[tuple[str, Callable[[int], float]]] = [
        ("low_frequency", lambda mode: 0.75 + 0.08 * mode),
        ("moderate_frequency", lambda mode: 3.7 + 0.05 * mode),
        ("near_interior_dirichlet_pole", lambda mode: first_zeros[mode] + 0.02),
    ]
    fem_by_k: dict[float, FEMHelmholtzDtN] = {}
    for mode in modes:
        values = mode_values(n_boundary, mode)
        for case_name, k_fn in case_specs:
            k = float(k_fn(mode))
            for kind in ("interior", "exterior_outgoing"):
                exact = helmholtz_dtn_lambda(float(mode), k, kind)
                q_value = q_repaid_helmholtz_dtn_lambda(n_q, mode, k, kind)
                raw_cycle_q_value = q_cycle_helmholtz_dtn_lambda(n_q, mode, k, kind)
                fem_value = None
                fem_error = None
                fem_build_ms = None
                if kind == "interior":
                    fem = fem_by_k.get(k)
                    if fem is None:
                        fem = build_fem_helmholtz_dtn(points, k, radial_levels=radial_levels)
                        fem_by_k[k] = fem
                    fem_flux = fem.apply_dtn(values)
                    fem_value = project_cosine_amplitude(fem_flux, mode)
                    fem_error = relative_error(fem_value, exact)
                    fem_build_ms = fem.build_ms
                rows.append(
                    {
                        "suite": "disk_modal_exact",
                        "case": case_name,
                        "kind": kind,
                        "mode": mode,
                        "wavenumber": k,
                        "q_boundary_samples": n_q,
                        "fem_boundary_samples": n_boundary if kind == "interior" else None,
                        "fem_radial_levels": radial_levels if kind == "interior" else None,
                        "fem_nodes": fem_by_k[k].node_count if kind == "interior" else None,
                        "fem_triangles": fem_by_k[k].triangle_count if kind == "interior" else None,
                        "exact_amplitude": complex_parts(exact),
                        "q_spectral_amplitude": complex_parts(q_value),
                        "q_raw_cycle_amplitude": complex_parts(raw_cycle_q_value),
                        "fem_amplitude": None if fem_value is None else complex_parts(fem_value),
                        "q_spectral_relative_error": relative_error(q_value, exact),
                        "q_raw_cycle_relative_error": relative_error(raw_cycle_q_value, exact),
                        "fem_relative_error": fem_error,
                        "winner": (
                            "q_spectral"
                            if fem_error is None or relative_error(q_value, exact) <= fem_error
                            else "fem"
                        ),
                        "q_order": continuum_repaid_dtn_eigenvalue(mode, n_q),
                        "q_raw_cycle_order": cycle_dtn_eigenvalue(mode, n_q),
                        "exact_order": float(mode),
                        "denominator_abs": helmholtz_denominator_abs(float(mode), k, kind),
                        "nearest_known_interior_bessel_zero": first_zeros[mode],
                        "distance_to_first_interior_bessel_zero": abs(k - first_zeros[mode]),
                        "q_dense_matrix_stored": False,
                        "fem_dense_schur_stored": kind == "interior",
                        "fem_build_ms": fem_build_ms,
                    }
                )
    return rows


def edge_outward_normal(left, right, orientation: float) -> tuple[tuple[float, float], float]:
    dx = float(right[0] - left[0])
    dy = float(right[1] - left[1])
    length = math.hypot(dx, dy)
    if length <= 0.0:
        raise ValueError("duplicate adjacent boundary points")
    if orientation >= 0.0:
        return (dy / length, -dx / length), length
    return (-dy / length, dx / length), length


def gauss_legendre_8() -> tuple[tuple[float, ...], tuple[float, ...]]:
    return (
        (
            0.019855071751231884,
            0.10166676129318664,
            0.2372337950418355,
            0.4082826787521751,
            0.5917173212478249,
            0.7627662049581645,
            0.8983332387068134,
            0.9801449282487681,
        ),
        (
            0.05061426814518813,
            0.11119051722668724,
            0.15685332293894363,
            0.18134189168918099,
            0.18134189168918099,
            0.15685332293894363,
            0.11119051722668724,
            0.05061426814518813,
        ),
    )


def boundary_values(points, field: ComplexField) -> list[complex]:
    pts = as_points(points)
    values = []
    for x, y in pts:
        value, _ = field(float(x), float(y))
        values.append(value)
    return values


def exact_weak_flux(points, field: ComplexField) -> np.ndarray:
    pts = as_points(points)
    orientation = 1.0 if polygon_area(pts) >= 0.0 else -1.0
    out = np.zeros(len(pts), dtype=np.complex128)
    gauss_x, gauss_w = gauss_legendre_8()
    for index in range(len(pts)):
        left = pts[index]
        right = pts[(index + 1) % len(pts)]
        normal, length = edge_outward_normal(left, right, orientation)
        for s, weight in zip(gauss_x, gauss_w, strict=True):
            point = (1.0 - s) * left + s * right
            _, gradient = field(float(point[0]), float(point[1]))
            normal_flux = gradient[0] * normal[0] + gradient[1] * normal[1]
            contribution = weight * length * normal_flux
            out[index] += contribution * (1.0 - s)
            out[(index + 1) % len(pts)] += contribution * s
    return out


def plane_wave_field(k: float, angle: float) -> ComplexField:
    direction = (math.cos(angle), math.sin(angle))

    def field(x: float, y: float):
        phase = k * (direction[0] * x + direction[1] * y)
        value = complex(math.cos(phase), math.sin(phase))
        gradient = (1j * k * direction[0] * value, 1j * k * direction[1] * value)
        return value, gradient

    return field


def point_source_field(k: float, source: tuple[float, float]) -> ComplexField:
    sx, sy = source

    def field(x: float, y: float):
        dx = x - sx
        dy = y - sy
        radius = math.hypot(dx, dy)
        if radius <= 0.0:
            raise ValueError("point source lies on the boundary")
        value = complex(special.hankel1(0, k * radius))
        radial_derivative = -k * complex(special.hankel1(1, k * radius))
        scale = radial_derivative / radius
        return value, (scale * dx, scale * dy)

    return field


def perimeter(points: Iterable[tuple[float, float]]) -> float:
    pts = as_points(points)
    total = 0.0
    for index in range(len(pts)):
        left = pts[index]
        right = pts[(index + 1) % len(pts)]
        total += math.hypot(float(left[0] - right[0]), float(left[1] - right[1]))
    return total


def run_manufactured_suite() -> list[dict[str, object]]:
    n_boundary = 96
    radial_levels = 32
    wavenumbers = (0.75, 2.5, 5.0, 8.0)
    rows: list[dict[str, object]] = []
    for shape, family, make_points in shape_suite():
        points = make_points(n_boundary)
        mass = boundary_lumped_mass(points)
        length = perimeter(points)
        for k in wavenumbers:
            direction_count = 8 if k < 1.0 else 16 if k < 4.0 else 32
            uniform_directions = tuple(TAU * index / direction_count for index in range(direction_count))
            plane_wave_directions = (0.37,) + uniform_directions
            qjet, q_build_ms = timed(
                lambda points=points, k=k: build_helmholtz_moment_corrected_planar_qjet(
                    points,
                    k,
                    moment_degree=2,
                    zeta_tail_degree=None,
                    plane_wave_directions=plane_wave_directions,
                    orthogonalization_tolerance=1.0e-8,
                )
            )
            fem = build_fem_helmholtz_dtn(points, k, radial_levels=radial_levels)
            fields = (
                ("plane_wave", plane_wave_field(k, angle=0.37)),
                ("exterior_point_source", point_source_field(k, source=(5.0, 3.6))),
            )
            for field_name, field in fields:
                values = boundary_values(points, field)
                exact_flux = exact_weak_flux(points, field) / mass
                q_result, q_ms = timed(lambda values=values, qjet=qjet: qjet.apply_helmholtz_dtn(values))
                fem_flux, fem_apply_ms = timed(lambda values=values, fem=fem: fem.apply_dtn(values))
                q_error = relative_weighted_l2(q_result.values, exact_flux, mass)
                fem_error = relative_weighted_l2(fem_flux, exact_flux, mass)
                rows.append(
                    {
                        "suite": "manufactured_funky_domain",
                        "shape": shape,
                        "family": family,
                        "field": field_name,
                        "n_boundary": n_boundary,
                        "wavenumber": k,
                        "points_per_wavelength_on_boundary": (TAU / k) / (length / n_boundary),
                        "q_method": "helmholtz_plane_wave_reproduction_repaid_qjet",
                        "fem_method": "true_volumetric_p1_helmholtz_schur_dtn",
                        "truth": "analytic manufactured Helmholtz field exact weak boundary flux",
                        "q_relative_l2_to_exact": q_error,
                        "fem_relative_l2_to_exact": fem_error,
                        "winner": "q_helmholtz" if q_error <= fem_error else "fem",
                        "q_error_over_fem_error": q_error / max(fem_error, 1.0e-14),
                        "q_status": q_result.ledger.status,
                        "q_error_type": q_result.stats.get("q_error_type"),
                        "q_recommended_q": q_result.stats.get("recommended_q"),
                        "correction_rank": q_result.stats.get("correction_rank"),
                        "helmholtz_correction_rank": q_result.stats.get("helmholtz_correction_rank"),
                        "helmholtz_plane_wave_direction_count": len(plane_wave_directions),
                        "zeta_tail_rank": q_result.stats.get("zeta_tail_rank"),
                        "q_build_ms": q_build_ms,
                        "q_apply_ms": q_ms,
                        "q_work_units": q_result.work_units,
                        "fem_build_ms": fem.build_ms,
                        "fem_apply_ms": fem_apply_ms,
                        "fem_nodes": fem.node_count,
                        "fem_triangles": fem.triangle_count,
                        "fem_radial_levels": fem.radial_levels,
                        "q_dense_matrix_stored": False,
                        "fem_dense_schur_stored": True,
                    }
                )
    return rows


def median(values: Iterable[float | None]) -> float | None:
    clean = sorted(float(value) for value in values if value is not None and math.isfinite(float(value)))
    if not clean:
        return None
    mid = len(clean) // 2
    if len(clean) % 2:
        return clean[mid]
    return 0.5 * (clean[mid - 1] + clean[mid])


def summarize(disk_rows: list[dict[str, object]], manufactured_rows: list[dict[str, object]]) -> dict[str, object]:
    disk_interior = [row for row in disk_rows if row["kind"] == "interior"]
    manufactured_fem_wins = [row for row in manufactured_rows if row["winner"] == "fem"]
    manufactured_q_wins = [row for row in manufactured_rows if row["winner"] == "q_helmholtz"]
    disk_fem_wins = [row for row in disk_interior if row["winner"] == "fem"]
    disk_q_wins = [row for row in disk_interior if row["winner"] == "q_spectral"]
    return {
        "disk_case_count": len(disk_rows),
        "disk_interior_case_count": len(disk_interior),
        "disk_q_spectral_win_count": len(disk_q_wins),
        "disk_fem_win_count": len(disk_fem_wins),
        "disk_median_q_spectral_relative_error": median(
            row["q_spectral_relative_error"] for row in disk_rows
        ),
        "disk_median_fem_relative_error": median(row["fem_relative_error"] for row in disk_interior),
        "disk_max_q_spectral_relative_error": max(row["q_spectral_relative_error"] for row in disk_rows),
        "disk_max_fem_relative_error": max(row["fem_relative_error"] for row in disk_interior),
        "manufactured_case_count": len(manufactured_rows),
        "manufactured_q_helmholtz_win_count": len(manufactured_q_wins),
        "manufactured_fem_win_count": len(manufactured_fem_wins),
        "manufactured_median_q_relative_l2_to_exact": median(
            row["q_relative_l2_to_exact"] for row in manufactured_rows
        ),
        "manufactured_median_fem_relative_l2_to_exact": median(
            row["fem_relative_l2_to_exact"] for row in manufactured_rows
        ),
        "manufactured_max_q_relative_l2_to_exact": max(
            row["q_relative_l2_to_exact"] for row in manufactured_rows
        ),
        "manufactured_max_fem_relative_l2_to_exact": max(
            row["fem_relative_l2_to_exact"] for row in manufactured_rows
        ),
        "top_disk_fem_wins": sorted(
            disk_fem_wins,
            key=lambda row: row["q_spectral_relative_error"] / max(row["fem_relative_error"], 1.0e-14),
            reverse=True,
        )[:8],
        "top_manufactured_fem_wins": sorted(
            manufactured_fem_wins,
            key=lambda row: row["q_error_over_fem_error"],
            reverse=True,
        )[:8],
    }


def main() -> int:
    disk_rows = run_disk_modal_suite()
    manufactured_rows = run_manufactured_suite()
    payload = {
        "parameters": {
            "disk_truth": "interior k J_m'(kR)/J_m(kR), exterior outgoing k H_m^{(1)'}(kR)/H_m^{(1)}(kR)",
            "manufactured_truth": "plane waves and exterior Hankel point sources integrated as weak normal flux",
            "fem_truth_role": "secondary comparator; true Helmholtz P1 Schur complement, not ground truth",
            "held_out_benchmark_registry_id": "dlmf_disk_bessel_helmholtz",
            "held_out_reference_url": "https://dlmf.nist.gov/10",
            "q_disk_role": "matrix-free repaid Q spectral Helmholtz DtN with Bessel/Hankel channel multipliers; raw finite-cycle order is a diagnostic only",
            "q_funky_role": "matrix-free arbitrary-domain Q principal DtN plus exact plane-wave Helmholtz reproduction repayment",
        },
        "summary": summarize(disk_rows, manufactured_rows),
        "disk_modal_rows": disk_rows,
        "manufactured_rows": manufactured_rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    printable = {
        "output": str(OUT),
        "disk_q_spectral_wins": payload["summary"]["disk_q_spectral_win_count"],
        "disk_fem_wins": payload["summary"]["disk_fem_win_count"],
        "manufactured_q_helmholtz_wins": payload["summary"]["manufactured_q_helmholtz_win_count"],
        "manufactured_fem_wins": payload["summary"]["manufactured_fem_win_count"],
        "disk_median_q": payload["summary"]["disk_median_q_spectral_relative_error"],
        "disk_median_fem": payload["summary"]["disk_median_fem_relative_error"],
        "manufactured_median_q": payload["summary"]["manufactured_median_q_relative_l2_to_exact"],
        "manufactured_median_fem": payload["summary"]["manufactured_median_fem_relative_l2_to_exact"],
    }
    print(json.dumps(printable, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
