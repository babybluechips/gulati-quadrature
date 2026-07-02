#!/usr/bin/env python3
"""Boundary-only Q/DtN PDE operators versus volumetric P1 FEM baseline."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from inverse_shape.q_dtn import (
    apply_continuum_repaid_dtn,
    apply_cycle_dtn,
    continuum_repaid_dtn_heat,
    continuum_repaid_dtn_helmholtz_resolvent,
    continuum_repaid_dtn_poisson_solve,
    continuum_repaid_dtn_wave,
    exact_disk_amplitude,
    q_cycle_disk_amplitude,
    q_disk_amplitude,
    relative_error,
)

ROOT = Path(__file__).resolve().parents[1]
TAU = 2.0 * math.pi


def timed(fn):
    start = time.perf_counter()
    value = fn()
    return value, 1000.0 * (time.perf_counter() - start)


def cosine_mode(n, mode):
    return [math.cos(TAU * mode * index / n) for index in range(n)]


def projected_amplitude(values, mode):
    n = len(values)
    basis = cosine_mode(n, mode)
    numerator = sum(complex(values[index]) * basis[index] for index in range(n))
    denominator = sum(value * value for value in basis)
    return numerator / denominator


def boundary_value(mode, x, y):
    theta = math.atan2(y, x)
    return math.cos(mode * theta)


def polar_node_index(radial_level, angular_index, angular_segments):
    if radial_level == 0:
        return 0
    return 1 + (radial_level - 1) * angular_segments + (angular_index % angular_segments)


def fem_p1_disk_dtn_eigenvalue(mode, radial_levels, angular_segments):
    nodes = [(0.0, 0.0)]
    for radial_level in range(1, radial_levels + 1):
        radius = radial_level / radial_levels
        for angular_index in range(angular_segments):
            theta = TAU * angular_index / angular_segments
            nodes.append((radius * math.cos(theta), radius * math.sin(theta)))

    triangles = []
    for angular_index in range(angular_segments):
        triangles.append(
            (
                0,
                polar_node_index(1, angular_index, angular_segments),
                polar_node_index(1, angular_index + 1, angular_segments),
            )
        )
    for radial_level in range(1, radial_levels):
        for angular_index in range(angular_segments):
            lower_left = polar_node_index(radial_level, angular_index, angular_segments)
            lower_right = polar_node_index(radial_level, angular_index + 1, angular_segments)
            upper_left = polar_node_index(radial_level + 1, angular_index, angular_segments)
            upper_right = polar_node_index(radial_level + 1, angular_index + 1, angular_segments)
            triangles.append((lower_left, upper_left, upper_right))
            triangles.append((lower_left, upper_right, lower_right))

    rows = []
    cols = []
    data = []
    for triangle in triangles:
        x = [nodes[index][0] for index in triangle]
        y = [nodes[index][1] for index in triangle]
        area = 0.5 * ((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))
        if area < 0.0:
            area = -area
        b = [y[1] - y[2], y[2] - y[0], y[0] - y[1]]
        c = [x[2] - x[1], x[0] - x[2], x[1] - x[0]]
        for local_row in range(3):
            for local_col in range(3):
                rows.append(triangle[local_row])
                cols.append(triangle[local_col])
                data.append((b[local_row] * b[local_col] + c[local_row] * c[local_col]) / (4.0 * area))

    node_count = len(nodes)
    stiffness = coo_matrix((data, (rows, cols)), shape=(node_count, node_count)).tocsr()
    boundary_nodes = [polar_node_index(radial_levels, index, angular_segments) for index in range(angular_segments)]
    boundary_mask = np.zeros(node_count, dtype=bool)
    boundary_mask[boundary_nodes] = True
    interior_nodes = np.where(~boundary_mask)[0]
    boundary_values = np.zeros(node_count, dtype=np.float64)
    for angular_index, node in enumerate(boundary_nodes):
        theta = TAU * angular_index / angular_segments
        boundary_values[node] = math.cos(mode * theta)

    interior_stiffness = stiffness[interior_nodes[:, None], interior_nodes]
    boundary_stiffness = stiffness[interior_nodes[:, None], np.asarray(boundary_nodes)]
    rhs = -(boundary_stiffness @ boundary_values[boundary_nodes])
    interior_values = spsolve(interior_stiffness, rhs)
    values = boundary_values.copy()
    values[interior_nodes] = interior_values
    energy = float(values @ (stiffness @ values))
    return energy / math.pi, node_count, len(triangles)


def fem_amplitude(problem, mu, parameters):
    if problem == "laplace_dtn":
        return complex(mu)
    if problem == "heat":
        return complex(math.exp(-parameters["time"] * mu))
    if problem == "poisson":
        return complex(1.0 / (mu + parameters["mass"]))
    if problem == "helmholtz":
        return 1.0 / (mu * mu - parameters["wavenumber"] * parameters["wavenumber"] + 1j * parameters["damping"])
    if problem == "wave":
        return complex(math.cos(parameters["time"] * math.sqrt(mu)))
    raise ValueError(problem)


def q_operator_amplitude(problem, n, mode, parameters):
    values = cosine_mode(n, mode)
    if problem == "laplace_dtn":
        return projected_amplitude(apply_continuum_repaid_dtn(values), mode)
    if problem == "heat":
        return projected_amplitude(continuum_repaid_dtn_heat(values, parameters["time"]), mode)
    if problem == "poisson":
        return projected_amplitude(continuum_repaid_dtn_poisson_solve(values, mass=parameters["mass"]), mode)
    if problem == "helmholtz":
        return projected_amplitude(
            continuum_repaid_dtn_helmholtz_resolvent(
                values,
                parameters["wavenumber"],
                damping=parameters["damping"],
            ),
            mode,
        )
    if problem == "wave":
        return projected_amplitude(continuum_repaid_dtn_wave(values, parameters["time"]), mode)
    raise ValueError(problem)


def q_cycle_operator_amplitude(problem, n, mode, parameters):
    values = cosine_mode(n, mode)
    if problem == "laplace_dtn":
        return projected_amplitude(apply_cycle_dtn(values), mode)
    if problem == "heat":
        from inverse_shape.q_dtn import cycle_dtn_heat

        return projected_amplitude(cycle_dtn_heat(values, parameters["time"]), mode)
    if problem == "poisson":
        from inverse_shape.q_dtn import cycle_dtn_poisson_solve

        return projected_amplitude(cycle_dtn_poisson_solve(values, mass=parameters["mass"]), mode)
    if problem == "helmholtz":
        from inverse_shape.q_dtn import cycle_dtn_helmholtz_resolvent

        return projected_amplitude(
            cycle_dtn_helmholtz_resolvent(
                values,
                parameters["wavenumber"],
                damping=parameters["damping"],
            ),
            mode,
        )
    if problem == "wave":
        from inverse_shape.q_dtn import cycle_dtn_wave

        return projected_amplitude(cycle_dtn_wave(values, parameters["time"]), mode)
    raise ValueError(problem)


def main():
    n_boundary = 8192
    fem_radial_levels = 40
    fem_angular_segments = 160
    modes = (1, 2, 4, 8, 12, 16, 24)
    problem_parameters = {
        "laplace_dtn": {},
        "heat": {"time": 0.17},
        "poisson": {"mass": 0.35},
        "helmholtz": {"wavenumber": 3.7, "damping": 0.02},
        "wave": {"time": 0.8},
    }

    fem_modes = {}
    for mode in modes:
        fem_result, elapsed_ms = timed(
            lambda mode=mode: fem_p1_disk_dtn_eigenvalue(
                mode,
                fem_radial_levels,
                fem_angular_segments,
            )
        )
        mu, node_count, triangle_count = fem_result
        fem_modes[mode] = {
            "mu": mu,
            "ms": elapsed_ms,
            "node_count": node_count,
            "triangle_count": triangle_count,
        }

    rows = []
    for mode in modes:
        for problem, parameters in problem_parameters.items():
            q_value, q_ms = timed(lambda problem=problem, mode=mode, parameters=parameters: q_operator_amplitude(problem, n_boundary, mode, parameters))
            q_formula, q_formula_ms = timed(lambda problem=problem, mode=mode, parameters=parameters: q_disk_amplitude(problem, mode, n_boundary, **parameters))
            q_cycle_value, q_cycle_ms = timed(lambda problem=problem, mode=mode, parameters=parameters: q_cycle_operator_amplitude(problem, n_boundary, mode, parameters))
            q_cycle_formula = q_cycle_disk_amplitude(problem, mode, n_boundary, **parameters)
            exact = exact_disk_amplitude(problem, mode, **parameters)
            fem_value = fem_amplitude(problem, fem_modes[mode]["mu"], parameters)
            rows.append(
                {
                    "mode": mode,
                    "problem": problem,
                    "q_operator_amplitude_real": float(complex(q_value).real),
                    "q_operator_amplitude_imag": float(complex(q_value).imag),
                    "q_formula_amplitude_real": float(complex(q_formula).real),
                    "q_formula_amplitude_imag": float(complex(q_formula).imag),
                    "q_cycle_operator_amplitude_real": float(complex(q_cycle_value).real),
                    "q_cycle_operator_amplitude_imag": float(complex(q_cycle_value).imag),
                    "q_cycle_formula_amplitude_real": float(complex(q_cycle_formula).real),
                    "q_cycle_formula_amplitude_imag": float(complex(q_cycle_formula).imag),
                    "fem_amplitude_real": float(complex(fem_value).real),
                    "fem_amplitude_imag": float(complex(fem_value).imag),
                    "exact_amplitude_real": float(complex(exact).real),
                    "exact_amplitude_imag": float(complex(exact).imag),
                    "q_operator_relative_error": relative_error(q_value, exact),
                    "q_formula_relative_error": relative_error(q_formula, exact),
                    "q_cycle_operator_relative_error": relative_error(q_cycle_value, exact),
                    "q_cycle_formula_relative_error": relative_error(q_cycle_formula, exact),
                    "fem_relative_error": relative_error(fem_value, exact),
                    "q_operator_ms": q_ms,
                    "q_formula_ms": q_formula_ms,
                    "q_cycle_operator_ms": q_cycle_ms,
                    "fem_ms": fem_modes[mode]["ms"],
                    "q_boundary_samples": n_boundary,
                    "fem_radial_levels": fem_radial_levels,
                    "fem_angular_segments": fem_angular_segments,
                    "fem_node_count": fem_modes[mode]["node_count"],
                    "fem_triangle_count": fem_modes[mode]["triangle_count"],
                    "fem_dtn_mu": fem_modes[mode]["mu"],
                    "exact_dtn_mu": float(mode),
                }
            )

    def median(values):
        ordered = sorted(values)
        mid = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[mid]
        return 0.5 * (ordered[mid - 1] + ordered[mid])

    summary = {
        "case_count": len(rows),
        "mode_count": len(modes),
        "problem_count": len(problem_parameters),
        "median_q_operator_relative_error": median([row["q_operator_relative_error"] for row in rows]),
        "median_q_formula_relative_error": median([row["q_formula_relative_error"] for row in rows]),
        "median_fem_relative_error": median([row["fem_relative_error"] for row in rows]),
        "median_q_cycle_operator_relative_error": median([row["q_cycle_operator_relative_error"] for row in rows]),
        "median_q_cycle_formula_relative_error": median([row["q_cycle_formula_relative_error"] for row in rows]),
        "max_q_operator_relative_error": max(row["q_operator_relative_error"] for row in rows),
        "max_q_formula_relative_error": max(row["q_formula_relative_error"] for row in rows),
        "max_q_cycle_operator_relative_error": max(row["q_cycle_operator_relative_error"] for row in rows),
        "max_q_cycle_formula_relative_error": max(row["q_cycle_formula_relative_error"] for row in rows),
        "max_fem_relative_error": max(row["fem_relative_error"] for row in rows),
        "median_q_operator_ms": median([row["q_operator_ms"] for row in rows]),
        "median_q_formula_ms": median([row["q_formula_ms"] for row in rows]),
        "median_fem_ms": median([row["fem_ms"] for row in rows]),
        "median_speedup_q_operator_vs_fem": median([row["fem_ms"] / max(row["q_operator_ms"], 1.0e-12) for row in rows]),
        "median_speedup_q_formula_vs_fem": median([row["fem_ms"] / max(row["q_formula_ms"], 1.0e-12) for row in rows]),
    }
    payload = {
        "parameters": {
            "boundary_samples": n_boundary,
            "fem_radial_levels": fem_radial_levels,
            "fem_angular_segments": fem_angular_segments,
            "modes": list(modes),
            "problem_parameters": problem_parameters,
            "q_dtn_normalization": "raw cycle Lambda_Q=(h/pi)Q has m-m^2/n dispersion; production Q repays the endpoint defect to |m| before exact continuum claims",
            "fem_baseline": "volumetric P1 triangular FEM harmonic solves; DtN eigenvalues extracted by energy quotients",
            "reference_policy": "exact disk modal amplitudes are ground truth; production Q uses continuum-repaid Q spectrum, raw cycle Q is reported only as a finite-n diagnostic",
            "held_out_benchmark_registry_ids": ["unit_disk_steklov_dtn", "dlmf_disk_bessel_helmholtz"],
            "held_out_reference_urls": [
                "https://dms.umontreal.ca/~iossif/steklov_spectral_geometry.pdf",
                "https://dlmf.nist.gov/10",
            ],
            "fem_truth_role": "competitor baseline only, not ground truth",
        },
        "summary": summary,
        "fem_modes": fem_modes,
        "rows": rows,
    }
    output = ROOT / "docs" / "assets" / "q_dtn_vs_fem_benchmark.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("problem,mode,q_rel,fem_rel,q_ms,fem_ms,speedup")
    for row in rows:
        print(
            f"{row['problem']},{row['mode']},"
            f"{row['q_operator_relative_error']:.6e},{row['fem_relative_error']:.6e},"
            f"{row['q_operator_ms']:.6e},{row['fem_ms']:.6e},"
            f"{row['fem_ms'] / max(row['q_operator_ms'], 1.0e-12):.2f}"
        )
    print("summary=" + json.dumps(summary, sort_keys=True))
    print(f"json={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
