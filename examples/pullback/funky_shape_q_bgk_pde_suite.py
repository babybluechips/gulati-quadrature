#!/usr/bin/env python3
"""Funky-shape Q+BGK ladder and boundary-only PDE suite.

This is the wider test of the combined slick path:

1. Represent each analytic funky shape by an exact exterior Laurent map
   ``psi(w) = c_1 w + c_0 + c_{-1} w^-1 + ...``.
2. Borrow the exterior circle coordinate.
3. Compute the log singularity by the Q/Fourier primitive.
4. Repay the analytic quotient kernel.
5. Repay the BGK endpoint displacement by a zeta/Taylor derivative ladder.
6. Push the same conformal-coordinate boundary data through the PDE pipeline:
   Laplace DtN, heat, Poisson/Steklov solve, Helmholtz resolvent, and wave.

The suite stores only Laurent/QJet generators, samples, and small summary
tables.  No dense Q matrix is built.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from inverse_shape.q_dtn import (  # noqa: E402
    ScalarQJet,
    build_boundary_pullback_qjet,
    cycle_dtn_eigenvalue,
    exact_disk_amplitude,
    qjet_cos,
    qjet_sin,
)


OUT = ROOT / "outputs" / "funky_shape_q_bgk_pde_suite"
TAU = 2.0 * math.pi
ZETA_HALF = -1.4603545088095868128894991525152980124672293310126
BGK_BETA = -ZETA_HALF / math.sqrt(2.0 * math.pi)
ERROR_FLOOR = 1.0e-16
CORRECTION_ORDERS = (0, 1, 2, 4, 8)


@dataclass(frozen=True)
class LaurentShape:
    name: str
    family: str
    coeffs: tuple[tuple[int, complex], ...]
    target_phase: float = 0.73

    def psi(self, w: complex) -> complex:
        return sum(coeff * (w**power) for power, coeff in self.coeffs)

    def dpsi(self, w: complex) -> complex:
        return sum(power * coeff * (w ** (power - 1)) for power, coeff in self.coeffs if power != 0)


def unit(theta: float) -> complex:
    return complex(math.cos(theta), math.sin(theta))


def fft(values: list[complex]) -> list[complex]:
    n = len(values)
    if n == 0 or n & (n - 1):
        raise ValueError("FFT length must be a positive power of two")
    data = list(values)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            data[i], data[j] = data[j], data[i]
    length = 2
    while length <= n:
        root = unit(-TAU / length)
        half = length // 2
        for start in range(0, n, length):
            twiddle = 1.0 + 0.0j
            for k in range(start, start + half):
                even = data[k]
                odd = data[k + half] * twiddle
                data[k] = even + odd
                data[k + half] = even - odd
                twiddle *= root
        length *= 2
    return data


def rising_factorial(base: int, order: int) -> float:
    value = 1.0
    for index in range(order):
        value *= base + index
    return value


def log_derivatives(values: list[complex]) -> list[complex]:
    max_order = len(values) - 1
    out = [0.0j for _ in range(max_order + 1)]
    out[0] = complex(math.log(max(abs(values[0]), 1.0e-300)), math.atan2(values[0].imag, values[0].real))
    for order in range(max_order):
        residual = 0.0j
        for split in range(1, order + 1):
            residual += math.comb(order, split) * values[split] * out[order + 1 - split]
        out[order + 1] = (values[order + 1] - residual) / values[0]
    return out


def density(theta: float) -> float:
    return math.exp(0.35 * math.cos(2.0 * theta - 0.2) + 0.15 * math.sin(3.0 * theta + 0.4)) * (
        1.0 + 0.07 * math.cos(5.0 * theta + 0.1)
    )


def sample_pullback(shape: LaurentShape, n: int) -> tuple[list[float], list[complex]]:
    weights: list[float] = []
    nodes: list[complex] = []
    for index in range(n):
        theta = TAU * index / n
        z = unit(theta)
        weights.append(density(theta) * abs(shape.dpsi(z)))
        nodes.append(z)
    return weights, nodes


def quotient_rho_derivatives(shape: LaurentShape, rho: float, phase_unit: complex, z: complex, max_order: int) -> list[complex]:
    derivatives = [0.0j for _ in range(max_order + 1)]
    for power, coeff in shape.coeffs:
        if power == 1:
            derivatives[0] += coeff
            continue
        if power >= 0:
            continue
        mode = -power
        for q in range(mode):
            exponent = q + 1
            factor = z ** (q - mode)
            phase_factor = phase_unit ** (-exponent)
            for order in range(max_order + 1):
                derivatives[order] -= (
                    coeff
                    * ((-1.0) ** order)
                    * rising_factorial(exponent, order)
                    * (rho ** (-exponent - order))
                    * phase_factor
                    * factor
                )
    return derivatives


def q_split_derivatives(shape: LaurentShape, rho: float, n: int, max_order: int) -> list[float]:
    phase_unit = unit(shape.target_phase)
    weights, nodes = sample_pullback(shape, n)
    coeffs = [value / n for value in fft([complex(weight, 0.0) for weight in weights])]

    derivatives = [0.0 for _ in range(max_order + 1)]
    derivatives[0] = TAU * coeffs[0].real * math.log(rho)
    for order in range(1, max_order + 1):
        derivatives[order] = TAU * coeffs[0].real * ((-1.0) ** (order - 1)) * math.factorial(order - 1) * rho ** (-order)
    for mode in range(1, n // 2):
        modal = (coeffs[mode] * unit(mode * shape.target_phase)).real
        derivatives[0] -= TAU * rho ** (-mode) * modal / mode
        for order in range(1, max_order + 1):
            derivatives[order] -= (
                TAU
                * modal
                * ((-1.0) ** order)
                * rising_factorial(mode, order)
                * rho ** (-mode - order)
                / mode
            )

    for weight, node in zip(weights, nodes, strict=True):
        log_terms = log_derivatives(quotient_rho_derivatives(shape, rho, phase_unit, node, max_order))
        for order in range(max_order + 1):
            derivatives[order] += TAU * weight * log_terms[order].real / n
    return derivatives


def taylor_repay(derivatives: list[float], shift: float, order: int) -> float:
    return sum(((-shift) ** index) * derivatives[index] / math.factorial(index) for index in range(order + 1))


def relative_error(value: complex, reference: complex) -> float:
    return abs(value - reference) / max(abs(reference), 1.0e-14)


def relative_l2(values: list[complex], reference: list[complex]) -> float:
    numerator = sum(abs(value - truth) ** 2 for value, truth in zip(values, reference, strict=True))
    denominator = sum(abs(truth) ** 2 for truth in reference)
    return math.sqrt(numerator / max(denominator, 1.0e-300))


def generated_q_amplitude(problem: str, mode: int, n: int, parameters: dict[str, float]) -> complex:
    mu = cycle_dtn_eigenvalue(mode, n)
    if problem == "laplace_dtn":
        return complex(mu)
    if problem == "heat":
        return complex(math.exp(-parameters.get("time", 0.2) * mu))
    if problem == "poisson":
        mass = parameters.get("mass", 0.25)
        return complex(1.0 / (mu + mass) if mu + mass != 0.0 else 0.0)
    if problem == "helmholtz":
        k = parameters.get("wavenumber", 3.7)
        damping = parameters.get("damping", 1.0e-3)
        return 1.0 / (mu * mu - k * k + 1j * damping)
    if problem == "wave":
        return complex(math.cos(parameters.get("time", 0.8) * math.sqrt(mu)))
    raise ValueError(f"unknown problem: {problem}")


def projected_cos_amplitude(values: list[complex], mode: int) -> complex:
    n = len(values)
    basis = [math.cos(TAU * mode * index / n) for index in range(n)]
    numerator = sum(complex(value) * basis[index] for index, value in enumerate(values))
    denominator = sum(value * value for value in basis)
    return numerator / denominator


def shape_suite() -> tuple[LaurentShape, ...]:
    golden_major = 3.0
    golden_minor = math.sqrt(5.0)
    return (
        LaurentShape("circle", "baseline", ((1, 1.0 + 0.0j),)),
        LaurentShape(
            "golden_ellipse",
            "closed_form_conic",
            ((1, 0.5 * (golden_major + golden_minor)), (-1, 0.5 * (golden_major - golden_minor))),
        ),
        LaurentShape("eccentric_ellipse", "closed_form_conic", ((1, 1.75 + 0.0j), (-1, 0.75 + 0.0j))),
        LaurentShape("three_petal", "smooth_nonconvex", ((1, 1.0 + 0.0j), (-2, 0.26 + 0.0j))),
        LaurentShape("four_lobed_wiggle", "smooth_multiscale", ((1, 1.0 + 0.0j), (-3, 0.16 + 0.0j), (-7, 0.0 + 0.035j))),
        LaurentShape("asymmetric_limaçon", "asymmetric", ((1, 1.05 + 0.0j), (-1, 0.16 + 0.06j), (-2, 0.12 - 0.04j))),
        LaurentShape("crescent_teardrop", "near_cusp_smooth", ((1, 1.08 + 0.0j), (-1, 0.24 + 0.0j), (-2, -0.13 + 0.03j))),
        LaurentShape("peanut", "pinched_smooth", ((1, 1.08 + 0.0j), (-1, -0.20 + 0.0j), (-3, 0.10 + 0.0j))),
        LaurentShape("high_frequency_gear", "smooth_high_frequency", ((1, 1.0 + 0.0j), (-4, 0.085 + 0.0j), (-9, 0.025 - 0.015j))),
        LaurentShape("rotated_mixed", "complex_coefficients", ((1, 1.06 + 0.0j), (-1, 0.11 + 0.09j), (-3, -0.055 + 0.07j), (-5, 0.025 - 0.02j))),
    )


def segment_intersections(points: list[complex]) -> int:
    def orient(a: complex, b: complex, c: complex) -> float:
        return ((b.real - a.real) * (c.imag - a.imag)) - ((b.imag - a.imag) * (c.real - a.real))

    def intersects(a: complex, b: complex, c: complex, d: complex) -> bool:
        o1 = orient(a, b, c)
        o2 = orient(a, b, d)
        o3 = orient(c, d, a)
        o4 = orient(c, d, b)
        return o1 * o2 < 0.0 and o3 * o4 < 0.0

    n = len(points)
    total = 0
    for i in range(n):
        a = points[i]
        b = points[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            c = points[j]
            d = points[(j + 1) % n]
            if intersects(a, b, c, d):
                total += 1
    return total


def shape_diagnostics(shape: LaurentShape, n: int = 512) -> dict[str, object]:
    points = [shape.psi(unit(TAU * index / n)) for index in range(n)]
    speeds = [abs(shape.dpsi(unit(TAU * index / n))) for index in range(n)]
    area = 0.5 * sum((points[i].real * points[(i + 1) % n].imag - points[i].imag * points[(i + 1) % n].real) for i in range(n))
    return {
        "shape": shape.name,
        "family": shape.family,
        "min_speed": min(speeds),
        "max_speed": max(speeds),
        "anisotropy": max(speeds) / min(speeds),
        "sample_area": area,
        "segment_intersections": segment_intersections(points),
    }


def run_bgk_rows() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    n = 2048
    reference_n = 32768
    true_offset = 1.0e-3
    monitor_h = 2.0**-26
    rows: list[dict[str, object]] = []
    for shape in shape_suite():
        reference = q_split_derivatives(shape, 1.0 + true_offset, reference_n, 0)[0]
        shift = BGK_BETA * math.sqrt(monitor_h)
        derivatives = q_split_derivatives(shape, 1.0 + true_offset + shift, n, max(CORRECTION_ORDERS))
        for order in CORRECTION_ORDERS:
            value = derivatives[0] if order == 0 else taylor_repay(derivatives, shift, order)
            rows.append(
                {
                    "shape": shape.name,
                    "family": shape.family,
                    "n": n,
                    "reference_n": reference_n,
                    "true_conformal_offset": true_offset,
                    "monitor_h": monitor_h,
                    "bgk_shift": shift,
                    "correction_order": order,
                    "reference_value": reference,
                    "corrected_value": value,
                    "rel_error": relative_error(value, reference),
                    "dense_q_matrix_stored": False,
                }
            )
    summary: list[dict[str, object]] = []
    for shape in shape_suite():
        subset = [row for row in rows if row["shape"] == shape.name]
        by_order = {int(row["correction_order"]): float(row["rel_error"]) for row in subset}
        summary.append(
            {
                "shape": shape.name,
                "family": shape.family,
                "raw_rel_error": by_order[0],
                "bgk1_rel_error": by_order[1],
                "bgk2_rel_error": by_order[2],
                "bgk4_rel_error": by_order[4],
                "bgk8_rel_error": by_order[8],
                "best_rel_error": min(by_order.values()),
            }
        )
    return rows, summary


def qjet_map(shape: LaurentShape):
    def boundary(theta: ScalarQJet):
        x = ScalarQJet(0.0)
        y = ScalarQJet(0.0)
        for power, coeff in shape.coeffs:
            angle = power * theta
            c = qjet_cos(angle)
            s = qjet_sin(angle)
            x += coeff.real * c - coeff.imag * s
            y += coeff.real * s + coeff.imag * c
        return x, y

    return boundary


def cosine_mode(n: int, mode: int) -> list[float]:
    return [math.cos(TAU * mode * index / n) for index in range(n)]


def run_pde_rows() -> list[dict[str, object]]:
    n = 2048
    mode = 7
    values = cosine_mode(n, mode)
    problem_parameters = {
        "laplace_dtn": {},
        "heat": {"time": 0.17},
        "poisson": {"mass": 0.35},
        "helmholtz": {"wavenumber": 3.7, "damping": 0.02},
        "wave": {"time": 0.8},
    }
    rows: list[dict[str, object]] = []
    for shape in shape_suite():
        start = perf_counter()
        boundary = build_boundary_pullback_qjet(n, qjet_map(shape))
        build_ms = 1000.0 * (perf_counter() - start)
        for problem, params in problem_parameters.items():
            start = perf_counter()
            result = boundary.solve_boundary_problem(problem, values, **params)
            ms = 1000.0 * (perf_counter() - start)
            output = [complex(value) for value in result.values]
            if problem == "laplace_dtn":
                continuum = [mode * math.cos(TAU * mode * index / n) / boundary.speeds[index] for index in range(n)]
                generated = [cycle_dtn_eigenvalue(mode, n) * math.cos(TAU * mode * index / n) / boundary.speeds[index] for index in range(n)]
                continuum_error = relative_l2(output, [complex(value) for value in continuum])
                generated_error = relative_l2(output, [complex(value) for value in generated])
                amplitude = projected_cos_amplitude([output[index] * boundary.speeds[index] for index in range(n)], mode)
                exact_amplitude = exact_disk_amplitude(problem, mode, **params)
                generated_amplitude = generated_q_amplitude(problem, mode, n, params)
            else:
                amplitude = projected_cos_amplitude(output, mode)
                exact_amplitude = exact_disk_amplitude(problem, mode, **params)
                generated_amplitude = generated_q_amplitude(problem, mode, n, params)
                continuum_error = relative_error(amplitude, exact_amplitude)
                generated_error = relative_error(amplitude, generated_amplitude)
            rows.append(
                {
                    "shape": shape.name,
                    "family": shape.family,
                    "problem": problem,
                    "n": n,
                    "mode": mode,
                    "continuum_rel_error": continuum_error,
                    "generated_rel_error": generated_error,
                    "amplitude_real": amplitude.real,
                    "amplitude_imag": amplitude.imag,
                    "exact_amplitude_real": exact_amplitude.real,
                    "exact_amplitude_imag": exact_amplitude.imag,
                    "generated_amplitude_real": generated_amplitude.real,
                    "generated_amplitude_imag": generated_amplitude.imag,
                    "build_ms": build_ms,
                    "solve_ms": ms,
                    "work_units": result.work_units,
                    "protocol": result.stats["protocol"],
                    "dense_q_matrix_stored": False,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_figure(path: Path, diagnostics: list[dict[str, object]], bgk_summary: list[dict[str, object]], pde_rows: list[dict[str, object]]) -> None:
    shapes = shape_suite()
    fig, axes = plt.subplots(3, 4, figsize=(14.0, 9.2), constrained_layout=True)
    for axis, shape in zip(axes[0], shapes[:4], strict=True):
        points = [shape.psi(unit(TAU * index / 700)) for index in range(701)]
        axis.plot([p.real for p in points], [p.imag for p in points], color="0.0", linewidth=1.1)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(shape.name.replace("_", " "), fontsize=8)
        axis.grid(color="0.92", linewidth=0.5)

    for axis, shape in zip(axes[1], shapes[4:8], strict=True):
        points = [shape.psi(unit(TAU * index / 700)) for index in range(701)]
        axis.plot([p.real for p in points], [p.imag for p in points], color="0.0", linewidth=1.1)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(shape.name.replace("_", " "), fontsize=8)
        axis.grid(color="0.92", linewidth=0.5)

    labels = [row["shape"].replace("_", "\n") for row in bgk_summary]
    x = list(range(len(labels)))
    axes[2][0].bar(x, [max(float(row["raw_rel_error"]), ERROR_FLOOR) for row in bgk_summary], color="0.70", label="raw")
    axes[2][0].bar(x, [max(float(row["bgk4_rel_error"]), ERROR_FLOOR) for row in bgk_summary], color="0.05", label="BGK-4")
    axes[2][0].set_yscale("log")
    axes[2][0].set_xticks(x)
    axes[2][0].set_xticklabels(labels, fontsize=6)
    axes[2][0].set_title("Q+BGK ladder")
    axes[2][0].set_ylabel("relative error")
    axes[2][0].grid(axis="y", color="0.90")
    axes[2][0].legend(frameon=False, fontsize=7)

    pde_max = []
    for shape in shapes:
        subset = [row for row in pde_rows if row["shape"] == shape.name]
        pde_max.append(max(float(row["continuum_rel_error"]) for row in subset))
    axes[2][1].bar(x, pde_max, color="0.25")
    axes[2][1].set_yscale("log")
    axes[2][1].set_xticks(x)
    axes[2][1].set_xticklabels(labels, fontsize=6)
    axes[2][1].set_title("max PDE rel. error")
    axes[2][1].grid(axis="y", color="0.90")

    problems = ("laplace_dtn", "heat", "poisson", "helmholtz", "wave")
    problem_errors = []
    for problem in problems:
        subset = [row for row in pde_rows if row["problem"] == problem]
        problem_errors.append(max(float(row["continuum_rel_error"]) for row in subset))
    axes[2][2].bar(range(len(problems)), problem_errors, color="0.35")
    axes[2][2].set_yscale("log")
    axes[2][2].set_xticks(range(len(problems)))
    axes[2][2].set_xticklabels([p.replace("_", "\n") for p in problems], fontsize=7)
    axes[2][2].set_title("PDE problems")
    axes[2][2].grid(axis="y", color="0.90")

    axes[2][3].bar(x, [float(row["anisotropy"]) for row in diagnostics], color="0.45")
    axes[2][3].set_yscale("log")
    axes[2][3].set_xticks(x)
    axes[2][3].set_xticklabels(labels, fontsize=6)
    axes[2][3].set_title("speed anisotropy")
    axes[2][3].grid(axis="y", color="0.90")

    fig.suptitle("Funky shapes: exterior Q+BGK ladder and boundary PDE pipeline", fontsize=13)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_report(
    path: Path,
    figure_path: Path,
    diagnostics: list[dict[str, object]],
    bgk_summary: list[dict[str, object]],
    pde_rows: list[dict[str, object]],
    diag_csv: Path,
    bgk_csv: Path,
    pde_csv: Path,
    json_path: Path,
) -> None:
    max_bgk4 = max(float(row["bgk4_rel_error"]) for row in bgk_summary)
    max_pde = max(float(row["continuum_rel_error"]) for row in pde_rows)
    max_generated = max(float(row["generated_rel_error"]) for row in pde_rows)
    lines = [
        "# Funky Shape Q+BGK/PDE Suite",
        "",
        "This tests the combined slick path on a wider exact-Laurent shape suite, then sends the same conformal-coordinate data through the boundary-only PDE pipeline.",
        "",
        f"![funky Q BGK PDE suite]({figure_path})",
        "",
        "## Shape Diagnostics",
        "",
        "| Shape | Family | Anisotropy | Intersections |",
        "|---|---|---:|---:|",
    ]
    for row in diagnostics:
        lines.append(
            f"| {row['shape']} | {row['family']} | `{float(row['anisotropy']):.3f}` | `{int(row['segment_intersections'])}` |"
        )
    lines.extend(
        [
            "",
            "## BGK Ladder",
            "",
            "| Shape | Raw | BGK-1 | BGK-2 | BGK-4 | BGK-8 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in bgk_summary:
        lines.append(
            "| {shape} | `{raw:.3e}` | `{one:.3e}` | `{two:.3e}` | `{four:.3e}` | `{eight:.3e}` |".format(
                shape=row["shape"],
                raw=float(row["raw_rel_error"]),
                one=float(row["bgk1_rel_error"]),
                two=float(row["bgk2_rel_error"]),
                four=float(row["bgk4_rel_error"]),
                eight=float(row["bgk8_rel_error"]),
            )
        )
    lines.extend(
        [
            "",
            f"Max BGK-4 relative error across the shape suite: `{max_bgk4:.3e}`.",
            "",
            "## PDE Pipeline",
            "",
            "| Problem | Max continuum error | Max generated-Q residual |",
            "|---|---:|---:|",
        ]
    )
    for problem in ("laplace_dtn", "heat", "poisson", "helmholtz", "wave"):
        subset = [row for row in pde_rows if row["problem"] == problem]
        lines.append(
            f"| {problem} | `{max(float(row['continuum_rel_error']) for row in subset):.3e}` | `{max(float(row['generated_rel_error']) for row in subset):.3e}` |"
        )
    lines.extend(
        [
            "",
            f"Max PDE continuum relative error across all shapes/problems: `{max_pde:.3e}`.",
            f"Max generated-Q implementation residual across all shapes/problems: `{max_generated:.3e}`.",
            "",
            "For Laplace DtN the continuum comparison is against physical conformal-map flux `m cos(m theta)/|psi'|`. For heat, Poisson, Helmholtz, and wave the continuum comparison is against the exact unit-circle modal multiplier after borrowing the conformal coordinate. The generated-Q residual compares against the finite Q spectrum actually used by the engine.",
            "",
            "## Artifacts",
            "",
            f"- Shape diagnostics CSV: `{diag_csv}`",
            f"- BGK rows CSV: `{bgk_csv}`",
            f"- PDE rows CSV: `{pde_csv}`",
            f"- JSON: `{json_path}`",
            f"- Figure: `{figure_path}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    diagnostics = [shape_diagnostics(shape) for shape in shape_suite()]
    bgk_rows, bgk_summary = run_bgk_rows()
    pde_rows = run_pde_rows()

    diag_csv = OUT / "funky_shape_diagnostics.csv"
    bgk_rows_csv = OUT / "funky_shape_bgk_ladder_rows.csv"
    bgk_summary_csv = OUT / "funky_shape_bgk_ladder_summary.csv"
    pde_csv = OUT / "funky_shape_pde_rows.csv"
    json_path = OUT / "funky_shape_q_bgk_pde_suite.json"
    figure_path = OUT / "funky_shape_q_bgk_pde_suite.png"
    report_path = OUT / "funky_shape_q_bgk_pde_suite.md"

    write_csv(diag_csv, diagnostics)
    write_csv(bgk_rows_csv, bgk_rows)
    write_csv(bgk_summary_csv, bgk_summary)
    write_csv(pde_csv, pde_rows)
    json_path.write_text(
        json.dumps(
            {
                "method": {
                    "dense_q_matrix_stored": False,
                    "shape_model": "exact finite exterior Laurent maps",
                    "quadrature": "exterior Q/Fourier kernel split plus BGK Taylor repayment ladder",
                    "pde": "borrow conformal circle coordinate, compute cycle Q/DtN PDE, repay metric for Laplace normal flux",
                    "bgk_beta": BGK_BETA,
                },
                "diagnostics": diagnostics,
                "bgk_rows": bgk_rows,
                "bgk_summary": bgk_summary,
                "pde_rows": pde_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    make_figure(figure_path, diagnostics, bgk_summary, pde_rows)
    write_report(report_path, figure_path, diagnostics, bgk_summary, pde_rows, diag_csv, bgk_rows_csv, pde_csv, json_path)
    print(json.dumps({"report": str(report_path), "figure": str(figure_path), "pde_csv": str(pde_csv)}, indent=2))


if __name__ == "__main__":
    main()
