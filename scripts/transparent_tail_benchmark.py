#!/usr/bin/env python3
"""Benchmark explicit shells, finite-tail symbols, and exact transparent caps."""

from __future__ import annotations

import csv
import gc
import json
import math
import statistics
import sys
import time
import tracemalloc
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inverse_shape.quadrature import TAU, _cos, _sin  # noqa: E402
from inverse_shape.transparent_tail import (  # noqa: E402
    CylindricalTransparentDtN,
    golden_tail_certificate,
)

OUT = ROOT / "outputs" / "transparent_tail_benchmark"
N_THETA = 128
RIGHT_HAND_SIDES = 8
DEPTHS = (4, 8, 16, 32, 64, 128, 256, 512)
CASES = (
    ("laplace", 0.0, 0.0),
    ("screened_poisson", 0.6, 0.0),
    ("heat_resolvent", 0.4, 0.0),
    ("helmholtz", 0.7, 0.2),
    ("wave_resolvent", 0.7, 0.2),
)


def trace(phase):
    return tuple(
        complex(
            _cos(TAU * index / N_THETA + phase)
            + 0.23 * _sin(3.0 * TAU * index / N_THETA - 0.4 * phase)
            - 0.08 * _cos(11.0 * TAU * index / N_THETA + 0.7 * phase),
            0.05 * _sin(5.0 * TAU * index / N_THETA + 0.2 * phase),
        )
        for index in range(N_THETA)
    )


TRACES = tuple(trace(0.19 * index) for index in range(RIGHT_HAND_SIDES))


def relative_l2(reference, candidate):
    numerator = sum(
        abs(complex(left) - complex(right)) ** 2
        for left, right in zip(reference, candidate, strict=True)
    )
    denominator = sum(abs(complex(value)) ** 2 for value in reference)
    return math.sqrt(numerator / max(denominator, 1.0e-300))


def measured(call):
    gc.collect()
    tracemalloc.start()
    started = time.perf_counter_ns()
    value = call()
    elapsed_ms = (time.perf_counter_ns() - started) * 1.0e-6
    _current, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return value, elapsed_ms, peak_bytes


def repeat_last(call):
    result = None
    for values in TRACES:
        result = call(values)
    return result


def direct_batch(cap, depth):
    return repeat_last(
        lambda values: cap.solve_direct_dirichlet_shells(values, depth)
    )


def compiled_batch(cap, symbols):
    return repeat_last(lambda values: cap.apply_mode_symbols(values, symbols))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=tuple(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def benchmark_case(problem, parameter, damping):
    cap, setup_ms, setup_peak = measured(
        lambda: CylindricalTransparentDtN.for_problem(
            N_THETA,
            problem,
            parameter=parameter,
            damping=damping,
        )
    )
    exact_first = cap.apply_boundary_dtn(TRACES[0]).values
    _warm = cap.apply_boundary_dtn(TRACES[0])
    _cap_last, cap_apply_ms, cap_peak = measured(
        lambda: repeat_last(lambda values: cap.apply_boundary_dtn(values).values)
    )
    mode_index = 1
    if cap.modes[mode_index].marginal:
        mode_index = 2
    rows = []
    certificates = []
    for depth in DEPTHS:
        direct_first = cap.solve_direct_dirichlet_shells(TRACES[0], depth)
        finite_first = cap.apply_finite_dirichlet(TRACES[0], depth, quantity="flux")
        agreement = relative_l2(direct_first, finite_first)
        boundary_error = relative_l2(exact_first, finite_first)

        _direct_last, direct_ms, direct_peak = measured(
            partial(direct_batch, cap, depth)
        )
        finite_symbols, finite_setup_ms, finite_setup_peak = measured(
            partial(cap.finite_dirichlet_symbols, depth, quantity="flux")
        )
        _finite_last, finite_apply_ms, finite_apply_peak = measured(
            partial(compiled_batch, cap, finite_symbols)
        )
        rows.extend(
            (
                {
                    "problem": problem,
                    "parameter": parameter,
                    "damping": damping,
                    "n_theta": N_THETA,
                    "right_hand_sides": RIGHT_HAND_SIDES,
                    "depth": depth,
                    "method": "direct_L_shell_thomas",
                    "setup_ms": 0.0,
                    "apply_ms": direct_ms,
                    "total_ms": direct_ms,
                    "measured_peak_bytes": direct_peak,
                    "shell_iterations": RIGHT_HAND_SIDES * N_THETA * depth,
                    "relative_boundary_error": boundary_error,
                    "direct_finite_agreement": agreement,
                    "time_big_o": "O(K*N_theta*L + K*N_theta*log(N_theta))",
                    "storage_big_o": "O(N_theta+L)",
                    "dense_matrix_stored": False,
                },
                {
                    "problem": problem,
                    "parameter": parameter,
                    "damping": damping,
                    "n_theta": N_THETA,
                    "right_hand_sides": RIGHT_HAND_SIDES,
                    "depth": depth,
                    "method": "compiled_finite_dirichlet",
                    "setup_ms": finite_setup_ms,
                    "apply_ms": finite_apply_ms,
                    "total_ms": finite_setup_ms + finite_apply_ms,
                    "measured_peak_bytes": max(finite_setup_peak, finite_apply_peak),
                    "shell_iterations": N_THETA * max(depth - 1, 0),
                    "relative_boundary_error": boundary_error,
                    "direct_finite_agreement": agreement,
                    "time_big_o": "O(N_theta*L + K*N_theta*log(N_theta))",
                    "storage_big_o": "O(N_theta)",
                    "dense_matrix_stored": False,
                },
                {
                    "problem": problem,
                    "parameter": parameter,
                    "damping": damping,
                    "n_theta": N_THETA,
                    "right_hand_sides": RIGHT_HAND_SIDES,
                    "depth": depth,
                    "method": "exact_fixed_point_cap",
                    "setup_ms": setup_ms,
                    "apply_ms": cap_apply_ms,
                    "total_ms": setup_ms + cap_apply_ms,
                    "measured_peak_bytes": max(setup_peak, cap_peak),
                    "shell_iterations": 0,
                    "relative_boundary_error": 0.0,
                    "direct_finite_agreement": agreement,
                    "time_big_o": "O(N_theta + K*N_theta*log(N_theta))",
                    "storage_big_o": "O(N_theta)",
                    "dense_matrix_stored": False,
                },
            )
        )
        certificate = cap.cross_ratio_certificate(mode_index, depth)
        certificates.append(
            {
                "problem": problem,
                "depth": depth,
                "mode": certificate["mode"],
                "contraction": certificate["contraction"],
                "pivot_error": certificate["pivot_error"],
                "pivot_error_bound": certificate["pivot_error_bound"],
                "cross_ratio_residual": certificate["linearization_residual"],
            }
        )
    mode_rows = [
        {
            "problem": problem,
            "parameter": parameter,
            "damping": damping,
            "spectral_shift_real": cap.spectral_shift.real,
            "spectral_shift_imag": cap.spectral_shift.imag,
            "maximum_root_modulus": max(abs(mode.root) for mode in cap.modes),
            "minimum_root_modulus": min(abs(mode.root) for mode in cap.modes),
            "marginal_modes": sum(mode.marginal for mode in cap.modes),
            "maximum_fixed_point_residual": max(
                mode.fixed_point_residual for mode in cap.modes
            ),
            "maximum_root_residual": max(mode.root_residual for mode in cap.modes),
        }
    ]
    return rows, certificates, mode_rows


def svg_plot(rows):
    width = 980
    height = 520
    left = 90
    right = 30
    top = 50
    bottom = 75
    methods = (
        ("direct_L_shell_thomas", "Direct L-shell", "#111111", ""),
        ("compiled_finite_dirichlet", "Finite symbol", "#666666", "6,5"),
        ("exact_fixed_point_cap", "Fixed-point cap", "#999999", "2,4"),
    )
    aggregates = {}
    for method, _label, _color, _dash in methods:
        aggregates[method] = []
        for depth in DEPTHS:
            values = [
                float(row["total_ms"])
                for row in rows
                if row["method"] == method and int(row["depth"]) == depth
            ]
            aggregates[method].append(statistics.median(values))
    all_values = [value for values in aggregates.values() for value in values]
    lower = math.floor(math.log10(max(min(all_values), 1.0e-6)))
    upper = math.ceil(math.log10(max(all_values)))
    if upper <= lower:
        upper = lower + 1

    def x_position(depth):
        index = DEPTHS.index(depth)
        return left + index * (width - left - right) / (len(DEPTHS) - 1)

    def y_position(value):
        scaled = (math.log10(max(value, 1.0e-12)) - lower) / (upper - lower)
        return top + (1.0 - scaled) * (height - top - bottom)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#171717;letter-spacing:0}'
        '.axis{stroke:#222;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}'
        '.series{fill:none;stroke-width:2.2}.point{stroke:white;stroke-width:1}'
        '</style>',
        '<text x="90" y="27" font-size="19" font-weight="600">'
        'Transparent tail closure: median total time across five PDE resolvents'
        '</text>',
    ]
    for exponent in range(lower, upper + 1):
        value = 10.0**exponent
        y_value = y_position(value)
        parts.append(
            f'<line class="grid" x1="{left}" y1="{y_value:.2f}" '
            f'x2="{width-right}" y2="{y_value:.2f}"/>'
        )
        parts.append(
            f'<text x="{left-12}" y="{y_value+5:.2f}" text-anchor="end" '
            f'font-size="12">10^{exponent} ms</text>'
        )
    for depth in DEPTHS:
        x_value = x_position(depth)
        parts.append(
            f'<line class="grid" x1="{x_value:.2f}" y1="{top}" '
            f'x2="{x_value:.2f}" y2="{height-bottom}"/>'
        )
        parts.append(
            f'<text x="{x_value:.2f}" y="{height-bottom+24}" '
            f'text-anchor="middle" font-size="12">{depth}</text>'
        )
    parts.extend(
        (
            f'<line class="axis" x1="{left}" y1="{height-bottom}" '
            f'x2="{width-right}" y2="{height-bottom}"/>',
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}"/>',
            f'<text x="{(left+width-right)/2:.2f}" y="{height-18}" '
            'text-anchor="middle" font-size="14">Explicit shell depth L</text>',
            f'<text x="20" y="{(top+height-bottom)/2:.2f}" '
            'text-anchor="middle" font-size="14" '
            f'transform="rotate(-90 20 {(top+height-bottom)/2:.2f})">'
            'Total time for K=8 right-hand sides (ms)</text>',
        )
    )
    for method, _label, color, dash in methods:
        points = " ".join(
            f"{x_position(depth):.2f},{y_position(value):.2f}"
            for depth, value in zip(DEPTHS, aggregates[method], strict=True)
        )
        dash_attribute = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(
            f'<polyline class="series" points="{points}" stroke="{color}"{dash_attribute}/>'
        )
        for depth, value in zip(DEPTHS, aggregates[method], strict=True):
            parts.append(
                f'<circle class="point" cx="{x_position(depth):.2f}" '
                f'cy="{y_position(value):.2f}" r="3.5" fill="{color}"/>'
            )
    legend_x = 590
    for offset, (_method, label, color, dash) in enumerate(methods):
        y_value = 72 + 24 * offset
        dash_attribute = f' stroke-dasharray="{dash}"' if dash else ""
        parts.append(
            f'<line x1="{legend_x}" y1="{y_value}" x2="{legend_x+34}" '
            f'y2="{y_value}" stroke="{color}" stroke-width="2.2"'
            f'{dash_attribute}/>'
        )
        parts.append(
            f'<text x="{legend_x+44}" y="{y_value+5}" font-size="13">{label}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def report(summary, rows, mode_rows, golden):
    lines = [
        "# Exact transparent-tail benchmark",
        "",
        "This campaign compares a streamed direct L-shell tridiagonal solve, a "
        "compiled finite-Dirichlet Riccati symbol, and the exact fixed-point "
        "transparent cap. All methods use the same custom angular QJet FFT; no "
        "dense shell or boundary matrix is formed.",
        "",
        "![Runtime against explicit shell depth](runtime_scaling.svg)",
        "",
        "## PDE coverage",
        "",
        "| Problem | Shift | max |w| | Fixed-point residual |",
        "|---|---:|---:|---:|",
    ]
    for row in mode_rows:
        shift = complex(
            float(row["spectral_shift_real"]),
            float(row["spectral_shift_imag"]),
        )
        lines.append(
            f"| {row['problem']} | `{shift.real:.3g}{shift.imag:+.3g}i` | "
            f"{float(row['maximum_root_modulus']):.6f} | "
            f"{float(row['maximum_fixed_point_residual']):.3e} |"
        )
    lines.extend(
        (
            "",
            "## Deep-tail comparison",
            "",
            f"The table uses `N_theta={N_THETA}`, `K={RIGHT_HAND_SIDES}`, and "
            f"`L={DEPTHS[-1]}`. Timings are measured wall times on this machine.",
            "",
            "| Problem | Direct ms | Finite-symbol ms | Cap ms | Direct/cap | Finite error |",
            "|---|---:|---:|---:|---:|---:|",
        )
    )
    for problem, _parameter, _damping in CASES:
        selected = [
            row
            for row in rows
            if row["problem"] == problem and int(row["depth"]) == DEPTHS[-1]
        ]
        by_method = {row["method"]: row for row in selected}
        direct = by_method["direct_L_shell_thomas"]
        finite = by_method["compiled_finite_dirichlet"]
        cap = by_method["exact_fixed_point_cap"]
        speedup = float(direct["total_ms"]) / max(float(cap["total_ms"]), 1.0e-15)
        lines.append(
            f"| {problem} | {float(direct['total_ms']):.3f} | "
            f"{float(finite['total_ms']):.3f} | {float(cap['total_ms']):.3f} | "
            f"{speedup:.1f}x | {float(finite['relative_boundary_error']):.3e} |"
        )
    lines.extend(
        (
            "",
            "## Certificates",
            "",
            f"- Maximum direct/finite disagreement: "
            f"`{summary['maximum_direct_finite_disagreement']:.3e}`.",
            f"- Maximum fixed-point residual: "
            f"`{summary['maximum_fixed_point_residual']:.3e}`.",
            f"- Maximum cross-ratio linearization residual: "
            f"`{summary['maximum_cross_ratio_residual']:.3e}`.",
            f"- Cylinder generator identity residual: "
            f"`{summary['cylinder_identity_residual']:.3e}`.",
            f"- Golden depth {golden['depth']} pivot: "
            f"`F_{2*golden['depth']+2}/F_{2*golden['depth']} = "
            f"{golden['numerator']}/{golden['denominator']}`; error-law residual "
            f"`{golden['error_law_residual']:.3e}`.",
            f"- Summably perturbed transition: actual pivot defect "
            f"`{summary['perturbed_transition_actual_error']:.3e}` under certified "
            f"bound `{summary['perturbed_transition_error_bound']:.3e}`.",
            "",
            "The fixed-point row has zero autonomous-tail truncation error by the "
            "proved Schur identity. That statement is stronger than convergence "
            "against a long finite tail, but narrower than a continuum claim on an "
            "arbitrary CAD surface.",
            "",
            "## Complexity",
            "",
            "- Direct streamed shells: `O(K N_theta L + K N_theta log N_theta)` "
            "time and `O(N_theta + L)` working storage.",
            "- Compiled finite tail: `O(N_theta L)` setup, then "
            "`O(K N_theta log N_theta)` application and `O(N_theta)` storage.",
            "- Exact cap: `O(N_theta)` setup, "
            "`O(K N_theta log N_theta)` application, and `O(N_theta)` storage, "
            "independent of `L`.",
            "",
            "## CAD accuracy scope",
            "",
            "This cap closes an autonomous cylindrical or conic end. The held-out "
            "public-CAD campaign instead compresses each source mesh to 48–155 "
            "operator vertices and tests an unseen degree-four continuum harmonic "
            "against a degree-three compiler. The separate NASA display gallery "
            "uses 24–42 nodes but reports only retained-channel or finite-equation "
            "audits. The cap does not reconstruct those discarded surface channels.",
        )
    )
    return "\n".join(lines) + "\n"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    certificates = []
    mode_rows = []
    for problem, parameter, damping in CASES:
        print(f"benchmarking {problem}", flush=True)
        case_rows, case_certificates, case_modes = benchmark_case(
            problem,
            parameter,
            damping,
        )
        rows.extend(case_rows)
        certificates.extend(case_certificates)
        mode_rows.extend(case_modes)
    cylinder = CylindricalTransparentDtN(N_THETA)
    golden = golden_tail_certificate(20)
    transition_cap = CylindricalTransparentDtN(N_THETA, spectral_shift=0.4)
    transition = transition_cap.perturbed_transition_certificate(
        3,
        tuple(0.02 * 0.5**shell for shell in range(16)),
        tuple(-0.004 * 0.5**shell for shell in range(16)),
    )
    maximum_agreement = max(float(row["direct_finite_agreement"]) for row in rows)
    maximum_fixed = max(
        float(row["maximum_fixed_point_residual"]) for row in mode_rows
    )
    maximum_cross = max(
        float(row["cross_ratio_residual"]) for row in certificates
    )
    deepest_speedups = []
    for problem, _parameter, _damping in CASES:
        selected = [
            row
            for row in rows
            if row["problem"] == problem and int(row["depth"]) == DEPTHS[-1]
        ]
        by_method = {row["method"]: row for row in selected}
        deepest_speedups.append(
            float(by_method["direct_L_shell_thomas"]["total_ms"])
            / float(by_method["exact_fixed_point_cap"]["total_ms"])
        )
    summary = {
        "campaign": "exact spectral fixed-point transparent-tail closure",
        "n_theta": N_THETA,
        "right_hand_sides": RIGHT_HAND_SIDES,
        "depths": DEPTHS,
        "pde_case_count": len(CASES),
        "benchmark_row_count": len(rows),
        "maximum_direct_finite_disagreement": maximum_agreement,
        "maximum_fixed_point_residual": maximum_fixed,
        "maximum_cross_ratio_residual": maximum_cross,
        "cylinder_identity_residual": cylinder.cylinder_identity_residual(),
        "minimum_deep_tail_direct_speedup": min(deepest_speedups),
        "median_deep_tail_direct_speedup": statistics.median(deepest_speedups),
        "maximum_deep_tail_direct_speedup": max(deepest_speedups),
        "golden_certificate": golden,
        "perturbed_transition_actual_error": transition["actual_pivot_error"],
        "perturbed_transition_error_bound": transition[
            "certified_pivot_error_bound"
        ],
        "perturbed_transition_maximum_lipschitz": transition[
            "maximum_local_lipschitz"
        ],
        "autonomous_tail_truncation_error_with_cap": 0.0,
        "cap_exactness_basis": "algebraic fixed-point Schur identity",
        "dense_shell_matrix_stored": False,
        "dense_boundary_matrix_stored": False,
        "cap_apply_big_o": "O(K*N_theta*log(N_theta))",
        "cap_storage_big_o": "O(N_theta)",
        "held_out_cad_machine_precision_claim": False,
        "all_gates_passed": (
            maximum_agreement < 3.0e-12
            and maximum_fixed < 3.0e-14
            and maximum_cross < 3.0e-12
            and cylinder.cylinder_identity_residual() < 3.0e-14
            and golden["error_law_residual"] < 1.0e-15
            and transition["actual_pivot_error"]
            <= transition["certified_pivot_error_bound"] + 2.0e-15
            and transition["maximum_local_lipschitz"] < 1.0
        ),
    }
    write_csv(OUT / "benchmark_rows.csv", rows)
    write_csv(OUT / "certificate_rows.csv", certificates)
    write_csv(OUT / "pde_mode_rows.csv", mode_rows)
    (OUT / "runtime_scaling.svg").write_text(svg_plot(rows), encoding="utf-8")
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (OUT / "report.md").write_text(
        report(summary, rows, mode_rows, golden),
        encoding="utf-8",
    )
    if not summary["all_gates_passed"]:
        raise SystemExit("transparent-tail benchmark gates failed")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
