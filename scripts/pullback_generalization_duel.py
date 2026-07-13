#!/usr/bin/env python3
# ruff: noqa: E501
"""Head-to-head Riemann/circulant versus cosecant pullback campaign.

Every reference action and residual diagnostic streams pairs.  The benchmark
does not form or retain a dense matrix.
"""

import csv
import json
from pathlib import Path
from time import perf_counter

from inverse_shape.axisymmetric3d import build_axisymmetric_surface_qjet, torus_qjet
from inverse_shape.pullback_generalization import (
    AxisymmetricMeridionalCosecantQJet,
    CosecantPullbackQJet,
    LagAveragedCirculantQJet,
    PeriodicCurveSamples,
    apply_physical_chord_qjet,
    equal_arclength_samples,
    streamed_pullback_diagnostics,
)
from inverse_shape.quadrature import TAU, _abs, _cos, _log, _sin, _sqrt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "pullback_generalization_duel"


def relative_error(reference, candidate):
    numerator = sum(
        _abs(complex(left) - complex(right)) ** 2
        for left, right in zip(reference, candidate, strict=True)
    )
    denominator = sum(_abs(complex(value)) ** 2 for value in reference)
    return _sqrt(numerator / max(denominator, 1.0e-300))


def radial(theta, radius):
    return radius * complex(_cos(theta), _sin(theta))


def circle(theta):
    return radial(theta, 1.0)


def golden_ellipse(theta):
    return complex(3.0 * _cos(theta), _sqrt(5.0) * _sin(theta))


def starfish(theta):
    return radial(theta, 1.0 + 0.24 * _cos(5.0 * theta))


def asymmetric_fourier(theta):
    radius = 1.0 + 0.17 * _cos(3.0 * theta) + 0.08 * _sin(5.0 * theta) - 0.05 * _cos(7.0 * theta)
    return radial(theta, radius)


def peanut(theta):
    return radial(theta, 1.0 + 0.43 * _cos(2.0 * theta))


def double_concave(theta):
    return radial(
        theta,
        1.0 + 0.27 * _cos(2.0 * theta) - 0.12 * _cos(6.0 * theta),
    )


def cardioid_cusp(theta):
    return radial(theta, 1.0 + _cos(theta))


def rounded_square(theta):
    cosine = _cos(theta)
    sine = _sin(theta)
    return complex(
        (1.0 if cosine >= 0.0 else -1.0) * _sqrt(_abs(cosine)),
        (1.0 if sine >= 0.0 else -1.0) * _sqrt(_abs(sine)),
    )


def square(theta):
    coordinate = 4.0 * (theta % TAU) / TAU
    side = int(coordinate) % 4
    local = coordinate - int(coordinate)
    if side == 0:
        return complex(1.0, -1.0 + 2.0 * local)
    if side == 1:
        return complex(1.0 - 2.0 * local, 1.0)
    if side == 2:
        return complex(-1.0, 1.0 - 2.0 * local)
    return complex(-1.0 + 2.0 * local, -1.0)


PLANAR_SHAPES = (
    ("circle", circle, "smooth"),
    ("golden_ellipse", golden_ellipse, "smooth"),
    ("starfish_5", starfish, "smooth"),
    ("asymmetric_fourier", asymmetric_fourier, "smooth"),
    ("peanut", peanut, "smooth_concave"),
    ("double_concave", double_concave, "smooth_concave"),
    ("rounded_square", rounded_square, "low_regular"),
    ("cardioid_cusp", cardioid_cusp, "cusp"),
    ("square", square, "corners"),
)


def field(n, mode):
    return tuple(
        _cos(mode * TAU * index / n) + 0.17 * _sin((mode + 3) * TAU * index / n)
        for index in range(n)
    )


def benchmark_planar():
    rows = []
    for shape_name, generator, regularity in PLANAR_SHAPES:
        for n in (64, 128, 256):
            if shape_name == "circle":
                samples = PeriodicCurveSamples(
                    tuple(circle(TAU * (index + 0.5) / n) for index in range(n)),
                    TAU,
                )
            else:
                samples = equal_arclength_samples(
                    generator,
                    n,
                    oversample_factor=48,
                    phase=0.5,
                )
            start = perf_counter()
            cosecant = CosecantPullbackQJet(samples)
            cosecant_setup_ms = 1000.0 * (perf_counter() - start)
            start = perf_counter()
            lagged = LagAveragedCirculantQJet(samples)
            lagged_setup_ms = 1000.0 * (perf_counter() - start)
            diagnostics = streamed_pullback_diagnostics(
                samples,
                cosecant,
                lagged,
                local_band=3,
            )
            for mode_label, mode in (("low", 3), ("high", max(5, n // 4 - 1))):
                values = field(n, mode)
                start = perf_counter()
                reference = apply_physical_chord_qjet(samples, values)
                direct_ms = 1000.0 * (perf_counter() - start)
                start = perf_counter()
                csc_raw = cosecant.apply(values)
                csc_raw_ms = 1000.0 * (perf_counter() - start)
                start = perf_counter()
                csc_repaid = cosecant.apply_repaid(values, bandwidth=3)
                csc_repaid_ms = 1000.0 * (perf_counter() - start)
                start = perf_counter()
                lag_result = lagged.apply(values)
                lag_apply_ms = 1000.0 * (perf_counter() - start)
                rows.append(
                    {
                        "shape": shape_name,
                        "regularity": regularity,
                        "n": n,
                        "mode_band": mode_label,
                        "mode": mode,
                        "segment_anisotropy": diagnostics["segment_anisotropy"],
                        "cosecant_action_rel_error": relative_error(reference, csc_raw),
                        "cosecant_repaid_action_rel_error": relative_error(
                            reference,
                            csc_repaid,
                        ),
                        "lag_average_action_rel_error": relative_error(
                            reference,
                            lag_result,
                        ),
                        "cosecant_far_kernel_rel_error": diagnostics[
                            "cosecant_far_relative_residual"
                        ],
                        "lag_average_far_kernel_rel_error": diagnostics[
                            "lag_average_far_relative_residual"
                        ],
                        "prime_form_far_rel_defect": diagnostics["prime_form_far_relative_defect"],
                        "prime_form_negative_fraction": diagnostics[
                            "prime_form_far_negative_fraction"
                        ],
                        "cosecant_residual_local_fraction": diagnostics[
                            "cosecant_residual_local_fraction"
                        ],
                        "cosecant_setup_ms": cosecant_setup_ms,
                        "lag_average_setup_ms": lagged_setup_ms,
                        "direct_apply_ms": direct_ms,
                        "cosecant_apply_ms": csc_raw_ms,
                        "cosecant_repaid_apply_ms": csc_repaid_ms,
                        "lag_average_apply_ms": lag_apply_ms,
                        "dense_matrix_stored": False,
                    }
                )
    return rows


def benchmark_axisymmetric():
    rows = []
    for geometry, major, minor in (
        ("slender_torus", 4.0, 0.25),
        ("moderate_torus", 2.0, 0.45),
        ("near_horn_torus", 1.25, 0.55),
    ):
        for n_meridian in (32, 64, 128):
            surface = torus_qjet(
                major,
                minor,
                n_meridian=n_meridian,
                n_theta=128,
            )
            mode = max(4, n_meridian // 4 - 1)
            amplitudes = tuple(
                _cos(mode * TAU * (index + 0.5) / n_meridian) for index in range(n_meridian)
            )
            start = perf_counter()
            reference = surface.apply_azimuthal_mode(amplitudes, mode=0)
            streamed_ms = 1000.0 * (perf_counter() - start)
            start = perf_counter()
            raw = AxisymmetricMeridionalCosecantQJet(
                surface,
                mode=0,
                bandwidth=0,
            )
            raw_setup_ms = 1000.0 * (perf_counter() - start)
            start = perf_counter()
            repaid = AxisymmetricMeridionalCosecantQJet(
                surface,
                mode=0,
                bandwidth=3,
            )
            repaid_setup_ms = 1000.0 * (perf_counter() - start)
            start = perf_counter()
            raw_values = raw.apply(amplitudes)
            raw_apply_ms = 1000.0 * (perf_counter() - start)
            start = perf_counter()
            repaid_values = repaid.apply(amplitudes)
            repaid_apply_ms = 1000.0 * (perf_counter() - start)
            rows.append(
                {
                    "geometry": geometry,
                    "major_radius": major,
                    "minor_radius": minor,
                    "n_meridian": n_meridian,
                    "n_theta": surface.n_theta,
                    "surface_nodes": surface.n_nodes,
                    "meridional_mode": mode,
                    "cosecant_principal_rel_error": relative_error(
                        reference,
                        raw_values,
                    ),
                    "cosecant_repaid_rel_error": relative_error(
                        reference,
                        repaid_values,
                    ),
                    "streamed_reference_ms": streamed_ms,
                    "cosecant_setup_ms": raw_setup_ms,
                    "cosecant_repaid_setup_ms": repaid_setup_ms,
                    "cosecant_apply_ms": raw_apply_ms,
                    "cosecant_repaid_apply_ms": repaid_apply_ms,
                    "stored_three_jets": repaid.stats()["stored_three_jets"],
                    "stored_sparse_edges": repaid.stats()["stored_sparse_edges"],
                    "dense_matrix_stored": False,
                }
            )
    return rows


def benchmark_log_channel():
    rows = []
    radius = 1.0
    target = 1.0 / (4.0 * radius * radius)
    for separation in (0.2, 0.1, 0.05, 0.025):
        surface = build_axisymmetric_surface_qjet(
            (radius, radius),
            (0.0, separation),
            (1.0, 1.0),
            8192,
        )
        reduced = surface.reduced_meridional_kernel(0, 1)
        leading = 2.0 / (separation * separation)
        coefficient = (reduced - leading + 3.0 / (8.0 * radius * radius)) / _log(
            8.0 * radius / separation
        )
        rows.append(
            {
                "radius": radius,
                "separation": separation,
                "reduced_kernel": reduced,
                "inverse_square_channel": leading,
                "measured_log_coefficient": coefficient,
                "predicted_log_coefficient": target,
                "coefficient_abs_error": _abs(coefficient - target),
                "dense_matrix_stored": False,
            }
        )
    return rows


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compact(value):
    return f"{float(value):.3e}"


def summarize(planar_rows, axisymmetric_rows, log_rows):
    circle_rows = [row for row in planar_rows if row["shape"] == "circle"]
    generic_rows = [row for row in planar_rows if row["shape"] != "circle"]
    smooth_high = [
        row
        for row in generic_rows
        if row["mode_band"] == "high" and row["regularity"].startswith("smooth")
    ]
    summary = {
        "planar_cases": len(planar_rows),
        "axisymmetric_cases": len(axisymmetric_rows),
        "circle_max_cosecant_action_error": max(
            row["cosecant_action_rel_error"] for row in circle_rows
        ),
        "circle_max_lag_average_action_error": max(
            row["lag_average_action_rel_error"] for row in circle_rows
        ),
        "generic_min_lag_average_far_kernel_error": min(
            row["lag_average_far_kernel_rel_error"] for row in generic_rows
        ),
        "generic_max_prime_form_far_defect": max(
            row["prime_form_far_rel_defect"] for row in generic_rows
        ),
        "smooth_high_median_cosecant_error": sorted(
            row["cosecant_action_rel_error"] for row in smooth_high
        )[len(smooth_high) // 2],
        "smooth_high_median_cosecant_repaid_error": sorted(
            row["cosecant_repaid_action_rel_error"] for row in smooth_high
        )[len(smooth_high) // 2],
        "axisymmetric_finest": {
            row["geometry"]: {
                "raw": row["cosecant_principal_rel_error"],
                "repaid": row["cosecant_repaid_rel_error"],
                "repeated_apply_speedup": row["streamed_reference_ms"]
                / max(row["cosecant_repaid_apply_ms"], 1.0e-12),
                "first_solve_speedup": row["streamed_reference_ms"]
                / max(
                    row["cosecant_repaid_setup_ms"] + row["cosecant_repaid_apply_ms"],
                    1.0e-12,
                ),
            }
            for row in axisymmetric_rows
            if row["n_meridian"] == 128
        },
        "log_channel_finest_coefficient": log_rows[-1]["measured_log_coefficient"],
        "log_channel_predicted_coefficient": log_rows[-1]["predicted_log_coefficient"],
        "log_channel_finest_abs_error": log_rows[-1]["coefficient_abs_error"],
        "winner": "cosecant principal channel plus sparse local repayment",
        "riemann_route_status": (
            "not a 3D DtN generalization; generic arclength kernels are not circulant, "
            "and the prime-form mixed Hessian is not the inverse-square chord weight off a circle"
        ),
        "remaining_requirement": (
            "repay the logarithmic reduced channel and compress the nonlocal remainder "
            "with a certified FMM/H-matrix; local three-jets alone do not determine it"
        ),
        "stored_dense_matrix": False,
    }
    return summary


def write_report(path, summary, planar_rows, axisymmetric_rows):
    finest_planar = [row for row in planar_rows if row["n"] == 256 and row["mode_band"] == "high"]
    lines = [
        "# Riemann versus cosecant pullback generalization",
        "",
        "## Verdict",
        "",
        "The cosecant route is the viable principal-channel generalization. It is exact on the circle, "
        "removes the universal inverse-square diagonal singularity on every regular closed curve, "
        "and survives azimuthal reduction of the three-dimensional inverse-cube kernel. Sparse local "
        "repayment improves it without storing a dense matrix.",
        "",
        "The Riemann manuscript does not provide an arbitrary-3D fast path. Its boundary is a "
        "one-dimensional loop in a two-dimensional Riemann surface, whereas the 3D volume DtN "
        "operator acts on a two-dimensional boundary. In addition, a generic arclength-parametrized "
        "prime-form or chord kernel depends on both boundary points, not only their lag. The "
        "lag-averaged experiment below is the best circulant projection and still leaves a nonzero "
        "geometry residual.",
        "",
        "## Structural checks",
        "",
        f"- planar action cases: `{summary['planar_cases']}`",
        f"- axisymmetric 3D cases: `{summary['axisymmetric_cases']}`",
        f"- circle cosecant maximum action error: `{compact(summary['circle_max_cosecant_action_error'])}`",
        f"- circle best-lag maximum action error: `{compact(summary['circle_max_lag_average_action_error'])}`",
        f"- smallest noncircle best-lag far-kernel residual: `{compact(summary['generic_min_lag_average_far_kernel_error'])}`",
        f"- largest off-circle prime-form/chord far defect: `{compact(summary['generic_max_prime_form_far_defect'])}`",
        f"- smooth high-mode median cosecant error: `{compact(summary['smooth_high_median_cosecant_error'])}`",
        f"- after three exact local repayment bands: `{compact(summary['smooth_high_median_cosecant_repaid_error'])}`",
        f"- finest measured logarithmic coefficient: `{summary['log_channel_finest_coefficient']:.8f}` (target `0.25`)",
        "- dense matrices or pair tables stored: `no`",
        "",
        "## Planar high-mode comparison at n=256",
        "",
        "| shape | regularity | cosecant | cosecant + 3 bands | best lag-only | prime/chord far defect |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in finest_planar:
        lines.append(
            "| {shape} | {regularity} | `{csc}` | `{repaid}` | `{lag}` | `{prime}` |".format(
                shape=row["shape"],
                regularity=row["regularity"],
                csc=compact(row["cosecant_action_rel_error"]),
                repaid=compact(row["cosecant_repaid_action_rel_error"]),
                lag=compact(row["lag_average_action_rel_error"]),
                prime=compact(row["prime_form_far_rel_defect"]),
            )
        )
    lines.extend(
        [
            "",
            "## Axisymmetric 3D meridional reduction",
            "",
            "For a periodic meridian,",
            "",
            "```text",
            "r' integral |X-X'|^-3 dtheta' ~ 2 / |s-s'|^2.",
            "```",
            "",
            "The subtraction leaves a logarithmic local channel. For a cylinder of radius R,",
            "",
            "```text",
            "A_R(h) - 2/h^2 = log(8R/|h|)/(4R^2) - 3/(8R^2) + O(h^2 log|h|).",
            "```",
            "",
            "This produces the same cosecant principal symbol after periodization. The table compares "
            "that FFT operator with the full streamed ring-pair calculation for azimuthal mode zero.",
            "The meridional probe is the refinement-relative high mode `n_s/4-1`; this is a principal-symbol "
            "stress test, not a fixed-mode convergence table.",
            "",
            "| geometry | n_s | mode | nodes | cosecant | + 3 bands | first solve | repeated apply |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in axisymmetric_rows:
        speedup = row["streamed_reference_ms"] / max(
            row["cosecant_repaid_apply_ms"],
            1.0e-12,
        )
        first_speedup = row["streamed_reference_ms"] / max(
            row["cosecant_repaid_setup_ms"] + row["cosecant_repaid_apply_ms"],
            1.0e-12,
        )
        lines.append(
            "| {geometry} | {n_meridian} | {mode} | {surface_nodes} | `{raw}` | `{repaid}` | `{first:.1f}x` | `{speedup:.1f}x` |".format(
                geometry=row["geometry"],
                n_meridian=row["n_meridian"],
                mode=row["meridional_mode"],
                surface_nodes=row["surface_nodes"],
                raw=compact(row["cosecant_principal_rel_error"]),
                repaid=compact(row["cosecant_repaid_rel_error"]),
                first=first_speedup,
                speedup=speedup,
            )
        )
    lines.extend(
        [
            "",
            "## Complexity",
            "",
            "| route | setup | apply | retained storage | status |",
            "|---|---:|---:|---:|---|",
            "| global Riemann/lag-only proxy | `O(n^2)` | `O(n log n)` | `O(n)` | not exact off homogeneous loops |",
            "| cosecant principal | `O(n)` | `O(n log n)` | `O(n)` | exact singular channel |",
            "| cosecant + b local bands | `O(bn)` | `O(n log n + bn)` | `O(n + bn)` | implemented |",
            "| complete arbitrary geometry | hierarchical | target `O(n log n)` | target `O(n)` | log-local and far certificates required |",
            "",
            "## Consequence",
            "",
            "Use the cosecant pullback as the borrowed principal operator. Repay the diagonal and "
            "corner/cusp and logarithmic reduced channels with local product integration. Compute the "
            "nonlocal geometry with a certified multipole or H-matrix layer. For non-axisymmetric 3D surfaces, "
            "the corresponding local normal form is the two-dimensional periodic Riesz kernel with "
            "symbol |xi|, not a one-dimensional cosecant kernel.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    planar_rows = benchmark_planar()
    axisymmetric_rows = benchmark_axisymmetric()
    log_rows = benchmark_log_channel()
    summary = summarize(planar_rows, axisymmetric_rows, log_rows)
    write_csv(OUT / "planar_pullback_comparison.csv", planar_rows)
    write_csv(OUT / "axisymmetric_cosecant_comparison.csv", axisymmetric_rows)
    write_csv(OUT / "cylinder_log_channel.csv", log_rows)
    (OUT / "pullback_generalization_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(
        OUT / "pullback_generalization_report.md",
        summary,
        planar_rows,
        axisymmetric_rows,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
