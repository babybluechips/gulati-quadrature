#!/usr/bin/env python3
"""BGK endpoint bookkeeping: Monte Carlo, Spitzer, and Q/DtN zeta.

This script verifies the point that the BGK constant is not an arbitrary
barrier tweak.  It is the discrete-vs-continuous endpoint defect:

    E max_{0<=k<=N} S_k
      = sqrt(2N/pi) + zeta(1/2)/sqrt(2pi) + O(N^-1/2).

The same half-integer endpoint debt appears in the cycle Q/DtN spectral zeta
sum.  No dense Q matrix is built.
"""

from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "bgk_endpoint_bookkeeping"

ZETA_HALF = -1.4603545088095868128894991525152980124672293310126
SQRT_2PI = math.sqrt(2.0 * math.pi)
BGK_BETA = -ZETA_HALF / SQRT_2PI


def continuous_brownian_max(n: int) -> float:
    return math.sqrt(2.0 * n / math.pi)


def spitzer_exact_discrete_max(n: int) -> float:
    return sum(k**-0.5 for k in range(1, n + 1)) / SQRT_2PI


def monte_carlo_discrete_max(n: int, paths: int, seed: int) -> dict[str, float]:
    rng = random.Random(seed)
    total = 0.0
    total2 = 0.0
    for _ in range(paths):
        position = 0.0
        maximum = 0.0
        for _ in range(n):
            position += rng.gauss(0.0, 1.0)
            if position > maximum:
                maximum = position
        total += maximum
        total2 += maximum * maximum
    mean = total / paths
    variance = max(total2 / paths - mean * mean, 0.0)
    stderr = math.sqrt(variance / paths)
    return {"mean": mean, "stderr": stderr}


def monte_carlo_rows() -> list[dict[str, float]]:
    schedule = (
        (64, 20000),
        (128, 12000),
        (256, 8000),
        (512, 5000),
        (1024, 3000),
        (2048, 1500),
    )
    rows: list[dict[str, float]] = []
    for n, paths in schedule:
        continuous = continuous_brownian_max(n)
        exact = spitzer_exact_discrete_max(n)
        bgk_approx = continuous - BGK_BETA
        mc = monte_carlo_discrete_max(n, paths, seed=982451653 + n)
        exact_defect = exact - continuous
        mc_defect = mc["mean"] - continuous
        rows.append(
            {
                "n": float(n),
                "paths": float(paths),
                "continuous_brownian_max": continuous,
                "spitzer_exact_discrete_max": exact,
                "bgk_approx_discrete_max": bgk_approx,
                "monte_carlo_mean": mc["mean"],
                "monte_carlo_stderr": mc["stderr"],
                "exact_endpoint_defect": exact_defect,
                "monte_carlo_endpoint_defect": mc_defect,
                "exact_beta_estimate": -exact_defect,
                "monte_carlo_beta_estimate": -mc_defect,
                "exact_zeta_half_estimate": exact_defect * SQRT_2PI,
                "monte_carlo_zeta_half_estimate": mc_defect * SQRT_2PI,
                "exact_bgk_residual": exact - bgk_approx,
                "monte_carlo_minus_spitzer": mc["mean"] - exact,
            }
        )
    return rows


def q_dtn_endpoint_rows(n_values: tuple[int, ...]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for n in n_values:
        spectral_sum = sum((k * (1.0 - k / n)) ** -0.5 for k in range(1, n))
        bulk = math.pi * math.sqrt(n)
        bgk_approx = bulk + 2.0 * ZETA_HALF
        rows.append(
            {
                "n": float(n),
                "cycle_dtn_zeta_sum_s_half": spectral_sum,
                "bulk_integral": bulk,
                "bulk_plus_bgk_endpoint": bgk_approx,
                "raw_endpoint_residual": spectral_sum - bulk,
                "bgk_corrected_residual": spectral_sum - bgk_approx,
                "half_residual_zeta_estimate": 0.5 * (spectral_sum - bulk),
            }
        )
    return rows


def chord_arc_rows(m_values: tuple[int, ...]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for m in m_values:
        arc = 2.0 * math.pi / m
        chord = 2.0 * math.sin(math.pi / m)
        arc_inverse_square = arc**-2
        chord_inverse_square = chord**-2
        rows.append(
            {
                "m": float(m),
                "arc_step": arc,
                "chord_step": chord,
                "arc_minus_chord": arc - chord,
                "relative_chord_arc_defect": (arc - chord) / arc,
                "inverse_square_arc": arc_inverse_square,
                "inverse_square_chord": chord_inverse_square,
                "relative_inverse_square_defect": chord_inverse_square / arc_inverse_square - 1.0,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fit_power(xs: list[float], ys: list[float]) -> float:
    lx = [math.log(x) for x in xs]
    ly = [math.log(max(y, 1.0e-300)) for y in ys]
    mx = sum(lx) / len(lx)
    my = sum(ly) / len(ly)
    numerator = sum((x - mx) * (y - my) for x, y in zip(lx, ly, strict=True))
    denominator = sum((x - mx) ** 2 for x in lx)
    return numerator / denominator


def make_figure(
    mc_rows: list[dict[str, float]],
    dtn_rows: list[dict[str, float]],
    chord_rows: list[dict[str, float]],
    path: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "0.1",
            "axes.labelcolor": "0.1",
            "xtick.color": "0.1",
            "ytick.color": "0.1",
            "text.color": "0.1",
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.2), constrained_layout=True)
    fig.patch.set_facecolor("white")

    ax = axes[0][0]
    n = [row["n"] for row in mc_rows]
    exact_defect = [-row["exact_endpoint_defect"] for row in mc_rows]
    mc_defect = [-row["monte_carlo_endpoint_defect"] for row in mc_rows]
    mc_err = [row["monte_carlo_stderr"] for row in mc_rows]
    ax.errorbar(n, mc_defect, yerr=mc_err, color="0.45", marker="s", linewidth=1.2, label="MC beta estimate")
    ax.plot(n, exact_defect, color="0.0", marker="o", linewidth=1.4, label="Spitzer exact")
    ax.axhline(BGK_BETA, color="0.2", linestyle=(0, (3, 2)), linewidth=1.0, label="BGK beta")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("walk steps N")
    ax.set_ylabel("continuous - discrete maximum")
    ax.set_title("Gaussian walk endpoint defect")
    ax.grid(color="0.88", linewidth=0.6)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0][1]
    raw = [abs(row["exact_endpoint_defect"]) for row in mc_rows]
    corrected = [abs(row["exact_bgk_residual"]) for row in mc_rows]
    ax.loglog(n, raw, color="0.0", marker="o", linewidth=1.4, label="without BGK")
    ax.loglog(n, corrected, color="0.5", marker="s", linewidth=1.4, label="after BGK")
    ax.set_xlabel("walk steps N")
    ax.set_ylabel("absolute residual")
    ax.set_title("BGK removes the constant endpoint rung")
    ax.grid(True, which="both", color="0.88", linewidth=0.6)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1][0]
    dn = [row["n"] for row in dtn_rows]
    dtn_raw = [abs(row["raw_endpoint_residual"]) for row in dtn_rows]
    dtn_corrected = [abs(row["bgk_corrected_residual"]) for row in dtn_rows]
    ax.loglog(dn, dtn_raw, color="0.0", marker="o", linewidth=1.4, label="bulk only")
    ax.loglog(dn, dtn_corrected, color="0.5", marker="s", linewidth=1.4, label="bulk + 2 zeta(1/2)")
    ax.set_xlabel("cycle samples n")
    ax.set_ylabel("spectral zeta residual")
    ax.set_title("Q/DtN half-integer endpoint")
    ax.grid(True, which="both", color="0.88", linewidth=0.6)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1][1]
    m = [row["m"] for row in chord_rows]
    chord_defect = [row["relative_chord_arc_defect"] for row in chord_rows]
    inv_defect = [row["relative_inverse_square_defect"] for row in chord_rows]
    ax.loglog(m, chord_defect, color="0.0", marker="o", linewidth=1.4, label="arc-chord")
    ax.loglog(m, inv_defect, color="0.5", marker="s", linewidth=1.4, label="inverse-square")
    ax.set_xlabel("regular polygon sides M")
    ax.set_ylabel("relative defect")
    ax.set_title("Chord-to-arc bookkeeping in Q")
    ax.grid(True, which="both", color="0.88", linewidth=0.6)
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("BGK continuity correction as endpoint/chord-to-arc bookkeeping", fontsize=13)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_report(
    report_path: Path,
    figure_path: Path,
    mc_csv: Path,
    dtn_csv: Path,
    chord_csv: Path,
    json_path: Path,
    mc_rows: list[dict[str, float]],
    dtn_rows: list[dict[str, float]],
    chord_rows: list[dict[str, float]],
) -> None:
    exact_power = fit_power(
        [1.0 / row["n"] for row in mc_rows],
        [abs(row["exact_bgk_residual"]) for row in mc_rows],
    )
    dtn_power = fit_power(
        [1.0 / row["n"] for row in dtn_rows[-5:]],
        [abs(row["bgk_corrected_residual"]) for row in dtn_rows[-5:]],
    )
    stable_mc_rows = [row for row in mc_rows if row["monte_carlo_stderr"] <= 0.1]
    representative_mc = stable_mc_rows[-1] if stable_mc_rows else mc_rows[0]
    latest_mc = mc_rows[-1]
    latest_dtn = dtn_rows[-1]
    latest_chord = chord_rows[-1]
    lines = [
        "# BGK Endpoint Bookkeeping",
        "",
        "This verifies the point from `geometry_of_money (16).pdf`: BGK is the translator between continuously monitored crossing and discretely monitored crossing. It is the half-integer endpoint defect, not an arbitrary empirical shift.",
        "",
        f"![BGK endpoint bookkeeping]({figure_path})",
        "",
        "## Gaussian Walk Ledger",
        "",
        "For a standard Gaussian walk, Spitzer's identity gives",
        "",
        "```text",
        "E max_{0<=k<=N} S_k = (1/sqrt(2 pi)) sum_{k=1}^N k^(-1/2)",
        "                         = sqrt(2N/pi) + zeta(1/2)/sqrt(2 pi) + O(N^(-1/2)).",
        "```",
        "",
        "Therefore the continuous-to-discrete translator is",
        "",
        "```text",
        "beta_BGK = -zeta(1/2)/sqrt(2 pi)",
        f"         = {BGK_BETA:.16f}.",
        "```",
        "",
        f"The Monte Carlo estimate is noisy at large `N` because the maximum itself fluctuates on the `sqrt(N)` scale. A representative row is `N={int(representative_mc['n'])}`: exact Spitzer gives beta estimate `{representative_mc['exact_beta_estimate']:.6f}`, while Monte Carlo gives `{representative_mc['monte_carlo_beta_estimate']:.6f}` with stderr `{representative_mc['monte_carlo_stderr']:.6f}` on the maximum itself.",
        "",
        f"The convergence proof is the exact Spitzer sum. At the largest exact row, `N={int(latest_mc['n'])}`, the beta estimate is `{latest_mc['exact_beta_estimate']:.6f}`. After the BGK subtraction the exact residual decays with fitted power `{exact_power:.3f}` in `1/N`.",
        "",
        "## Q/DtN Ledger",
        "",
        "The same half-integer endpoint defect appears in the cycle Q/DtN spectral zeta sum:",
        "",
        "```text",
        "sum_{k=1}^{n-1} [k(1-k/n)]^(-1/2) = pi sqrt(n) + 2 zeta(1/2) + lower terms.",
        "```",
        "",
        f"At `n={int(latest_dtn['n'])}`, the raw endpoint residual is `{latest_dtn['raw_endpoint_residual']:.6f}` and the BGK-corrected residual is `{latest_dtn['bgk_corrected_residual']:.6f}`. The last five corrected levels fit power `{dtn_power:.3f}` in `1/n`.",
        "",
        "## Chord-To-Arc Ledger",
        "",
        "The inverse-square chord operator is the boundary version of the same bookkeeping. The continuum sees arc distance; the finite operator sees chord distance. The correction ladder records the defect between those two ledgers.",
        "",
        f"At `M={int(latest_chord['m'])}`, the relative arc-chord defect is `{latest_chord['relative_chord_arc_defect']:.6e}`, while the inverse-square Q defect is `{latest_chord['relative_inverse_square_defect']:.6e}`.",
        "",
        "## Interpretation",
        "",
        "In the heat/barrier language, continuous monitoring is the killed semigroup and discrete monitoring is the projected product. The missed Brownian-bridge crossings produce the `sqrt(h)` boundary-flux term. In Q language, that boundary flux is the DtN/Q operator; BGK is the endpoint translator that repays the discrete monitoring mesh.",
        "",
        "So the full bookkeeping is:",
        "",
        "```text",
        "continuous heat/barrier operator",
        "  -> discrete monitoring dates",
        "  -> Brownian bridge endpoint defect",
        "  -> zeta(1/2) / sqrt(2 pi)",
        "  -> beta_BGK",
        "  -> Q/DtN boundary flux correction",
        "  -> chord-to-arc inverse-square repayment",
        "```",
        "",
        "## Artifacts",
        "",
        f"- Monte Carlo CSV: `{mc_csv}`",
        f"- Q/DtN endpoint CSV: `{dtn_csv}`",
        f"- chord/arc CSV: `{chord_csv}`",
        f"- JSON: `{json_path}`",
        f"- figure: `{figure_path}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mc_rows = monte_carlo_rows()
    dtn_rows = q_dtn_endpoint_rows((64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536))
    chord_rows = chord_arc_rows((16, 32, 64, 128, 256, 512, 1024, 2048, 4096))

    mc_csv = OUT / "bgk_monte_carlo_spitzer.csv"
    dtn_csv = OUT / "bgk_q_dtn_endpoint.csv"
    chord_csv = OUT / "bgk_chord_arc_bookkeeping.csv"
    json_path = OUT / "bgk_endpoint_bookkeeping.json"
    figure_path = OUT / "bgk_endpoint_bookkeeping.png"
    report_path = OUT / "bgk_endpoint_bookkeeping.md"

    write_csv(mc_csv, mc_rows)
    write_csv(dtn_csv, dtn_rows)
    write_csv(chord_csv, chord_rows)
    make_figure(mc_rows, dtn_rows, chord_rows, figure_path)
    payload = {
        "constants": {
            "zeta_half": ZETA_HALF,
            "bgk_beta": BGK_BETA,
            "endpoint_defect": ZETA_HALF / SQRT_2PI,
        },
        "monte_carlo": mc_rows,
        "q_dtn_endpoint": dtn_rows,
        "chord_arc": chord_rows,
        "method": {
            "dense_q_matrix_stored": False,
            "monte_carlo_seed_rule": "982451653 + N",
            "normalization": "standard Gaussian walk increments N(0,1)",
            "continuous_reference": "E sup_{0<=t<=N} B_t = sqrt(2N/pi)",
        },
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(report_path, figure_path, mc_csv, dtn_csv, chord_csv, json_path, mc_rows, dtn_rows, chord_rows)
    print(json.dumps({"report": str(report_path), "figure": str(figure_path), "json": str(json_path)}, indent=2))


if __name__ == "__main__":
    main()
