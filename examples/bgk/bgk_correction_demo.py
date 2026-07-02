#!/usr/bin/env python3
"""Show where the BGK correction helps in the DtN/zeta setting.

No dense Q matrix is assembled.  The demo uses only generated endpoint lattices
and the closed-form cycle DtN spectrum

    mu_{n,k} = k (1 - k/n),  k=1,...,n-1,

which is the rescaled spectrum appearing in the DtN/BGK note.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "bgk_correction_help"

ZETA_HALF = -1.4603545088095868128894991525152980124672293310126
BGK_BETA = -ZETA_HALF / math.sqrt(2.0 * math.pi)


def compact(value: float) -> str:
    if value == 0.0:
        return "0"
    if abs(value) < 1.0e-3 or abs(value) >= 1.0e4:
        return f"{value:.3e}"
    return f"{value:.6f}"


def fit_power(xs: list[float], ys: list[float]) -> float:
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    mx = sum(lx) / len(lx)
    my = sum(ly) / len(ly)
    num = sum((x - mx) * (y - my) for x, y in zip(lx, ly))
    den = sum((x - mx) ** 2 for x in lx)
    return num / den


def endpoint_rows(n_values: tuple[int, ...]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    exact = 2.0
    for n in n_values:
        h = 1.0 / float(n)
        j = torch.arange(1, n + 1, dtype=torch.float64)
        raw = float(math.sqrt(h) * torch.sum(torch.rsqrt(j)).item())
        corrected = raw - ZETA_HALF * math.sqrt(h)
        raw_error = raw - exact
        corrected_error = corrected - exact
        rows.append(
            {
                "n": float(n),
                "h": h,
                "raw_value": raw,
                "bgk_corrected_value": corrected,
                "raw_error": raw_error,
                "bgk_corrected_error": corrected_error,
                "raw_abs_error": abs(raw_error),
                "bgk_corrected_abs_error": abs(corrected_error),
                "zeta_half_estimate": raw_error / math.sqrt(h),
                "bgk_beta_estimate": -raw_error / math.sqrt(2.0 * math.pi * h),
            }
        )
    return rows


def dtn_rows(n_values: tuple[int, ...]) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for n in n_values:
        k = torch.arange(1, n, dtype=torch.float64)
        mu = k * (1.0 - k / float(n))
        spectral_sum = float(torch.sum(torch.rsqrt(mu)).item())
        bulk = math.pi * math.sqrt(float(n))
        raw_endpoint_residual = spectral_sum - bulk
        bgk_corrected_residual = spectral_sum - (bulk + 2.0 * ZETA_HALF)
        rows.append(
            {
                "n": float(n),
                "spectral_sum_s_half": spectral_sum,
                "bulk_beta_counterterm": bulk,
                "raw_endpoint_residual": raw_endpoint_residual,
                "bgk_corrected_residual": bgk_corrected_residual,
                "half_residual_zeta_estimate": 0.5 * raw_endpoint_residual,
                "raw_abs_residual": abs(raw_endpoint_residual),
                "bgk_corrected_abs_residual": abs(bgk_corrected_residual),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_figure(endpoint: list[dict[str, float]], dtn: list[dict[str, float]], path: Path) -> None:
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
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), constrained_layout=True)
    fig.patch.set_facecolor("white")

    ax = axes[0]
    h = [row["h"] for row in endpoint]
    raw = [row["raw_abs_error"] for row in endpoint]
    corrected = [row["bgk_corrected_abs_error"] for row in endpoint]
    ax.loglog(h, raw, color="0.05", marker="o", linewidth=1.6, label="raw endpoint sum")
    ax.loglog(h, corrected, color="0.45", marker="s", linewidth=1.6, label="after BGK rung")
    ax.invert_xaxis()
    ax.set_xlabel("h = 1/n")
    ax.set_ylabel("absolute error")
    ax.set_title("Square-root endpoint lattice")
    ax.grid(True, which="both", color="0.88", linewidth=0.6)
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    n = [row["n"] for row in dtn]
    raw_res = [abs(row["raw_endpoint_residual"]) for row in dtn]
    corr_res = [row["bgk_corrected_abs_residual"] for row in dtn]
    ax.loglog(n, raw_res, color="0.05", marker="o", linewidth=1.6, label="bulk only")
    ax.loglog(n, corr_res, color="0.45", marker="s", linewidth=1.6, label="bulk + 2 zeta(1/2)")
    ax.set_xlabel("cycle nodes n")
    ax.set_ylabel("residual magnitude")
    ax.set_title("DtN spectral zeta endpoint")
    ax.grid(True, which="both", color="0.88", linewidth=0.6)
    ax.legend(frameon=False, fontsize=9)

    fig.suptitle("BGK correction removes the half-integer sampling debt", fontsize=13)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_report(
    path: Path,
    figure_path: Path,
    endpoint_csv: Path,
    dtn_csv: Path,
    payload_path: Path,
    endpoint: list[dict[str, float]],
    dtn: list[dict[str, float]],
) -> None:
    raw_power = fit_power([row["h"] for row in endpoint], [row["raw_abs_error"] for row in endpoint])
    corrected_power = fit_power(
        [row["h"] for row in endpoint], [row["bgk_corrected_abs_error"] for row in endpoint]
    )
    dtn_power = fit_power(
        [1.0 / row["n"] for row in dtn[-5:]], [row["bgk_corrected_abs_residual"] for row in dtn[-5:]]
    )
    latest_endpoint = endpoint[-1]
    latest_dtn = dtn[-1]

    lines = [
        "# BGK Correction Help",
        "",
        "This is the concrete place where the BGK continuity correction helps the attached DtN/harmonic notes.",
        "",
        f"![BGK correction help]({figure_path})",
        "",
        "## What Is Being Corrected",
        "",
        "`dtn_bgk_merge (9).pdf` identifies the rescaled cycle DtN spectrum",
        "",
        "```text",
        "mu[n,k] = k (1 - k/n)",
        "```",
        "",
        "and says that the half-integer endpoint term of its spectral zeta sum is the BGK constant. `harmonic_sp (41).pdf` says the same thing in harmonic-language: when a continuum harmonic object is sampled on a finite cyclic grid, the residual is a zeta-coded sampling defect.",
        "",
        "The correction does not replace Q. It repays the finite lattice after the principal DtN/Q symbol has already been borrowed:",
        "",
        "```text",
        "phase:       de Moivre/Fourier modes",
        "operator:    pi |D| or the generated chord-QJet",
        "metric:      pullback Jacobian J(theta)",
        "sampling:    BGK/Hurwitz-zeta endpoint repayment",
        "```",
        "",
        "## Endpoint Model",
        "",
        "The square-root endpoint channel is the local model behind the BGK rung:",
        "",
        "```text",
        "h sum_{j=1}^n (j h)^(-1/2) - integral_0^1 x^(-1/2) dx",
        "  = zeta(1/2) h^(1/2) + O(h).",
        "```",
        "",
        f"Since `zeta(1/2) = {ZETA_HALF:.16f}`, the BGK barrier constant is",
        "",
        "```text",
        "beta_BGK = -zeta(1/2) / sqrt(2 pi)",
        f"         = {BGK_BETA:.16f}.",
        "```",
        "",
        f"In this run the raw endpoint error fits power `{raw_power:.3f}` in `h`; after subtracting the BGK rung it fits power `{corrected_power:.3f}`. At `n={int(latest_endpoint['n'])}`, the raw error is `{compact(latest_endpoint['raw_abs_error'])}` and the corrected error is `{compact(latest_endpoint['bgk_corrected_abs_error'])}`.",
        "",
        "## DtN Spectral Zeta Model",
        "",
        "For the cycle DtN spectrum at `s=1/2`, the bulk integral is `pi sqrt(n)`. The endpoint residual is",
        "",
        "```text",
        "S_n(1/2) - pi sqrt(n) -> 2 zeta(1/2).",
        "```",
        "",
        "So the BGK-corrected bulk approximation is",
        "",
        "```text",
        "S_n(1/2) ~= pi sqrt(n) + 2 zeta(1/2).",
        "```",
        "",
        f"At `n={int(latest_dtn['n'])}`, the bulk-only endpoint residual is `{compact(latest_dtn['raw_endpoint_residual'])}`. After adding `2 zeta(1/2)`, the residual drops to `{compact(latest_dtn['bgk_corrected_residual'])}`. The corrected residual over the last five levels fits power `{dtn_power:.3f}` in `1/n`.",
        "",
        "## Practical Meaning For Q",
        "",
        "BGK helps when the Q/DtN computation has the right principal operator but the finite boundary sampling still sees a square-root endpoint or survival/barrier channel. It removes the leading half-integer sampling debt. For polygons and sharper corners, the same slot is filled by the Kondrat'ev/Hurwitz rule `zeta(1-lambda,beta) h^lambda`; BGK is the crack/square-root special case `lambda=1/2`.",
        "",
        "## Artifacts",
        "",
        f"- endpoint CSV: `{endpoint_csv}`",
        f"- DtN CSV: `{dtn_csv}`",
        f"- JSON: `{payload_path}`",
        f"- figure: `{figure_path}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    endpoint_n = (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144)
    dtn_n = (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072)
    endpoint = endpoint_rows(endpoint_n)
    dtn = dtn_rows(dtn_n)

    endpoint_csv = OUT / "endpoint_bgk_correction.csv"
    dtn_csv = OUT / "dtn_spectral_bgk_correction.csv"
    payload_path = OUT / "bgk_correction_help.json"
    figure_path = OUT / "bgk_correction_help.png"
    report_path = OUT / "bgk_correction_help.md"

    write_csv(endpoint_csv, endpoint)
    write_csv(dtn_csv, dtn)
    payload = {
        "constants": {
            "zeta_half": ZETA_HALF,
            "bgk_beta": BGK_BETA,
            "interpretation": "BGK is the leading half-integer endpoint sampling repayment, not the principal Q kernel.",
        },
        "endpoint_rows": endpoint,
        "dtn_rows": dtn,
    }
    payload_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    make_figure(endpoint, dtn, figure_path)
    write_report(report_path, figure_path, endpoint_csv, dtn_csv, payload_path, endpoint, dtn)
    print(json.dumps({"report": str(report_path), "figure": str(figure_path)}, indent=2))


if __name__ == "__main__":
    main()
