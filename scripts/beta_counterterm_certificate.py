#!/usr/bin/env python3
"""Executable beta-counterterm certificate.

This script tests the finite-cycle sum

    S_n(s) = sum_{k=1}^{n-1} [k (1 - k/n)]^{-s}

against the analytic-continuation ledger

    S_n(s) = n^{1-s} B(1-s,1-s)
             + 2 sum_{j>=0} (s)_j zeta(s-j) n^{-j}/j!
             + residual.

The beta term is the bulk continuum channel.  Subtracting it is the one-
dimensional analogue of subtracting pi R^2 from the Gauss-circle count before
studying the boundary error.  No dense Q matrix is formed here; this is a
scalar arithmetic certificate for the repayment bookkeeping used by Q/BGK.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mpmath as mp


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = ROOT / "outputs" / "beta_counterterm_certificate"


@dataclass(frozen=True)
class CycleCase:
    label: str
    s_re: str
    s_im: str = "0"


@dataclass(frozen=True)
class ResidualRow:
    label: str
    n: int
    raw_abs: float
    bulk_abs: float
    beta_only_error_abs: float
    after_j1_abs: float
    after_j2_abs: float
    rung1_ratio_re: float
    rung1_ratio_im: float
    rung1_ratio_abs_error: float


@dataclass(frozen=True)
class SlopeRow:
    label: str
    beta_only_order: float
    after_j1_order: float
    after_j2_order: float
    rung1_ratio_abs_error_at_max_n: float
    pass_endpoint_decay: bool


@dataclass(frozen=True)
class GaussCircleRow:
    radius: int
    raw_count: int
    area_channel: float
    residual: float
    relative_area_error: float
    residual_over_sqrt_r: float


def mp_complex(case: CycleCase) -> mp.mpc:
    return mp.mpc(mp.mpf(case.s_re), mp.mpf(case.s_im))


def complex_payload(z: mp.mpc | mp.mpf) -> dict[str, float]:
    return {"re": float(mp.re(z)), "im": float(mp.im(z)), "abs": float(abs(z))}


def cycle_sum(s: mp.mpc, n: int) -> mp.mpc:
    return mp.fsum(
        mp.power(mp.mpf(k) * (1 - mp.mpf(k) / n), -s) for k in range(1, n)
    )


def beta_bulk(s: mp.mpc, n: int) -> mp.mpc:
    return mp.power(n, 1 - s) * mp.beta(1 - s, 1 - s)


def endpoint_coeff(s: mp.mpc, j: int) -> mp.mpc:
    return 2 * mp.rf(s, j) * mp.zeta(s - j) / mp.factorial(j)


def endpoint_series(s: mp.mpc, n: int, max_j: int) -> mp.mpc:
    return mp.fsum(endpoint_coeff(s, j) * mp.power(n, -j) for j in range(max_j + 1))


def fit_decay_order(rows: list[ResidualRow], field: str) -> float:
    # Use the last four refinements so the measured order reflects the
    # asymptotic endpoint ledger, not the coarse transient.
    tail = rows[-4:]
    xs = [math.log(row.n) for row in tail]
    ys = [math.log(max(float(getattr(row, field)), 1e-300)) for row in tail]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator = sum((x - mean_x) ** 2 for x in xs)
    return -numerator / denominator


def evaluate_cycle_case(case: CycleCase, ns: list[int]) -> tuple[list[ResidualRow], SlopeRow]:
    s = mp_complex(case)
    rows: list[ResidualRow] = []
    rung1 = endpoint_coeff(s, 1)
    for n in ns:
        raw = cycle_sum(s, n)
        bulk = beta_bulk(s, n)
        beta_repaid = raw - bulk
        beta_only_error = beta_repaid - endpoint_series(s, n, 0)
        after_j1 = beta_repaid - endpoint_series(s, n, 1)
        after_j2 = beta_repaid - endpoint_series(s, n, 2)
        ratio = beta_only_error / (rung1 / n)
        rows.append(
            ResidualRow(
                label=case.label,
                n=n,
                raw_abs=float(abs(raw)),
                bulk_abs=float(abs(bulk)),
                beta_only_error_abs=float(abs(beta_only_error)),
                after_j1_abs=float(abs(after_j1)),
                after_j2_abs=float(abs(after_j2)),
                rung1_ratio_re=float(mp.re(ratio)),
                rung1_ratio_im=float(mp.im(ratio)),
                rung1_ratio_abs_error=float(abs(ratio - 1)),
            )
        )

    slope = SlopeRow(
        label=case.label,
        beta_only_order=fit_decay_order(rows, "beta_only_error_abs"),
        after_j1_order=fit_decay_order(rows, "after_j1_abs"),
        after_j2_order=fit_decay_order(rows, "after_j2_abs"),
        rung1_ratio_abs_error_at_max_n=rows[-1].rung1_ratio_abs_error,
        pass_endpoint_decay=(
            fit_decay_order(rows, "beta_only_error_abs") > 0.95
            and fit_decay_order(rows, "after_j1_abs") > 1.90
            and fit_decay_order(rows, "after_j2_abs") > 2.80
            and rows[-1].rung1_ratio_abs_error < 5e-4
        ),
    )
    return rows, slope


def gauss_circle_count(radius: int) -> int:
    r2 = radius * radius
    total = 0
    for a in range(-radius, radius + 1):
        bmax = math.isqrt(max(r2 - a * a, 0))
        total += 2 * bmax + 1
    return total


def gauss_circle_rows(radii: list[int]) -> list[GaussCircleRow]:
    rows: list[GaussCircleRow] = []
    for radius in radii:
        count = gauss_circle_count(radius)
        area = math.pi * radius * radius
        residual = count - area
        rows.append(
            GaussCircleRow(
                radius=radius,
                raw_count=count,
                area_channel=area,
                residual=residual,
                relative_area_error=residual / area,
                residual_over_sqrt_r=residual / math.sqrt(radius),
            )
        )
    return rows


def polyline(points: list[tuple[float, float]], stroke: str, width: float = 1.5) -> str:
    encoded = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return (
        f'<polyline points="{encoded}" fill="none" stroke="{stroke}" '
        f'stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round" />'
    )


def write_decay_svg(
    path: Path, by_case: dict[str, list[ResidualRow]], width: int = 920, height: int = 540
) -> None:
    margin_left = 76
    margin_right = 28
    margin_top = 34
    margin_bottom = 62
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom

    all_rows = [row for rows in by_case.values() for row in rows]
    xs = [math.log10(row.n) for row in all_rows]
    ys = []
    for row in all_rows:
        ys.extend(
            [
                math.log10(max(row.beta_only_error_abs, 1e-300)),
                math.log10(max(row.after_j1_abs, 1e-300)),
                math.log10(max(row.after_j2_abs, 1e-300)),
            ]
        )
    min_x, max_x = min(xs), max(xs)
    min_y = math.floor(min(ys)) - 0.2
    max_y = math.ceil(max(ys)) + 0.2

    def sx(log_n: float) -> float:
        return margin_left + (log_n - min_x) * plot_w / (max_x - min_x)

    def sy(log_e: float) -> float:
        return margin_top + (max_y - log_e) * plot_h / (max_y - min_y)

    grid = []
    for p in range(math.floor(min_y), math.ceil(max_y) + 1):
        y = sy(p)
        grid.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" '
            f'y2="{y:.2f}" stroke="#e5e5e5" stroke-width="1" />'
        )
        grid.append(
            f'<text x="18" y="{y + 4:.2f}" font-size="12" fill="#555">1e{p}</text>'
        )
    for n in [128, 256, 512, 1024, 2048, 4096, 8192]:
        log_n = math.log10(n)
        x = sx(log_n)
        grid.append(
            f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" '
            f'y2="{height - margin_bottom}" stroke="#eeeeee" stroke-width="1" />'
        )
        grid.append(
            f'<text x="{x - 16:.2f}" y="{height - 28}" font-size="12" fill="#555">{n}</text>'
        )

    # Plot the real half-order case as the visual anchor.
    rows = by_case["s=1/2"]
    beta_points = [
        (sx(math.log10(row.n)), sy(math.log10(row.beta_only_error_abs))) for row in rows
    ]
    j1_points = [(sx(math.log10(row.n)), sy(math.log10(row.after_j1_abs))) for row in rows]
    j2_points = [(sx(math.log10(row.n)), sy(math.log10(row.after_j2_abs))) for row in rows]

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff" />',
        *grid,
        f'<rect x="{margin_left}" y="{margin_top}" width="{plot_w}" height="{plot_h}" '
        'fill="none" stroke="#222" stroke-width="1" />',
        polyline(beta_points, "#111111", 2.2),
        polyline(j1_points, "#666666", 2.2),
        polyline(j2_points, "#aaaaaa", 2.2),
        '<text x="76" y="24" font-size="17" font-family="serif" fill="#111">'
        'Beta counterterm exposes endpoint-rung orders for s=1/2</text>',
        '<text x="380" y="514" font-size="14" font-family="serif" fill="#111">n</text>',
        '<text x="16" y="302" transform="rotate(-90 16 302)" '
        'font-size="14" font-family="serif" fill="#111">absolute residual</text>',
        '<line x1="664" y1="56" x2="718" y2="56" stroke="#111" stroke-width="2.2" />',
        '<text x="728" y="61" font-size="13" fill="#111">after beta only: O(n^-1)</text>',
        '<line x1="664" y1="79" x2="718" y2="79" stroke="#666" stroke-width="2.2" />',
        '<text x="728" y="84" font-size="13" fill="#111">after first rung: O(n^-2)</text>',
        '<line x1="664" y1="102" x2="718" y2="102" stroke="#aaa" stroke-width="2.2" />',
        '<text x="728" y="107" font-size="13" fill="#111">after second rung: O(n^-3)</text>',
        "</svg>",
    ]
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |"]
    out.append("|" + "|".join(["---"] * len(headers)) + "|")
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def fmt_float(x: float, digits: int = 6) -> str:
    if x == 0:
        return "0"
    if abs(x) < 1e-3 or abs(x) >= 1e4:
        return f"{x:.{digits}e}"
    return f"{x:.{digits}f}"


def write_markdown(
    path: Path,
    payload: dict[str, Any],
    title: str,
    include_pedagogy: bool,
) -> None:
    slope_rows = [
        [
            row["label"],
            fmt_float(row["beta_only_order"], 4),
            fmt_float(row["after_j1_order"], 4),
            fmt_float(row["after_j2_order"], 4),
            fmt_float(row["rung1_ratio_abs_error_at_max_n"], 4),
            "PASS" if row["pass_endpoint_decay"] else "FAIL",
        ]
        for row in payload["slope_rows"]
    ]
    slope_table = markdown_table(
        [
            "case",
            "after beta",
            "after J=1",
            "after J=2",
            "rung ratio err",
            "status",
        ],
        slope_rows,
    )
    half_rows = [
        [
            str(row["n"]),
            fmt_float(row["raw_abs"], 4),
            fmt_float(row["bulk_abs"], 4),
            fmt_float(row["beta_only_error_abs"], 4),
            fmt_float(row["after_j1_abs"], 4),
            fmt_float(row["after_j2_abs"], 4),
        ]
        for row in payload["cycle_rows"]["s=1/2"]
    ]
    half_table = markdown_table(
        ["n", "|S_n|", "|bulk|", "beta-only error", "after J=1", "after J=2"],
        half_rows,
    )
    gauss_rows = [
        [
            str(row["radius"]),
            str(row["raw_count"]),
            fmt_float(row["area_channel"], 3),
            fmt_float(row["residual"], 3),
            fmt_float(row["relative_area_error"], 4),
        ]
        for row in payload["gauss_circle_rows"]
    ]
    gauss_table = markdown_table(["R", "N(R)", "pi R^2", "E(R)", "E(R)/pi R^2"], gauss_rows)

    intro = """# %s

This certificate tests the beta counterterm as the common bookkeeping bridge
between finite-cycle zeta summation, Gauss-circle area subtraction, BGK endpoint
correction, and self-certifying Q quadrature.

The tested finite-cycle sum is

```text
S_n(s) = sum_{k=1}^{n-1} [k (1 - k/n)]^{-s}.
```

The certified ledger is

```text
S_n(s) = n^(1-s) B(1-s,1-s)
       + 2 sum_{j>=0} (s)_j zeta(s-j) n^(-j)/j!
       + residual.
```

The beta term is the bulk continuum channel.  Removing it exposes `2*zeta(s)`;
the higher powers are endpoint rungs.  This is the same structural move as
subtracting `pi R^2` from a lattice count before studying Gauss-circle boundary
error.

![beta counterterm decay](beta_counterterm_decay.svg)

## Current Audit

%s

For the half-order case, the endpoint rungs line up as follows.

%s

The BGK half-order endpoint constant is

```text
beta_BGK = -zeta(1/2)/sqrt(2*pi)
         = %.15f.
```

The arithmetic Gauss-circle analogue in the same run is:

%s
""" % (
        title,
        slope_table,
        half_table,
        payload["bgk_constant"],
        gauss_table,
    )

    proof = """
## Why the Beta Counterterm Is Not a Hack

Write `x=k/n`.  The raw sum has a bulk Riemann part:

```text
[k(1-k/n)]^(-s) = n^(-s) [x(1-x)]^(-s),
sum_k n^(-s) [x_k(1-x_k)]^(-s)
  ~ n^(1-s) int_0^1 [x(1-x)]^(-s) dx
  = n^(1-s) B(1-s,1-s).
```

That term is the wrong object if the target is the zeta finite part.  It is the
same situation as Gauss circle:

```text
N(R) = pi R^2 + E(R).
```

No one studies the raw count `N(R)` as the boundary error.  The publishable
object is `E(R)` after the area channel is repaid.  Here the publishable object
is the finite part after the beta channel is repaid.

## Endpoint Rungs

Near either endpoint,

```text
[k(1-k/n)]^(-s)
  = k^(-s) (1-k/n)^(-s)
  = k^(-s) sum_{j>=0} (s)_j (k/n)^j / j!.
```

Summing the endpoint model gives

```text
sum_k k^(j-s) = zeta(s-j)
```

by analytic continuation.  There are two endpoints, so the local repayment is

```text
2 (s)_j zeta(s-j) n^(-j) / j!.
```

The numerical certificate confirms the first three orders:

```text
after beta only:        O(n^-1)
after first endpoint:   O(n^-2)
after second endpoint:  O(n^-3)
```

## Diagram

```mermaid
flowchart LR
    A["raw finite cycle sum S_n(s)"] --> B["bulk beta channel"]
    A --> C["endpoint channels"]
    B --> D["subtract n^(1-s)B(1-s,1-s)"]
    C --> E["2 zeta(s) plus endpoint rungs"]
    D --> F["certified residual"]
    E --> F
```

```mermaid
flowchart TB
    G["Gauss circle N(R)"] --> H["area pi R^2"]
    G --> I["boundary error E(R)"]
    J["cycle sum S_n(s)"] --> K["beta bulk"]
    J --> L["zeta finite part and BGK rungs"]
    H --> M["bulk subtraction"]
    K --> M
    I --> N["self-certifying residual"]
    L --> N
```

## What This Does Not Prove

This certificate does not prove RH, and it does not by itself prove a global
continuum Q theorem.  The theorem-level target would need an explicit uniform
collar/block bound of the form

```text
sup_{s in boundary block} |R_{n,J}(s)| <= certified_bound(n,J,T),
```

strong enough for a Rouche or argument-principle zero-count transfer.  What the
present certificate proves operationally is narrower and useful: the beta
counterterm removes the continuum bulk artifact, the first endpoint rungs have
the predicted zeta coefficients, and the remaining residual decays at the
predicted powers on the tested real and complex cases.
"""

    footer = f"""
## Reproduction

From the repository root:

```sh
PYTHONPATH=src python3 scripts/beta_counterterm_certificate.py \\
  --out-dir outputs/beta_counterterm_certificate
```

Machine-readable output:

```text
outputs/beta_counterterm_certificate/beta_counterterm_certificate.json
```

Generated in `{payload["elapsed_ms"]:.1f} ms` with `mpmath` precision
`{payload["mpmath_dps"]}` decimal digits.
"""

    if include_pedagogy:
        path.write_text(intro + proof + footer, encoding="utf-8")
    else:
        path.write_text(intro + footer, encoding="utf-8")


def build_payload(ns: list[int]) -> dict[str, Any]:
    cases = [
        CycleCase("s=1/4", "0.25"),
        CycleCase("s=1/2", "0.5"),
        CycleCase("s=3/4", "0.75"),
        CycleCase("s=1/2+0.3i", "0.5", "0.3"),
        CycleCase("s=1/2+2i", "0.5", "2.0"),
    ]

    cycle_rows: dict[str, list[dict[str, Any]]] = {}
    slope_rows = []
    endpoints = []
    for case in cases:
        rows, slope = evaluate_cycle_case(case, ns)
        cycle_rows[case.label] = [asdict(row) for row in rows]
        slope_rows.append(asdict(slope))
        s = mp_complex(case)
        endpoints.append(
            {
                "label": case.label,
                "s": complex_payload(s),
                "bulk_beta_at_n_max": complex_payload(beta_bulk(s, ns[-1])),
                "two_zeta_s": complex_payload(2 * mp.zeta(s)),
                "first_rung_coefficient": complex_payload(endpoint_coeff(s, 1)),
                "second_rung_coefficient": complex_payload(endpoint_coeff(s, 2)),
            }
        )

    gauss_rows = gauss_circle_rows([16, 32, 64, 128, 256])
    bgk = -mp.zeta(mp.mpf("0.5")) / mp.sqrt(2 * mp.pi)
    return {
        "certificate": "beta_counterterm",
        "mpmath_dps": mp.mp.dps,
        "n_values": ns,
        "cycle_rows": cycle_rows,
        "slope_rows": slope_rows,
        "endpoint_coefficients": endpoints,
        "gauss_circle_rows": [asdict(row) for row in gauss_rows],
        "bgk_constant": float(bgk),
        "all_cycle_cases_pass": all(row["pass_endpoint_decay"] for row in slope_rows),
        "formula": (
            "S_n(s)=n^(1-s)B(1-s,1-s)+"
            "2*sum_j (s)_j*zeta(s-j)*n^(-j)/j!+R"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dps", type=int, default=80)
    parser.add_argument("--max-n", type=int, default=8192)
    args = parser.parse_args()

    t0 = time.perf_counter()
    mp.mp.dps = args.dps
    ns = [128, 256, 512, 1024, 2048, 4096, args.max_n]
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = build_payload(ns)
    payload["elapsed_ms"] = (time.perf_counter() - t0) * 1000.0

    json_path = out_dir / "beta_counterterm_certificate.json"
    md_path = out_dir / "beta_counterterm_certificate.md"
    readme_path = out_dir / "README.md"
    svg_path = out_dir / "beta_counterterm_decay.svg"

    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_decay_svg(
        svg_path,
        {
            label: [ResidualRow(**row) for row in rows]
            for label, rows in payload["cycle_rows"].items()
        },
    )
    write_markdown(md_path, payload, "Beta Counterterm Certificate", include_pedagogy=False)
    write_markdown(readme_path, payload, "Beta Counterterm Certificate", include_pedagogy=True)

    status = "PASS" if payload["all_cycle_cases_pass"] else "FAIL"
    print(f"{status}: wrote {json_path}")
    print(f"README: {readme_path}")
    print(f"SVG: {svg_path}")


if __name__ == "__main__":
    main()
