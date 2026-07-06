#!/usr/bin/env python3
"""Hardy-Voronoi flux RMS obstruction and alias-tower certificate.

The first half checks a negative claim: a dyadic Hardy-Voronoi block already has
mean-square size

    RMS(E_Q) ~ R0^(1/2) Q^(-1/2) (log Q)^(1/2),

so it cannot satisfy a pointwise target of order R0^(1/2)/Q on the whole
collar.  Supremum bounds cannot sit below the block RMS.

The second half checks the positive alias structure.  Equispaced angular
sampling has alias rungs A(m).  On the Gaussian-integer lattice every shell is
invariant under multiplication by i, so the angular coefficient W(nu,m)
vanishes unless 4 divides m.  Odd sample counts therefore kill the first three
alias levels automatically.
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
DEFAULT_OUT_DIR = ROOT / "outputs" / "hardy_voronoi_flux_certificate"


@dataclass(frozen=True)
class HardyBlockRow:
    q: int
    n_min: int
    n_max_exclusive: int
    represented_shells: int
    min_frequency_gap: float
    diagonal_sum: float
    theory_rms: float
    grid_rms: float
    grid_sup: float
    claimed_sqrt_x_over_q: float
    rms_over_claim: float
    theory_grid_relative_error: float


@dataclass(frozen=True)
class HardyCorrelationRow:
    radius_min: float
    radius_max: float
    qmax: int
    samples: int
    correlation_exact_vs_hardy: float
    residual_rms: float
    exact_error_rms: float


@dataclass(frozen=True)
class WVanishingRow:
    m: int
    divisible_by_four: bool
    max_abs_w: float
    shell_at_max: int
    pass_unit_group_rule: bool


@dataclass(frozen=True)
class AliasRungRow:
    sample_n: int
    k: int
    m: int
    divisible_by_four: bool
    alias_abs: float
    structural_zero: bool


def shell_points(max_k: int) -> tuple[list[int], dict[int, list[tuple[int, int]]]]:
    counts = [0] * (max_k + 1)
    points: dict[int, list[tuple[int, int]]] = {}
    max_abs = math.isqrt(max_k)
    for a in range(-max_abs, max_abs + 1):
        aa = a * a
        for b in range(-max_abs, max_abs + 1):
            k = aa + b * b
            if k > max_k:
                continue
            counts[k] += 1
            if k > 0:
                points.setdefault(k, []).append((a, b))
    return counts, points


def hardy_block_value(radius: float, q: int, counts: list[int]) -> float:
    total = 0.0
    lo = q * q
    hi = (2 * q) * (2 * q)
    for n in range(lo, hi):
        r2 = counts[n]
        if r2 == 0:
            continue
        total += r2 * n ** (-0.75) * math.cos(
            2.0 * math.pi * radius * math.sqrt(n) - 0.75 * math.pi
        )
    return math.sqrt(radius) * total / math.pi


def represented_frequencies(q: int, counts: list[int]) -> list[float]:
    lo = q * q
    hi = (2 * q) * (2 * q)
    return [math.sqrt(n) for n in range(lo, hi) if counts[n] > 0]


def diagonal_rms(radius_start: float, q: int, counts: list[int]) -> tuple[float, float]:
    lo = q * q
    hi = (2 * q) * (2 * q)
    diagonal = sum(counts[n] * counts[n] * n ** (-1.5) for n in range(lo, hi))
    mean_radius = 1.5 * radius_start
    rms = math.sqrt(mean_radius * diagonal / (2.0 * math.pi * math.pi))
    return rms, diagonal


def block_certificate_rows(
    radius_start: float, q_values: list[int], grid_samples: int, counts: list[int]
) -> list[HardyBlockRow]:
    rows: list[HardyBlockRow] = []
    for q in q_values:
        values = []
        for j in range(grid_samples):
            radius = radius_start + (j + 0.5) * radius_start / grid_samples
            values.append(hardy_block_value(radius, q, counts))
        grid_rms = math.sqrt(sum(v * v for v in values) / len(values))
        grid_sup = max(abs(v) for v in values)
        theory_rms, diagonal = diagonal_rms(radius_start, q, counts)
        freqs = represented_frequencies(q, counts)
        gaps = [b - a for a, b in zip(freqs, freqs[1:])]
        claim = math.sqrt(radius_start) / q
        rows.append(
            HardyBlockRow(
                q=q,
                n_min=q * q,
                n_max_exclusive=(2 * q) * (2 * q),
                represented_shells=len(freqs),
                min_frequency_gap=min(gaps) if gaps else float("nan"),
                diagonal_sum=diagonal,
                theory_rms=theory_rms,
                grid_rms=grid_rms,
                grid_sup=grid_sup,
                claimed_sqrt_x_over_q=claim,
                rms_over_claim=grid_rms / claim,
                theory_grid_relative_error=abs(grid_rms - theory_rms) / theory_rms,
            )
        )
    return rows


def lattice_count(radius: float) -> int:
    r2 = radius * radius
    limit = int(math.floor(radius))
    total = 0
    for a in range(-limit, limit + 1):
        b2 = r2 - a * a
        if b2 >= 0.0:
            total += 2 * int(math.floor(math.sqrt(b2))) + 1
    return total


def hardy_sum(radius: float, qmax: int, counts: list[int]) -> float:
    total = 0.0
    for n in range(1, qmax * qmax + 1):
        r2 = counts[n]
        if r2 == 0:
            continue
        total += r2 * n ** (-0.75) * math.cos(
            2.0 * math.pi * radius * math.sqrt(n) - 0.75 * math.pi
        )
    return math.sqrt(radius) * total / math.pi


def correlation(xs: list[float], ys: list[float]) -> float:
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = sum((x - mean_x) ** 2 for x in xs)
    den_y = sum((y - mean_y) ** 2 for y in ys)
    return numerator / math.sqrt(den_x * den_y)


def hardy_exact_correlation(
    radius_min: float, radius_max: float, qmax: int, samples: int, counts: list[int]
) -> HardyCorrelationRow:
    exact = []
    model = []
    for j in range(samples):
        radius = radius_min + (j + 0.37) * (radius_max - radius_min) / samples
        exact_error = lattice_count(radius) - math.pi * radius * radius
        exact.append(exact_error)
        model.append(hardy_sum(radius, qmax, counts))
    residual_rms = math.sqrt(sum((a - b) ** 2 for a, b in zip(exact, model)) / samples)
    exact_rms = math.sqrt(sum(a * a for a in exact) / samples)
    return HardyCorrelationRow(
        radius_min=radius_min,
        radius_max=radius_max,
        qmax=qmax,
        samples=samples,
        correlation_exact_vs_hardy=correlation(exact, model),
        residual_rms=residual_rms,
        exact_error_rms=exact_rms,
    )


def shell_w(points: list[tuple[int, int]], m: int) -> float:
    total = 0j
    for a, b in points:
        radius = math.hypot(a, b)
        total += complex(a / radius, b / radius) ** m
    return float(total.real)


def w_vanishing_rows(points: dict[int, list[tuple[int, int]]], max_m: int) -> list[WVanishingRow]:
    rows = []
    for m in range(1, max_m + 1):
        max_abs = 0.0
        shell_at_max = 0
        for nu, pts in points.items():
            value = abs(shell_w(pts, m))
            if value > max_abs:
                max_abs = value
                shell_at_max = nu
        divisible = (m % 4) == 0
        rows.append(
            WVanishingRow(
                m=m,
                divisible_by_four=divisible,
                max_abs_w=max_abs,
                shell_at_max=shell_at_max,
                pass_unit_group_rule=divisible or max_abs < 1e-12,
            )
        )
    return rows


def bessel_j_prime(order: int, x: mp.mpf) -> mp.mpf:
    return (mp.besselj(order - 1, x) - mp.besselj(order + 1, x)) / 2


def alias_amplitude(
    m: int,
    radius: float,
    eta: float,
    max_nu: int,
    points: dict[int, list[tuple[int, int]]],
) -> mp.mpf:
    total = mp.mpf("0")
    radius_mp = mp.mpf(radius)
    eta_mp = mp.mpf(eta)
    for nu, pts in points.items():
        if nu > max_nu:
            continue
        w = shell_w(pts, m)
        if abs(w) < 1e-30:
            continue
        root = mp.sqrt(nu)
        total += (
            mp.e ** (-eta_mp * root)
            * bessel_j_prime(m, 2 * mp.pi * radius_mp * root)
            * mp.mpf(w)
            / root
        )
    return -2 * radius_mp * total


def alias_rows(
    sample_ns: list[int],
    max_k: int,
    radius: float,
    eta: float,
    max_nu: int,
    points: dict[int, list[tuple[int, int]]],
) -> list[AliasRungRow]:
    rows = []
    for sample_n in sample_ns:
        for k in range(1, max_k + 1):
            m = k * sample_n
            value = alias_amplitude(m, radius, eta, max_nu, points)
            rows.append(
                AliasRungRow(
                    sample_n=sample_n,
                    k=k,
                    m=m,
                    divisible_by_four=(m % 4) == 0,
                    alias_abs=float(abs(value)),
                    structural_zero=(m % 4) != 0 and abs(value) < mp.mpf("1e-12"),
                )
            )
    return rows


def fmt_float(value: float, digits: int = 4) -> str:
    if value == 0:
        return "0"
    if abs(value) < 1e-3 or abs(value) >= 1e4:
        return f"{value:.{digits}e}"
    return f"{value:.{digits}f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    output = ["| " + " | ".join(headers) + " |"]
    output.append("|" + "|".join(["---"] * len(headers)) + "|")
    output.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(output)


def write_svg(path: Path, block_rows: list[HardyBlockRow], alias: list[AliasRungRow]) -> None:
    width = 960
    height = 520
    left = 70
    top = 38
    plot_w = 400
    plot_h = 360
    max_y = max(row.grid_rms for row in block_rows) * 1.12

    def x_for_q(q: int) -> float:
        q_min = math.log2(block_rows[0].q)
        q_max = math.log2(block_rows[-1].q)
        return left + (math.log2(q) - q_min) * plot_w / (q_max - q_min)

    def y_for_value(value: float) -> float:
        return top + (max_y - value) * plot_h / max_y

    def poly(points: list[tuple[float, float]], color: str, width_: float) -> str:
        encoded = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        return (
            f'<polyline points="{encoded}" fill="none" stroke="{color}" '
            f'stroke-width="{width_}" stroke-linejoin="round" stroke-linecap="round" />'
        )

    grid = []
    for row in block_rows:
        x = x_for_q(row.q)
        grid.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" '
            'stroke="#eeeeee" />'
        )
        grid.append(
            f'<text x="{x - 8:.2f}" y="{top + plot_h + 26}" font-size="12">{row.q}</text>'
        )
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        y = top + frac * plot_h
        value = max_y * (1 - frac)
        grid.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" '
            'stroke="#eeeeee" />'
        )
        grid.append(f'<text x="20" y="{y + 4:.2f}" font-size="12">{value:.1f}</text>')

    theory_points = [(x_for_q(row.q), y_for_value(row.theory_rms)) for row in block_rows]
    grid_points = [(x_for_q(row.q), y_for_value(row.grid_rms)) for row in block_rows]
    claim_points = [(x_for_q(row.q), y_for_value(row.claimed_sqrt_x_over_q)) for row in block_rows]

    odd_rows = [row for row in alias if row.sample_n == 17]
    bar_left = 580
    bar_top = 84
    bar_w = 34
    bar_gap = 15
    bar_h = 260
    max_alias = max(row.alias_abs for row in odd_rows) or 1.0
    bars = []
    for idx, row in enumerate(odd_rows):
        h = bar_h * row.alias_abs / max_alias
        x = bar_left + idx * (bar_w + bar_gap)
        y = bar_top + bar_h - h
        color = "#111111" if row.divisible_by_four else "#bdbdbd"
        bars.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w}" height="{h:.2f}" '
            f'fill="{color}" />'
        )
        bars.append(
            f'<text x="{x + 7:.2f}" y="{bar_top + bar_h + 22}" '
            f'font-size="12">{row.k}</text>'
        )

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff" />',
        '<text x="70" y="24" font-size="17" font-family="serif">Hardy block RMS obstruction</text>',
        '<text x="580" y="24" font-size="17" font-family="serif">Odd-n alias tower</text>',
        *grid,
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" '
        'fill="none" stroke="#222" />',
        poly(theory_points, "#111111", 2.2),
        poly(grid_points, "#666666", 2.2),
        poly(claim_points, "#bdbdbd", 2.2),
        '<text x="208" y="438" font-size="13">Q</text>',
        '<text x="20" y="252" transform="rotate(-90 20 252)" font-size="13">RMS</text>',
        '<line x1="290" y1="62" x2="334" y2="62" stroke="#111" stroke-width="2.2" />',
        '<text x="342" y="66" font-size="12">diagonal theory</text>',
        '<line x1="290" y1="82" x2="334" y2="82" stroke="#666" stroke-width="2.2" />',
        '<text x="342" y="86" font-size="12">grid RMS</text>',
        '<line x1="290" y1="102" x2="334" y2="102" stroke="#bdbdbd" stroke-width="2.2" />',
        '<text x="342" y="106" font-size="12">claimed sqrt(X)/Q</text>',
        f'<rect x="{bar_left - 18}" y="{bar_top}" width="390" height="{bar_h}" '
        'fill="none" stroke="#222" />',
        *bars,
        '<text x="724" y="394" font-size="13">alias level k for n=17</text>',
        '<text x="596" y="372" font-size="12" fill="#777">grey = structural zero</text>',
        '<text x="768" y="372" font-size="12" fill="#111">black = 4 | kn</text>',
        "</svg>",
    ]
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def write_markdown(path: Path, payload: dict[str, Any], include_proof: bool) -> None:
    block_rows = payload["hardy_block_rows"]
    alias_rows_payload = payload["alias_rows"]
    w_rows = payload["w_vanishing_rows"]
    corr = payload["hardy_exact_correlation"]

    block_table = markdown_table(
        [
            "Q",
            "theory RMS",
            "grid RMS",
            "sqrt(X)/Q",
            "RMS/claim",
            "grid sup",
            "rel. gap",
        ],
        [
            [
                str(row["q"]),
                fmt_float(row["theory_rms"], 3),
                fmt_float(row["grid_rms"], 3),
                fmt_float(row["claimed_sqrt_x_over_q"], 3),
                fmt_float(row["rms_over_claim"], 3),
                fmt_float(row["grid_sup"], 3),
                fmt_float(row["theory_grid_relative_error"], 3),
            ]
            for row in block_rows
        ],
    )
    w_table = markdown_table(
        ["m", "4 divides m", "max abs W(nu,m)", "shell", "status"],
        [
            [
                str(row["m"]),
                "yes" if row["divisible_by_four"] else "no",
                fmt_float(row["max_abs_w"], 3),
                str(row["shell_at_max"]),
                "PASS" if row["pass_unit_group_rule"] else "FAIL",
            ]
            for row in w_rows[:12]
        ],
    )
    alias_table = markdown_table(
        ["n", "k", "m=kn", "4 divides m", "abs A(m)", "structural zero"],
        [
            [
                str(row["sample_n"]),
                str(row["k"]),
                str(row["m"]),
                "yes" if row["divisible_by_four"] else "no",
                fmt_float(row["alias_abs"], 3),
                "yes" if row["structural_zero"] else "no",
            ]
            for row in alias_rows_payload
        ],
    )

    intro = f"""# Hardy-Voronoi Flux Certificate

This certificate records two facts that should be kept separate.

1. The dyadic Hardy-Voronoi block has a diagonal RMS floor.  A pointwise target
   below that floor cannot hold on a full collar.
2. The angular alias tower has exact Gaussian-unit cancellation: alias rungs
   vanish unless `4 | m`.

The radius interval in this script is `R in [X,2X]` with `X =
{payload["radius_start"]}`.  Li-Yang's Gauss-circle variable is usually
`x = R^2`, so exponent comparisons must be translated before being quoted.

![Hardy-Voronoi flux certificate](hardy_voronoi_flux_certificate.svg)

## Dyadic Block RMS

{block_table}

The RMS/claim ratio grows like `Q^(1/2)` up to logarithms, exactly as the
diagonal computation predicts.  Since `sup >= RMS`, this falsifies a uniform
block target of order `X^(1/2)/Q` on the full collar.

As a sanity check, the exact lattice-count error and the truncated
Hardy-Voronoi sum correlate strongly:

```text
corr(E_exact, Hardy_sum) = {corr["correlation_exact_vs_hardy"]:.6f}
residual_rms             = {corr["residual_rms"]:.6f}
exact_error_rms          = {corr["exact_error_rms"]:.6f}
```

## Unit-Group Alias Tower

For a lattice shell, define

```text
W(nu,m) = sum over a^2+b^2=nu of cos(m arg(a+ib)).
```

The certificate checks the Gaussian-unit rule:

{w_table}

For angular sampling with `n` equispaced nodes, the alias rungs have `m = k*n`.
Odd `n` kills `k = 1,2,3` because none of `n,2n,3n` is divisible by four:

{alias_table}
"""

    proof = """
## Diagonal Mean-Square Derivation

The Hardy-Voronoi block used here is

```text
E_Q(R) = R^(1/2)/pi
         sum_{Q^2 <= n < (2Q)^2}
           r_2(n) n^(-3/4) cos(2*pi*R*sqrt(n) - 3*pi/4).
```

The frequencies are `sqrt(n)`.  In the block `n ~ Q^2`, adjacent frequencies
are separated by about `1/Q`.  Averaging over `R in [X,2X]` with `X >> Q`,
off-diagonal integrals have denominator `sqrt(n)-sqrt(m)` and wash out.  The
diagonal gives

```text
1/X int_X^{2X} |E_Q(R)|^2 dR
  ~ X sum_{n~Q^2} r_2(n)^2 n^(-3/2)
  ~ X log(Q)/Q.
```

Therefore

```text
RMS(E_Q) ~ X^(1/2) Q^(-1/2) (log Q)^(1/2).
```

That is larger than `X^(1/2)/Q` by a factor of order `Q^(1/2)`, ignoring the
log.  Collar restriction does not remove this average-size obstruction if the
collar has full measure in the averaging variable.

This lines up with the Li-Yang diagnosis of the current circle-problem wall:
the Bombieri-Iwaniec method recovers only partial orthogonality between short
sums through large-sieve estimates; the missing cancellation is pointwise, not
merely L2.  See Li and Yang, *An improvement on Gauss's Circle Problem and
Dirichlet's Divisor Problem*, arXiv:2308.14859.

## Alias-Rung Derivation

The angular sampling error has rungs

```text
A(m) = -2R sum_nu exp(-eta*sqrt(nu))
       J_m'(2*pi*R*sqrt(nu)) W(nu,m)/sqrt(nu).
```

If a shell contains `omega`, it also contains `i*omega`, `-omega`, and
`-i*omega`.  In complex notation,

```text
1 + i^m + (-1)^m + (-i)^m = 0,    unless 4 divides m.
```

Thus `W(nu,m)=0` for every shell unless `4 | m`.  The rungs are Bessel
transforms of the angular shell coefficients, so the same structural zero
propagates into `A(m)`.
"""

    footer = f"""
## Reproduction

```sh
PYTHONPATH=src python3 scripts/hardy_voronoi_flux_certificate.py \\
  --out-dir outputs/hardy_voronoi_flux_certificate
```

Machine-readable output:

```text
outputs/hardy_voronoi_flux_certificate/hardy_voronoi_flux_certificate.json
```

Generated in `{payload["elapsed_ms"]:.1f} ms`.
"""

    path.write_text(intro + (proof if include_proof else "") + footer, encoding="utf-8")


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    q_values = [8, 16, 32, 64]
    max_block_n = (2 * max(q_values)) ** 2
    max_needed = max(max_block_n, args.alias_max_nu, args.correlation_qmax**2)
    counts, points = shell_points(max_needed)

    block_rows = block_certificate_rows(args.radius_start, q_values, args.grid_samples, counts)
    corr = hardy_exact_correlation(
        args.correlation_radius_min,
        args.correlation_radius_max,
        args.correlation_qmax,
        args.correlation_samples,
        counts,
    )
    small_points = {nu: pts for nu, pts in points.items() if nu <= args.alias_max_nu}
    w_rows = w_vanishing_rows(small_points, args.max_w_m)
    aliases = alias_rows(
        [17, 16],
        args.alias_levels,
        args.alias_radius,
        args.alias_eta,
        args.alias_max_nu,
        small_points,
    )
    return {
        "certificate": "hardy_voronoi_flux",
        "radius_start": args.radius_start,
        "radius_interval": [args.radius_start, 2 * args.radius_start],
        "grid_samples": args.grid_samples,
        "hardy_block_rows": [asdict(row) for row in block_rows],
        "hardy_exact_correlation": asdict(corr),
        "w_vanishing_rows": [asdict(row) for row in w_rows],
        "alias_rows": [asdict(row) for row in aliases],
        "alias_radius": args.alias_radius,
        "alias_eta": args.alias_eta,
        "alias_max_nu": args.alias_max_nu,
        "all_unit_group_checks_pass": all(row.pass_unit_group_rule for row in w_rows),
        "all_odd_low_aliases_zero": all(
            row.structural_zero for row in aliases if row.sample_n == 17 and row.k <= 3
        ),
        "li_yang_source": "https://arxiv.org/pdf/2308.14859",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--radius-start", type=float, default=1500.0)
    parser.add_argument("--grid-samples", type=int, default=4096)
    parser.add_argument("--correlation-radius-min", type=float, default=200.0)
    parser.add_argument("--correlation-radius-max", type=float, default=400.0)
    parser.add_argument("--correlation-qmax", type=int, default=600)
    parser.add_argument("--correlation-samples", type=int, default=80)
    parser.add_argument("--alias-radius", type=float, default=10.0)
    parser.add_argument("--alias-eta", type=float, default=0.03)
    parser.add_argument("--alias-max-nu", type=int, default=400)
    parser.add_argument("--alias-levels", type=int, default=8)
    parser.add_argument("--max-w-m", type=int, default=16)
    parser.add_argument("--dps", type=int, default=50)
    args = parser.parse_args()

    t0 = time.perf_counter()
    mp.mp.dps = args.dps
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = build_payload(args)
    payload["elapsed_ms"] = (time.perf_counter() - t0) * 1000.0

    json_path = args.out_dir / "hardy_voronoi_flux_certificate.json"
    md_path = args.out_dir / "hardy_voronoi_flux_certificate.md"
    readme_path = args.out_dir / "README.md"
    svg_path = args.out_dir / "hardy_voronoi_flux_certificate.svg"

    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    block_rows = [HardyBlockRow(**row) for row in payload["hardy_block_rows"]]
    alias_payload = [AliasRungRow(**row) for row in payload["alias_rows"]]
    write_svg(svg_path, block_rows, alias_payload)
    write_markdown(md_path, payload, include_proof=False)
    write_markdown(readme_path, payload, include_proof=True)

    status = (
        "PASS"
        if payload["all_unit_group_checks_pass"] and payload["all_odd_low_aliases_zero"]
        else "FAIL"
    )
    print(f"{status}: wrote {json_path}")
    print(f"README: {readme_path}")
    print(f"SVG: {svg_path}")


if __name__ == "__main__":
    main()
