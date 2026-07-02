#!/usr/bin/env python3
"""Represent hard complex-plane shape pieces by tangent lines and asymptotes.

This is an algebraic atlas demo.  Smooth and cusp pieces are encoded by
complex polynomial charts z(w).  Far-field or pole-like pieces are encoded by
a rational polynomial chart

    f(t) = N(t) / D(t) = Q(t) + R(t) / D(t).

The stored carrier is the simple line/polynomial Q plus the small defect R/D;
no dense matrix is built.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "polynomial_shape_tangent_asymptote"
TAU = 2.0 * math.pi


def trim(poly: list[complex], eps: float = 1.0e-14) -> list[complex]:
    out = list(poly)
    while len(out) > 1 and abs(out[-1]) < eps:
        out.pop()
    return out


def poly_add(a: list[complex], b: list[complex]) -> list[complex]:
    n = max(len(a), len(b))
    out = [0j] * n
    for i in range(n):
        out[i] = (a[i] if i < len(a) else 0j) + (b[i] if i < len(b) else 0j)
    return trim(out)


def poly_mul(a: list[complex], b: list[complex]) -> list[complex]:
    out = [0j] * (len(a) + len(b) - 1)
    for i, av in enumerate(a):
        for j, bv in enumerate(b):
            out[i + j] += av * bv
    return trim(out)


def poly_eval(poly: list[complex], x: complex) -> complex:
    value = 0j
    for coeff in reversed(poly):
        value = value * x + coeff
    return value


def poly_divmod(numerator: list[complex], denominator: list[complex]) -> tuple[list[complex], list[complex]]:
    num = trim(numerator)
    den = trim(denominator)
    if len(den) == 1 and abs(den[0]) < 1.0e-14:
        raise ZeroDivisionError("zero denominator polynomial")
    if len(num) < len(den):
        return [0j], num
    quotient = [0j] * (len(num) - len(den) + 1)
    remainder = list(num)
    while len(remainder) >= len(den) and not (len(remainder) == 1 and abs(remainder[0]) < 1.0e-14):
        shift = len(remainder) - len(den)
        coeff = remainder[-1] / den[-1]
        quotient[shift] = coeff
        for i in range(len(den)):
            remainder[shift + i] -= coeff * den[i]
        remainder = trim(remainder)
    return trim(quotient), trim(remainder)


def factor_poly(root: float, multiplicity: int, scale: float = 1.0) -> list[complex]:
    poly = [complex(scale, 0.0)]
    for _ in range(multiplicity):
        poly = poly_mul(poly, [complex(-root, 0.0), 1.0 + 0j])
    return poly


def cardioid(theta: float) -> complex:
    w = complex(math.cos(theta), math.sin(theta))
    return w - 0.5 * (w * w + 1.0)


def nephroid(theta: float) -> complex:
    w = complex(math.cos(theta), math.sin(theta))
    return 3.0 * w - w * w * w


def tangent_direction(curve, theta0: float, eps: float = 1.0e-4) -> complex:
    z0 = curve(theta0)
    dz = curve(theta0 + eps) - z0
    if abs(dz) < 1.0e-12:
        dz = curve(theta0 + 2.0 * eps) - z0
    return dz / abs(dz)


def line_points(point: complex, direction: complex, span: float) -> tuple[list[float], list[float]]:
    return (
        [(point - span * direction).real, (point + span * direction).real],
        [(point - span * direction).imag, (point + span * direction).imag],
    )


def rational_chart() -> dict[str, object]:
    q = [-0.02 + 0j, 0.18 + 0j]
    d = [1.0 + 0j, 0j, 0j, 0j, 0j, 0j, 1.0 + 0j]
    r_touch = factor_poly(0.45, 2, scale=0.025)
    r_cross = factor_poly(-0.85, 3, scale=1.0)
    r = poly_mul(r_touch, r_cross)
    n = poly_add(poly_mul(q, d), r)
    q_recovered, r_recovered = poly_divmod(n, d)
    return {"N": n, "D": d, "Q": q_recovered, "R": r_recovered}


def rational_value(chart: dict[str, object], t: float) -> complex:
    n = chart["N"]
    d = chart["D"]
    assert isinstance(n, list)
    assert isinstance(d, list)
    return poly_eval(n, t) / poly_eval(d, t)


def asymptote_value(chart: dict[str, object], t: float) -> complex:
    q = chart["Q"]
    assert isinstance(q, list)
    return poly_eval(q, t)


def plot_polynomial_shape(ax, curve, cusp_thetas: tuple[float, ...], title: str, span: float) -> None:
    theta_values = [TAU * i / 1400 for i in range(1401)]
    zs = [curve(theta) for theta in theta_values]
    ax.plot([z.real for z in zs], [z.imag for z in zs], color="0.15", linewidth=1.4)
    for theta0 in cusp_thetas:
        z0 = curve(theta0)
        direction = tangent_direction(curve, theta0)
        xs, ys = line_points(z0, direction, span)
        ax.plot(xs, ys, color="0.55", linewidth=1.1, linestyle="--")
        ax.scatter([z0.real], [z0.imag], facecolors="white", edgecolors="0.05", s=42, zorder=5)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="0.9", linewidth=0.6)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Re z")
    ax.set_ylabel("Im z")


def make_figure(chart: dict[str, object], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.2), constrained_layout=True)
    fig.patch.set_facecolor("white")
    fig.suptitle("Tangent and asymptote carriers for complex polynomial shape charts", fontsize=13)

    plot_polynomial_shape(
        axes[0][0],
        cardioid,
        (0.0,),
        "polynomial chart: cardioid z(w)=w-(w^2+1)/2",
        0.45,
    )
    plot_polynomial_shape(
        axes[0][1],
        nephroid,
        (0.0, math.pi),
        "polynomial chart: nephroid z(w)=3w-w^3",
        0.65,
    )

    t_values = [-3.0 + 6.0 * i / 1600 for i in range(1601)]
    f_values = [rational_value(chart, t).real for t in t_values]
    q_values = [asymptote_value(chart, t).real for t in t_values]
    ax = axes[1][0]
    ax.plot(t_values, f_values, color="0.1", linewidth=1.4, label="f=N/D")
    ax.plot(t_values, q_values, color="0.55", linewidth=1.2, linestyle="--", label="Q asymptote")
    for root, multiplicity, label in ((0.45, 2, "touch m=2"), (-0.85, 3, "cross m=3")):
        y = rational_value(chart, root).real
        ax.scatter([root], [y], facecolors="white", edgecolors="0.05", s=42, zorder=5)
        ax.text(root, y + 0.08, label, ha="center", fontsize=8)
    ax.set_title("rational chart: f(t)=Q(t)+R(t)/D(t)")
    ax.set_xlabel("t = Re chart coordinate")
    ax.set_ylabel("Im z = f(t)")
    ax.grid(color="0.9", linewidth=0.6)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1][1]
    residual = [abs(rational_value(chart, t).real - asymptote_value(chart, t).real) for t in t_values]
    ax.semilogy(t_values, [max(v, 1.0e-12) for v in residual], color="0.1", linewidth=1.4)
    for root in (0.45, -0.85):
        ax.axvline(root, color="0.55", linestyle="--", linewidth=0.9)
    ax.set_title("hard defect stored separately: |R(t)/D(t)|")
    ax.set_xlabel("t")
    ax.set_ylabel("residual magnitude")
    ax.grid(color="0.9", linewidth=0.6)

    fig.savefig(path, dpi=220)
    plt.close(fig)


def coeffs_for_json(poly: list[complex]) -> list[list[float]]:
    return [[float(c.real), float(c.imag)] for c in poly]


def write_report(chart: dict[str, object], report_path: Path, figure_path: Path, csv_path: Path, json_path: Path) -> None:
    q = chart["Q"]
    r = chart["R"]
    d = chart["D"]
    assert isinstance(q, list)
    assert isinstance(r, list)
    assert isinstance(d, list)
    lines = [
        "# Complex Polynomial Shape Tangent/Asymptote Atlas",
        "",
        "The pasted rational-function rule becomes a shape atlas rule once the boundary is represented as a loop of complex numbers.",
        "",
        f"![tangent and asymptote atlas]({figure_path})",
        "",
        "## Shape Encoding",
        "",
        "A planar boundary is stored as a closed complex curve",
        "",
        "```text",
        "Gamma = { z(theta) = x(theta) + i y(theta) : 0 <= theta < 2 pi }.",
        "```",
        "",
        "After sampling, the shape is the ordered vector",
        "",
        "```text",
        "z = (z_0, ..., z_{n-1}) in C^n.",
        "```",
        "",
        "The Q operator is built from complex chords only:",
        "",
        "```text",
        "Q_ij = -1 / |z_i - z_j|^2,  i != j",
        "Q_ii = sum_{j != i} 1 / |z_i - z_j|^2.",
        "```",
        "",
        "So the complex loop is the primary shape object. Translation and rotation do not change Q, while scaling by `r` sends `Q -> r^(-2) Q`.",
        "",
        "For analytic boundaries, the circle is the normal form:",
        "",
        "```text",
        "z(theta) = Phi^(-1)(exp(i theta)),",
        "w = Phi(z) = exp(rho + i theta).",
        "```",
        "",
        "That separates phase `theta`, scale/standoff `rho`, and shape deformation. The tangent/asymptote atlas below is the local algebra used when this complex loop has hard pieces.",
        "",
        "## Rule",
        "",
        "For a rational chart, divide once:",
        "",
        "```text",
        "z(t) = x(t) + i N(t)/D(t)",
        "N(t)/D(t) = Q(t) + R(t)/D(t).",
        "```",
        "",
        "`Q` is the carrier geometry. If it is constant it is a horizontal asymptote; if it is linear it is a slant asymptote; if it has higher degree it is the polynomial asymptote. The hard piece is the residual `R/D`.",
        "",
        "For a finite polynomial chart, expand at the hard point:",
        "",
        "```text",
        "z(t0+u) = z0 + a_m u^m + a_{m+1} u^{m+1} + ...",
        "```",
        "",
        "The first nonzero coefficient `a_m` gives the tangent line or tangent cone. If `m=1` the boundary is smooth. If `m>1` the chart has a cusp or singular contact, but the tangent carrier is still explicit.",
        "",
        "## Examples",
        "",
        "- `cardioid_single_cusp`: `z(w)=w-(w^2+1)/2`, one derivative-zero cusp. The hard part is represented by the cusp tangent cone plus higher polynomial jets.",
        "- `nephroid_two_cusps`: `z(w)=3w-w^3`, two derivative-zero cusps. The two local tangent cones separate the hard endpoints.",
        "- rational chart: the quotient `Q` is the asymptote and `R/D` is the encoded defect. Roots of `R` are contacts with the asymptote. Even multiplicity touches/bounces; odd multiplicity crosses.",
        "",
        "## Recovered Division",
        "",
        f"- `deg Q = {len(q) - 1}`",
        f"- `deg R = {len(r) - 1}`",
        f"- `deg D = {len(d) - 1}`",
        "- carrier: `Q(t) = -0.02 + 0.18 t`",
        "- residual roots: `(t-0.45)^2 (t+0.85)^3` up to the stored scale",
        "",
        "## Use In Q",
        "",
        "This is the representation layer for hard geometry. Store the generating QJets for the carrier line/asymptote and the residual polynomial factors. The Q kernel still uses physical complex chords, but the singular/cusp/asymptotic behavior is classified by the tangent/asymptote atlas before the Mellin/zeta or BGK repayment is selected.",
        "",
        "## Artifacts",
        "",
        f"- CSV: `{csv_path}`",
        f"- JSON: `{json_path}`",
        f"- Figure: `{figure_path}`",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_samples(chart: dict[str, object], csv_path: Path) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t", "f", "Q", "R_over_D", "abs_residual"])
        for i in range(401):
            t = -3.0 + 6.0 * i / 400
            f = rational_value(chart, t).real
            q = asymptote_value(chart, t).real
            writer.writerow([t, f, q, f - q, abs(f - q)])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    chart = rational_chart()
    figure_path = OUT / "complex_polynomial_shape_tangent_asymptote.png"
    csv_path = OUT / "rational_asymptote_samples.csv"
    json_path = OUT / "complex_polynomial_shape_tangent_asymptote.json"
    report_path = OUT / "complex_polynomial_shape_tangent_asymptote.md"

    make_figure(chart, figure_path)
    write_samples(chart, csv_path)
    payload = {
        "cardioid_polynomial": "z(w)=w-(w^2+1)/2 on |w|=1; cusp at w=1",
        "nephroid_polynomial": "z(w)=3w-w^3 on |w|=1; cusps at w=+1 and w=-1",
        "rational_chart": {
            "N": coeffs_for_json(chart["N"]),
            "D": coeffs_for_json(chart["D"]),
            "Q": coeffs_for_json(chart["Q"]),
            "R": coeffs_for_json(chart["R"]),
            "division_identity": "N/D = Q + R/D",
        },
        "interpretation": "tangent lines/tangent cones encode finite hard points; Q encodes the asymptotic carrier; R/D encodes the hard residual.",
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(chart, report_path, figure_path, csv_path, json_path)
    print(json.dumps({"report": str(report_path), "figure": str(figure_path)}, indent=2))


if __name__ == "__main__":
    main()
