#!/usr/bin/env python3
"""Audit the exact scale-phase chord gate on representative 3D surfaces."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inverse_shape.conic_pencil_surface import (  # noqa: E402
    aircraft_conic_bundle_qjet,
    bent_conic_tube_qjet,
    straight_conic_tube_qjet,
    tapered_conic_tube_qjet,
    toroidal_conic_bundle_qjet,
    twisted_ellipse_tube_qjet,
)
from inverse_shape.quadrature import TAU, _cos, _exp, _log, _sin, _sqrt  # noqa: E402
from inverse_shape.scale_phase_geometry import (  # noqa: E402
    certify_scale_phase_chord,
)


OUT = ROOT / "outputs" / "scale_phase_geometry"
N_SCALE = 12
N_THETA = 16


def exact_annulus():
    rhos = tuple(-0.9 + 1.8 * index / (N_SCALE - 1) for index in range(N_SCALE))
    points = tuple(
        (
            1.2
            + _exp(rho) * _cos(TAU * phase / N_THETA),
            -0.7
            + _exp(rho) * _sin(TAU * phase / N_THETA),
            2.1,
        )
        for rho in rhos
        for phase in range(N_THETA)
    )
    return points, rhos


def fitted_ring_rhos(points):
    output = []
    for scale in range(N_SCALE):
        row = points[scale * N_THETA : (scale + 1) * N_THETA]
        center = tuple(
            sum(point[axis] for point in row) / N_THETA for axis in range(3)
        )
        radius_squared = sum(
            sum((point[axis] - center[axis]) ** 2 for axis in range(3))
            for point in row
        ) / N_THETA
        output.append(_log(_sqrt(radius_squared)))
    return tuple(output)


def surface_points(surface):
    generated = surface.generate_nodes()
    return tuple(generated.points)


def cases():
    exact_points, exact_rhos = exact_annulus()
    surfaces = (
        (
            "straight_cylinder",
            straight_conic_tube_qjet(3.0, 0.7, 0.7, N_SCALE, N_THETA),
        ),
        (
            "tapered_circular_cone",
            tapered_conic_tube_qjet(
                3.0,
                0.3,
                1.0,
                0.3,
                1.0,
                N_SCALE,
                N_THETA,
            ),
        ),
        (
            "bent_circular_tube",
            bent_conic_tube_qjet(2.5, 1.4, 0.45, 0.45, N_SCALE, N_THETA),
        ),
        (
            "twisted_ellipse",
            twisted_ellipse_tube_qjet(
                3.0,
                0.7,
                0.3,
                1.8,
                N_SCALE,
                N_THETA,
            ),
        ),
        (
            "toroidal_bundle",
            toroidal_conic_bundle_qjet(
                2.0,
                0.45,
                0.3,
                N_SCALE,
                N_THETA,
            ),
        ),
        (
            "aircraft_bundle",
            aircraft_conic_bundle_qjet(4.0, N_SCALE, N_THETA),
        ),
    )
    output = [("exact_flat_annulus", exact_points, exact_rhos)]
    for name, surface in surfaces:
        points = surface_points(surface)
        output.append((name, points, fitted_ring_rhos(points)))
    return tuple(output)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, points, rhos in cases():
        sampled = certify_scale_phase_chord(points, rhos, N_THETA)
        exhaustive = certify_scale_phase_chord(
            points,
            rhos,
            N_THETA,
            exhaustive=True,
        )
        rows.append(
            {
                "shape": name,
                "nodes": len(points),
                "sampled_max_relative_residual": (
                    sampled.maximum_relative_residual
                ),
                "exhaustive_max_relative_residual": (
                    exhaustive.maximum_relative_residual
                ),
                "exhaustive_rms_relative_residual": (
                    exhaustive.relative_rms_residual
                ),
                "sampled_pairs": sampled.audited_pairs,
                "exhaustive_pairs": exhaustive.audited_pairs,
                "accepted": exhaustive.accepted,
                "stored_dense_matrix": False,
            }
        )
    with (OUT / "audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "normal_form": (
            "|X_i-X_j|^2=2 exp(rho_i+rho_j) "
            "(cosh(rho_i-rho_j)-cos(theta_i-theta_j))"
        ),
        "tolerance": 5.0e-13,
        "rows": rows,
        "production_rule": (
            "accept exact scale-phase path only when the geometry certificate "
            "passes; otherwise compile a geometry residual"
        ),
        "stored_dense_matrix": False,
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Scale-phase geometry acceptance audit",
        "",
        "| shape | sampled max | exhaustive max | exhaustive RMS | accepted |",
        "|---|---:|---:|---:|:---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['shape']} | "
            f"{row['sampled_max_relative_residual']:.3e} | "
            f"{row['exhaustive_max_relative_residual']:.3e} | "
            f"{row['exhaustive_rms_relative_residual']:.3e} | "
            f"{'yes' if row['accepted'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "The flat annulus is the exact exponential normal form, including "
            "rigid translation. The other surfaces require a geometry "
            "repayment; applying the Cauchy core to them without that residual "
            "would change the physical operator.",
            "",
            "The sampled pass is an `O(N log n_scale)` diagnostic. The "
            "fail-closed point-cloud wrapper defaults to the exhaustive "
            "streamed audit; structurally generated exponential charts use "
            "the exact core directly and need no pair audit.",
            "",
        ]
    )
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
