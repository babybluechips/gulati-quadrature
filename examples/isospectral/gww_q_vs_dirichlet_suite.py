#!/usr/bin/env python3
"""Run a small suite of GWW isospectral-pair Q/Dirichlet comparisons."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import torch

from gww_q_vs_dirichlet import (
    GWWConfig,
    GWW_LEFT,
    GWW_RIGHT,
    TAU,
    closed,
    finite_difference_dirichlet,
    normalize_polygon,
    polygon_area,
    polygon_perimeter,
    q_apply_real,
    q_lanczos_ritz,
    relative_l2,
    sample_polygon_boundary,
)
from inverse_shape.q_dtn import build_planar_domain_qjet


ROOT = Path(__file__).resolve().parents[2]


Point = tuple[float, float]


@dataclass(frozen=True)
class SuiteCase:
    name: str
    area: float
    rotation_deg: float
    reflect_x: bool
    boundary_samples: int
    lanczos_steps: int
    dirichlet_grid_size: int


SUITE_CASES: tuple[SuiteCase, ...] = (
    SuiteCase("base_unit_n64", 1.0, 0.0, False, 64, 42, 56),
    SuiteCase("base_unit_n96", 1.0, 0.0, False, 96, 56, 64),
    SuiteCase("base_unit_n128", 1.0, 0.0, False, 128, 64, 72),
    SuiteCase("rotated_37deg_n96", 1.0, 37.0, False, 96, 56, 64),
    SuiteCase("reflected_rotated_n96", 1.0, -21.0, True, 96, 56, 64),
    SuiteCase("large_area_2p25_n96", 2.25, 12.0, False, 96, 56, 64),
    SuiteCase("small_area_0p64_n96", 0.64, -33.0, False, 96, 56, 64),
)


def transform_polygon(
    points: tuple[Point, ...],
    *,
    rotation_deg: float,
    reflect_x: bool,
) -> tuple[Point, ...]:
    angle = math.radians(rotation_deg)
    c = math.cos(angle)
    s = math.sin(angle)
    out: list[Point] = []
    for x, y in points:
        if reflect_x:
            x = -x
        out.append((c * x - s * y, s * x + c * y))
    return tuple(out)


def probe_vectors(sample_count: int) -> dict[str, torch.Tensor]:
    index = torch.arange(sample_count, dtype=torch.float64)
    theta = TAU * index / sample_count
    center = 0.63 * TAU
    wrapped = torch.atan2(torch.sin(theta - center), torch.cos(theta - center))
    probes = {
        "mixed_low": torch.cos(3.0 * theta + 0.2) + 0.35 * torch.sin(7.0 * theta - 0.4),
        "mixed_high": torch.cos(9.0 * theta - 0.1) - 0.28 * torch.sin(15.0 * theta + 0.6),
        "corner_localized": torch.exp(-0.5 * (wrapped / 0.18).square())
        - 0.45 * torch.exp(-0.5 * (torch.atan2(torch.sin(theta - 0.16 * TAU), torch.cos(theta - 0.16 * TAU)) / 0.12).square()),
    }
    return {name: values - torch.mean(values) for name, values in probes.items()}


def q_probe_separations(left_q: object, right_q: object, sample_count: int) -> dict[str, float]:
    rows: dict[str, float] = {}
    for name, probe in probe_vectors(sample_count).items():
        left_response = q_apply_real(left_q, probe)
        right_response = q_apply_real(right_q, probe)
        scale = max(
            float(torch.linalg.norm(left_response)),
            float(torch.linalg.norm(right_response)),
            1.0e-30,
        )
        rows[name] = float(torch.linalg.norm(left_response - right_response) / scale)
    return rows


def run_case(case: SuiteCase, *, q_ritz_count: int, dirichlet_eigen_count: int) -> dict[str, object]:
    left = normalize_polygon(GWW_LEFT, area=case.area)
    right = normalize_polygon(GWW_RIGHT, area=case.area)
    left = transform_polygon(left, rotation_deg=case.rotation_deg, reflect_x=case.reflect_x)
    right = transform_polygon(right, rotation_deg=case.rotation_deg, reflect_x=case.reflect_x)
    left_samples = sample_polygon_boundary(left, case.boundary_samples)
    right_samples = sample_polygon_boundary(right, case.boundary_samples)
    left_q = build_planar_domain_qjet(left_samples)
    right_q = build_planar_domain_qjet(right_samples)
    left_ritz, left_ms = q_lanczos_ritz(left_q, steps=case.lanczos_steps, count=q_ritz_count)
    right_ritz, right_ms = q_lanczos_ritz(right_q, steps=case.lanczos_steps, count=q_ritz_count)
    fd = finite_difference_dirichlet(
        left,
        right,
        count=dirichlet_eigen_count,
        grid_size=case.dirichlet_grid_size,
    )
    left_ritz_tensor = torch.tensor(left_ritz, dtype=torch.float64)
    right_ritz_tensor = torch.tensor(right_ritz, dtype=torch.float64)
    probe_separation = q_probe_separations(left_q, right_q, case.boundary_samples)
    return {
        "case": asdict(case),
        "left_area": abs(polygon_area(left)),
        "right_area": abs(polygon_area(right)),
        "left_perimeter": polygon_perimeter(left),
        "right_perimeter": polygon_perimeter(right),
        "left_vertices": left,
        "right_vertices": right,
        "left_samples": left_samples,
        "right_samples": right_samples,
        "dirichlet": {
            "exact_spectral_distance": 0.0,
            "distinguishes_domains": False,
            "finite_difference_check": fd,
        },
        "q": {
            "dense_q_matrix_stored": False,
            "left_top_ritz": left_ritz,
            "right_top_ritz": right_ritz,
            "top_ritz_relative_l2_difference": relative_l2(left_ritz_tensor, right_ritz_tensor),
            "probe_response_relative_differences": probe_separation,
            "left_lanczos_elapsed_ms": left_ms,
            "right_lanczos_elapsed_ms": right_ms,
            "protocol": left_q.apply_dtn(probe_vectors(case.boundary_samples)["mixed_low"].tolist()).stats["protocol"],
        },
    }


def write_csv(payload: dict[str, object], path: Path) -> None:
    rows = payload["cases"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "case",
                "boundary_samples",
                "area",
                "rotation_deg",
                "reflect_x",
                "dirichlet_exact_spectral_distance",
                "fd_dirichlet_relative_l2_difference",
                "q_top_ritz_relative_l2_difference",
                "q_probe_mixed_low",
                "q_probe_mixed_high",
                "q_probe_corner_localized",
                "q_dense_matrix_stored",
            ),
        )
        writer.writeheader()
        for row in rows:
            case = row["case"]
            probes = row["q"]["probe_response_relative_differences"]
            writer.writerow(
                {
                    "case": case["name"],
                    "boundary_samples": case["boundary_samples"],
                    "area": case["area"],
                    "rotation_deg": case["rotation_deg"],
                    "reflect_x": case["reflect_x"],
                    "dirichlet_exact_spectral_distance": row["dirichlet"]["exact_spectral_distance"],
                    "fd_dirichlet_relative_l2_difference": row["dirichlet"]["finite_difference_check"]["relative_l2_difference"],
                    "q_top_ritz_relative_l2_difference": row["q"]["top_ritz_relative_l2_difference"],
                    "q_probe_mixed_low": probes["mixed_low"],
                    "q_probe_mixed_high": probes["mixed_high"],
                    "q_probe_corner_localized": probes["corner_localized"],
                    "q_dense_matrix_stored": row["q"]["dense_q_matrix_stored"],
                }
            )


def write_plot(payload: dict[str, object], path: Path) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "0.05",
            "axes.labelcolor": "0.05",
            "axes.titlecolor": "0.05",
            "xtick.color": "0.05",
            "ytick.color": "0.05",
            "grid.color": "0.82",
            "font.family": "serif",
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
        }
    )
    cases = payload["cases"]
    names = [row["case"]["name"] for row in cases]
    x = list(range(len(cases)))
    q_ritz = [row["q"]["top_ritz_relative_l2_difference"] for row in cases]
    fd_dirichlet = [
        row["dirichlet"]["finite_difference_check"]["relative_l2_difference"] for row in cases
    ]
    exact_dirichlet = [0.0 for _ in cases]
    probes = ("mixed_low", "mixed_high", "corner_localized")
    heat = [
        [row["q"]["probe_response_relative_differences"][probe] for probe in probes]
        for row in cases
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.8), constrained_layout=True)

    ax = axes[0, 0]
    base = next(row for row in cases if row["case"]["name"] == "base_unit_n96")
    for offset, points, label, style, color in (
        (-1.75, base["left_vertices"], "left", "-", "0.00"),
        (1.75, base["right_vertices"], "right", (0, (3, 2)), "0.28"),
    ):
        shifted = tuple((point[0] + offset, point[1]) for point in points)
        curve = closed(shifted)
        ax.plot(
            [point[0] for point in curve],
            [point[1] for point in curve],
            linestyle=style,
            color=color,
            linewidth=1.8,
            label=label,
        )
        ax.scatter([point[0] for point in shifted], [point[1] for point in shifted], color=color, s=8)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("representative GWW pair")
    ax.grid(True, linewidth=0.45)
    ax.legend(loc="best", frameon=False)

    ax = axes[0, 1]
    ax.plot(x, exact_dirichlet, color="0.0", linewidth=1.4, label="exact Dirichlet")
    ax.plot(x, fd_dirichlet, color="0.58", linestyle=(0, (4, 2)), marker="o", markersize=3, label="FD artifact")
    ax.plot(x, q_ritz, color="0.18", linestyle=(0, (1, 1)), marker="s", markersize=3, label="Q Ritz")
    ax.set_title("Dirichlet blind spot versus Q separation")
    ax.set_ylabel("relative difference")
    ax.set_xticks(x)
    ax.set_xticklabels([str(index + 1) for index in x])
    ax.grid(True, linewidth=0.45)
    ax.legend(loc="best", frameon=False)

    ax = axes[1, 0]
    image = ax.imshow(heat, cmap="Greys", aspect="auto", interpolation="nearest")
    ax.set_title("Q response separation across probes")
    ax.set_xlabel("boundary probe")
    ax.set_ylabel("suite case")
    ax.set_xticks(range(len(probes)))
    ax.set_xticklabels(probes, rotation=20, ha="right")
    ax.set_yticks(x)
    ax.set_yticklabels([str(index + 1) for index in x])
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.02)
    colorbar.set_label("relative response difference")
    colorbar.outline.set_linewidth(0.45)

    ax = axes[1, 1]
    ax.axis("off")
    table_rows = [
        [
            str(index + 1),
            name.replace("_", " "),
            f"{q_ritz[index]:.3f}",
            f"{max(heat[index]):.3f}",
            f"{fd_dirichlet[index]:.3f}",
        ]
        for index, name in enumerate(names)
    ]
    table = ax.table(
        cellText=table_rows,
        colLabels=("id", "case", "Q Ritz", "max Qf", "FD Dir."),
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=(0.08, 0.47, 0.14, 0.14, 0.14),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.4)
    table.scale(1.0, 1.25)
    for cell in table.get_celld().values():
        cell.set_edgecolor("0.72")
        cell.set_linewidth(0.4)
    ax.set_title("suite summary", pad=8)

    for ax in axes.ravel()[:3]:
        for spine in ax.spines.values():
            spine.set_linewidth(0.7)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_suite(*, q_ritz_count: int, dirichlet_eigen_count: int) -> dict[str, object]:
    cases = [
        run_case(case, q_ritz_count=q_ritz_count, dirichlet_eigen_count=dirichlet_eigen_count)
        for case in SUITE_CASES
    ]
    return {
        "description": "Similarity/resolution/probe sweep for the GWW isospectral pair.",
        "truth": "All cases are exact Dirichlet-isospectral pairs; finite-difference differences are discretization artifacts.",
        "q_operator": "matrix-free PlanarDomainQJet inverse-square chord DtN generator",
        "dense_q_matrix_stored": False,
        "single_case_config_reference": asdict(GWWConfig()),
        "references": (
            "https://arxiv.org/abs/math/9207215",
            "https://www.comsol.com/model/download/1166951/models.mph.isospectral_drums.pdf",
            "https://eudml.org/doc/144038",
        ),
        "cases": cases,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs" / "gww_isospectral_q_vs_dirichlet_suite")
    parser.add_argument("--q-ritz-count", type=int, default=10)
    parser.add_argument("--dirichlet-eigen-count", type=int, default=8)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = run_suite(q_ritz_count=args.q_ritz_count, dirichlet_eigen_count=args.dirichlet_eigen_count)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / "gww_q_vs_dirichlet_suite.json"
    csv_path = args.out_dir / "gww_q_vs_dirichlet_suite.csv"
    png_path = args.out_dir / "gww_q_vs_dirichlet_suite.png"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(payload, csv_path)
    if not args.no_plot:
        write_plot(payload, png_path)
    rows = payload["cases"]
    q_ritz = [row["q"]["top_ritz_relative_l2_difference"] for row in rows]
    q_probe = [
        max(row["q"]["probe_response_relative_differences"].values())
        for row in rows
    ]
    fd_dirichlet = [
        row["dirichlet"]["finite_difference_check"]["relative_l2_difference"]
        for row in rows
    ]
    print(
        json.dumps(
            {
                "case_count": len(rows),
                "json": str(json_path),
                "csv": str(csv_path),
                "png": None if args.no_plot else str(png_path),
                "dirichlet_exact_spectral_distance_all_cases": 0.0,
                "fd_dirichlet_relative_l2_range": [min(fd_dirichlet), max(fd_dirichlet)],
                "q_top_ritz_relative_l2_range": [min(q_ritz), max(q_ritz)],
                "q_probe_response_relative_difference_range": [min(q_probe), max(q_probe)],
                "dense_q_matrix_stored": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
