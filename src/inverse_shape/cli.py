"""Command-line interface for inverse-shape."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from inverse_shape.dirichlet import dirichlet_eigenvalues
from inverse_shape.geometry import BoundaryCurve, StarShapeModel, hausdorff_distance
from inverse_shape.io import load_points_csv, save_points_csv, save_vector_csv, write_json
from inverse_shape.operators import (
    dressed_gulati_hessian,
    extract_flux_from_hessian,
    gulati_laplacian,
)
from inverse_shape.reconstruction import fit_star_shape_model, reconstruct_polygon_from_gulati
from inverse_shape.spectra import fit_heat_trace_coefficients


def _cmd_summarize_boundary(args: argparse.Namespace) -> int:
    curve = BoundaryCurve(load_points_csv(args.points))
    summary = {
        "points": curve.n,
        "signed_area": curve.area,
        "perimeter": curve.perimeter,
        "centroid": curve.centroid.tolist(),
    }
    if args.json:
        write_json(args.json, summary)
    print(summary)
    return 0


def _cmd_polygon_from_gulati(args: argparse.Namespace) -> int:
    gu = np.load(args.gulati_matrix)
    result = reconstruct_polygon_from_gulati(gu)
    save_points_csv(args.out, result.points)
    print({"method": result.method, "residual_norm": result.residual_norm, "out": str(args.out)})
    return 0


def _cmd_flux_from_hessian(args: argparse.Namespace) -> int:
    points = load_points_csv(args.points)
    h_res = np.load(args.hessian)
    flux = extract_flux_from_hessian(points, h_res, neighbor_window=args.neighbor_window)
    save_vector_csv(args.out, flux, header="flux")
    print({"out": str(args.out), "min_flux": float(flux.min()), "max_flux": float(flux.max())})
    return 0


def _cmd_fit_heat(args: argparse.Namespace) -> int:
    eigenvalues = np.loadtxt(args.eigenvalues, delimiter=",", dtype=np.float64)
    t_values = np.geomspace(args.t_min, args.t_max, args.samples)
    fit = fit_heat_trace_coefficients(eigenvalues, t_values, max_order=args.max_order)
    payload = {
        "area": fit.area,
        "perimeter": fit.perimeter,
        "coefficients": fit.coefficients.tolist(),
        "exponents": fit.exponents.tolist(),
        "residual_norm": fit.residual_norm,
    }
    write_json(args.out, payload)
    print({"out": str(args.out), "area": fit.area, "perimeter": fit.perimeter})
    return 0


def _cmd_dirichlet_spectrum(args: argparse.Namespace) -> int:
    points = load_points_csv(args.points)
    values = dirichlet_eigenvalues(
        points,
        k=args.count,
        grid_size=args.grid_size,
        padding=args.padding,
    )
    save_vector_csv(args.out, values, header="eigenvalue")
    print(
        {
            "out": str(args.out),
            "count": int(len(values)),
            "grid_size": args.grid_size,
            "first": float(values[0]),
            "last": float(values[-1]),
        }
    )
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = StarShapeModel(
        center=np.array([0.0, 0.0]),
        base_radius=1.0,
        cos=np.array([0.12, -0.04, 0.03]),
        sin=np.array([0.00, 0.07, -0.02]),
    )
    points = model.boundary_points(args.samples)
    curve = BoundaryCurve(points).normalized()
    flux_true = 1.0 + 0.18 * np.cos(np.linspace(0, 2 * np.pi, curve.n, endpoint=False) * 2.0)
    h_res = dressed_gulati_hessian(curve.points, flux_true)
    flux_hat = extract_flux_from_hessian(curve.points, h_res, neighbor_window=4)
    gu = gulati_laplacian(curve.points)
    polygon_result = reconstruct_polygon_from_gulati(gu)
    fitted = fit_star_shape_model(curve.points, modes=3)
    fitted_points = fitted.boundary_points(args.samples)

    save_points_csv(out_dir / "boundary.csv", curve.points)
    save_points_csv(out_dir / "gulati_reconstruction.csv", polygon_result.points)
    save_vector_csv(out_dir / "flux_true.csv", flux_true, header="flux")
    save_vector_csv(out_dir / "flux_recovered.csv", flux_hat, header="flux")
    np.save(out_dir / "gulati.npy", gu)
    np.save(out_dir / "hadamard_residual.npy", h_res)
    write_json(
        out_dir / "summary.json",
        {
            "samples": curve.n,
            "area": curve.area,
            "perimeter": curve.perimeter,
            "gulati_reconstruction_residual": polygon_result.residual_norm,
            "flux_relative_error": float(
                np.linalg.norm(flux_hat - flux_true) / np.linalg.norm(flux_true)
            ),
            "star_model_hausdorff": hausdorff_distance(curve.points, fitted_points),
            "outputs": [
                "boundary.csv",
                "gulati.npy",
                "gulati_reconstruction.csv",
                "hadamard_residual.npy",
                "flux_true.csv",
                "flux_recovered.csv",
            ],
        },
    )
    print(
        {
            "out": str(out_dir),
            "flux_relative_error": float(
                np.linalg.norm(flux_hat - flux_true) / np.linalg.norm(flux_true)
            ),
        }
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="inverse-shape")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="run a synthetic end-to-end reconstruction demo")
    demo.add_argument("--out", type=Path, default=Path("artifacts/demo"))
    demo.add_argument("--samples", type=int, default=160)
    demo.set_defaults(func=_cmd_demo)

    summarize = sub.add_parser("summarize-boundary", help="summarize a boundary CSV")
    summarize.add_argument("points", type=Path)
    summarize.add_argument("--json", type=Path)
    summarize.set_defaults(func=_cmd_summarize_boundary)

    poly = sub.add_parser(
        "polygon-from-gulati",
        help="reconstruct boundary points from a Gulati matrix .npy",
    )
    poly.add_argument("gulati_matrix", type=Path)
    poly.add_argument("--out", type=Path, required=True)
    poly.set_defaults(func=_cmd_polygon_from_gulati)

    flux = sub.add_parser("flux-from-hessian", help="extract flux from sampled Hadamard residual")
    flux.add_argument("points", type=Path)
    flux.add_argument("hessian", type=Path)
    flux.add_argument("--out", type=Path, required=True)
    flux.add_argument("--neighbor-window", type=int, default=4)
    flux.set_defaults(func=_cmd_flux_from_hessian)

    heat = sub.add_parser("fit-heat", help="fit heat-trace coefficients from eigenvalue CSV")
    heat.add_argument("eigenvalues", type=Path)
    heat.add_argument("--out", type=Path, required=True)
    heat.add_argument("--max-order", type=int, default=4)
    heat.add_argument("--samples", type=int, default=40)
    heat.add_argument("--t-min", type=float, default=1e-3)
    heat.add_argument("--t-max", type=float, default=1e-1)
    heat.set_defaults(func=_cmd_fit_heat)

    spectrum = sub.add_parser(
        "dirichlet-spectrum",
        help="compute finite-difference Dirichlet eigenvalues from a boundary CSV",
    )
    spectrum.add_argument("points", type=Path)
    spectrum.add_argument("--out", type=Path, required=True)
    spectrum.add_argument("--count", type=int, default=8)
    spectrum.add_argument("--grid-size", type=int, default=56)
    spectrum.add_argument("--padding", type=float, default=0.16)
    spectrum.set_defaults(func=_cmd_dirichlet_spectrum)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
