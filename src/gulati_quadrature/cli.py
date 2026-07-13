"""Small CLI for the production Gulati Q public API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from gulati_quadrature import (
    SurfaceQConfig,
    build_engine,
    build_spheroid_engine,
    cosine_trace,
    cycle_certificate,
    star_boundary,
)


def _cmd_demo(args: argparse.Namespace) -> int:
    points = star_boundary(args.samples)
    values = cosine_trace(args.samples, args.mode)
    engine = build_engine(points)
    dtn = engine.apply_dtn(values)
    heat = engine.solve("heat", values, time=args.time)
    payload = {
        "samples": args.samples,
        "mode": args.mode,
        "dtn_protocol": dtn.stats["protocol"],
        "heat_protocol": heat.stats["protocol"],
        "dtn_work_units": dtn.work_units,
        "heat_work_units": heat.work_units,
        "dense_q_matrix_stored": False,
        "engine": engine.stats(),
    }
    _write_or_print(payload, args.out)
    return 0


def _cmd_certificate(args: argparse.Namespace) -> int:
    payload = cycle_certificate(args.n, radius=args.radius)
    _write_or_print(payload, args.out)
    return 0


def _cmd_surface_demo(args: argparse.Namespace) -> int:
    engine = build_spheroid_engine(
        args.equatorial_radius,
        args.polar_radius,
        args.rings,
        args.phases,
        config=SurfaceQConfig(kernel_power=args.kernel_power, leaf_size=4),
    )
    values = tuple(x + 0.2 * y - 0.1 * z * z for x, y, z in engine.points)
    result = (
        engine.apply_dtn_principal(values)
        if args.kernel_power == 3.0
        else engine.apply(values)
    )
    payload = {
        "nodes": engine.n,
        "kernel_power": args.kernel_power,
        "compression_inf_bound": result.compression_inf_bound,
        "ledger_status": result.ledger.status,
        "dense_q_matrix_stored": False,
        "engine": engine.stats(),
    }
    _write_or_print(payload, args.out)
    return 0


def _write_or_print(payload: dict[str, object], out: Path | None) -> None:
    text = json.dumps(payload, indent=2)
    if out is None:
        print(text)
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
        print({"out": str(out)})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gulati-q")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="run a production Q API smoke demo")
    demo.add_argument("--samples", type=int, default=128)
    demo.add_argument("--mode", type=int, default=5)
    demo.add_argument("--time", type=float, default=0.05)
    demo.add_argument("--out", type=Path)
    demo.set_defaults(func=_cmd_demo)

    certificate = sub.add_parser("certificate", help="emit the regular-cycle arithmetic checksum")
    certificate.add_argument("--n", type=int, default=16)
    certificate.add_argument("--radius", type=float, default=1.0)
    certificate.add_argument("--out", type=Path)
    certificate.set_defaults(func=_cmd_certificate)

    surface = sub.add_parser(
        "surface-demo",
        help="run the production matrix-free 3D surface QJet",
    )
    surface.add_argument("--rings", type=int, default=6)
    surface.add_argument("--phases", type=int, default=8)
    surface.add_argument("--equatorial-radius", type=float, default=1.0)
    surface.add_argument("--polar-radius", type=float, default=0.8)
    surface.add_argument("--kernel-power", type=float, choices=(2.0, 3.0), default=3.0)
    surface.add_argument("--out", type=Path)
    surface.set_defaults(func=_cmd_surface_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
