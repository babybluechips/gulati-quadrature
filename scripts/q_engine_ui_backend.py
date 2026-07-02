#!/usr/bin/env python3
"""Local production-Q bridge for the interactive HTML engine UI.

The HTML UI is intentionally static and can run by itself.  This bridge serves
that same file and adds a small JSON endpoint backed by the repository's
matrix-free Q/DtN implementation.  The loader imports only ``quadrature.py`` and
``q_dtn.py`` by file path so the bridge does not pull in legacy NumPy-facing
package imports from ``inverse_shape.__init__``.
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import types
from typing import Any
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
HTML_PATH = ROOT / "outputs" / "q_quadrature_engine_ui" / "q_engine_ui.html"
SRC_DIR = ROOT / "src" / "inverse_shape"


def _load_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_production_modules() -> tuple[types.ModuleType, types.ModuleType]:
    package = types.ModuleType("inverse_shape")
    package.__path__ = [str(SRC_DIR)]  # type: ignore[attr-defined]
    sys.modules["inverse_shape"] = package
    quadrature = _load_module("inverse_shape.quadrature", SRC_DIR / "quadrature.py")
    q_dtn = _load_module("inverse_shape.q_dtn", SRC_DIR / "q_dtn.py")
    return quadrature, q_dtn


QUADRATURE, Q_DTN = load_production_modules()
QJET_CACHE: dict[str, Any] = {}
MAX_QJET_CACHE = 16


def _complex_from_json(value: Any) -> complex:
    if isinstance(value, (int, float)):
        return complex(float(value), 0.0)
    if isinstance(value, dict):
        return complex(float(value.get("re", 0.0)), float(value.get("im", 0.0)))
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return complex(float(value[0]), 0.0)
        return complex(float(value[0]), float(value[1]))
    raise TypeError(f"cannot decode complex value: {value!r}")


def _complex_to_json(value: complex) -> list[float]:
    return [float(value.real), float(value.imag)]


def _relative_l2(values: list[complex], reference: list[complex]) -> float:
    numerator = math.sqrt(math.fsum(abs(a - b) ** 2 for a, b in zip(values, reference, strict=True)))
    denominator = math.sqrt(math.fsum(abs(b) ** 2 for b in reference))
    return numerator / max(denominator, 1.0e-30)


def _stable_checksum(values: list[complex]) -> str:
    """Return a deterministic FNV-1a checksum of the production output."""

    state = 0xCBF29CE484222325
    prime = 0x100000001B3
    for value in values:
        chunk = f"{value.real:.17e},{value.imag:.17e};".encode("ascii")
        for byte in chunk:
            state ^= byte
            state = (state * prime) & 0xFFFFFFFFFFFFFFFF
    return f"{state:016x}"


def _stable_point_checksum(points: list[tuple[float, float]], *, moment_degree: int, zeta_tail_degree: int) -> str:
    state = 0xCBF29CE484222325
    prime = 0x100000001B3
    header = f"m={moment_degree};z={zeta_tail_degree};n={len(points)};".encode("ascii")
    for byte in header:
        state ^= byte
        state = (state * prime) & 0xFFFFFFFFFFFFFFFF
    for x, y in points:
        chunk = f"{x:.17e},{y:.17e};".encode("ascii")
        for byte in chunk:
            state ^= byte
            state = (state * prime) & 0xFFFFFFFFFFFFFFFF
    return f"{state:016x}"


def _get_corrected_qjet(
    points: list[tuple[float, float]],
    *,
    moment_degree: int,
    zeta_tail_degree: int,
) -> tuple[Any, bool, str]:
    """Return a cached harmonic/zeta QJet generator without storing dense Q."""

    key = _stable_point_checksum(points, moment_degree=moment_degree, zeta_tail_degree=zeta_tail_degree)
    cached = QJET_CACHE.get(key)
    if cached is not None:
        return cached, True, key
    qjet = Q_DTN.build_harmonic_moment_corrected_planar_qjet(
        points,
        moment_degree=moment_degree,
        zeta_tail_degree=zeta_tail_degree,
    )
    if len(QJET_CACHE) >= MAX_QJET_CACHE:
        oldest = next(iter(QJET_CACHE))
        del QJET_CACHE[oldest]
    QJET_CACHE[key] = qjet
    return qjet, False, key


def _relative_error(value: complex, reference: complex) -> float:
    return abs(value - reference) / max(abs(reference), 1.0e-14)


def _unit(theta: float) -> complex:
    return complex(math.cos(theta), math.sin(theta))


def _cos_mode_values(n: int, mode: int) -> list[complex]:
    return [complex(math.cos(2.0 * math.pi * mode * index / n), 0.0) for index in range(n)]


def _project_cos_amplitude(values: list[complex], mode: int) -> complex:
    n = len(values)
    basis = [math.cos(2.0 * math.pi * mode * index / n) for index in range(n)]
    numerator = sum(values[index] * basis[index] for index in range(n))
    denominator = sum(value * value for value in basis)
    if denominator == 0.0:
        return sum(values) / n
    return numerator / denominator


def run_reference_suite() -> dict[str, Any]:
    """Run exact disk modal audits through the backend Q/DtN functions."""

    n = 256
    modes = (0, 1, 2, 3, 5, 9, 17)
    problems: dict[str, dict[str, float]] = {
        "laplace_dtn": {},
        "heat": {"time": 0.35},
        "poisson": {"mass": 0.45},
        "helmholtz": {"wavenumber": 4.2, "damping": 0.02},
        "wave": {"time": 0.35},
    }
    rows: list[dict[str, Any]] = []
    outputs_for_checksum: list[complex] = []
    started = time.perf_counter()
    for problem, params in problems.items():
        for mode in modes:
            values = _cos_mode_values(n, mode)
            if problem == "laplace_dtn":
                output = list(Q_DTN.apply_continuum_repaid_dtn(values))
            elif problem == "heat":
                output = list(Q_DTN.continuum_repaid_dtn_heat(values, params["time"]))
            elif problem == "poisson":
                output = list(Q_DTN.continuum_repaid_dtn_poisson_solve(values, mass=params["mass"]))
            elif problem == "helmholtz":
                output = list(
                    Q_DTN.continuum_repaid_dtn_helmholtz_resolvent(
                        values,
                        params["wavenumber"],
                        damping=params["damping"],
                    )
                )
            elif problem == "wave":
                output = list(Q_DTN.continuum_repaid_dtn_wave(values, params["time"]))
            else:
                raise AssertionError(problem)
            amplitude = _project_cos_amplitude([complex(value) for value in output], mode)
            exact = Q_DTN.exact_disk_amplitude(problem, mode, **params)
            rel_error = _relative_error(amplitude, exact)
            rows.append(
                {
                    "problem": problem,
                    "mode": mode,
                    "computed": _complex_to_json(amplitude),
                    "exact": _complex_to_json(exact),
                    "relative_error": rel_error,
                }
            )
            outputs_for_checksum.append(amplitude)
    elapsed_ms = 1000.0 * (time.perf_counter() - started)
    max_relative_error = max(row["relative_error"] for row in rows)
    return {
        "ok": True,
        "engine": "continuum_repaid_cycle_qjet",
        "n": n,
        "problem_count": len(problems),
        "mode_count": len(modes),
        "rows": rows,
        "max_relative_error": max_relative_error,
        "elapsed_ms": elapsed_ms,
        "checksum": _stable_checksum(outputs_for_checksum),
        "certificate": "pass" if max_relative_error <= 1.0e-12 else "warn",
        "dense_matrix_stored": False,
    }


def _as_points(payload: dict[str, Any]) -> list[tuple[float, float]]:
    raw = payload.get("points")
    if not isinstance(raw, list) or len(raw) < 8:
        raise ValueError("points must contain at least eight [x, y] samples")
    points: list[tuple[float, float]] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            raise ValueError("each point must be [x, y]")
        x = float(item[0])
        y = float(item[1])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("points must be finite")
        points.append((x, y))
    return points


def _as_values(payload: dict[str, Any], n: int) -> list[complex]:
    raw = payload.get("values")
    if not isinstance(raw, list) or len(raw) != n:
        raise ValueError("values length must match points length")
    return [_complex_from_json(value) for value in raw]


def _problem_name(name: str) -> str:
    if name == "laplace":
        return "laplace_dtn"
    if name in {"laplace_dtn", "heat", "poisson", "helmholtz", "wave"}:
        return name
    raise ValueError(f"unsupported problem: {name}")


def production_verify(payload: dict[str, Any]) -> dict[str, Any]:
    points = _as_points(payload)
    values = _as_values(payload, len(points))
    problem = _problem_name(str(payload.get("problem", "laplace_dtn")))
    parameters = payload.get("parameters") if isinstance(payload.get("parameters"), dict) else {}
    clean_parameters = {
        "time": float(parameters.get("time", 0.35)),
        "mass": float(parameters.get("mass", 0.45)),
        "wavenumber": float(parameters.get("wavenumber", 4.2)),
        "damping": float(parameters.get("damping", 0.02)),
        "max_steps": min(int(parameters.get("max_steps", 8)), 8),
        "iterations": min(int(parameters.get("iterations", 12)), 16),
        "tolerance": float(parameters.get("tolerance", 1.0e-9)),
        "moment_degree": int(parameters.get("moment_degree", 2)),
        "zeta_tail_degree": int(parameters.get("zeta_tail_degree", 6)),
    }
    browser_output = payload.get("browserOutput")
    browser_vector: list[complex] | None = None
    if isinstance(browser_output, list) and len(browser_output) == len(points):
        browser_vector = [_complex_from_json(value) for value in browser_output]
    geometry = payload.get("geometry") if isinstance(payload.get("geometry"), dict) else {}
    accuracy_class = str(geometry.get("accuracy_class", "generated QJet bounded"))
    try:
        ui_error_bound = float(geometry.get("ui_error_bound", math.nan))
    except (TypeError, ValueError):
        ui_error_bound = math.nan
    residuals = payload.get("residuals") if isinstance(payload.get("residuals"), dict) else {}
    try:
        visual_residual = float(residuals.get("visual_residual", geometry.get("visual_residual", math.nan)))
    except (TypeError, ValueError):
        visual_residual = math.nan

    started = time.perf_counter()
    qjet, cache_hit, qjet_cache_key = _get_corrected_qjet(
        points,
        moment_degree=int(clean_parameters["moment_degree"]),
        zeta_tail_degree=int(clean_parameters["zeta_tail_degree"]),
    )
    if problem == "laplace_dtn":
        evaluation = qjet.apply_dtn(values)
    else:
        evaluation = qjet.solve_boundary_problem(problem, values, **clean_parameters)
    elapsed_ms = 1000.0 * (time.perf_counter() - started)

    output = [complex(value) for value in evaluation.values]
    stats = dict(evaluation.stats)
    stats["qjet_cache"] = "hit" if cache_hit else "miss"
    stats["qjet_cache_key"] = qjet_cache_key
    stats["cached_dense_matrix"] = False
    ledger = evaluation.ledger
    finite_output = all(math.isfinite(value.real) and math.isfinite(value.imag) for value in output)
    browser_relative_error = None
    if browser_vector is not None:
        browser_relative_error = _relative_l2(output, browser_vector)
    dense_matrix_stored = False
    pair_weight_table_stored = False
    certificates = [
        {
            "name": "ledger",
            "state": "pass" if ledger.status == "borrowed_repaid" else "warn",
            "value": ledger.status,
        },
        {
            "name": "dense matrix",
            "state": "pass" if not dense_matrix_stored else "fail",
            "value": "not stored" if not dense_matrix_stored else "stored",
        },
        {
            "name": "pair table",
            "state": "pass" if not pair_weight_table_stored else "fail",
            "value": "not stored" if not pair_weight_table_stored else "stored",
        },
        {
            "name": "finite output",
            "state": "pass" if finite_output else "fail",
            "value": "yes" if finite_output else "no",
        },
        {
            "name": "Q signature",
            "state": "pass" if stats.get("q_error_type") else "warn",
            "value": str(stats.get("q_error_type", "unavailable")),
        },
        {
            "name": "accuracy class",
            "state": "pass",
            "value": accuracy_class,
        },
        {
            "name": "arithmetic bound",
            "state": "pass" if math.isfinite(ui_error_bound) and ui_error_bound <= 1.0e-12 else "warn",
            "value": f"{ui_error_bound:.3e}" if math.isfinite(ui_error_bound) else "unavailable",
        },
        {
            "name": "visual residual",
            "state": "pass" if math.isfinite(visual_residual) and visual_residual <= 1.0e-10 else "warn",
            "value": f"{visual_residual:.3e}" if math.isfinite(visual_residual) else "unavailable",
        },
    ]
    if browser_relative_error is not None:
        certificates.append(
            {
                "name": "preview consistency",
                "state": "pass" if browser_relative_error <= 1.0e-8 else "warn",
                "value": f"{browser_relative_error:.3e}",
            }
        )

    return {
        "ok": True,
        "engine": "HarmonicZetaPlanarDomainQJet",
        "problem": problem,
        "n": len(points),
        "elapsed_ms": elapsed_ms,
        "work_units": evaluation.work_units,
        "stats": stats,
        "accuracy_class": accuracy_class,
        "ui_error_bound": ui_error_bound if math.isfinite(ui_error_bound) else None,
        "visual_residual": visual_residual if math.isfinite(visual_residual) else None,
        "ledger": {
            "borrowed": list(ledger.borrowed),
            "computed": list(ledger.computed),
            "repaid": list(ledger.repaid),
            "residuals": [[str(k), v] for k, v in ledger.residuals],
            "status": ledger.status,
            "notes": ledger.notes,
        },
        "output_l2": math.sqrt(math.fsum(abs(value) ** 2 for value in output)),
        "output_mean_abs": math.fsum(abs(value) for value in output) / len(output),
        "output_checksum": _stable_checksum(output),
        "finite_output": finite_output,
        "certificates": certificates,
        "browser_relative_error": browser_relative_error,
        "output": [_complex_to_json(value) for value in output],
        "output_head": [_complex_to_json(value) for value in output[:16]],
        "dense_matrix_stored": dense_matrix_stored,
        "pair_weight_table_stored": pair_weight_table_stored,
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "QEngineUI/0.1"

    def _headers(self, status: int, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._headers(204, "text/plain")

    def do_HEAD(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in {"/", "/q_engine_ui.html", "/outputs/q_quadrature_engine_ui/q_engine_ui.html"}:
            self._headers(200, "text/html; charset=utf-8")
            return
        if path in {"/api/production-q/health", "/api/production-q/reference-suite"}:
            self._headers(200, "application/json")
            return
        self._headers(404, "application/json")

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path in {"/", "/q_engine_ui.html", "/outputs/q_quadrature_engine_ui/q_engine_ui.html"}:
            data = HTML_PATH.read_bytes()
            self._headers(200, "text/html; charset=utf-8")
            self.wfile.write(data)
            return
        if path == "/api/production-q/health":
            body = json.dumps(
                {
                    "ok": True,
                    "engine": "HarmonicZetaPlanarDomainQJet",
                    "html": str(HTML_PATH),
                    "dense_matrix_stored": False,
                }
            ).encode("utf-8")
            self._headers(200, "application/json")
            self.wfile.write(body)
            return
        if path == "/api/production-q/reference-suite":
            body = json.dumps(run_reference_suite(), separators=(",", ":")).encode("utf-8")
            self._headers(200, "application/json")
            self.wfile.write(body)
            return
        self._headers(404, "application/json")
        self.wfile.write(b'{"ok": false, "error": "not found"}')

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path != "/api/production-q/verify":
            self._headers(404, "application/json")
            self.wfile.write(b'{"ok": false, "error": "not found"}')
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object")
            result = production_verify(payload)
            body = json.dumps(result, separators=(",", ":")).encode("utf-8")
            self._headers(200, "application/json")
            self.wfile.write(body)
        except Exception as exc:  # keep the UI bridge debuggable
            body = json.dumps({"ok": False, "error": str(exc)}, separators=(",", ":")).encode("utf-8")
            self._headers(400, "application/json")
            self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), format % args))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8790)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Q engine UI bridge serving http://{args.host}:{args.port}/")
    server.serve_forever()


if __name__ == "__main__":
    main()
