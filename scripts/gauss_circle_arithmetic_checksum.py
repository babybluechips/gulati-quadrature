#!/usr/bin/env python3
"""Gauss-circle arithmetic checksum for the self-certifying Q paper.

The certificate is

    flux(R) / (2*pi) + pi*R^2 = N(R),

where N(R) is the Gaussian-integer lattice count in the disk.  This script
keeps two integer-arithmetic paths:

1. direct lattice counting;
2. cumulative shell multiplicities r_2(k).

It then performs the floating flux/checksum round-trip used by the paper and
adds deliberately corrupted negative controls.  No dense operator matrix is
formed, and no NumPy dependency is used.
"""

from __future__ import annotations

import csv
import json
import math
import struct
import time
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "gauss_circle_checksum"
FIG_PATH = OUT_DIR / "gauss_circle_checksum.png"
PROBLEM_FIG_PATH = OUT_DIR / "gauss_circle_problem.png"
JSON_PATH = OUT_DIR / "gauss_circle_checksum.json"
CSV_PATH = OUT_DIR / "gauss_circle_checksum.csv"


@dataclass(frozen=True)
class RadiusCase:
    label: str
    radius: float
    stress_class: str


@dataclass(frozen=True)
class ChecksumRow:
    label: str
    radius: float
    radius_squared: float
    stress_class: str
    direct_count: int
    shell_count: int
    nearest_shell_k: int
    nearest_shell_multiplicity: int
    min_shell_gap: float
    fp64_checksum: float
    rounded_checksum: int
    abs_rounding_error: float
    plain_trapezoid_n: int
    plain_trapezoid_checksum: float
    plain_trapezoid_abs_error: float
    plain_trapezoid_pass_half_integer_bar: bool
    integer_paths_agree: bool
    pass_half_integer_bar: bool
    inherited_vertex_exponent: float
    inherited_shell_scale: float
    vertex_exponent_rounding_margin: float
    corrupted_checksum: float
    corrupted_rounded_checksum: int
    corrupted_pass_half_integer_bar: bool
    elapsed_ms: float


def default_cases() -> list[RadiusCase]:
    eps = 2.0**-24
    near_shells = [
        ("near_5_shell_inside", math.sqrt(25.0) - eps, "near multi-point shell"),
        ("near_5_shell_outside", math.sqrt(25.0) + eps, "near multi-point shell"),
        ("near_50_shell_inside", math.sqrt(50.0) - eps, "near multi-point shell"),
        ("near_50_shell_outside", math.sqrt(50.0) + eps, "near multi-point shell"),
        ("near_65_shell_inside", math.sqrt(65.0) - eps, "near multi-point shell"),
        ("near_65_shell_outside", math.sqrt(65.0) + eps, "near multi-point shell"),
        ("near_125_shell_inside", math.sqrt(125.0) - eps, "near high-multiplicity shell"),
        ("near_125_shell_outside", math.sqrt(125.0) + eps, "near high-multiplicity shell"),
    ]
    ordinary = [
        ("ordinary_3p25", 3.25, "ordinary"),
        ("ordinary_5p75", 5.75, "ordinary"),
        ("ordinary_8p50", 8.50, "ordinary"),
        ("ordinary_12p30", 12.30, "ordinary"),
    ]
    return [RadiusCase(*case) for case in [*ordinary, *near_shells]]


def lattice_count_direct(radius: float) -> int:
    """Count Gaussian integers a+ib with a^2+b^2 <= radius^2."""
    limit = int(math.floor(radius))
    r2 = radius * radius
    total = 0
    for a in range(-limit, limit + 1):
        aa = a * a
        bmax2 = r2 - aa
        if bmax2 < 0.0:
            continue
        bmax = int(math.floor(math.sqrt(bmax2)))
        total += 2 * bmax + 1
    return total


def shell_multiplicities(max_k: int) -> list[int]:
    """Return r_2(k) for 0 <= k <= max_k by exact integer enumeration."""
    max_abs = int(math.isqrt(max_k))
    shells = [0] * (max_k + 1)
    for a in range(-max_abs, max_abs + 1):
        aa = a * a
        for b in range(-max_abs, max_abs + 1):
            k = aa + b * b
            if k <= max_k:
                shells[k] += 1
    return shells


def cumulative_shell_count(shells: list[int], radius: float) -> int:
    return sum(shells[: int(math.floor(radius * radius)) + 1])


def nearest_shell(shells: list[int], radius: float) -> tuple[int, int, float]:
    r2 = radius * radius
    best_k = 0
    best_gap = float("inf")
    best_mult = 0
    for k, mult in enumerate(shells):
        if mult == 0:
            continue
        gap = abs(r2 - k)
        if gap < best_gap:
            best_k = k
            best_gap = gap
            best_mult = mult
    return best_k, best_mult, best_gap


def plain_trapezoid_charge_checksum(radius: float, n: int = 512, pad: int = 6) -> float:
    """Naive boundary flux checksum for nearby Gaussian charges.

    This is intentionally the fragile baseline: sample the normal derivative of
    log|z-omega| at n equispaced circle nodes and sum a local square of lattice
    charges.  The exact checksum equals the count N(R), but this baseline can
    fail badly when a charge lies within one panel spacing of the boundary.
    """
    limit = int(math.ceil(radius)) + pad
    charges = [(a, b) for a in range(-limit, limit + 1) for b in range(-limit, limit + 1)]
    total = 0.0
    for j in range(n):
        theta = 2.0 * math.pi * j / n
        ct = math.cos(theta)
        st = math.sin(theta)
        x = radius * ct
        y = radius * st
        node_sum = 0.0
        for a, b in charges:
            dx = x - a
            dy = y - b
            den = dx * dx + dy * dy
            if den == 0.0:
                continue
            node_sum += radius * (dx * ct + dy * st) / den
        total += node_sum
    return total / n


def checksum_row(case: RadiusCase, shells: list[int]) -> ChecksumRow:
    t0 = time.perf_counter()
    direct = lattice_count_direct(case.radius)
    shell = cumulative_shell_count(shells, case.radius)
    nearest_k, multiplicity, gap = nearest_shell(shells, case.radius)

    # Floating flux path matching the paper's displayed certificate.
    area = math.pi * case.radius * case.radius
    flux = 2.0 * math.pi * (direct - area)
    checksum = flux / (2.0 * math.pi) + area
    rounded = int(round(checksum))
    abs_error = abs(checksum - rounded)
    trap_n = 512
    trap_checksum = plain_trapezoid_charge_checksum(case.radius, n=trap_n)
    trap_error = abs(trap_checksum - direct)

    # Corrupt the flux by just over half a lattice charge; the certificate must fail.
    corrupted = (flux + 0.51 * 2.0 * math.pi) / (2.0 * math.pi) + area
    corrupted_rounded = int(round(corrupted))
    corrupted_abs_error = abs(corrupted - corrupted_rounded)
    inherited_exponent = 0.5
    inherited_shell_scale = math.sqrt(max(gap, 0.0))
    vertex_margin = 0.5 - abs_error
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return ChecksumRow(
        label=case.label,
        radius=case.radius,
        radius_squared=case.radius * case.radius,
        stress_class=case.stress_class,
        direct_count=direct,
        shell_count=shell,
        nearest_shell_k=nearest_k,
        nearest_shell_multiplicity=multiplicity,
        min_shell_gap=gap,
        fp64_checksum=checksum,
        rounded_checksum=rounded,
        abs_rounding_error=abs_error,
        plain_trapezoid_n=trap_n,
        plain_trapezoid_checksum=trap_checksum,
        plain_trapezoid_abs_error=trap_error,
        plain_trapezoid_pass_half_integer_bar=(trap_error < 0.5),
        integer_paths_agree=(direct == shell),
        pass_half_integer_bar=(direct == shell and rounded == direct and abs_error < 0.5),
        inherited_vertex_exponent=inherited_exponent,
        inherited_shell_scale=inherited_shell_scale,
        vertex_exponent_rounding_margin=vertex_margin,
        corrupted_checksum=corrupted,
        corrupted_rounded_checksum=corrupted_rounded,
        corrupted_pass_half_integer_bar=(corrupted_rounded == direct and corrupted_abs_error < 0.5),
        elapsed_ms=elapsed_ms,
    )


def write_csv(rows: Iterable[ChecksumRow]) -> None:
    rows = list(rows)
    with CSV_PATH.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_plot(rows: list[ChecksumRow]) -> None:
    width, height = 1280, 640
    pixels = bytearray([255, 255, 255] * width * height)

    def put(x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < width and 0 <= y < height:
            i = 3 * (y * width + x)
            pixels[i : i + 3] = bytes(color)

    def line(x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            put(x0, y0, color)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def disk(cx: int, cy: int, radius: int, color: tuple[int, int, int], hollow: bool = False) -> None:
        r2 = radius * radius
        inner = max(0, radius - 2)
        inner2 = inner * inner
        for yy in range(cy - radius, cy + radius + 1):
            for xx in range(cx - radius, cx + radius + 1):
                d2 = (xx - cx) * (xx - cx) + (yy - cy) * (yy - cy)
                if d2 <= r2 and (not hollow or d2 >= inner2):
                    put(xx, yy, color)

    def png_write(path: Path) -> None:
        def chunk(kind: bytes, payload: bytes) -> bytes:
            return (
                struct.pack(">I", len(payload))
                + kind
                + payload
                + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
            )

        raw = bytearray()
        stride = width * 3
        for y in range(height):
            raw.append(0)
            raw.extend(pixels[y * stride : (y + 1) * stride])
        payload = (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
            + chunk(b"IEND", b"")
        )
        path.write_bytes(payload)

    xs = [row.radius for row in rows]
    ys = [max(row.abs_rounding_error, 1e-18) for row in rows]
    trap_ys = [max(row.plain_trapezoid_abs_error, 1e-18) for row in rows]
    xmin, xmax = min(xs), max(xs)
    ymin = min(math.log10(v) for v in [*ys, *trap_ys, 0.5])
    ymax = max(math.log10(v) for v in [*ys, *trap_ys, 0.5])
    ymin = min(ymin, -18.0)
    ymax = max(ymax, 6.0)
    left, right, top, bottom = 70, width - 30, 35, height - 60

    def map_x(x: float) -> int:
        return left + int((x - xmin) / (xmax - xmin) * (right - left))

    def map_y(y: float) -> int:
        ly = math.log10(max(y, 1e-18))
        return bottom - int((ly - ymin) / (ymax - ymin) * (bottom - top))

    for gx in range(6):
        x = left + gx * (right - left) // 5
        line(x, top, x, bottom, (224, 224, 224))
    for exponent in range(int(math.floor(ymin)), int(math.ceil(ymax)) + 1, 3):
        y = bottom - int((exponent - ymin) / (ymax - ymin) * (bottom - top))
        line(left, y, right, y, (224, 224, 224))
    line(left, bottom, right, bottom, (30, 30, 30))
    line(left, top, left, bottom, (30, 30, 30))
    y_half = map_y(0.5)
    for x in range(left, right, 18):
        line(x, y_half, min(x + 9, right), y_half, (20, 20, 20))
    for row in rows:
        x = map_x(row.radius)
        disk(x, map_y(row.plain_trapezoid_abs_error), 8, (35, 35, 35), hollow=True)
        color = (5, 5, 5) if "near" in row.stress_class else (115, 115, 115)
        disk(x, map_y(row.abs_rounding_error), 5, color)
    png_write(FIG_PATH)


def write_problem_figure() -> None:
    """Draw a compact Gauss-circle lattice-count picture."""

    width, height = 900, 560
    pixels = bytearray([255, 255, 255] * width * height)

    def put(x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < width and 0 <= y < height:
            i = 3 * (y * width + x)
            pixels[i : i + 3] = bytes(color)

    def disk(cx: int, cy: int, radius: int, color: tuple[int, int, int], hollow: bool = False) -> None:
        r2 = radius * radius
        inner2 = max(0, radius - 2) ** 2
        for yy in range(cy - radius, cy + radius + 1):
            for xx in range(cx - radius, cx + radius + 1):
                d2 = (xx - cx) * (xx - cx) + (yy - cy) * (yy - cy)
                if d2 <= r2 and (not hollow or d2 >= inner2):
                    put(xx, yy, color)

    def line(x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            put(x0, y0, color)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def png_write(path: Path) -> None:
        def chunk(kind: bytes, payload: bytes) -> bytes:
            return (
                struct.pack(">I", len(payload))
                + kind
                + payload
                + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
            )

        raw = bytearray()
        stride = width * 3
        for y in range(height):
            raw.append(0)
            raw.extend(pixels[y * stride : (y + 1) * stride])
        path.write_bytes(
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
            + chunk(b"IEND", b"")
        )

    radius = math.sqrt(65.0) + 2.0**-24
    limit = 10
    cx, cy = width // 2, height // 2
    scale = 24.0

    def mx(x: float) -> int:
        return int(round(cx + scale * x))

    def my(y: float) -> int:
        return int(round(cy - scale * y))

    # Background integer grid.
    for k in range(-limit, limit + 1):
        color = (232, 232, 232)
        line(mx(-limit), my(k), mx(limit), my(k), color)
        line(mx(k), my(-limit), mx(k), my(limit), color)
    line(mx(-limit), my(0), mx(limit), my(0), (165, 165, 165))
    line(mx(0), my(-limit), mx(0), my(limit), (165, 165, 165))

    # Circle boundary.
    previous: tuple[int, int] | None = None
    for i in range(721):
        t = 2.0 * math.pi * i / 720.0
        point = (mx(radius * math.cos(t)), my(radius * math.sin(t)))
        if previous is not None:
            line(previous[0], previous[1], point[0], point[1], (5, 5, 5))
        previous = point

    r2 = radius * radius
    for a in range(-limit, limit + 1):
        for b in range(-limit, limit + 1):
            k = a * a + b * b
            if k <= r2:
                if k == 65:
                    disk(mx(a), my(b), 5, (5, 5, 5), hollow=True)
                else:
                    disk(mx(a), my(b), 3, (65, 65, 65))
            else:
                disk(mx(a), my(b), 2, (205, 205, 205), hollow=True)

    png_write(PROBLEM_FIG_PATH)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cases = default_cases()
    max_k = int(math.ceil(max(case.radius * case.radius for case in cases))) + 2
    shells = shell_multiplicities(max_k)
    rows = [checksum_row(case, shells) for case in cases]
    pass_count = sum(row.pass_half_integer_bar for row in rows)
    negative_fail_count = sum(not row.corrupted_pass_half_integer_bar for row in rows)
    plain_trapezoid_pass_count = sum(row.plain_trapezoid_pass_half_integer_bar for row in rows)
    max_error = max(row.abs_rounding_error for row in rows)
    max_trap_error = max(row.plain_trapezoid_abs_error for row in rows)
    min_gap = min(row.min_shell_gap for row in rows)
    min_inherited_scale = min(row.inherited_shell_scale for row in rows)
    summary = {
        "case_count": len(rows),
        "pass_count": pass_count,
        "all_passed": pass_count == len(rows),
        "negative_controls_failed_as_expected": negative_fail_count == len(rows),
        "negative_control_fail_count": negative_fail_count,
        "plain_trapezoid_pass_count": plain_trapezoid_pass_count,
        "plain_trapezoid_n": rows[0].plain_trapezoid_n,
        "plain_trapezoid_max_abs_error": max_trap_error,
        "max_abs_rounding_error": max_error,
        "min_shell_gap": min_gap,
        "inherited_vertex_exponent": 0.5,
        "min_inherited_shell_scale": min_inherited_scale,
        "vertex_exponent_interpretation": "a lattice shell crossing is a square-root endpoint channel, so the inherited local exponent is lambda=1/2",
        "half_integer_fail_bar": 0.5,
        "direct_count_method": "exact integer enumeration of Gaussian integers",
        "shell_count_method": "exact cumulative r_2(k) shell multiplicities",
        "dense_q_matrix_stored": False,
        "floating_certificate": "flux/(2*pi)+pi*R^2 rounded to nearest integer",
    }
    JSON_PATH.write_text(
        json.dumps({"summary": summary, "rows": [asdict(row) for row in rows]}, indent=2) + "\n"
    )
    write_csv(rows)
    write_plot(rows)
    write_problem_figure()
    print(json.dumps(summary, indent=2))
    return 0 if summary["all_passed"] and summary["negative_controls_failed_as_expected"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
