#!/usr/bin/env python3
"""
Numerical falsification checks for completed finite cycle traces.

The tests here are necessary conditions, not proofs. Bernoulli/exponential
completion factors do not move zeros, so the first rejection test is simply:
does the finite determinant in the completed variable have non-real zeros?

For each natural transfer rule, the script also checks finite Jensen
polynomials, contiguous Toeplitz/Pólya-frequency minors, and a sampled
Hermite-Biehler inequality for the candidate E = A - iB where A/B are the
even/odd parts of the determinant polynomial.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
from dataclasses import dataclass

import numpy as np

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from prime_orbit_trace_lab import (  # noqa: E402
    TRANSFER_RULES,
    make_boundary,
    parse_int_list,
    parse_rule_list,
    transfer_for_args,
)


@dataclass(frozen=True)
class ZeroCheck:
    count: int
    max_abs_imag: float
    max_rel_imag: float


@dataclass(frozen=True)
class JensenCheck:
    checked: int
    failures: int
    worst_rel_imag: float


@dataclass(frozen=True)
class PFCheck:
    checked: int
    failures: int
    min_minor: float
    orientation: str


@dataclass(frozen=True)
class HBCheck:
    checked: int
    failures: int
    min_margin: float


def parse_float_list(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("empty float list")
    return values


def determinant_coefficients(transfer: np.ndarray, completed_form: str) -> np.ndarray:
    eigvals = np.linalg.eigvals(transfer)
    coeffs = np.array([1.0 + 0.0j])
    for lam in eigvals:
        coeffs = np.convolve(coeffs, np.array([1.0 + 0.0j, -lam], dtype=np.complex128))

    if completed_form == "det":
        out = coeffs
    elif completed_form == "even":
        signs = (-1.0) ** np.arange(len(coeffs))
        out = np.convolve(coeffs, coeffs * signs)
    else:
        raise ValueError(f"unsupported completed form: {completed_form}")

    real = np.real_if_close(out, tol=1000)
    if np.iscomplexobj(real):
        imag = float(np.max(np.abs(np.imag(real))))
        scale = float(max(1.0, np.max(np.abs(real))))
        if imag / scale > 1.0e-8:
            raise ValueError(f"determinant coefficients are not numerically real: rel_imag={imag / scale:.3e}")
        real = np.real(real)
    return trim_coefficients(np.asarray(real, dtype=np.float64))


def determinant_zeros(transfer: np.ndarray, completed_form: str, eig_floor: float = 1.0e-12) -> np.ndarray:
    eigvals = np.linalg.eigvals(transfer)
    finite = eigvals[np.abs(eigvals) > eig_floor]
    roots = 1.0 / finite
    if completed_form == "even":
        roots = np.concatenate([roots, -roots])
    elif completed_form != "det":
        raise ValueError(f"unsupported completed form: {completed_form}")
    return roots


def trim_coefficients(coeffs: np.ndarray, tol: float = 1.0e-14) -> np.ndarray:
    out = np.asarray(coeffs, dtype=np.float64).copy()
    scale = float(max(1.0, np.max(np.abs(out)))) if len(out) else 1.0
    while len(out) > 1 and abs(out[-1]) <= tol * scale:
        out = out[:-1]
    return out


def zero_check(roots: np.ndarray, tol: float) -> ZeroCheck:
    if len(roots) == 0:
        return ZeroCheck(count=0, max_abs_imag=0.0, max_rel_imag=0.0)
    abs_imag = np.abs(np.imag(roots))
    rel_imag = abs_imag / np.maximum(1.0, np.abs(roots))
    return ZeroCheck(
        count=int(np.count_nonzero(rel_imag > tol)),
        max_abs_imag=float(np.max(abs_imag)),
        max_rel_imag=float(np.max(rel_imag)),
    )


def jensen_polynomial_coefficients(coeffs: np.ndarray, shift: int, degree: int) -> np.ndarray:
    parts = []
    max_log = -math.inf
    for j in range(degree + 1):
        k = shift + j
        c = float(coeffs[k])
        if c == 0.0:
            parts.append((0.0, -math.inf))
            continue
        log_abs = math.log(abs(c)) + math.lgamma(k + 1.0) + math.log(math.comb(degree, j))
        parts.append((math.copysign(1.0, c), log_abs))
        max_log = max(max_log, log_abs)

    if not math.isfinite(max_log):
        return np.zeros(degree + 1, dtype=np.float64)

    out = np.zeros(degree + 1, dtype=np.float64)
    for j, (sign, log_abs) in enumerate(parts):
        if math.isfinite(log_abs):
            out[j] = sign * math.exp(log_abs - max_log)
    return trim_coefficients(out)


def check_jensen(coeffs: np.ndarray, max_degree: int, tol: float) -> JensenCheck:
    n = len(coeffs) - 1
    checked = 0
    failures = 0
    worst = 0.0
    for degree in range(2, min(max_degree, n) + 1):
        for shift in range(0, n - degree + 1):
            poly = jensen_polynomial_coefficients(coeffs, shift, degree)
            if len(poly) <= 1:
                continue
            roots = np.roots(poly[::-1])
            rel_imag = np.abs(np.imag(roots)) / np.maximum(1.0, np.abs(roots))
            max_rel = float(np.max(rel_imag)) if len(rel_imag) else 0.0
            checked += 1
            worst = max(worst, max_rel)
            if max_rel > tol:
                failures += 1
    return JensenCheck(checked=checked, failures=failures, worst_rel_imag=worst)


def oriented_pf_sequence(coeffs: np.ndarray, tol: float) -> tuple[np.ndarray, str]:
    k = np.arange(len(coeffs))
    candidates = [
        (coeffs, "raw"),
        (-coeffs, "-raw"),
        (coeffs * ((-1.0) ** k), "alternating"),
        (-coeffs * ((-1.0) ** k), "-alternating"),
    ]
    best_seq = candidates[0][0]
    best_name = candidates[0][1]
    best_key = (math.inf, -math.inf)
    for seq, name in candidates:
        negatives = int(np.count_nonzero(seq < -tol))
        min_value = float(np.min(seq)) if len(seq) else 0.0
        key = (negatives, -min_value)
        if key < best_key:
            best_seq = seq
            best_name = name
            best_key = key
    scale = float(max(1.0, np.max(np.abs(best_seq)))) if len(best_seq) else 1.0
    return np.asarray(best_seq / scale, dtype=np.float64), best_name


def toeplitz_entry(seq: np.ndarray, row: int, col: int) -> float:
    k = col - row
    if k < 0 or k >= len(seq):
        return 0.0
    return float(seq[k])


def check_pf_minors(coeffs: np.ndarray, max_order: int, tol: float) -> PFCheck:
    seq, orientation = oriented_pf_sequence(coeffs, tol)
    width = len(seq)
    checked = 0
    failures = 0
    min_minor = math.inf

    for order in range(1, max_order + 1):
        for row0 in range(width):
            for col0 in range(width):
                minor = np.empty((order, order), dtype=np.float64)
                for i in range(order):
                    for j in range(order):
                        minor[i, j] = toeplitz_entry(seq, row0 + i, col0 + j)
                det = float(np.linalg.det(minor))
                checked += 1
                min_minor = min(min_minor, det)
                norm = float(max(1.0, np.linalg.norm(minor, ord=2) ** order))
                if det < -tol * norm:
                    failures += 1

    if not math.isfinite(min_minor):
        min_minor = float("nan")
    return PFCheck(checked=checked, failures=failures, min_minor=min_minor, orientation=orientation)


def polyval_ascending(coeffs: np.ndarray, z: complex) -> complex:
    acc = 0.0 + 0.0j
    for c in reversed(coeffs):
        acc = acc * z + float(c)
    return acc


def check_hermite_biehler(
    coeffs: np.ndarray,
    *,
    x_radius: float,
    x_count: int,
    y_values: list[float],
    tol: float,
) -> HBCheck:
    scaled = coeffs / max(1.0, float(np.max(np.abs(coeffs))))
    even = scaled.copy()
    odd = scaled.copy()
    even[1::2] = 0.0
    odd[0::2] = 0.0

    checked = 0
    failures = 0
    min_margin = math.inf
    for x in np.linspace(-x_radius, x_radius, x_count):
        for y in y_values:
            if y <= 0.0:
                continue
            z = complex(float(x), float(y))
            a = polyval_ascending(even, z)
            b = polyval_ascending(odd, z)
            e = a - 1j * b
            e_sharp = a + 1j * b
            denom = max(1.0, abs(e) ** 2 + abs(e_sharp) ** 2)
            margin = (abs(e) ** 2 - abs(e_sharp) ** 2) / denom
            checked += 1
            if not np.isfinite(margin):
                margin = -math.inf
            min_margin = min(min_margin, float(margin))
            if margin <= tol:
                failures += 1

    if not math.isfinite(min_margin):
        min_margin = float("nan")
    return HBCheck(checked=checked, failures=failures, min_margin=min_margin)


def rejection_reason(
    zeros: ZeroCheck,
    jensen: JensenCheck,
    pf: PFCheck,
    hb: HBCheck,
) -> str:
    reasons = []
    if zeros.count:
        reasons.append("nonreal_zeros")
    if jensen.failures:
        reasons.append("jensen_fail")
    if pf.failures:
        reasons.append("pf_minor_fail")
    if hb.failures:
        reasons.append("hb_inequality_fail")
    return "|".join(reasons) if reasons else "none"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--boundary", choices=("circle", "star", "random_star"), default="star")
    parser.add_argument("--n", type=int, default=14)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rules", default="inverse_square,curvature,dtn,curvature_dtn")
    parser.add_argument("--top-k-values", default="3,4")
    parser.add_argument("--min-sep-values", default="2")
    parser.add_argument("--spectral-radius", type=float, default=0.85)
    parser.add_argument("--curvature-strength", type=float, default=0.35)
    parser.add_argument("--completed-form", choices=("det", "even"), default="det")
    parser.add_argument("--nonbacktracking", action="store_true")
    parser.add_argument("--include-nonbacktracking", action="store_true")
    parser.add_argument("--zero-tol", type=float, default=1.0e-8)
    parser.add_argument("--jensen-degree", type=int, default=4)
    parser.add_argument("--pf-order", type=int, default=3)
    parser.add_argument("--hb-x-radius", type=float, default=2.0)
    parser.add_argument("--hb-x-count", type=int, default=41)
    parser.add_argument("--hb-y-values", default="0.1,0.25,0.5,1.0")
    args = parser.parse_args()

    if args.hb_x_count < 2:
        raise ValueError("--hb-x-count must be at least 2")

    points = make_boundary(args.boundary, args.n, args.seed)
    rules = parse_rule_list(args.rules)
    top_k_values = parse_int_list(args.top_k_values, 4)
    min_sep_values = parse_int_list(args.min_sep_values, 2)
    nonbacktracking_values = [False, True] if args.include_nonbacktracking else [args.nonbacktracking]
    y_values = parse_float_list(args.hb_y_values)

    print("section=falsification_setup")
    print(
        "boundary,n,completed_form,spectral_radius,curvature_strength,zero_tol,"
        "jensen_degree,pf_order,hb_x_radius,hb_x_count,hb_y_values"
    )
    print(
        f"{args.boundary},{args.n},{args.completed_form},{args.spectral_radius},"
        f"{args.curvature_strength},{args.zero_tol},{args.jensen_degree},"
        f"{args.pf_order},{args.hb_x_radius},{args.hb_x_count},"
        f"{'|'.join(str(v) for v in y_values)}"
    )

    print("section=falsification")
    print(
        "boundary,n,rule,nonbacktracking,top_k,min_sep,degree,edge_count,"
        "nonreal_zero_count,max_zero_abs_imag,max_zero_rel_imag,real_rooted,"
        "jensen_checked,jensen_failures,jensen_worst_rel_imag,"
        "pf_checked,pf_failures,pf_min_minor,pf_orientation,"
        "hb_checked,hb_failures,hb_min_margin,rejected,rejection_reason,status"
    )

    summary: list[tuple[bool, str, bool, int, int, str]] = []
    for rule in rules:
        if rule not in TRANSFER_RULES:
            raise ValueError(f"unsupported rule: {rule}")
        for nonbacktracking in nonbacktracking_values:
            for top_k in top_k_values:
                for min_sep in min_sep_values:
                    status = "ok"
                    try:
                        transfer, _, _ = transfer_for_args(
                            points,
                            top_k=top_k,
                            min_sep=min_sep,
                            spectral_radius=args.spectral_radius,
                            rule=rule,
                            curvature_strength=args.curvature_strength,
                            nonbacktracking=nonbacktracking,
                        )
                        coeffs = determinant_coefficients(transfer, args.completed_form)
                        roots = determinant_zeros(transfer, args.completed_form)
                        zeros = zero_check(roots, args.zero_tol)
                        jensen = check_jensen(coeffs, args.jensen_degree, args.zero_tol)
                        pf = check_pf_minors(coeffs, args.pf_order, args.zero_tol)
                        hb = check_hermite_biehler(
                            coeffs,
                            x_radius=args.hb_x_radius,
                            x_count=args.hb_x_count,
                            y_values=y_values,
                            tol=args.zero_tol,
                        )
                        reason = rejection_reason(zeros, jensen, pf, hb)
                        rejected = reason != "none"
                        summary.append((rejected, rule, nonbacktracking, top_k, min_sep, reason))
                        print(
                            f"{args.boundary},{args.n},{rule},{str(nonbacktracking).lower()},"
                            f"{top_k},{min_sep},{len(coeffs) - 1},{int(np.count_nonzero(transfer))},"
                            f"{zeros.count},{zeros.max_abs_imag:.12e},{zeros.max_rel_imag:.12e},"
                            f"{str(zeros.count == 0).lower()},"
                            f"{jensen.checked},{jensen.failures},{jensen.worst_rel_imag:.12e},"
                            f"{pf.checked},{pf.failures},{pf.min_minor:.12e},{pf.orientation},"
                            f"{hb.checked},{hb.failures},{hb.min_margin:.12e},"
                            f"{str(rejected).lower()},{reason},{status}"
                        )
                    except ValueError as exc:
                        status = str(exc).replace(",", ";")
                        summary.append((True, rule, nonbacktracking, top_k, min_sep, "error"))
                        print(
                            f"{args.boundary},{args.n},{rule},{str(nonbacktracking).lower()},"
                            f"{top_k},{min_sep},0,0,0,nan,nan,false,0,0,nan,"
                            f"0,0,nan,none,0,0,nan,true,error,{status}"
                        )

    print("section=falsification_summary")
    print("total,rejected,not_rejected")
    rejected_count = sum(1 for row in summary if row[0])
    print(f"{len(summary)},{rejected_count},{len(summary) - rejected_count}")
    print("section=survivors")
    print("rule,nonbacktracking,top_k,min_sep,rejection_reason")
    for rejected, rule, nonbacktracking, top_k, min_sep, reason in summary:
        if not rejected:
            print(f"{rule},{str(nonbacktracking).lower()},{top_k},{min_sep},{reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
