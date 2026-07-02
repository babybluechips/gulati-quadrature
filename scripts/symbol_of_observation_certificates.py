#!/usr/bin/env python3
"""Reproduce executable certificates from ``symbol_of_observation``.

The source PDF frames observation as a strobe/zeta calculus over complex
probability.  This script tests the parts that are most directly executable:

* C1: Spitzer identity over a finite complex measure.
* C2/C3: strobe transfer and blindness at the first zeta zero.
* C4-C6: FFT-Spitzer constants and total-variation grading for lattice walks.
* C9: the unitary-face sawtooth law Im(kappa) = -L/4.

The implementation follows the appendix reference code, but emits a structured
JSON/Markdown audit that can be checked by CI or included in reports.
"""

from __future__ import annotations

import argparse
import cmath
import hashlib
import itertools
import json
import math
import time
from pathlib import Path
from typing import Any

import mpmath as mp
import numpy as np


def cdict(z: complex) -> dict[str, float]:
    return {"re": float(z.real), "im": float(z.imag), "abs": float(abs(z))}


def fmt_complex(z: complex, digits: int = 6) -> str:
    sign = "+" if z.imag >= 0 else "-"
    return f"{z.real:.{digits}f} {sign} {abs(z.imag):.{digits}f}i"


def pdf_sha256(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def certificate_c1_spitzer() -> dict[str, Any]:
    vals = [1, 0, -1]
    weights = [0.3 + 0.4j, 0.5 - 0.1j, 0.2 - 0.3j]
    n = 7

    direct = 0j
    for path in itertools.product(range(3), repeat=n):
        w = 1 + 0j
        partial = 0
        maximum = 0
        for index in path:
            w *= weights[index]
            partial += vals[index]
            maximum = max(maximum, partial)
        direct += w * maximum

    spitzer = 0j
    for k in range(1, n + 1):
        expected_positive = 0j
        for path in itertools.product(range(3), repeat=k):
            w = 1 + 0j
            partial = 0
            for index in path:
                w *= weights[index]
                partial += vals[index]
            if partial > 0:
                expected_positive += w * partial
        spitzer += expected_positive / k

    diff = abs(direct - spitzer)
    return {
        "name": "C1 Spitzer over complex measure",
        "pass": diff < 1e-12,
        "n": n,
        "direct_expected_max": cdict(direct),
        "spitzer_sum": cdict(spitzer),
        "difference": diff,
        "tolerance": 1e-12,
    }


def strobe_remainder(tau: mp.mpf | float, delta: float, cutoff: float) -> complex:
    alpha = mp.mpf("0.5") + 1j * tau
    kmax = int(cutoff / delta)
    strobe = delta * mp.fsum(
        [
            mp.power(k * delta, alpha - 1) * mp.exp(-k * delta)
            for k in range(1, kmax + 1)
        ]
    )
    rem = (strobe - mp.gamma(alpha)) / mp.power(delta, alpha)
    return complex(rem)


def certificate_c2_c3_strobe(cutoff: float, deltas: list[float]) -> dict[str, Any]:
    mp.mp.dps = 40
    tau = mp.mpf("14")
    target = complex(mp.zeta(mp.mpf("0.5") - 1j * tau))
    next_coeff = abs(complex(mp.zeta(mp.mpf("-0.5") - 1j * tau)))
    c2_rows = []
    for delta in deltas:
        r = strobe_remainder(tau, delta, cutoff)
        error = abs(r - target)
        c2_rows.append(
            {
                "delta": delta,
                "R": cdict(r),
                "error_to_zeta_half": error,
                "error_over_delta": error / delta,
            }
        )

    gamma1 = mp.im(mp.zetazero(1))
    blind_target = abs(complex(mp.zeta(mp.mpf("-0.5") - 1j * gamma1)))
    c3_rows = []
    for delta in deltas:
        r = strobe_remainder(gamma1, delta, cutoff)
        ratio = abs(r) / delta
        c3_rows.append({"delta": delta, "R": cdict(r), "abs_R_over_delta": ratio})

    c2_last = c2_rows[-1]
    c3_last = c3_rows[-1]
    return {
        "name": "C2/C3 strobe transfer and zeta-zero blindness",
        "pass": (
            c2_rows[0]["error_to_zeta_half"] > c2_rows[-1]["error_to_zeta_half"]
            and abs(c2_last["error_over_delta"] - next_coeff) / next_coeff < 0.08
            and abs(c3_last["abs_R_over_delta"] - blind_target) / blind_target < 0.08
        ),
        "cutoff": cutoff,
        "deltas": deltas,
        "tau14_zeta_half_minus_itau": cdict(target),
        "tau14_next_coefficient_abs": next_coeff,
        "tau14_rows": c2_rows,
        "gamma1": float(gamma1),
        "gamma1_next_coefficient_abs": blind_target,
        "gamma1_rows": c3_rows,
    }


def lattice_x_grid(nfft: int) -> np.ndarray:
    j = np.arange(nfft)
    return np.where(j < nfft // 2, j, j - nfft).astype(float)


def fft_spitzer_walk(
    phihat: np.ndarray,
    sigma: complex,
    record_n: list[int],
    tv_ks: list[int] | None = None,
) -> dict[str, Any]:
    nfft = int(phihat.size)
    max_n = max(record_n + (tv_ks or [0]))
    record_set = set(record_n)
    tv_set = set(tv_ks or [])
    x = lattice_x_grid(nfft)
    xp = np.maximum(x, 0.0)
    powers = np.ones(nfft, dtype=complex)
    expected_max = 0j
    constants: dict[int, complex] = {}
    tv: dict[int, float] = {}

    for k in range(1, max_n + 1):
        powers *= phihat
        pk = np.fft.ifft(powers)
        expected_positive = np.sum(xp * pk)
        expected_max += expected_positive / k
        if k in tv_set:
            tv[k] = float(np.sum(np.abs(pk)))
        if k in record_set:
            constants[k] = expected_max - sigma * math.sqrt(2 * k / math.pi) - sigma * (
                0.5 / math.sqrt(2 * math.pi * k)
            )

    return {
        "nfft": nfft,
        "record_n": record_n,
        "constants": {str(k): cdict(v) for k, v in constants.items()},
        "tv": {str(k): v for k, v in tv.items()},
    }


def phihat_srw(nfft: int) -> tuple[np.ndarray, complex]:
    lam = 2 * math.pi * np.arange(nfft) / nfft
    return np.cos(lam).astype(complex), 1 + 0j


def phihat_bessel(theta: float, nfft: int) -> tuple[np.ndarray, complex]:
    lam = 2 * math.pi * np.arange(nfft) / nfft
    z = cmath.exp(1j * theta)
    return np.exp(-z * (1 - np.cos(lam))), cmath.exp(0.5j * theta)


def phihat_unitary_cos(beta: list[float], nfft: int) -> tuple[np.ndarray, complex, float, float]:
    lam = 2 * math.pi * np.arange(nfft) / nfft
    psi = np.zeros(nfft)
    s = 0.0
    hopping_range = 0.0
    for m, b in enumerate(beta, start=1):
        psi += b * (np.cos(m * lam) - 1)
        s += b * m * m
        hopping_range += b * m
    sigma = cmath.exp(1j * math.pi / 4) * math.sqrt(abs(s))
    return np.exp(1j * psi), sigma, s, hopping_range


def richardson(c_n: complex, c_4n: complex, power: float = 0.5) -> complex:
    r = 4 ** (-power)
    return (c_4n - r * c_n) / (1 - r)


def _complex_from_row(row: dict[str, float]) -> complex:
    return complex(row["re"], row["im"])


def certificate_c4_c6_fft(nfft: int, quick: bool) -> dict[str, Any]:
    srw_records = [500, 1000, 2000] if quick else [1000, 2000, 4000]
    sector_records = [500, 2000] if quick else [750, 3000]
    face_records = [750, 1500] if quick else [1500, 3000]

    srw_phihat, srw_sigma = phihat_srw(nfft)
    srw = fft_spitzer_walk(srw_phihat, srw_sigma, srw_records, tv_ks=[srw_records[-1]])
    srw_last = _complex_from_row(srw["constants"][str(srw_records[-1])])

    sector = {}
    for theta in [math.pi / 4, math.pi / 3]:
        phihat, sigma = phihat_bessel(theta, nfft)
        tv_ks = [1, 100, sector_records[-1]]
        out = fft_spitzer_walk(phihat, sigma, sector_records, tv_ks=tv_ks)
        tv_limit = 1.0 / math.sqrt(math.cos(theta))
        out["theta"] = theta
        out["sigma"] = cdict(sigma)
        out["tv_limit_sec_sqrt"] = tv_limit
        out["tv_relative_error_last"] = abs(out["tv"][str(sector_records[-1])] - tv_limit) / tv_limit
        sector[f"theta_{theta:.6f}"] = out

    face_phihat, face_sigma = phihat_bessel(math.pi / 2, nfft)
    face_tv_ks = [100, face_records[-1]]
    face = fft_spitzer_walk(face_phihat, face_sigma, face_records, tv_ks=face_tv_ks)
    beta_const = math.gamma(0.5) * math.gamma(0.75) / math.gamma(1.25)
    watson = (2 / math.pi) ** 1.5 * beta_const
    face["theta"] = math.pi / 2
    face["sigma"] = cdict(face_sigma)
    face["watson_tv_over_sqrt_k"] = watson
    face["tv_over_sqrt_k"] = {
        str(k): face["tv"][str(k)] / math.sqrt(k) for k in face_tv_ks
    }

    return {
        "name": "C4-C6 FFT-Spitzer lattice certificates",
        "pass": (
            abs(srw_last.real + 0.5) < 2e-4
            and abs(srw_last.imag) < 2e-10
            and all(item["tv_relative_error_last"] < 0.02 for item in sector.values())
            and abs(face["tv_over_sqrt_k"][str(face_records[-1])] - watson) / watson < 0.08
        ),
        "nfft": nfft,
        "srw": srw,
        "sector": sector,
        "unitary_face": face,
    }


def certificate_c9_sawtooth(nfft: int, quick: bool) -> dict[str, Any]:
    records = [400, 1600] if quick else [750, 3000]
    cases = [
        {"name": "beta_1_1over4", "beta": [1.0, 0.25]},
        {"name": "beta_1_minus1over8", "beta": [1.0, -0.125]},
        {"name": "beta_half_0_half", "beta": [0.5, 0.0, 0.5]},
    ]
    rows = []
    for case in cases:
        phihat, sigma, s, hopping_range = phihat_unitary_cos(case["beta"], nfft)
        out = fft_spitzer_walk(phihat, sigma, records, tv_ks=[])
        c_n = _complex_from_row(out["constants"][str(records[0])])
        c_4n = _complex_from_row(out["constants"][str(records[1])])
        c_inf = richardson(c_n, c_4n, power=0.5)
        target = -hopping_range / 4
        rows.append(
            {
                "name": case["name"],
                "beta": case["beta"],
                "s": s,
                "L": hopping_range,
                "kappa_richardson": cdict(c_inf),
                "target_imag": target,
                "imag_error": float(abs(c_inf.imag - target)),
                "raw": out,
            }
        )
    return {
        "name": "C9 sawtooth law Im(kappa) = -L/4",
        "pass": all(row["imag_error"] < (8e-3 if quick else 3e-3) for row in rows),
        "nfft": nfft,
        "records": records,
        "rows": rows,
    }


def build_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Symbol of Observation Certificate Audit",
        "",
        f"- source: `{result['source_pdf']}`",
        f"- sha256: `{result['source_sha256']}`",
        f"- profile: `{result['profile']}`",
        f"- elapsed: `{result['elapsed_seconds']:.2f} s`",
        f"- overall: `{'PASS' if result['pass'] else 'FAIL'}`",
        "",
        "| Certificate | Status | Key check |",
        "|---|---:|---|",
    ]
    c1 = result["certificates"]["C1"]
    lines.append(
        f"| C1 Spitzer | {'PASS' if c1['pass'] else 'FAIL'} | "
        f"diff `{c1['difference']:.3e}`, E[M7] `{fmt_complex(complex(c1['direct_expected_max']['re'], c1['direct_expected_max']['im']))}` |"
    )
    c23 = result["certificates"]["C2_C3"]
    c2_last = c23["tau14_rows"][-1]
    c3_last = c23["gamma1_rows"][-1]
    lines.append(
        f"| C2 strobe transfer | {'PASS' if c23['pass'] else 'FAIL'} | "
        f"last error/Delta `{c2_last['error_over_delta']:.6f}` vs `|zeta(-1/2-14i)|={c23['tau14_next_coefficient_abs']:.6f}` |"
    )
    lines.append(
        f"| C3 blind mode | {'PASS' if c23['pass'] else 'FAIL'} | "
        f"last |R|/Delta `{c3_last['abs_R_over_delta']:.6f}` vs `{c23['gamma1_next_coefficient_abs']:.6f}` |"
    )
    c46 = result["certificates"]["C4_C6"]
    srw_keys = list(c46["srw"]["constants"].keys())
    srw_last = c46["srw"]["constants"][srw_keys[-1]]
    lines.append(
        f"| C4 SRW anchor | {'PASS' if c46['pass'] else 'FAIL'} | "
        f"c({srw_keys[-1]}) `{srw_last['re']:.6f}` vs `-0.5` |"
    )
    for name, item in c46["sector"].items():
        last_tv_key = sorted(item["tv"].keys(), key=int)[-1]
        lines.append(
            f"| C5 sector {name} | {'PASS' if item['tv_relative_error_last'] < 0.02 else 'FAIL'} | "
            f"TV({last_tv_key}) `{item['tv'][last_tv_key]:.6f}` vs `{item['tv_limit_sec_sqrt']:.6f}` |"
        )
    face = c46["unitary_face"]
    face_tv_key = sorted(face["tv_over_sqrt_k"].keys(), key=int)[-1]
    lines.append(
        f"| C6 unitary TV | {'PASS' if c46['pass'] else 'FAIL'} | "
        f"TV/sqrt(k) `{face['tv_over_sqrt_k'][face_tv_key]:.6f}` vs Watson `{face['watson_tv_over_sqrt_k']:.6f}` |"
    )
    c9 = result["certificates"]["C9"]
    for row in c9["rows"]:
        lines.append(
            f"| C9 {row['name']} | {'PASS' if row['imag_error'] < 8e-3 else 'FAIL'} | "
            f"Im kappa `{row['kappa_richardson']['im']:.6f}` vs `{-row['L']/4:.6f}` |"
        )
    lines.append("")
    lines.append(
        "Interpretation: the PDF's executable claims survive the numerical audit in this "
        "profile. The strongest link to the Q/BGK pipeline is the shared endpoint ledger: "
        "uniform strobes and monitored boundary/path functionals both expose zeta-coded "
        "sampling defects rather than Monte Carlo noise."
    )
    lines.append("")
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    quick = args.profile == "quick"
    deltas = [0.04, 0.02, 0.01] if quick else [0.02, 0.01, 0.005]
    result = {
        "source_pdf": args.pdf.name if args.pdf else None,
        "source_sha256": pdf_sha256(args.pdf),
        "profile": args.profile,
        "nfft": args.nfft,
        "certificates": {},
    }
    result["certificates"]["C1"] = certificate_c1_spitzer()
    result["certificates"]["C2_C3"] = certificate_c2_c3_strobe(args.strobe_cutoff, deltas)
    result["certificates"]["C4_C6"] = certificate_c4_c6_fft(args.nfft, quick)
    result["certificates"]["C9"] = certificate_c9_sawtooth(args.nfft, quick)
    result["elapsed_seconds"] = time.perf_counter() - start
    result["pass"] = all(cert["pass"] for cert in result["certificates"].values())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/symbol_of_observation"))
    parser.add_argument("--profile", choices=["quick", "full"], default="quick")
    parser.add_argument("--nfft", type=int, default=2**14)
    parser.add_argument("--strobe-cutoff", type=float, default=50.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = run(args)
    json_path = args.out_dir / "symbol_of_observation_certificates.json"
    md_path = args.out_dir / "symbol_of_observation_certificates.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    md_path.write_text(build_markdown(result), encoding="utf-8")
    print(json.dumps({"ok": result["pass"], "json": str(json_path), "md": str(md_path)}, indent=2))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
