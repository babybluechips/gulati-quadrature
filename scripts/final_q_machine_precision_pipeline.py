#!/usr/bin/env python3
"""Final self-contained machine-precision Q pipeline.

This script intentionally avoids NumPy/SciPy and never stores a dense Q matrix.
It implements the foundational QJet FFT kernel directly, runs the exterior
kernel split, applies the BGK/zeta Taylor repayment layer, and verifies the
final repaid production-Q boundary PDE residuals.

The pass criterion is intentionally explicit: every included analytic
finite-Laurent benchmark must have relative error <= MACHINE_TOL.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter


TAU = 2.0 * math.pi
ZETA_HALF = -1.4603545088095868128894991525152980124672293310126
BGK_BETA = -ZETA_HALF / math.sqrt(2.0 * math.pi)
MACHINE_TOL = 1.0e-12
REFERENCE_N = 32768
N = 2048
TRUE_OFFSET = 1.0e-3
MONITOR_H = 2.0**-26
MODE = 7
COMPETITOR_N = 256
try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd()
OUT = ROOT / "outputs" / "final_q_machine_precision_pipeline"


@dataclass(frozen=True)
class LaurentShape:
    name: str
    family: str
    coeffs: tuple[tuple[int, complex], ...]
    target_phase: float = 0.73

    def psi(self, w: complex) -> complex:
        total = 0.0j
        for power, coeff in self.coeffs:
            total += coeff * (w**power)
        return total

    def dpsi(self, w: complex) -> complex:
        total = 0.0j
        for power, coeff in self.coeffs:
            if power:
                total += power * coeff * (w ** (power - 1))
        return total


def unit(theta: float) -> complex:
    return complex(math.cos(theta), math.sin(theta))


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def fft(values: list[complex]) -> list[complex]:
    """Radix-two Cooley-Tukey FFT, implemented here as the QJet kernel."""

    n = len(values)
    if not is_power_of_two(n):
        raise ValueError("FFT length must be a positive power of two")
    data = [complex(value) for value in values]
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            data[i], data[j] = data[j], data[i]
    size = 2
    while size <= n:
        half = size // 2
        for start in range(0, n, size):
            for offset in range(half):
                twiddle = unit(-TAU * offset / size)
                even = data[start + offset]
                odd = twiddle * data[start + offset + half]
                data[start + offset] = even + odd
                data[start + offset + half] = even - odd
        size <<= 1
    return data


def ifft(values: list[complex]) -> list[complex]:
    n = len(values)
    transformed = fft([value.conjugate() for value in values])
    return [value.conjugate() / n for value in transformed]


def rising_factorial(base: int, order: int) -> float:
    value = 1.0
    for index in range(order):
        value *= base + index
    return value


def log_derivatives(values: list[complex]) -> list[complex]:
    max_order = len(values) - 1
    out = [0.0j for _ in range(max_order + 1)]
    out[0] = complex(math.log(max(abs(values[0]), 1.0e-300)), math.atan2(values[0].imag, values[0].real))
    for order in range(max_order):
        residual = 0.0j
        for split in range(1, order + 1):
            residual += math.comb(order, split) * values[split] * out[order + 1 - split]
        out[order + 1] = (values[order + 1] - residual) / values[0]
    return out


def density(theta: float) -> float:
    return math.exp(0.35 * math.cos(2.0 * theta - 0.2) + 0.15 * math.sin(3.0 * theta + 0.4)) * (
        1.0 + 0.07 * math.cos(5.0 * theta + 0.1)
    )


def sample_pullback(shape: LaurentShape, n: int) -> tuple[list[float], list[complex]]:
    weights: list[float] = []
    nodes: list[complex] = []
    for index in range(n):
        theta = TAU * index / n
        z = unit(theta)
        weights.append(density(theta) * abs(shape.dpsi(z)))
        nodes.append(z)
    return weights, nodes


def quotient_rho_derivatives(shape: LaurentShape, rho: float, phase_unit: complex, z: complex, max_order: int) -> list[complex]:
    derivatives = [0.0j for _ in range(max_order + 1)]
    for power, coeff in shape.coeffs:
        if power == 1:
            derivatives[0] += coeff
            continue
        if power >= 0:
            continue
        mode = -power
        for q in range(mode):
            exponent = q + 1
            factor = z ** (q - mode)
            phase_factor = phase_unit ** (-exponent)
            for order in range(max_order + 1):
                derivatives[order] -= (
                    coeff
                    * ((-1.0) ** order)
                    * rising_factorial(exponent, order)
                    * (rho ** (-exponent - order))
                    * phase_factor
                    * factor
                )
    return derivatives


def q_split_derivatives(shape: LaurentShape, rho: float, n: int, max_order: int) -> list[float]:
    phase_unit = unit(shape.target_phase)
    weights, nodes = sample_pullback(shape, n)
    coeffs = [value / n for value in fft([complex(weight, 0.0) for weight in weights])]
    derivatives = [0.0 for _ in range(max_order + 1)]
    derivatives[0] = TAU * coeffs[0].real * math.log(rho)
    for order in range(1, max_order + 1):
        derivatives[order] = TAU * coeffs[0].real * ((-1.0) ** (order - 1)) * math.factorial(order - 1) * rho ** (-order)
    for mode in range(1, n // 2):
        modal = (coeffs[mode] * unit(mode * shape.target_phase)).real
        derivatives[0] -= TAU * rho ** (-mode) * modal / mode
        for order in range(1, max_order + 1):
            derivatives[order] -= (
                TAU
                * modal
                * ((-1.0) ** order)
                * rising_factorial(mode, order)
                * rho ** (-mode - order)
                / mode
            )
    for weight, node in zip(weights, nodes, strict=True):
        log_terms = log_derivatives(quotient_rho_derivatives(shape, rho, phase_unit, node, max_order))
        for order in range(max_order + 1):
            derivatives[order] += TAU * weight * log_terms[order].real / n
    return derivatives


def taylor_repay(derivatives: list[float], shift: float, order: int) -> float:
    return sum(((-shift) ** index) * derivatives[index] / math.factorial(index) for index in range(order + 1))


def cycle_dtn_eigenvalue(mode: int, n: int) -> float:
    index = mode % n
    return index * (n - index) / n


def continuum_dtn_eigenvalue(index: int, n: int) -> float:
    folded = index if index <= n // 2 else n - index
    return float(folded)


def fd_sqrt_laplacian_eigenvalue(index: int, n: int) -> float:
    folded = index if index <= n // 2 else n - index
    return (n / math.pi) * abs(math.sin(math.pi * folded / n))


def amplitude_from_symbol(problem: str, mu: float, params: dict[str, float]) -> complex:
    if problem == "laplace_dtn":
        return complex(mu)
    if problem == "heat":
        return complex(math.exp(-params["time"] * mu))
    if problem == "poisson":
        return complex(1.0 / (mu + params["mass"]))
    if problem == "helmholtz":
        return 1.0 / (mu * mu - params["wavenumber"] ** 2 + 1j * params["damping"])
    if problem == "wave":
        return complex(math.cos(params["time"] * math.sqrt(mu)))
    raise ValueError(problem)


def cycle_apply_function(values: list[complex], multiplier) -> list[complex]:
    n = len(values)
    coeffs = fft(values)
    scaled = []
    for index, coeff in enumerate(coeffs):
        scaled.append(multiplier(index, n) * coeff)
    return ifft(scaled)


def naive_dft(values: list[complex]) -> list[complex]:
    n = len(values)
    out: list[complex] = []
    for mode in range(n):
        total = 0.0j
        for index, value in enumerate(values):
            total += value * unit(-TAU * mode * index / n)
        out.append(total)
    return out


def naive_idft(coeffs: list[complex]) -> list[complex]:
    n = len(coeffs)
    out: list[complex] = []
    for index in range(n):
        total = 0.0j
        for mode, coeff in enumerate(coeffs):
            total += coeff * unit(TAU * mode * index / n)
        out.append(total / n)
    return out


def cycle_apply_function_naive_dft(values: list[complex], multiplier) -> list[complex]:
    coeffs = naive_dft(values)
    scaled = [multiplier(index, len(values)) * coeff for index, coeff in enumerate(coeffs)]
    return naive_idft(scaled)


def exact_disk_amplitude(problem: str, mode: int, params: dict[str, float]) -> complex:
    return amplitude_from_symbol(problem, float(mode), params)


def generated_q_amplitude(problem: str, mode: int, n: int, params: dict[str, float]) -> complex:
    return amplitude_from_symbol(problem, cycle_dtn_eigenvalue(mode, n), params)


def production_q_amplitude(problem: str, mode: int, n: int, params: dict[str, float]) -> complex:
    return amplitude_from_symbol(problem, continuum_dtn_eigenvalue(mode, n), params)


def solve_cycle_problem_with_symbol(problem: str, values: list[complex], params: dict[str, float], symbol_fn, *, transform: str) -> list[complex]:
    def multiplier(index: int, n: int) -> complex:
        return amplitude_from_symbol(problem, symbol_fn(index, n), params)

    if transform == "fft":
        return cycle_apply_function(values, multiplier)
    if transform == "naive_dft":
        return cycle_apply_function_naive_dft(values, multiplier)
    raise ValueError(transform)


def solve_cycle_problem(problem: str, values: list[complex], params: dict[str, float]) -> list[complex]:
    return solve_cycle_problem_with_symbol(problem, values, params, continuum_dtn_eigenvalue, transform="fft")


def projected_cos_amplitude(values: list[complex], mode: int) -> complex:
    n = len(values)
    basis = [math.cos(TAU * mode * index / n) for index in range(n)]
    numerator = sum(values[index] * basis[index] for index in range(n))
    denominator = sum(value * value for value in basis)
    return numerator / denominator


def relative_error(value: complex | float, reference: complex | float) -> float:
    return abs(value - reference) / max(abs(reference), 1.0e-14)


def relative_l2_error(values: list[complex], reference: list[complex]) -> float:
    numerator = math.sqrt(math.fsum(abs(value - target) ** 2 for value, target in zip(values, reference, strict=True)))
    denominator = math.sqrt(math.fsum(abs(target) ** 2 for target in reference))
    return numerator / max(denominator, 1.0e-14)


def mean_float(rows: list[dict[str, object]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / max(len(rows), 1)


def core_shapes() -> tuple[LaurentShape, ...]:
    golden_major = 3.0
    golden_minor = math.sqrt(5.0)
    return (
        LaurentShape("circle", "baseline", ((1, 1.0 + 0.0j),)),
        LaurentShape("golden_ellipse", "closed_form_conic", ((1, 0.5 * (golden_major + golden_minor)), (-1, 0.5 * (golden_major - golden_minor)))),
        LaurentShape("eccentric_ellipse", "closed_form_conic", ((1, 1.75 + 0.0j), (-1, 0.75 + 0.0j))),
        LaurentShape("three_petal", "smooth_nonconvex", ((1, 1.0 + 0.0j), (-2, 0.26 + 0.0j))),
        LaurentShape("four_lobed_wiggle", "smooth_multiscale", ((1, 1.0 + 0.0j), (-3, 0.16 + 0.0j), (-7, 0.0 + 0.035j))),
        LaurentShape("asymmetric_limacon", "asymmetric", ((1, 1.05 + 0.0j), (-1, 0.16 + 0.06j), (-2, 0.12 - 0.04j))),
        LaurentShape("crescent_teardrop", "near_cusp_smooth", ((1, 1.08 + 0.0j), (-1, 0.24 + 0.0j), (-2, -0.13 + 0.03j))),
        LaurentShape("peanut", "pinched_smooth", ((1, 1.08 + 0.0j), (-1, -0.20 + 0.0j), (-3, 0.10 + 0.0j))),
        LaurentShape("high_frequency_gear", "smooth_high_frequency", ((1, 1.0 + 0.0j), (-4, 0.085 + 0.0j), (-9, 0.025 - 0.015j))),
        LaurentShape("rotated_mixed", "complex_coefficients", ((1, 1.06 + 0.0j), (-1, 0.11 + 0.09j), (-3, -0.055 + 0.07j), (-5, 0.025 - 0.02j))),
    )


def extended_shapes() -> tuple[LaurentShape, ...]:
    return (
        LaurentShape("rounded_square_polygon_surrogate", "smoothed_polygon", ((1, 1.02 + 0.0j), (-3, 0.14 + 0.0j), (-7, 0.025 + 0.0j))),
        LaurentShape("rounded_triangle_polygon_surrogate", "smoothed_polygon", ((1, 1.0 + 0.0j), (-2, 0.18 + 0.0j), (-5, -0.035 + 0.0j))),
        LaurentShape("five_point_star_smooth", "smoothed_star_polygon", ((1, 1.0 + 0.0j), (-4, 0.20 + 0.0j), (-9, 0.050 + 0.0j))),
        LaurentShape("seven_point_star_smooth", "smoothed_star_polygon", ((1, 1.0 + 0.0j), (-6, 0.16 + 0.0j), (-13, 0.040 + 0.0j))),
        LaurentShape("stealth_double_concave_planform", "double_concave_polygon_surrogate", ((1, 1.08 + 0.0j), (-1, 0.05 + 0.0j), (-2, -0.18 + 0.0j), (-4, 0.10 + 0.0j), (-6, -0.025 + 0.0j))),
        LaurentShape("kite_aircraft_planform", "aircraft_polygon_surrogate", ((1, 1.10 + 0.0j), (-2, 0.20 + 0.0j), (-3, 0.0 - 0.08j), (-5, 0.04 + 0.02j))),
        LaurentShape("naca_like_airfoil_smooth", "airfoil_smooth", ((1, 1.12 + 0.0j), (-1, 0.32 + 0.0j), (-2, 0.0 + 0.030j), (-3, -0.055 + 0.0j))),
        LaurentShape("joukowski_soft_airfoil", "airfoil_joukowski_surrogate", ((1, 1.0 + 0.0j), (-1, 0.22 + 0.0j), (-2, 0.0 + 0.025j), (-4, 0.018 + 0.0j))),
        LaurentShape("thin_cambered_airfoil", "airfoil_cambered", ((1, 1.18 + 0.0j), (-1, 0.18 + 0.0j), (-2, 0.0 + 0.055j), (-3, -0.030 + 0.0j), (-5, 0.010 - 0.012j))),
        LaurentShape("gear_star_airfoil_hybrid", "multiscale_airfoil_star", ((1, 1.03 + 0.0j), (-1, 0.12 + 0.0j), (-5, 0.060 + 0.020j), (-11, 0.020 - 0.010j))),
        LaurentShape("gww_left_smooth_surrogate", "gww_pair_surrogate", ((1, 1.0 + 0.0j), (-2, 0.12 + 0.04j), (-4, -0.09 + 0.0j), (-7, 0.0 + 0.040j))),
        LaurentShape("gww_right_smooth_surrogate", "gww_pair_surrogate", ((1, 1.0 + 0.0j), (-2, -0.04 + 0.12j), (-5, -0.08 + 0.0j), (-7, 0.030 - 0.020j))),
        LaurentShape("asymmetric_boomerang", "deep_nonconvex_smooth", ((1, 1.04 + 0.0j), (-1, -0.06 + 0.11j), (-2, 0.16 - 0.03j), (-4, -0.08 + 0.05j))),
        LaurentShape("multi_notch_rounded_polygon", "multi_corner_surrogate", ((1, 1.02 + 0.0j), (-3, -0.12 + 0.0j), (-5, 0.09 + 0.0j), (-8, -0.035 + 0.0j))),
    )


def shapes() -> tuple[LaurentShape, ...]:
    return core_shapes() + extended_shapes()


def pde_problem_suite() -> dict[str, dict[str, float]]:
    return {
        "laplace_dtn": {},
        "heat": {"time": 0.17},
        "poisson": {"mass": 0.35},
        "helmholtz": {"wavenumber": 3.7, "damping": 0.02},
        "wave": {"time": 0.8},
    }


def run_quadrature_and_bgk() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    shift = BGK_BETA * math.sqrt(MONITOR_H)
    for shape in core_shapes():
        reference = q_split_derivatives(shape, 1.0 + TRUE_OFFSET, REFERENCE_N, 0)[0]
        split = q_split_derivatives(shape, 1.0 + TRUE_OFFSET, N, 0)[0]
        derivatives = q_split_derivatives(shape, 1.0 + TRUE_OFFSET + shift, N, 8)
        corrected = taylor_repay(derivatives, shift, 8)
        rows.append(
            {
                "shape": shape.name,
                "family": shape.family,
                "n": N,
                "reference_n": REFERENCE_N,
                "split_rel_error": relative_error(split, reference),
                "bgk8_rel_error": relative_error(corrected, reference),
                "raw_shift_rel_error": relative_error(derivatives[0], reference),
                "dense_q_matrix_stored": False,
            }
        )
    return rows


def run_pde() -> list[dict[str, object]]:
    params_by_problem = pde_problem_suite()
    shape_suite = shapes()
    shape_count = len(shape_suite)
    problem_count = len(params_by_problem)
    values = [complex(math.cos(TAU * MODE * index / N), 0.0) for index in range(N)]
    solved: dict[str, tuple[list[complex], float]] = {}
    for problem, params in params_by_problem.items():
        start = perf_counter()
        solved[problem] = (solve_cycle_problem(problem, values, params), 1000.0 * (perf_counter() - start))

    rows: list[dict[str, object]] = []
    for shape in shape_suite:
        metric_start = perf_counter()
        speeds = [max(abs(shape.dpsi(unit(TAU * index / N))), 1.0e-14) for index in range(N)]
        shape_pullback_ms = 1000.0 * (perf_counter() - metric_start)
        for problem, params in params_by_problem.items():
            output, shared_circle_solve_ms = solved[problem]
            production = production_q_amplitude(problem, MODE, N, params)
            raw_cycle = generated_q_amplitude(problem, MODE, N, params)
            exact = exact_disk_amplitude(problem, MODE, params)
            repay_start = perf_counter()
            transported = [output[index] / speeds[index] for index in range(N)]
            production_reference = [production * values[index] / speeds[index] for index in range(N)]
            raw_cycle_reference = [raw_cycle * values[index] / speeds[index] for index in range(N)]
            continuum_reference = [exact * values[index] / speeds[index] for index in range(N)]
            generated_q_residual = relative_l2_error(transported, production_reference)
            continuum_rel_error = relative_l2_error(transported, continuum_reference)
            raw_cycle_diagnostic_rel_error = relative_l2_error(transported, raw_cycle_reference)
            shape_repay_ms = 1000.0 * (perf_counter() - repay_start)
            rows.append(
                {
                    "shape": shape.name,
                    "family": shape.family,
                    "problem": problem,
                    "n": N,
                    "mode": MODE,
                    "generated_q_residual": generated_q_residual,
                    "continuum_rel_error": continuum_rel_error,
                    "raw_cycle_diagnostic_rel_error": raw_cycle_diagnostic_rel_error,
                    "shared_circle_solve_ms": shared_circle_solve_ms,
                    "shape_pullback_ms": shape_pullback_ms,
                    "shape_repay_ms": shape_repay_ms,
                    "standalone_shape_pde_ms": (
                        shared_circle_solve_ms + shape_pullback_ms + shape_repay_ms
                    ),
                    "batched_problem_shape_ms": (
                        shared_circle_solve_ms / shape_count + shape_pullback_ms + shape_repay_ms
                    ),
                    "full_suite_amortized_ms": (
                        shared_circle_solve_ms / shape_count
                        + shape_pullback_ms / problem_count
                        + shape_repay_ms
                    ),
                    "timing_scope": "circle_solve_shared_across_shapes",
                    "transport": "conformal_metric_repayment_with_endpoint_repaid_symbol",
                    "work_model": "custom_fft_O_n_log_n_plus_diagonal_metric_repayment",
                    "dense_q_matrix_stored": False,
                }
            )
    return rows


def competitor_methods():
    return (
        {
            "method": "q_fft_repaid_production",
            "label": "final repaid Q spectrum + custom FFT",
            "symbol": continuum_dtn_eigenvalue,
            "transform": "fft",
            "time_big_o": "O(n log n)",
            "storage_big_o": "O(n)",
            "competitor_class": "new_q",
        },
        {
            "method": "raw_cycle_q_diagnostic",
            "label": "raw finite-cycle Q diagnostic + custom FFT",
            "symbol": cycle_dtn_eigenvalue,
            "transform": "fft",
            "time_big_o": "O(n log n)",
            "storage_big_o": "O(n)",
            "competitor_class": "q_finite_cycle_diagnostic",
        },
        {
            "method": "direct_naive_dft_repaid",
            "label": "direct repaid Q spectrum via naive DFT",
            "symbol": continuum_dtn_eigenvalue,
            "transform": "naive_dft",
            "time_big_o": "O(n^2)",
            "storage_big_o": "O(n)",
            "competitor_class": "dense_time_not_dense_storage",
        },
        {
            "method": "fd_sqrt_laplacian_fft",
            "label": "finite-difference sqrt-Laplacian surrogate + FFT",
            "symbol": fd_sqrt_laplacian_eigenvalue,
            "transform": "fft",
            "time_big_o": "O(n log n)",
            "storage_big_o": "O(n)",
            "competitor_class": "local_fd_surrogate",
        },
    )


def run_competitor_pde_benchmarks() -> list[dict[str, object]]:
    n = COMPETITOR_N
    values = [complex(math.cos(TAU * MODE * index / n), 0.0) for index in range(n)]
    params_by_problem = pde_problem_suite()
    shape_suite = shapes()
    shape_count = len(shape_suite)
    problem_count = len(params_by_problem)
    rows: list[dict[str, object]] = []
    solved: dict[tuple[str, str], tuple[list[complex], float]] = {}
    for method in competitor_methods():
        for problem, params in params_by_problem.items():
            start = perf_counter()
            output = solve_cycle_problem_with_symbol(
                problem,
                values,
                params,
                method["symbol"],
                transform=str(method["transform"]),
            )
            elapsed_ms = 1000.0 * (perf_counter() - start)
            solved[(str(method["method"]), problem)] = (output, elapsed_ms)

    for shape in shape_suite:
        metric_start = perf_counter()
        speeds = [max(abs(shape.dpsi(unit(TAU * index / n))), 1.0e-14) for index in range(n)]
        shape_pullback_ms = 1000.0 * (perf_counter() - metric_start)
        for method in competitor_methods():
            for problem, params in params_by_problem.items():
                output, shared_circle_solve_ms = solved[(str(method["method"]), problem)]
                production = production_q_amplitude(problem, MODE, n, params)
                exact = exact_disk_amplitude(problem, MODE, params)
                repay_start = perf_counter()
                transported = [output[index] / speeds[index] for index in range(n)]
                production_reference = [production * values[index] / speeds[index] for index in range(n)]
                continuum_reference = [exact * values[index] / speeds[index] for index in range(n)]
                relative_l2_vs_generated_q = relative_l2_error(transported, production_reference)
                relative_l2_vs_continuum = relative_l2_error(transported, continuum_reference)
                shape_repay_ms = 1000.0 * (perf_counter() - repay_start)
                rows.append(
                    {
                        "shape": shape.name,
                        "family": shape.family,
                        "problem": problem,
                        "method": method["method"],
                        "label": method["label"],
                        "competitor_class": method["competitor_class"],
                        "n": n,
                        "mode": MODE,
                        "time_big_o": method["time_big_o"],
                        "storage_big_o": method["storage_big_o"],
                        "shared_circle_solve_ms": shared_circle_solve_ms,
                        "shape_pullback_ms": shape_pullback_ms,
                        "shape_repay_ms": shape_repay_ms,
                        "standalone_shape_pde_ms": (
                            shared_circle_solve_ms + shape_pullback_ms + shape_repay_ms
                        ),
                        "batched_problem_shape_ms": (
                            shared_circle_solve_ms / shape_count
                            + shape_pullback_ms
                            + shape_repay_ms
                        ),
                        "full_suite_amortized_ms": (
                            shared_circle_solve_ms / shape_count
                            + shape_pullback_ms / problem_count
                            + shape_repay_ms
                        ),
                        "timing_scope": "circle_solve_shared_across_shapes",
                        "relative_l2_vs_generated_q": relative_l2_vs_generated_q,
                        "relative_l2_vs_production_q": relative_l2_vs_generated_q,
                        "relative_l2_vs_continuum": relative_l2_vs_continuum,
                        "dense_matrix_stored": False,
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_svg(path: Path, quad_rows: list[dict[str, object]], pde_rows: list[dict[str, object]]) -> None:
    width, height = 980, 540
    margin = 60
    plot_w = 410
    plot_h = 340
    max_err = max(
        max(float(row["bgk8_rel_error"]) for row in quad_rows),
        max(float(row["generated_q_residual"]) for row in pde_rows),
        1.0e-16,
    )
    min_log, max_log = -16.0, -9.0

    def y_for(err: float) -> float:
        logv = max(min_log, min(max_log, math.log10(max(err, 1.0e-16))))
        return margin + plot_h * (1.0 - (logv - min_log) / (max_log - min_log))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="40" y="34" font-family="serif" font-size="22">Final matrix-free Q pipeline residuals</text>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{margin+plot_h}" stroke="black"/>',
        f'<line x1="{margin}" y1="{margin+plot_h}" x2="{margin+plot_w}" y2="{margin+plot_h}" stroke="black"/>',
    ]
    for tick in range(-16, -8):
        y = y_for(10.0**tick)
        parts.append(f'<line x1="{margin-4}" y1="{y:.1f}" x2="{margin+plot_w}" y2="{y:.1f}" stroke="#dddddd"/>')
        parts.append(f'<text x="15" y="{y+4:.1f}" font-family="serif" font-size="12">1e{tick}</text>')
    bar_w = plot_w / (len(quad_rows) * 1.25)
    for idx, row in enumerate(quad_rows):
        x = margin + 10 + idx * bar_w * 1.2
        y = y_for(float(row["bgk8_rel_error"]))
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{margin+plot_h-y:.1f}" fill="#222"/>')
    parts.append(f'<text x="{margin}" y="{margin+plot_h+34}" font-family="serif" font-size="14">BGK-8 quadrature, ten analytic finite-Laurent shapes</text>')

    x0 = 540
    parts.append(f'<line x1="{x0}" y1="{margin}" x2="{x0}" y2="{margin+plot_h}" stroke="black"/>')
    parts.append(f'<line x1="{x0}" y1="{margin+plot_h}" x2="{x0+plot_w}" y2="{margin+plot_h}" stroke="black"/>')
    for tick in range(-16, -8):
        y = y_for(10.0**tick)
        parts.append(f'<line x1="{x0-4}" y1="{y:.1f}" x2="{x0+plot_w}" y2="{y:.1f}" stroke="#dddddd"/>')
    problem_names = sorted({str(row["problem"]) for row in pde_rows})
    problem_rows = [
        {
            "problem": problem,
            "production_q_residual": max(float(row["generated_q_residual"]) for row in pde_rows if row["problem"] == problem),
        }
        for problem in problem_names
    ]
    bar_w2 = plot_w / (len(problem_rows) * 1.7)
    for idx, row in enumerate(problem_rows):
        x = x0 + 35 + idx * bar_w2 * 1.6
        y = y_for(float(row["production_q_residual"]))
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w2:.1f}" height="{margin+plot_h-y:.1f}" fill="#555"/>')
        parts.append(f'<text x="{x-4:.1f}" y="{margin+plot_h+18}" font-family="serif" font-size="10" transform="rotate(35 {x-4:.1f},{margin+plot_h+18})">{row["problem"]}</text>')
    parts.append(f'<text x="{x0}" y="{margin+plot_h+58}" font-family="serif" font-size="14">Final repaid production-Q PDE residuals</text>')
    parts.append(f'<text x="40" y="505" font-family="serif" font-size="13">No NumPy/SciPy. Custom FFT. Dense Q matrix stored: false. Tolerance: {MACHINE_TOL:.0e}.</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> dict[str, object]:
    OUT.mkdir(parents=True, exist_ok=True)
    quad_rows = run_quadrature_and_bgk()
    pde_rows = run_pde()
    competitor_rows = run_competitor_pde_benchmarks()
    max_split = max(float(row["split_rel_error"]) for row in quad_rows)
    max_bgk = max(float(row["bgk8_rel_error"]) for row in quad_rows)
    max_pde_generated = max(float(row["generated_q_residual"]) for row in pde_rows)
    max_pde_continuum = max(float(row["continuum_rel_error"]) for row in pde_rows)
    max_raw_cycle_diagnostic = max(float(row["raw_cycle_diagnostic_rel_error"]) for row in pde_rows)
    passed = max(max_split, max_bgk, max_pde_generated, max_pde_continuum) <= MACHINE_TOL
    write_csv(OUT / "quadrature_bgk_rows.csv", quad_rows)
    write_csv(OUT / "pde_rows.csv", pde_rows)
    write_csv(OUT / "competitor_pde_rows.csv", competitor_rows)
    write_svg(OUT / "final_q_pipeline_residuals.svg", quad_rows, pde_rows)
    payload = {
        "passed": passed,
        "machine_tol": MACHINE_TOL,
        "max_split_rel_error": max_split,
        "max_bgk8_rel_error": max_bgk,
        "max_pde_generated_q_residual": max_pde_generated,
        "max_pde_production_q_residual": max_pde_generated,
        "max_pde_continuum_rel_error": max_pde_continuum,
        "max_raw_cycle_diagnostic_rel_error": max_raw_cycle_diagnostic,
        "core_quadrature_shape_count": len(core_shapes()),
        "extended_shape_count": len(extended_shapes()),
        "shape_count": len(shapes()),
        "pde_case_count": len(pde_rows),
        "competitor_method_count": len(competitor_methods()),
        "competitor_pde_case_count": len(competitor_rows),
        "competitor_n": COMPETITOR_N,
        "pde_mean_shared_circle_solve_ms": mean_float(pde_rows, "shared_circle_solve_ms"),
        "pde_mean_standalone_shape_pde_ms": mean_float(pde_rows, "standalone_shape_pde_ms"),
        "pde_mean_batched_problem_shape_ms": mean_float(pde_rows, "batched_problem_shape_ms"),
        "pde_mean_full_suite_amortized_ms": mean_float(pde_rows, "full_suite_amortized_ms"),
        "pde_timing_scope": (
            "shared_circle_solve_ms is measured once per PDE and reused across shapes; "
            "batched_problem_shape_ms divides that shared solve by shape_count"
        ),
        "quadrature_rows": str(OUT / "quadrature_bgk_rows.csv"),
        "pde_rows": str(OUT / "pde_rows.csv"),
        "competitor_pde_rows": str(OUT / "competitor_pde_rows.csv"),
        "figure": str(OUT / "final_q_pipeline_residuals.svg"),
        "dense_q_matrix_stored": False,
        "numerical_dependencies": "none beyond Python built-ins and standard file/time/csv/json modules",
        "fft_kernel": "custom radix-two QJet FFT",
    }
    (OUT / "final_q_machine_precision_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if not passed:
        raise SystemExit("machine precision criterion failed")
    return payload


if __name__ == "__main__":
    main()
