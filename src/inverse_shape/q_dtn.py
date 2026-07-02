"""Boundary-only PDE operators from the Q/DtN correspondence.

For a regular sampled circle, the cycle Gulati spectrum

    lambda_m = m(n-m)/2

becomes the unit-disk Dirichlet-to-Neumann spectrum |m| after the boundary
normalization h/pi = length/(pi n).  The functions here use that normalized
Q spectrum as the load-bearing boundary generator for Laplace, heat, Poisson,
Helmholtz-resolvent, and wave boundary dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable

from inverse_shape.quadrature import (
    BoundaryQJet,
    BorrowComputeRepayLedger,
    CycleQJet,
    NumericVector,
    q_spectral_error_signature,
)


@dataclass(frozen=True)
class BoundaryPDEModeResult:
    """Single-mode boundary PDE result."""

    mode: int
    problem: str
    q_amplitude: complex
    exact_amplitude: complex
    relative_error: float


@dataclass(frozen=True)
class ScalarQJet:
    """First-order scalar QJet for autodiff boundary pullbacks."""

    value: float
    derivative: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", float(self.value))
        object.__setattr__(self, "derivative", float(self.derivative))

    @staticmethod
    def coerce(value: float | ScalarQJet) -> ScalarQJet:
        if isinstance(value, ScalarQJet):
            return value
        return ScalarQJet(float(value), 0.0)

    def __add__(self, other: float | ScalarQJet) -> ScalarQJet:
        rhs = ScalarQJet.coerce(other)
        return ScalarQJet(self.value + rhs.value, self.derivative + rhs.derivative)

    def __radd__(self, other: float | ScalarQJet) -> ScalarQJet:
        return self.__add__(other)

    def __sub__(self, other: float | ScalarQJet) -> ScalarQJet:
        rhs = ScalarQJet.coerce(other)
        return ScalarQJet(self.value - rhs.value, self.derivative - rhs.derivative)

    def __rsub__(self, other: float | ScalarQJet) -> ScalarQJet:
        lhs = ScalarQJet.coerce(other)
        return ScalarQJet(lhs.value - self.value, lhs.derivative - self.derivative)

    def __mul__(self, other: float | ScalarQJet) -> ScalarQJet:
        rhs = ScalarQJet.coerce(other)
        return ScalarQJet(
            self.value * rhs.value,
            self.derivative * rhs.value + self.value * rhs.derivative,
        )

    def __rmul__(self, other: float | ScalarQJet) -> ScalarQJet:
        return self.__mul__(other)

    def __truediv__(self, other: float | ScalarQJet) -> ScalarQJet:
        rhs = ScalarQJet.coerce(other)
        if rhs.value == 0.0:
            raise ZeroDivisionError("division by zero QJet value")
        return ScalarQJet(
            self.value / rhs.value,
            (self.derivative * rhs.value - self.value * rhs.derivative) / (rhs.value * rhs.value),
        )

    def __rtruediv__(self, other: float | ScalarQJet) -> ScalarQJet:
        lhs = ScalarQJet.coerce(other)
        return lhs.__truediv__(self)

    def __neg__(self) -> ScalarQJet:
        return ScalarQJet(-self.value, -self.derivative)

    def __pow__(self, exponent: float) -> ScalarQJet:
        if self.value <= 0.0:
            raise ValueError("QJet power requires a positive base")
        out = self.value**exponent
        return ScalarQJet(out, exponent * self.value ** (exponent - 1.0) * self.derivative)


@dataclass(frozen=True)
class PullbackPDEEvaluation:
    """Matrix-free arbitrary-domain PDE evaluation through a circle pullback."""

    values: NumericVector
    ledger: BorrowComputeRepayLedger
    problem: str
    work_units: int
    stats: dict[str, float | int | str]


@dataclass(frozen=True)
class BoundaryPullbackQJet:
    """Autodiff boundary map QJets for pullback-to-circle PDE solves.

    ``points`` are physical boundary samples ``z(theta_i)`` and ``speeds`` are
    the autodiff metric factors ``|dz/dtheta|``.  The class stores only these
    generating jets and the regular cycle QJet; it never materializes a dense
    pullback or DtN matrix.
    """

    points: tuple[tuple[float, float], ...]
    speeds: tuple[float, ...]
    tangents: tuple[tuple[float, float], ...]
    normals: tuple[tuple[float, float], ...]
    theta_step: float

    @property
    def n(self) -> int:
        return len(self.points)

    @property
    def min_speed(self) -> float:
        return min(self.speeds)

    @property
    def max_speed(self) -> float:
        return max(self.speeds)

    @property
    def anisotropy(self) -> float:
        return self.max_speed / self.min_speed

    def apply_dtn(self, values: Iterable[complex]) -> PullbackPDEEvaluation:
        """Apply the physical Laplace DtN map via borrow-compute-repay.

        The circle computes ``Lambda_circle(f o z)``.  The repay step maps the
        normal derivative back to the physical boundary by multiplying by
        ``1 / |dz/dtheta|`` at each boundary QJet.
        """

        vector = tuple(values)
        if len(vector) != self.n:
            raise ValueError("values length must match the pullback QJet size")
        circle_flux = apply_cycle_dtn(vector)
        physical_flux = NumericVector(circle_flux[index] / self.speeds[index] for index in range(self.n))
        ledger = self._ledger(
            problem="laplace_dtn",
            computed=("cycle QJet DtN applied on borrowed circle coordinates",),
            repaid=(
                "normal derivative scaled by autodiff speed factor |dz/dtheta|^-1",
                "physical flux returned at arbitrary-domain boundary samples",
                "dense DtN matrix was never formed",
            ),
        )
        return PullbackPDEEvaluation(
            physical_flux,
            ledger,
            "laplace_dtn",
            self._work_units(),
            self._stats(),
        )

    def solve_boundary_problem(self, problem: str, values: Iterable[complex], **parameters: float) -> PullbackPDEEvaluation:
        """Solve a pulled-back boundary PDE on an arbitrary domain."""

        vector = tuple(values)
        if len(vector) != self.n:
            raise ValueError("values length must match the pullback QJet size")
        if problem == "laplace_dtn":
            return self.apply_dtn(vector)
        if problem == "heat":
            output = cycle_dtn_heat(vector, parameters.get("time", 0.2))
            computed = ("cycle QJet heat semigroup exp(-t Lambda_circle)",)
        elif problem == "poisson":
            output = cycle_dtn_poisson_solve(vector, mass=parameters.get("mass", 0.25))
            computed = ("cycle QJet Poisson/Steklov solve (Lambda_circle + mass)^-1",)
        elif problem == "helmholtz":
            output = cycle_dtn_helmholtz_resolvent(
                vector,
                parameters.get("wavenumber", 3.7),
                damping=parameters.get("damping", 1.0e-3),
            )
            computed = ("cycle QJet Helmholtz boundary resolvent",)
        elif problem == "wave":
            output = cycle_dtn_wave(vector, parameters.get("time", 0.8))
            computed = ("cycle QJet wave propagator cos(t sqrt(Lambda_circle))",)
        else:
            raise ValueError(f"unknown boundary PDE problem: {problem}")
        ledger = self._ledger(
            problem=problem,
            computed=computed,
            repaid=(
                "pulled-back boundary field returned at arbitrary-domain samples",
                "autodiff geometry QJets retained for physical normal replay",
                "dense operator matrix was never formed",
            ),
        )
        return PullbackPDEEvaluation(output, ledger, problem, self._work_units(), self._stats())

    def _stats(self) -> dict[str, float | int | str]:
        return {
            "n": self.n,
            "theta_step": self.theta_step,
            "min_speed": self.min_speed,
            "max_speed": self.max_speed,
            "anisotropy": self.anisotropy,
            "protocol": "borrow_compute_repay_pullback",
        }

    def _work_units(self) -> int:
        return self.n

    def _ledger(
        self,
        *,
        problem: str,
        computed: tuple[str, ...],
        repaid: tuple[str, ...],
    ) -> BorrowComputeRepayLedger:
        return BorrowComputeRepayLedger(
            borrowed=(
                "autodiff boundary QJets z(theta), dz/dtheta",
                "regular-cycle modal QJet generator on the unit circle",
                "Dirichlet samples pulled back by theta-index identity",
            ),
            computed=computed,
            repaid=repaid,
            residuals=(
                ("min_autodiff_speed", self.min_speed),
                ("max_autodiff_speed", self.max_speed),
                ("speed_anisotropy", self.anisotropy),
            ),
            residual_norm=0.0,
            status="borrowed_repaid",
            notes=f"{problem} solved by matrix-free circle pullback and autodiff metric repay.",
        )


@dataclass(frozen=True)
class PlanarDomainQJet:
    """Matrix-free Q/DtN engine for arbitrary sampled planar domains.

    This is the sampled-domain path from the paper: arclength samples generate
    the inverse-square chord QJet on the actual curve.  That QJet contains the
    bounded geometry correction ``K0`` relative to the circle principal part,
    and the spectrum signature records which error channel is active.  No dense
    matrix is stored.
    """

    qjet: BoundaryQJet
    correction_order: float = 2.0

    @property
    def n(self) -> int:
        return self.qjet.n

    @property
    def length(self) -> float:
        return self.qjet.perimeter

    @property
    def scale(self) -> float:
        return dtn_scale(self.n, length=self.length)

    def apply_dtn(self, values: Iterable[complex]) -> PullbackPDEEvaluation:
        """Apply the arbitrary-planar-domain DtN generator."""

        vector = tuple(values)
        self._check_vector(vector)
        output = self._apply_operator(vector)
        ledger = self._ledger(
            problem="laplace_dtn",
            computed=("arbitrary-domain chord QJet applied with arclength scaling h/pi",),
            repaid=(
                "bounded K0 geometry correction retained by physical chord distances",
                "Q spectral error channel attached to the evaluation",
                "dense DtN matrix was never formed",
            ),
        )
        return PullbackPDEEvaluation(output, ledger, "laplace_dtn", self._work_units(1), self._stats())

    def solve_boundary_problem(
        self,
        problem: str,
        values: Iterable[complex],
        **parameters: float,
    ) -> PullbackPDEEvaluation:
        """Solve a boundary PDE on an arbitrary sampled planar domain."""

        vector = tuple(values)
        self._check_vector(vector)
        if problem == "laplace_dtn":
            return self.apply_dtn(vector)
        if problem == "heat":
            output, applications = self._heat(
                vector,
                parameters.get("time", 0.2),
                max_steps=int(parameters.get("max_steps", 256)),
            )
            computed = ("matrix-free heat stepping under the arbitrary-domain Q/DtN generator",)
        elif problem == "poisson":
            output, applications = self._poisson(
                vector,
                mass=parameters.get("mass", 0.25),
                iterations=int(parameters.get("iterations", 160)),
                tolerance=parameters.get("tolerance", 1.0e-10),
            )
            computed = ("matrix-free Richardson solve for (Lambda_Q + mass) u = f",)
        elif problem == "helmholtz":
            output, applications = self._helmholtz(
                vector,
                wavenumber=parameters.get("wavenumber", 3.7),
                damping=parameters.get("damping", 1.0e-2),
                iterations=int(parameters.get("iterations", 120)),
                tolerance=parameters.get("tolerance", 1.0e-8),
            )
            computed = ("matrix-free damped normal-equation solve for Helmholtz boundary resolvent",)
        elif problem == "wave":
            output, applications = self._wave(
                vector,
                time=parameters.get("time", 0.8),
                max_steps=int(parameters.get("max_steps", 256)),
            )
            computed = ("matrix-free leapfrog propagation under the arbitrary-domain Q/DtN generator",)
        else:
            raise ValueError(f"unknown boundary PDE problem: {problem}")
        ledger = self._ledger(
            problem=problem,
            computed=computed,
            repaid=(
                "smooth bounded-correction channel carried by the chord QJet",
                "corner/cusp channel exposed through the Q spectral signature",
                "dense operator matrix was never formed",
            ),
        )
        return PullbackPDEEvaluation(output, ledger, problem, self._work_units(applications), self._stats())

    def _apply_operator(self, values: Iterable[complex]) -> NumericVector:
        raw = self.qjet.apply(values)
        return NumericVector(self.scale * raw[index] for index in range(self.n))

    def _check_vector(self, vector: tuple[complex, ...]) -> None:
        if len(vector) != self.n:
            raise ValueError("values length must match the planar-domain QJet size")

    def _heat(
        self,
        vector: tuple[complex, ...],
        time: float,
        *,
        max_steps: int,
    ) -> tuple[NumericVector, int]:
        if time < 0.0:
            raise ValueError("time must be non-negative")
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        bound = self._operator_bound()
        steps = max(1, min(max_steps, int(math.ceil(max(time * bound / 0.25, 1.0)))))
        dt = time / steps if steps else 0.0
        values = [complex(value) for value in vector]
        applications = 0
        for _ in range(steps):
            k1 = self._scaled_operator(values, -1.0)
            k2 = self._scaled_operator(_add_scaled(values, k1, 0.5 * dt), -1.0)
            k3 = self._scaled_operator(_add_scaled(values, k2, 0.5 * dt), -1.0)
            k4 = self._scaled_operator(_add_scaled(values, k3, dt), -1.0)
            values = [
                values[index] + (dt / 6.0) * (k1[index] + 2.0 * k2[index] + 2.0 * k3[index] + k4[index])
                for index in range(self.n)
            ]
            applications += 4
        return NumericVector(values), applications

    def _poisson(
        self,
        vector: tuple[complex, ...],
        *,
        mass: float,
        iterations: int,
        tolerance: float,
    ) -> tuple[NumericVector, int]:
        if mass < 0.0:
            raise ValueError("mass must be non-negative")
        if iterations < 1:
            raise ValueError("iterations must be positive")
        rhs = [complex(value) for value in vector]
        if mass == 0.0:
            rhs = _project_mean_zero(rhs)
        rhs_norm = max(_norm_inf(rhs), 1.0e-14)
        values = [0.0 + 0.0j for _ in rhs]

        def apply_system(candidate: Iterable[complex]) -> list[complex]:
            candidate_values = [complex(value) for value in candidate]
            av = self._apply_operator(candidate_values)
            return [complex(av[index]) + mass * candidate_values[index] for index in range(self.n)]

        residual = rhs[:]
        direction = residual[:]
        residual_energy = _inner_real(residual, residual)
        applications = 0
        for _ in range(iterations):
            if math.sqrt(max(residual_energy, 0.0)) / rhs_norm <= tolerance:
                break
            applied = apply_system(direction)
            applications += 1
            denominator = _inner_real(direction, applied)
            if denominator <= 1.0e-30:
                break
            step = residual_energy / denominator
            values = [values[index] + step * direction[index] for index in range(self.n)]
            residual = [residual[index] - step * applied[index] for index in range(self.n)]
            next_energy = _inner_real(residual, residual)
            if math.sqrt(max(next_energy, 0.0)) / rhs_norm <= tolerance:
                residual_energy = next_energy
                break
            beta = next_energy / max(residual_energy, 1.0e-300)
            direction = [residual[index] + beta * direction[index] for index in range(self.n)]
            residual_energy = next_energy
        return NumericVector(values), applications

    def _helmholtz(
        self,
        vector: tuple[complex, ...],
        *,
        wavenumber: float,
        damping: float,
        iterations: int,
        tolerance: float,
    ) -> tuple[NumericVector, int]:
        if wavenumber < 0.0:
            raise ValueError("wavenumber must be non-negative")
        if damping <= 0.0:
            raise ValueError("damping must be positive")
        rhs = [complex(value) for value in vector]
        values = [0.0 + 0.0j for _ in rhs]
        shift = -wavenumber * wavenumber + 1j * damping
        rhs_normal, applications = self._helmholtz_apply(rhs, shift.conjugate())
        rhs_norm = max(_norm_inf(rhs_normal), 1.0e-14)
        residual = [complex(value) for value in rhs_normal]
        direction = residual[:]
        residual_energy = _inner_real(residual, residual)
        for _ in range(iterations):
            if math.sqrt(max(residual_energy, 0.0)) / rhs_norm <= tolerance:
                break
            bd, count_b = self._helmholtz_apply(direction, shift)
            normal_direction, count_bt = self._helmholtz_apply(bd, shift.conjugate())
            applications += count_b + count_bt
            denominator = _inner_real(direction, normal_direction)
            if denominator <= 1.0e-30:
                break
            step = residual_energy / denominator
            values = [values[index] + step * direction[index] for index in range(self.n)]
            residual = [residual[index] - step * normal_direction[index] for index in range(self.n)]
            next_energy = _inner_real(residual, residual)
            if math.sqrt(max(next_energy, 0.0)) / rhs_norm <= tolerance:
                residual_energy = next_energy
                break
            beta = next_energy / max(residual_energy, 1.0e-300)
            direction = [residual[index] + beta * direction[index] for index in range(self.n)]
            residual_energy = next_energy
        return NumericVector(values), applications

    def _helmholtz_apply(self, values: Iterable[complex], shift: complex) -> tuple[NumericVector, int]:
        av = self._apply_operator(values)
        aav = self._apply_operator(av)
        return NumericVector(aav[index] + shift * complex(values[index]) for index in range(self.n)), 2

    def _wave(
        self,
        vector: tuple[complex, ...],
        *,
        time: float,
        max_steps: int,
    ) -> tuple[NumericVector, int]:
        if max_steps < 1:
            raise ValueError("max_steps must be positive")
        bound = self._operator_bound()
        steps = max(1, min(max_steps, int(math.ceil(max(time * math.sqrt(max(bound, 0.0)) / 0.5, 1.0)))))
        dt = time / steps if steps else 0.0
        displacement = [complex(value) for value in vector]
        velocity = [0.0 + 0.0j for _ in vector]
        applications = 0
        for _ in range(steps):
            accel = self._scaled_operator(displacement, -1.0)
            velocity = [velocity[index] + 0.5 * dt * accel[index] for index in range(self.n)]
            displacement = [displacement[index] + dt * velocity[index] for index in range(self.n)]
            accel = self._scaled_operator(displacement, -1.0)
            velocity = [velocity[index] + 0.5 * dt * accel[index] for index in range(self.n)]
            applications += 2
        return NumericVector(displacement), applications

    def _scaled_operator(self, values: Iterable[complex], scale: float) -> list[complex]:
        av = self._apply_operator(values)
        return [scale * complex(value) for value in av]

    def _operator_bound(self) -> float:
        points = self.qjet.points
        row_sums = [0.0 for _ in range(self.n)]
        for i in range(self.n):
            pi = points.row_tuple(i)
            for j in range(i + 1, self.n):
                pj = points.row_tuple(j)
                dx = pi[0] - pj[0]
                dy = pi[1] - pj[1]
                distance2 = dx * dx + dy * dy
                if distance2 <= 0.0:
                    raise ValueError("duplicate boundary points produce singular Q weights")
                weight = 1.0 / distance2
                row_sums[i] += weight
                row_sums[j] += weight
        return 2.0 * self.scale * max(row_sums)

    def _signature_stats(self) -> dict[str, float | int | str]:
        try:
            signature = q_spectral_error_signature(self.qjet.points)
            return {
                "q_error_type": signature.error_type,
                "recommended_q": signature.recommended_q,
                "symbol_power": signature.symbol_power,
                "median_pair_split": signature.median_pair_split,
                "max_pair_split": signature.max_pair_split,
                "symbol_variation": signature.normalized_symbol_variation,
            }
        except ValueError as exc:
            return {
                "q_error_type": "insufficient_spectral_window",
                "recommended_q": "multipole_zeta_q",
                "signature_failure": str(exc),
            }

    def _stats(self) -> dict[str, float | int | str]:
        stats = {
            "n": self.n,
            "length": self.length,
            "scale_h_over_pi": self.scale,
            "operator_bound": self._operator_bound(),
            "protocol": "planar_chord_qjet_error_corrected",
            "correction_order": self.correction_order,
        }
        stats.update(self._signature_stats())
        return stats

    def _work_units(self, applications: int) -> int:
        return int(applications) * self.n * max(1, self.n - 1) // 2

    def _ledger(
        self,
        *,
        problem: str,
        computed: tuple[str, ...],
        repaid: tuple[str, ...],
    ) -> BorrowComputeRepayLedger:
        signature = self._signature_stats()
        return BorrowComputeRepayLedger(
            borrowed=(
                "arbitrary planar arclength/chord QJet generator",
                "inverse-square chord graph as Q = pi*Lambda + K0",
                "Q spectral error signature and recommended correction channel",
            ),
            computed=computed,
            repaid=repaid,
            residuals=(
                ("q_error_type", signature.get("q_error_type", "")),
                ("recommended_q", signature.get("recommended_q", "")),
                ("operator_bound", self._operator_bound()),
            ),
            residual_norm=0.0,
            status="borrowed_repaid",
            notes=f"{problem} evaluated on arbitrary planar samples with Q error-correction channels.",
        )


@dataclass(frozen=True)
class HarmonicCorrectionMode:
    """One orthonormal trace mode and its exact weak-flux residual."""

    name: str
    degree: int
    role: str
    trace: tuple[float, ...]
    residual_weak_flux: tuple[complex, ...]
    source_norm: float
    residual_norm: float


@dataclass(frozen=True)
class HarmonicMomentCorrection:
    """Low-rank harmonic reproduction and zeta-tail repayment layer.

    The correction stores only orthonormal trace jets and weak-flux residual
    columns.  Applying it costs ``O(n r)`` and does not form a dense matrix.
    """

    mass: tuple[float, ...]
    modes: tuple[HarmonicCorrectionMode, ...]
    moment_degree: int
    zeta_tail_degree: int
    orthogonalization_tolerance: float

    @property
    def rank(self) -> int:
        return len(self.modes)

    @property
    def moment_rank(self) -> int:
        return sum(1 for mode in self.modes if mode.role == "harmonic_moment")

    @property
    def zeta_tail_rank(self) -> int:
        return sum(1 for mode in self.modes if mode.role == "zeta_tail")

    @property
    def max_residual_norm(self) -> float:
        return max((mode.residual_norm for mode in self.modes), default=0.0)

    def apply(self, values: Iterable[complex], raw_flux: Iterable[complex]) -> NumericVector:
        vector = tuple(complex(value) for value in values)
        flux = [complex(value) for value in raw_flux]
        if len(vector) != len(self.mass) or len(flux) != len(self.mass):
            raise ValueError("correction dimensions do not match input values")
        weak_correction = [0.0 + 0.0j for _ in self.mass]
        for mode in self.modes:
            coefficient = sum(
                self.mass[index] * mode.trace[index] * vector[index]
                for index in range(len(self.mass))
            )
            if coefficient == 0.0:
                continue
            for index, residual in enumerate(mode.residual_weak_flux):
                weak_correction[index] += coefficient * residual
        return NumericVector(
            flux[index] + weak_correction[index] / self.mass[index]
            for index in range(len(self.mass))
        )


@dataclass(frozen=True)
class HelmholtzCorrectionMode:
    """One complex Helmholtz trace mode and its exact weak-flux residual."""

    name: str
    trace: tuple[complex, ...]
    residual_weak_flux: tuple[complex, ...]
    source_norm: float
    residual_norm: float


@dataclass(frozen=True)
class HelmholtzMomentCorrection:
    """Low-rank Helmholtz DtN reproduction layer.

    The layer stores complex trace QJets and weak-flux residuals only.  It
    corrects the Q principal operator by reproducing exact Helmholtz fluxes on
    a finite Herglotz/plane-wave basis without forming a dense boundary matrix.
    """

    mass: tuple[float, ...]
    wavenumber: float
    modes: tuple[HelmholtzCorrectionMode, ...]
    orthogonalization_tolerance: float

    @property
    def rank(self) -> int:
        return len(self.modes)

    @property
    def max_residual_norm(self) -> float:
        return max((mode.residual_norm for mode in self.modes), default=0.0)

    def apply(self, values: Iterable[complex], raw_flux: Iterable[complex]) -> NumericVector:
        vector = tuple(complex(value) for value in values)
        flux = [complex(value) for value in raw_flux]
        if len(vector) != len(self.mass) or len(flux) != len(self.mass):
            raise ValueError("Helmholtz correction dimensions do not match input values")
        weak_correction = [0.0 + 0.0j for _ in self.mass]
        for mode in self.modes:
            coefficient = sum(
                self.mass[index] * mode.trace[index].conjugate() * vector[index]
                for index in range(len(self.mass))
            )
            if coefficient == 0.0:
                continue
            for index, residual in enumerate(mode.residual_weak_flux):
                weak_correction[index] += coefficient * residual
        return NumericVector(
            flux[index] + weak_correction[index] / self.mass[index]
            for index in range(len(self.mass))
        )


@dataclass(frozen=True)
class HarmonicMomentCorrectedPlanarQJet(PlanarDomainQJet):
    """Planar QJet corrected to reproduce low harmonic DtN moments exactly."""

    harmonic_correction: HarmonicMomentCorrection | None = None

    def _apply_operator(self, values: Iterable[complex]) -> NumericVector:
        vector = tuple(values)
        raw = PlanarDomainQJet._apply_operator(self, vector)
        if self.harmonic_correction is None:
            return raw
        return self.harmonic_correction.apply(vector, raw)

    def _stats(self) -> dict[str, float | int | str]:
        stats = PlanarDomainQJet._stats(self)
        if self.harmonic_correction is None:
            return stats
        stats.update(
            {
                "protocol": "planar_chord_qjet_harmonic_zeta_repaid",
                "harmonic_moment_degree": self.harmonic_correction.moment_degree,
                "zeta_tail_degree": self.harmonic_correction.zeta_tail_degree,
                "correction_rank": self.harmonic_correction.rank,
                "harmonic_moment_rank": self.harmonic_correction.moment_rank,
                "zeta_tail_rank": self.harmonic_correction.zeta_tail_rank,
                "max_harmonic_residual_norm": self.harmonic_correction.max_residual_norm,
                "zeta_tail_projection": "orthogonalized_after_harmonic_moments",
            }
        )
        return stats

    def _work_units(self, applications: int) -> int:
        raw = PlanarDomainQJet._work_units(self, applications)
        rank = 0 if self.harmonic_correction is None else self.harmonic_correction.rank
        return raw + int(applications) * self.n * max(rank, 0)

    def _ledger(
        self,
        *,
        problem: str,
        computed: tuple[str, ...],
        repaid: tuple[str, ...],
    ) -> BorrowComputeRepayLedger:
        signature = self._signature_stats()
        correction = self.harmonic_correction
        correction_residuals: tuple[tuple[str, float | int | str], ...]
        if correction is None:
            correction_residuals = ()
        else:
            correction_residuals = (
                ("harmonic_moment_degree", correction.moment_degree),
                ("zeta_tail_degree", correction.zeta_tail_degree),
                ("correction_rank", correction.rank),
                ("harmonic_moment_rank", correction.moment_rank),
                ("zeta_tail_rank", correction.zeta_tail_rank),
                ("max_harmonic_residual_norm", correction.max_residual_norm),
            )
        return BorrowComputeRepayLedger(
            borrowed=(
                "arbitrary planar arclength/chord QJet generator",
                "mass-lumped weak boundary pairing",
                "exact harmonic polynomial weak flux moments",
                "projected zeta-tail harmonic multipole correction modes",
            ),
            computed=tuple(computed)
            + (
                "raw planar QJet applied matrix-free",
                "low-rank harmonic/zeta repayment applied without dense operator storage",
            ),
            repaid=tuple(repaid)
            + (
                "constant and low harmonic DtN moments reproduced in weak form",
                "zeta-tail modes orthogonalized against reproduced harmonic moments",
            ),
            residuals=(
                ("q_error_type", signature.get("q_error_type", "")),
                ("recommended_q", signature.get("recommended_q", "")),
                ("operator_bound", self._operator_bound()),
            )
            + correction_residuals,
            residual_norm=0.0,
            status="borrowed_repaid",
            notes=f"{problem} evaluated with harmonic moment and zeta-tail repayment.",
        )


@dataclass(frozen=True)
class HelmholtzMomentCorrectedPlanarQJet(HarmonicMomentCorrectedPlanarQJet):
    """Planar QJet with a Helmholtz-specific reproduction layer."""

    helmholtz_correction: HelmholtzMomentCorrection | None = None

    def apply_helmholtz_dtn(self, values: Iterable[complex]) -> PullbackPDEEvaluation:
        """Apply the corrected interior Helmholtz DtN map.

        The raw operator is the existing harmonic/zeta-corrected Q principal
        DtN.  The Helmholtz layer repays exact weak flux on stored basis
        solutions, which installs the lower-order ``k``-dependent channel.
        """

        vector = tuple(complex(value) for value in values)
        self._check_vector(vector)
        raw = HarmonicMomentCorrectedPlanarQJet._apply_operator(self, vector)
        output = raw if self.helmholtz_correction is None else self.helmholtz_correction.apply(vector, raw)
        ledger = self._ledger(
            problem="helmholtz_dtn",
            computed=(
                "harmonic/zeta Q principal DtN applied matrix-free",
                "complex Helmholtz reproduction modes projected in mass pairing",
            ),
            repaid=(
                "wavenumber-dependent weak flux residual repaid on Helmholtz basis",
                "boundary normal flux returned without dense Helmholtz matrix storage",
            ),
        )
        return PullbackPDEEvaluation(
            output,
            ledger,
            "helmholtz_dtn",
            self._work_units(1),
            self._stats(),
        )

    def solve_boundary_problem(
        self,
        problem: str,
        values: Iterable[complex],
        **parameters: float,
    ) -> PullbackPDEEvaluation:
        if problem == "helmholtz_dtn":
            return self.apply_helmholtz_dtn(values)
        return HarmonicMomentCorrectedPlanarQJet.solve_boundary_problem(self, problem, values, **parameters)

    def _stats(self) -> dict[str, float | int | str]:
        stats = HarmonicMomentCorrectedPlanarQJet._stats(self)
        if self.helmholtz_correction is None:
            return stats
        stats.update(
            {
                "protocol": "planar_chord_qjet_harmonic_zeta_helmholtz_repaid",
                "helmholtz_wavenumber": self.helmholtz_correction.wavenumber,
                "helmholtz_correction_rank": self.helmholtz_correction.rank,
                "max_helmholtz_residual_norm": self.helmholtz_correction.max_residual_norm,
                "helmholtz_basis": "complex plane-wave Herglotz traces",
            }
        )
        return stats

    def _work_units(self, applications: int) -> int:
        raw = HarmonicMomentCorrectedPlanarQJet._work_units(self, applications)
        rank = 0 if self.helmholtz_correction is None else self.helmholtz_correction.rank
        return raw + int(applications) * self.n * max(rank, 0)


BoundaryQJetMap = Callable[[ScalarQJet], tuple[ScalarQJet, ScalarQJet]]


def qjet_sin(value: float | ScalarQJet) -> ScalarQJet:
    jet = ScalarQJet.coerce(value)
    return ScalarQJet(math.sin(jet.value), math.cos(jet.value) * jet.derivative)


def qjet_cos(value: float | ScalarQJet) -> ScalarQJet:
    jet = ScalarQJet.coerce(value)
    return ScalarQJet(math.cos(jet.value), -math.sin(jet.value) * jet.derivative)


def qjet_exp(value: float | ScalarQJet) -> ScalarQJet:
    jet = ScalarQJet.coerce(value)
    out = math.exp(jet.value)
    return ScalarQJet(out, out * jet.derivative)


def qjet_theta(value: float) -> ScalarQJet:
    """Return the autodiff seed for a boundary angle."""

    return ScalarQJet(value, 1.0)


def circle_qjet_map(radius: float = 1.0, center: tuple[float, float] = (0.0, 0.0)) -> BoundaryQJetMap:
    """Return an autodiff QJet map for a circle."""

    if radius <= 0.0:
        raise ValueError("radius must be positive")

    def boundary(theta: ScalarQJet) -> tuple[ScalarQJet, ScalarQJet]:
        return (
            center[0] + radius * qjet_cos(theta),
            center[1] + radius * qjet_sin(theta),
        )

    return boundary


def ellipse_qjet_map(
    a: float,
    b: float,
    *,
    center: tuple[float, float] = (0.0, 0.0),
    phase: float = 0.0,
) -> BoundaryQJetMap:
    """Return an autodiff QJet map for an ellipse boundary."""

    if a <= 0.0 or b <= 0.0:
        raise ValueError("ellipse axes must be positive")

    def boundary(theta: ScalarQJet) -> tuple[ScalarQJet, ScalarQJet]:
        shifted = theta + phase
        return (
            center[0] + a * qjet_cos(shifted),
            center[1] + b * qjet_sin(shifted),
        )

    return boundary


def radial_fourier_qjet_map(
    base_radius: float,
    cos_coefficients: Iterable[float] = (),
    sin_coefficients: Iterable[float] = (),
    *,
    center: tuple[float, float] = (0.0, 0.0),
) -> BoundaryQJetMap:
    """Return an autodiff QJet map for a smooth star-shaped domain."""

    cos_values = tuple(float(value) for value in cos_coefficients)
    sin_values = tuple(float(value) for value in sin_coefficients)
    if base_radius <= 0.0:
        raise ValueError("base_radius must be positive")

    def boundary(theta: ScalarQJet) -> tuple[ScalarQJet, ScalarQJet]:
        radius = ScalarQJet(base_radius)
        for mode, coefficient in enumerate(cos_values, start=1):
            radius += coefficient * qjet_cos(mode * theta)
        for mode, coefficient in enumerate(sin_values, start=1):
            radius += coefficient * qjet_sin(mode * theta)
        if radius.value <= 0.0:
            raise ValueError("radial map left the star-shaped regime")
        return (
            center[0] + radius * qjet_cos(theta),
            center[1] + radius * qjet_sin(theta),
        )

    return boundary


def build_boundary_pullback_qjet(n: int, boundary_map: BoundaryQJetMap) -> BoundaryPullbackQJet:
    """Sample an arbitrary boundary map as autodiff QJets."""

    if n < 3:
        raise ValueError("n must be at least three")
    points: list[tuple[float, float]] = []
    speeds: list[float] = []
    tangents: list[tuple[float, float]] = []
    normals: list[tuple[float, float]] = []
    theta_step = 2.0 * math.pi / n
    for index in range(n):
        theta = qjet_theta(theta_step * index)
        x_jet, y_jet = boundary_map(theta)
        x_jet = ScalarQJet.coerce(x_jet)
        y_jet = ScalarQJet.coerce(y_jet)
        speed = math.hypot(x_jet.derivative, y_jet.derivative)
        if speed <= 0.0 or not math.isfinite(speed):
            raise ValueError("boundary map has a degenerate autodiff tangent")
        tangent = (x_jet.derivative / speed, y_jet.derivative / speed)
        normal = (tangent[1], -tangent[0])
        points.append((x_jet.value, y_jet.value))
        speeds.append(speed)
        tangents.append(tangent)
        normals.append(normal)
    return BoundaryPullbackQJet(
        tuple(points),
        tuple(speeds),
        tuple(tangents),
        tuple(normals),
        theta_step,
    )


def apply_pullback_dtn(
    values: Iterable[complex],
    boundary_map: BoundaryQJetMap,
) -> PullbackPDEEvaluation:
    """Apply arbitrary-domain Laplace DtN through a circle pullback."""

    vector = tuple(values)
    return build_boundary_pullback_qjet(len(vector), boundary_map).apply_dtn(vector)


def solve_pullback_boundary_pde(
    problem: str,
    values: Iterable[complex],
    boundary_map: BoundaryQJetMap,
    **parameters: float,
) -> PullbackPDEEvaluation:
    """Solve a boundary PDE on an arbitrary mapped domain without dense matrices."""

    vector = tuple(values)
    return build_boundary_pullback_qjet(len(vector), boundary_map).solve_boundary_problem(
        problem,
        vector,
        **parameters,
    )


def _polygon_area_tuple(points: tuple[tuple[float, float], ...]) -> float:
    return 0.5 * sum(
        points[index][0] * points[(index + 1) % len(points)][1]
        - points[index][1] * points[(index + 1) % len(points)][0]
        for index in range(len(points))
    )


def _boundary_lumped_mass_tuple(points: tuple[tuple[float, float], ...]) -> tuple[float, ...]:
    mass = []
    n = len(points)
    for index in range(n):
        left = points[(index - 1) % n]
        center = points[index]
        right = points[(index + 1) % n]
        previous_length = math.hypot(center[0] - left[0], center[1] - left[1])
        next_length = math.hypot(right[0] - center[0], right[1] - center[1])
        value = 0.5 * (previous_length + next_length)
        if value <= 0.0:
            raise ValueError("boundary contains duplicate adjacent samples")
        mass.append(value)
    return tuple(mass)


def _edge_outward_normal(
    left: tuple[float, float],
    right: tuple[float, float],
    orientation: float,
) -> tuple[tuple[float, float], float]:
    dx = right[0] - left[0]
    dy = right[1] - left[1]
    length = math.hypot(dx, dy)
    if length <= 0.0:
        raise ValueError("boundary contains duplicate adjacent samples")
    if orientation >= 0.0:
        return (dy / length, -dx / length), length
    return (-dy / length, dx / length), length


def _harmonic_value_gradient(
    x: float,
    y: float,
    degree: int,
    component: str,
) -> tuple[float, tuple[float, float]]:
    if degree == 0:
        return 1.0, (0.0, 0.0)
    z_power = complex(x, y) ** degree
    derivative_power = degree * (complex(x, y) ** (degree - 1))
    if component == "cos":
        return z_power.real, (derivative_power.real, -derivative_power.imag)
    if component == "sin":
        return z_power.imag, (derivative_power.imag, derivative_power.real)
    raise ValueError(f"unknown harmonic component: {component}")


def harmonic_polynomial_trace(
    points: Iterable[Iterable[float]],
    degree: int,
    component: str = "cos",
) -> tuple[float, ...]:
    """Sample ``Re(z^degree)`` or ``Im(z^degree)`` on boundary points."""

    rows = tuple((float(row[0]), float(row[1])) for row in points)
    if degree < 0:
        raise ValueError("degree must be non-negative")
    if degree == 0 and component != "cos":
        raise ValueError("degree-zero harmonic has only the cos/constant component")
    return tuple(_harmonic_value_gradient(x, y, degree, component)[0] for x, y in rows)


def harmonic_polynomial_weak_flux(
    points: Iterable[Iterable[float]],
    degree: int,
    component: str = "cos",
) -> tuple[float, ...]:
    """Return exact weak boundary flux for a harmonic polynomial trace.

    Flux is integrated against boundary hat functions on the polygonal sampled
    boundary, so corner normals never need to be assigned pointwise.
    """

    rows = tuple((float(row[0]), float(row[1])) for row in points)
    if len(rows) < 3:
        raise ValueError("at least three boundary points are required")
    if degree < 0:
        raise ValueError("degree must be non-negative")
    if degree == 0 and component != "cos":
        raise ValueError("degree-zero harmonic has only the cos/constant component")
    out = [0.0 for _ in rows]
    orientation = 1.0 if _polygon_area_tuple(rows) >= 0.0 else -1.0
    # Eight-point Gauss-Legendre on [0, 1], exact for polynomial integrands
    # through degree 15 and still robust for the external/mapped audit path.
    gauss_x = (
        0.019855071751231884,
        0.10166676129318664,
        0.2372337950418355,
        0.4082826787521751,
        0.5917173212478249,
        0.7627662049581645,
        0.8983332387068134,
        0.9801449282487681,
    )
    gauss_w = (
        0.05061426814518813,
        0.11119051722668724,
        0.15685332293894363,
        0.18134189168918088,
        0.18134189168918088,
        0.15685332293894363,
        0.11119051722668724,
        0.05061426814518813,
    )
    for index, left in enumerate(rows):
        right = rows[(index + 1) % len(rows)]
        normal, length = _edge_outward_normal(left, right, orientation)
        for s, weight in zip(gauss_x, gauss_w, strict=True):
            x = (1.0 - s) * left[0] + s * right[0]
            y = (1.0 - s) * left[1] + s * right[1]
            _, gradient = _harmonic_value_gradient(x, y, degree, component)
            normal_flux = gradient[0] * normal[0] + gradient[1] * normal[1]
            contribution = weight * length * normal_flux
            out[index] += contribution * (1.0 - s)
            out[(index + 1) % len(rows)] += contribution * s
    return tuple(out)


def helmholtz_plane_wave_trace(
    points: Iterable[Iterable[float]],
    wavenumber: float,
    angle: float,
) -> tuple[complex, ...]:
    """Sample ``exp(i k d.angle x)`` on boundary points."""

    if wavenumber < 0.0:
        raise ValueError("wavenumber must be non-negative")
    dx = math.cos(angle)
    dy = math.sin(angle)
    rows = tuple((float(row[0]), float(row[1])) for row in points)
    return tuple(
        complex(math.cos(wavenumber * (dx * x + dy * y)), math.sin(wavenumber * (dx * x + dy * y)))
        for x, y in rows
    )


def helmholtz_plane_wave_weak_flux(
    points: Iterable[Iterable[float]],
    wavenumber: float,
    angle: float,
) -> tuple[complex, ...]:
    """Return exact weak normal flux for a Helmholtz plane wave.

    Flux is integrated against boundary hat functions on polygon edges, matching
    the finite boundary pairing used by the Q correction layer.
    """

    if wavenumber < 0.0:
        raise ValueError("wavenumber must be non-negative")
    rows = tuple((float(row[0]), float(row[1])) for row in points)
    if len(rows) < 3:
        raise ValueError("at least three boundary points are required")
    direction = (math.cos(angle), math.sin(angle))
    out = [0.0 + 0.0j for _ in rows]
    orientation = 1.0 if _polygon_area_tuple(rows) >= 0.0 else -1.0
    gauss_x = (
        0.019855071751231884,
        0.10166676129318664,
        0.2372337950418355,
        0.4082826787521751,
        0.5917173212478249,
        0.7627662049581645,
        0.8983332387068134,
        0.9801449282487681,
    )
    gauss_w = (
        0.05061426814518813,
        0.11119051722668724,
        0.15685332293894363,
        0.18134189168918088,
        0.18134189168918088,
        0.15685332293894363,
        0.11119051722668724,
        0.05061426814518813,
    )
    for index, left in enumerate(rows):
        right = rows[(index + 1) % len(rows)]
        normal, length = _edge_outward_normal(left, right, orientation)
        normal_factor = direction[0] * normal[0] + direction[1] * normal[1]
        for s, weight in zip(gauss_x, gauss_w, strict=True):
            x = (1.0 - s) * left[0] + s * right[0]
            y = (1.0 - s) * left[1] + s * right[1]
            phase = wavenumber * (direction[0] * x + direction[1] * y)
            value = complex(math.cos(phase), math.sin(phase))
            normal_flux = 1j * wavenumber * normal_factor * value
            contribution = weight * length * normal_flux
            out[index] += contribution * (1.0 - s)
            out[(index + 1) % len(rows)] += contribution * s
    return tuple(out)


def _mass_inner_real(left: Iterable[float], right: Iterable[float], mass: Iterable[float]) -> float:
    return sum(m * a * b for a, b, m in zip(left, right, mass, strict=True))


def _weighted_norm_real(values: Iterable[float], mass: Iterable[float]) -> float:
    return math.sqrt(max(_mass_inner_real(values, values, mass), 0.0))


def _weighted_norm_complex(values: Iterable[complex], mass: Iterable[float]) -> float:
    return math.sqrt(
        max(sum(m * abs(complex(value)) ** 2 for value, m in zip(values, mass, strict=True)), 0.0)
    )


def _mass_inner_complex(
    left: Iterable[complex],
    right: Iterable[complex],
    mass: Iterable[float],
) -> complex:
    return sum(
        m * complex(a).conjugate() * complex(b)
        for a, b, m in zip(left, right, mass, strict=True)
    )


def _select_zeta_tail_degree(
    qjet: PlanarDomainQJet,
    moment_degree: int,
    zeta_tail_degree: int | None,
) -> int:
    if zeta_tail_degree is not None:
        return max(moment_degree, int(zeta_tail_degree))
    try:
        error_type = q_spectral_error_signature(qjet.qjet.points).error_type
    except ValueError:
        error_type = "insufficient_spectral_window"
    if "cusp" in error_type or "corner" in error_type or "vertex" in error_type:
        return max(moment_degree, 7)
    if "low_regularity" in error_type or "mixed" in error_type:
        return max(moment_degree, 6)
    return max(moment_degree, 5)


def _build_harmonic_moment_correction(
    raw_qjet: PlanarDomainQJet,
    *,
    moment_degree: int,
    zeta_tail_degree: int | None,
    orthogonalization_tolerance: float,
) -> HarmonicMomentCorrection:
    if moment_degree < 0:
        raise ValueError("moment_degree must be non-negative")
    points = tuple(raw_qjet.qjet.points.row_tuple(index) for index in range(raw_qjet.n))
    mass = _boundary_lumped_mass_tuple(points)
    tail_degree = _select_zeta_tail_degree(raw_qjet, moment_degree, zeta_tail_degree)
    modes: list[HarmonicCorrectionMode] = []
    for degree in range(tail_degree + 1):
        components = ("cos",) if degree == 0 else ("cos", "sin")
        for component in components:
            trace = list(harmonic_polynomial_trace(points, degree, component))
            exact_weak_flux = harmonic_polynomial_weak_flux(points, degree, component)
            raw_flux = raw_qjet._apply_operator(trace)
            residual = [
                complex(exact_weak_flux[index]) - mass[index] * complex(raw_flux[index])
                for index in range(raw_qjet.n)
            ]
            source_norm = _weighted_norm_real(trace, mass)
            if source_norm <= orthogonalization_tolerance:
                continue
            for mode in modes:
                coefficient = _mass_inner_real(mode.trace, trace, mass)
                if coefficient == 0.0:
                    continue
                trace = [trace[index] - coefficient * mode.trace[index] for index in range(raw_qjet.n)]
                residual = [
                    residual[index] - coefficient * mode.residual_weak_flux[index]
                    for index in range(raw_qjet.n)
                ]
            norm = _weighted_norm_real(trace, mass)
            if norm <= orthogonalization_tolerance * max(1.0, source_norm):
                continue
            normalized_trace = tuple(value / norm for value in trace)
            normalized_residual = tuple(value / norm for value in residual)
            role = "harmonic_moment" if degree <= moment_degree else "zeta_tail"
            modes.append(
                HarmonicCorrectionMode(
                    name=f"{component}_z^{degree}" if degree else "constant",
                    degree=degree,
                    role=role,
                    trace=normalized_trace,
                    residual_weak_flux=normalized_residual,
                    source_norm=source_norm,
                    residual_norm=_weighted_norm_complex(normalized_residual, (1.0 / value for value in mass)),
                )
            )
    return HarmonicMomentCorrection(
        mass=mass,
        modes=tuple(modes),
        moment_degree=moment_degree,
        zeta_tail_degree=tail_degree,
        orthogonalization_tolerance=orthogonalization_tolerance,
    )


def _build_helmholtz_moment_correction(
    raw_qjet: HarmonicMomentCorrectedPlanarQJet,
    *,
    wavenumber: float,
    plane_wave_directions: Iterable[float],
    orthogonalization_tolerance: float,
) -> HelmholtzMomentCorrection:
    if wavenumber < 0.0:
        raise ValueError("wavenumber must be non-negative")
    points = tuple(raw_qjet.qjet.points.row_tuple(index) for index in range(raw_qjet.n))
    mass = (
        raw_qjet.harmonic_correction.mass
        if raw_qjet.harmonic_correction is not None
        else _boundary_lumped_mass_tuple(points)
    )
    modes: list[HelmholtzCorrectionMode] = []
    for mode_index, angle in enumerate(tuple(float(value) for value in plane_wave_directions)):
        trace = list(helmholtz_plane_wave_trace(points, wavenumber, angle))
        exact_weak_flux = helmholtz_plane_wave_weak_flux(points, wavenumber, angle)
        raw_flux = raw_qjet._apply_operator(trace)
        residual = [
            complex(exact_weak_flux[index]) - mass[index] * complex(raw_flux[index])
            for index in range(raw_qjet.n)
        ]
        source_norm = _weighted_norm_complex(trace, mass)
        if source_norm <= orthogonalization_tolerance:
            continue
        for _ in range(2):
            for mode in modes:
                coefficient = _mass_inner_complex(mode.trace, trace, mass)
                if coefficient == 0.0:
                    continue
                trace = [trace[index] - coefficient * mode.trace[index] for index in range(raw_qjet.n)]
                residual = [
                    residual[index] - coefficient * mode.residual_weak_flux[index]
                    for index in range(raw_qjet.n)
                ]
        norm = _weighted_norm_complex(trace, mass)
        if norm <= orthogonalization_tolerance * max(1.0, source_norm):
            continue
        normalized_trace = tuple(value / norm for value in trace)
        normalized_residual = tuple(value / norm for value in residual)
        modes.append(
            HelmholtzCorrectionMode(
                name=f"plane_wave_{mode_index}",
                trace=normalized_trace,
                residual_weak_flux=normalized_residual,
                source_norm=source_norm,
                residual_norm=_weighted_norm_complex(
                    normalized_residual,
                    (1.0 / value for value in mass),
                ),
            )
        )
    return HelmholtzMomentCorrection(
        mass=mass,
        wavenumber=float(wavenumber),
        modes=tuple(modes),
        orthogonalization_tolerance=orthogonalization_tolerance,
    )


def build_planar_domain_qjet(
    points: Iterable[Iterable[float]],
    *,
    correction_order: float = 2.0,
) -> PlanarDomainQJet:
    """Build an error-corrected Q/DtN engine for arbitrary planar samples."""

    return PlanarDomainQJet(BoundaryQJet(points), correction_order=correction_order)


def build_harmonic_moment_corrected_planar_qjet(
    points: Iterable[Iterable[float]],
    *,
    correction_order: float = 2.0,
    moment_degree: int = 2,
    zeta_tail_degree: int | None = None,
    orthogonalization_tolerance: float = 1.0e-11,
) -> HarmonicMomentCorrectedPlanarQJet:
    """Build a matrix-free planar QJet with harmonic/zeta repayment.

    The low-rank correction enforces exact weak DtN flux for harmonic
    polynomials through ``moment_degree``.  Higher harmonic multipoles through
    ``zeta_tail_degree`` form the projected zeta-tail correction channel.
    """

    raw = PlanarDomainQJet(BoundaryQJet(points), correction_order=correction_order)
    correction = _build_harmonic_moment_correction(
        raw,
        moment_degree=moment_degree,
        zeta_tail_degree=zeta_tail_degree,
        orthogonalization_tolerance=orthogonalization_tolerance,
    )
    return HarmonicMomentCorrectedPlanarQJet(
        raw.qjet,
        correction_order=correction_order,
        harmonic_correction=correction,
    )


def build_helmholtz_moment_corrected_planar_qjet(
    points: Iterable[Iterable[float]],
    wavenumber: float,
    *,
    correction_order: float = 2.0,
    moment_degree: int = 2,
    zeta_tail_degree: int | None = None,
    plane_wave_count: int = 24,
    plane_wave_directions: Iterable[float] | None = None,
    orthogonalization_tolerance: float = 1.0e-8,
) -> HelmholtzMomentCorrectedPlanarQJet:
    """Build a matrix-free Helmholtz DtN QJet for planar samples.

    The builder keeps the harmonic/zeta corrected Q principal part and adds a
    low-rank exact Helmholtz reproduction layer on complex plane-wave traces.
    """

    if plane_wave_directions is None:
        if plane_wave_count < 1:
            raise ValueError("plane_wave_count must be positive")
        directions = tuple(2.0 * math.pi * index / plane_wave_count for index in range(plane_wave_count))
    else:
        directions = tuple(float(value) for value in plane_wave_directions)
        if not directions:
            raise ValueError("plane_wave_directions must not be empty")
    harmonic = build_harmonic_moment_corrected_planar_qjet(
        points,
        correction_order=correction_order,
        moment_degree=moment_degree,
        zeta_tail_degree=zeta_tail_degree,
        orthogonalization_tolerance=orthogonalization_tolerance,
    )
    helmholtz_correction = _build_helmholtz_moment_correction(
        harmonic,
        wavenumber=wavenumber,
        plane_wave_directions=directions,
        orthogonalization_tolerance=orthogonalization_tolerance,
    )
    return HelmholtzMomentCorrectedPlanarQJet(
        harmonic.qjet,
        correction_order=correction_order,
        harmonic_correction=harmonic.harmonic_correction,
        helmholtz_correction=helmholtz_correction,
    )


def apply_planar_domain_dtn(
    values: Iterable[complex],
    points: Iterable[Iterable[float]],
    *,
    correction_order: float = 2.0,
) -> PullbackPDEEvaluation:
    """Apply the arbitrary-planar-domain DtN generator without a dense matrix."""

    return build_planar_domain_qjet(points, correction_order=correction_order).apply_dtn(values)


def apply_harmonic_moment_corrected_planar_dtn(
    values: Iterable[complex],
    points: Iterable[Iterable[float]],
    *,
    correction_order: float = 2.0,
    moment_degree: int = 2,
    zeta_tail_degree: int | None = None,
) -> PullbackPDEEvaluation:
    """Apply the harmonic/zeta corrected planar DtN generator."""

    return build_harmonic_moment_corrected_planar_qjet(
        points,
        correction_order=correction_order,
        moment_degree=moment_degree,
        zeta_tail_degree=zeta_tail_degree,
    ).apply_dtn(values)


def solve_planar_domain_boundary_pde(
    problem: str,
    values: Iterable[complex],
    points: Iterable[Iterable[float]],
    *,
    correction_order: float = 2.0,
    **parameters: float,
) -> PullbackPDEEvaluation:
    """Solve a planar-domain boundary PDE with Q error-correction channels."""

    return build_planar_domain_qjet(points, correction_order=correction_order).solve_boundary_problem(
        problem,
        values,
        **parameters,
    )


def solve_harmonic_moment_corrected_planar_boundary_pde(
    problem: str,
    values: Iterable[complex],
    points: Iterable[Iterable[float]],
    *,
    correction_order: float = 2.0,
    moment_degree: int = 2,
    zeta_tail_degree: int | None = None,
    **parameters: float,
) -> PullbackPDEEvaluation:
    """Solve a planar boundary PDE with harmonic/zeta corrected QJets."""

    return build_harmonic_moment_corrected_planar_qjet(
        points,
        correction_order=correction_order,
        moment_degree=moment_degree,
        zeta_tail_degree=zeta_tail_degree,
    ).solve_boundary_problem(problem, values, **parameters)


def apply_helmholtz_moment_corrected_planar_dtn(
    values: Iterable[complex],
    points: Iterable[Iterable[float]],
    wavenumber: float,
    *,
    correction_order: float = 2.0,
    moment_degree: int = 2,
    zeta_tail_degree: int | None = None,
    plane_wave_count: int = 24,
    plane_wave_directions: Iterable[float] | None = None,
) -> PullbackPDEEvaluation:
    """Apply the Helmholtz reproduction-corrected planar DtN generator."""

    return build_helmholtz_moment_corrected_planar_qjet(
        points,
        wavenumber,
        correction_order=correction_order,
        moment_degree=moment_degree,
        zeta_tail_degree=zeta_tail_degree,
        plane_wave_count=plane_wave_count,
        plane_wave_directions=plane_wave_directions,
    ).apply_helmholtz_dtn(values)


def _add_scaled(left: Iterable[complex], right: Iterable[complex], scale: float) -> list[complex]:
    return [complex(a) + scale * complex(b) for a, b in zip(left, right, strict=True)]


def _project_mean_zero(values: Iterable[complex]) -> list[complex]:
    vector = [complex(value) for value in values]
    mean = sum(vector) / len(vector)
    return [value - mean for value in vector]


def _norm_inf(values: Iterable[complex]) -> float:
    return max((abs(value) for value in values), default=0.0)


def _inner_real(left: Iterable[complex], right: Iterable[complex]) -> float:
    return float(sum((complex(a).conjugate() * complex(b)).real for a, b in zip(left, right, strict=True)))


def dtn_scale(n: int, *, length: float = 2.0 * math.pi) -> float:
    """Return the Q-to-DtN scale h/pi."""

    if n < 2:
        raise ValueError("n must be at least two")
    if length <= 0.0:
        raise ValueError("length must be positive")
    return length / (math.pi * n)


def cycle_dtn_eigenvalue(index: int, n: int, *, length: float = 2.0 * math.pi) -> float:
    """Return the normalized Q/DtN eigenvalue for a Fourier index."""

    qjet = CycleQJet(n)
    return dtn_scale(n, length=length) * qjet.eigenvalue(index)


def continuum_repaid_dtn_eigenvalue(index: int, n: int, *, length: float = 2.0 * math.pi) -> float:
    """Return the moment/zeta-repaid circle DtN eigenvalue for a Fourier index.

    The raw finite cycle symbol is ``m - m^2/n`` on the unit circle.  The
    production disk/conformal-pullback path repays that known endpoint defect
    before making a continuum PDE error claim, so the applied spectrum is the
    exact sampled-circle DtN symbol ``|m|``.
    """

    folded = abs(CycleQJet(n).signed_mode(index))
    return (2.0 * math.pi / length) * folded


def cycle_dtn_eigenvalues(n: int, *, length: float = 2.0 * math.pi) -> NumericVector:
    """Return all generated Q/DtN eigenvalues."""

    scale = dtn_scale(n, length=length)
    return NumericVector(scale * value for value in CycleQJet(n).eigenvalues())


def continuum_repaid_dtn_eigenvalues(n: int, *, length: float = 2.0 * math.pi) -> NumericVector:
    """Return all continuum-repaid circle DtN eigenvalues."""

    return NumericVector(continuum_repaid_dtn_eigenvalue(index, n, length=length) for index in range(n))


def apply_cycle_dtn(values: Iterable[complex], *, length: float = 2.0 * math.pi) -> NumericVector:
    """Apply the boundary-only DtN generator Λ_Q."""

    vector = tuple(values)
    scale = dtn_scale(len(vector), length=length)
    return CycleQJet(len(vector)).apply_function(vector, lambda lam: scale * lam, zero_mode=0.0)


def apply_continuum_repaid_dtn(values: Iterable[complex], *, length: float = 2.0 * math.pi) -> NumericVector:
    """Apply the final continuum-repaid circle DtN generator."""

    vector = tuple(values)
    n = len(vector)
    return CycleQJet(n).apply_index_function(
        vector,
        lambda index: continuum_repaid_dtn_eigenvalue(index, n, length=length),
        zero_mode=0.0,
    )


def cycle_dtn_heat(
    values: Iterable[complex],
    time: float,
    *,
    length: float = 2.0 * math.pi,
) -> NumericVector:
    """Apply exp(-t Λ_Q) to boundary data."""

    if time < 0.0:
        raise ValueError("time must be non-negative")
    vector = tuple(values)
    scale = dtn_scale(len(vector), length=length)
    return CycleQJet(len(vector)).apply_function(
        vector,
        lambda lam: math.exp(-time * scale * lam),
        zero_mode=1.0,
    )


def continuum_repaid_dtn_heat(
    values: Iterable[complex],
    time: float,
    *,
    length: float = 2.0 * math.pi,
) -> NumericVector:
    """Apply exp(-t Λ) with the continuum-repaid circle DtN symbol."""

    if time < 0.0:
        raise ValueError("time must be non-negative")
    vector = tuple(values)
    n = len(vector)
    return CycleQJet(n).apply_index_function(
        vector,
        lambda index: math.exp(-time * continuum_repaid_dtn_eigenvalue(index, n, length=length)),
        zero_mode=1.0,
    )


def cycle_dtn_poisson_solve(
    values: Iterable[complex],
    *,
    mass: float = 0.0,
    length: float = 2.0 * math.pi,
) -> NumericVector:
    """Solve (Λ_Q + mass) u = f on the boundary.

    With ``mass=0`` the constant mode is projected out.
    """

    if mass < 0.0:
        raise ValueError("mass must be non-negative")
    vector = tuple(values)
    scale = dtn_scale(len(vector), length=length)
    zero_mode = 0.0 if mass == 0.0 else 1.0 / mass
    return CycleQJet(len(vector)).apply_function(
        vector,
        lambda lam: 1.0 / (scale * lam + mass),
        zero_mode=zero_mode,
    )


def continuum_repaid_dtn_poisson_solve(
    values: Iterable[complex],
    *,
    mass: float = 0.0,
    length: float = 2.0 * math.pi,
) -> NumericVector:
    """Solve (Λ + mass) u = f with the continuum-repaid circle DtN symbol."""

    if mass < 0.0:
        raise ValueError("mass must be non-negative")
    vector = tuple(values)
    n = len(vector)
    zero_mode = 0.0 if mass == 0.0 else 1.0 / mass
    return CycleQJet(n).apply_index_function(
        vector,
        lambda index: 1.0 / (continuum_repaid_dtn_eigenvalue(index, n, length=length) + mass),
        zero_mode=zero_mode,
    )


def cycle_dtn_helmholtz_resolvent(
    values: Iterable[complex],
    wavenumber: float,
    *,
    damping: float = 1.0e-3,
    length: float = 2.0 * math.pi,
) -> NumericVector:
    """Apply the boundary Helmholtz resolvent (Λ_Q^2 - k^2 + i damping)^-1."""

    if wavenumber < 0.0:
        raise ValueError("wavenumber must be non-negative")
    if damping <= 0.0:
        raise ValueError("damping must be positive")
    vector = tuple(values)
    scale = dtn_scale(len(vector), length=length)
    z_damping = 1j * damping
    return CycleQJet(len(vector)).apply_function(
        vector,
        lambda lam: 1.0 / ((scale * lam) * (scale * lam) - wavenumber * wavenumber + z_damping),
        zero_mode=1.0 / (-wavenumber * wavenumber + z_damping),
    )


def continuum_repaid_dtn_helmholtz_resolvent(
    values: Iterable[complex],
    wavenumber: float,
    *,
    damping: float = 1.0e-3,
    length: float = 2.0 * math.pi,
) -> NumericVector:
    """Apply (Λ^2 - k^2 + i damping)^-1 with the repaid circle DtN symbol."""

    if wavenumber < 0.0:
        raise ValueError("wavenumber must be non-negative")
    if damping <= 0.0:
        raise ValueError("damping must be positive")
    vector = tuple(values)
    n = len(vector)
    z_damping = 1j * damping
    return CycleQJet(n).apply_index_function(
        vector,
        lambda index: 1.0
        / (
            continuum_repaid_dtn_eigenvalue(index, n, length=length) ** 2
            - wavenumber * wavenumber
            + z_damping
        ),
        zero_mode=1.0 / (-wavenumber * wavenumber + z_damping),
    )


def cycle_dtn_wave(
    values: Iterable[complex],
    time: float,
    *,
    length: float = 2.0 * math.pi,
) -> NumericVector:
    """Apply cos(t sqrt(Λ_Q)) to boundary displacement data."""

    vector = tuple(values)
    scale = dtn_scale(len(vector), length=length)
    return CycleQJet(len(vector)).apply_function(
        vector,
        lambda lam: math.cos(time * math.sqrt(scale * lam)),
        zero_mode=1.0,
    )


def continuum_repaid_dtn_wave(
    values: Iterable[complex],
    time: float,
    *,
    length: float = 2.0 * math.pi,
) -> NumericVector:
    """Apply cos(t sqrt(Λ)) with the continuum-repaid circle DtN symbol."""

    vector = tuple(values)
    n = len(vector)
    return CycleQJet(n).apply_index_function(
        vector,
        lambda index: math.cos(time * math.sqrt(continuum_repaid_dtn_eigenvalue(index, n, length=length))),
        zero_mode=1.0,
    )


def ellipse_eccentric_parameter(a: float, b: float) -> float:
    """Return ``mu`` for an aligned ellipse with ``b / a = tanh(mu)``."""

    a = float(a)
    b = float(b)
    if a <= 0.0 or b <= 0.0:
        raise ValueError("ellipse axes must be positive")
    if b > a:
        raise ValueError("ellipse_weighted_dtn expects a >= b")
    ratio = b / a
    if ratio <= 0.0 or ratio >= 1.0:
        raise ValueError("ellipse must be noncircular and nondegenerate for this weighted model")
    return math.atanh(ratio)


def ellipse_anomaly_speed(theta: float, a: float, b: float) -> float:
    """Return ``|(a cos theta, b sin theta)'|``."""

    return math.hypot(a * math.sin(theta), b * math.cos(theta))


def ellipse_weighted_dtn_weak_density(
    values: Iterable[complex],
    a: float,
    b: float,
) -> NumericVector:
    """Apply the exact weighted ellipse DtN in eccentric-anomaly coordinates.

    The output is the weak flux density with respect to ``dtheta``:
    ``h(theta) * partial_n u``.  The implementation is matrix-free and stores
    only Fourier channel coefficients.
    """

    vector = tuple(complex(value) for value in values)
    n = len(vector)
    if n < 3:
        raise ValueError("at least three boundary samples are required")
    mu = ellipse_eccentric_parameter(a, b)
    out = [0.0 + 0.0j for _ in vector]
    max_mode = n // 2
    for mode in range(1, max_mode + 1):
        nyquist = n % 2 == 0 and mode == max_mode
        factor = (1.0 / n) if nyquist else (2.0 / n)
        cos_values = [math.cos(2.0 * math.pi * mode * index / n) for index in range(n)]
        sin_values = [math.sin(2.0 * math.pi * mode * index / n) for index in range(n)]
        cos_coefficient = factor * sum(vector[index] * cos_values[index] for index in range(n))
        sin_coefficient = 0.0 if nyquist else factor * sum(
            vector[index] * sin_values[index] for index in range(n)
        )
        tanh_value = math.tanh(mode * mu)
        cos_lambda = mode * tanh_value
        sin_lambda = mode / tanh_value
        for index in range(n):
            out[index] += (
                cos_lambda * cos_coefficient * cos_values[index]
                + sin_lambda * sin_coefficient * sin_values[index]
            )
    return NumericVector(out)


def ellipse_weighted_dtn(
    values: Iterable[complex],
    a: float,
    b: float,
) -> NumericVector:
    """Apply the exact aligned-ellipse DtN and return physical normal flux."""

    weak_density = ellipse_weighted_dtn_weak_density(values, a, b)
    n = len(weak_density)
    return NumericVector(
        weak_density[index] / ellipse_anomaly_speed(2.0 * math.pi * index / n, a, b)
        for index in range(n)
    )


def exact_disk_amplitude(problem: str, mode: int, **parameters: float) -> complex:
    """Exact unit-disk modal amplitude for the boundary PDE model."""

    if mode < 0:
        raise ValueError("mode must be non-negative")
    mu = float(mode)
    if problem == "laplace_dtn":
        return complex(mu)
    if problem == "heat":
        return complex(math.exp(-parameters.get("time", 0.2) * mu))
    if problem == "poisson":
        mass = parameters.get("mass", 0.25)
        return complex(1.0 / (mu + mass) if mu + mass != 0.0 else 0.0)
    if problem == "helmholtz":
        k = parameters.get("wavenumber", 3.7)
        damping = parameters.get("damping", 1.0e-3)
        return 1.0 / (mu * mu - k * k + 1j * damping)
    if problem == "wave":
        return complex(math.cos(parameters.get("time", 0.8) * math.sqrt(mu)))
    raise ValueError(f"unknown boundary PDE problem: {problem}")


def q_disk_amplitude(problem: str, mode: int, n: int, **parameters: float) -> complex:
    """Final repaid Q/DtN unit-disk modal amplitude without a dense matrix."""

    mu = continuum_repaid_dtn_eigenvalue(mode % n, n)
    return _disk_amplitude_from_mu(problem, mu, parameters)


def q_cycle_disk_amplitude(problem: str, mode: int, n: int, **parameters: float) -> complex:
    """Raw finite-cycle Q/DtN modal amplitude for discretization diagnostics."""

    mu = cycle_dtn_eigenvalue(mode % n, n)
    return _disk_amplitude_from_mu(problem, mu, parameters)


def _disk_amplitude_from_mu(problem: str, mu: float, parameters: dict[str, float]) -> complex:
    if problem == "laplace_dtn":
        return complex(mu)
    if problem == "heat":
        return complex(math.exp(-parameters.get("time", 0.2) * mu))
    if problem == "poisson":
        mass = parameters.get("mass", 0.25)
        return complex(1.0 / (mu + mass) if mu + mass != 0.0 else 0.0)
    if problem == "helmholtz":
        k = parameters.get("wavenumber", 3.7)
        damping = parameters.get("damping", 1.0e-3)
        return 1.0 / (mu * mu - k * k + 1j * damping)
    if problem == "wave":
        return complex(math.cos(parameters.get("time", 0.8) * math.sqrt(mu)))
    raise ValueError(f"unknown boundary PDE problem: {problem}")


def relative_error(value: complex, reference: complex) -> float:
    return abs(value - reference) / max(abs(reference), 1.0e-14)


__all__ = [
    "BoundaryPDEModeResult",
    "BoundaryPullbackQJet",
    "BoundaryQJetMap",
    "HarmonicCorrectionMode",
    "HarmonicMomentCorrectedPlanarQJet",
    "HarmonicMomentCorrection",
    "HelmholtzCorrectionMode",
    "HelmholtzMomentCorrectedPlanarQJet",
    "HelmholtzMomentCorrection",
    "PlanarDomainQJet",
    "PullbackPDEEvaluation",
    "ScalarQJet",
    "apply_continuum_repaid_dtn",
    "apply_cycle_dtn",
    "apply_harmonic_moment_corrected_planar_dtn",
    "apply_helmholtz_moment_corrected_planar_dtn",
    "apply_planar_domain_dtn",
    "apply_pullback_dtn",
    "build_boundary_pullback_qjet",
    "build_harmonic_moment_corrected_planar_qjet",
    "build_helmholtz_moment_corrected_planar_qjet",
    "build_planar_domain_qjet",
    "circle_qjet_map",
    "continuum_repaid_dtn_eigenvalue",
    "continuum_repaid_dtn_eigenvalues",
    "continuum_repaid_dtn_heat",
    "continuum_repaid_dtn_helmholtz_resolvent",
    "continuum_repaid_dtn_poisson_solve",
    "continuum_repaid_dtn_wave",
    "cycle_dtn_eigenvalue",
    "cycle_dtn_eigenvalues",
    "cycle_dtn_heat",
    "cycle_dtn_helmholtz_resolvent",
    "cycle_dtn_poisson_solve",
    "cycle_dtn_wave",
    "dtn_scale",
    "ellipse_anomaly_speed",
    "ellipse_eccentric_parameter",
    "ellipse_qjet_map",
    "ellipse_weighted_dtn",
    "ellipse_weighted_dtn_weak_density",
    "exact_disk_amplitude",
    "harmonic_polynomial_trace",
    "harmonic_polynomial_weak_flux",
    "helmholtz_plane_wave_trace",
    "helmholtz_plane_wave_weak_flux",
    "q_cycle_disk_amplitude",
    "q_disk_amplitude",
    "qjet_cos",
    "qjet_exp",
    "qjet_sin",
    "qjet_theta",
    "radial_fourier_qjet_map",
    "relative_error",
    "solve_harmonic_moment_corrected_planar_boundary_pde",
    "solve_planar_domain_boundary_pde",
    "solve_pullback_boundary_pde",
]
