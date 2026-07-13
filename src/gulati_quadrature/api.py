"""Stable public facade for production Q quadrature and boundary PDE solves."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

from inverse_shape.q_dtn import (
    PullbackPDEEvaluation,
    build_harmonic_moment_corrected_planar_qjet,
    build_helmholtz_moment_corrected_planar_qjet,
)
from inverse_shape.quadrature import (
    MultipoleZetaQEvaluation,
    q_spectral_error_signature,
    log_layer_multipole_zeta_q_borrow_compute_repay,
)

Point = tuple[float, float]


@dataclass(frozen=True)
class PreparedBoundary:
    """Canonical boundary samples accepted by the production Q engine."""

    points: tuple[Point, ...]
    signed_area: float
    perimeter: float
    removed_duplicate_endpoint: bool
    orientation: str

    @property
    def n(self) -> int:
        return len(self.points)


@dataclass(frozen=True)
class ProductionQConfig:
    """Production defaults for arbitrary planar domains.

    The defaults intentionally choose the load-bearing path from the paper:
    harmonic reproduction through degree two, zeta-tail repayment through
    degree eight, and a Helmholtz plane-wave correction when Helmholtz DtN is
    requested.
    """

    correction_order: float = 2.0
    harmonic_moment_degree: int = 2
    zeta_tail_degree: int = 8
    helmholtz_plane_wave_count: int = 24
    multipole_order: int = 18
    multipole_leaf_size: int = 32
    multipole_theta: float = 0.45


def prepare_boundary(
    points: Iterable[Iterable[float]],
    *,
    ensure_ccw: bool = True,
    duplicate_tolerance: float = 1.0e-12,
) -> PreparedBoundary:
    """Validate and canonicalize an ordered closed-curve sample.

    The closing endpoint should normally be omitted.  If a user supplies the
    first point again as the last point, this function removes the duplicate
    before the QJet is generated.
    """

    raw = tuple((float(point[0]), float(point[1])) for point in points)
    if len(raw) < 3:
        raise ValueError("at least three boundary samples are required")
    removed = False
    if _distance(raw[0], raw[-1]) <= duplicate_tolerance:
        raw = raw[:-1]
        removed = True
    if len(raw) < 3:
        raise ValueError("at least three non-duplicate boundary samples are required")
    _reject_duplicate_samples(raw, duplicate_tolerance)
    area = _signed_area(raw)
    if abs(area) <= duplicate_tolerance:
        raise ValueError("boundary samples have near-zero signed area")
    orientation = "ccw" if area > 0.0 else "cw"
    canonical = raw
    if ensure_ccw and area < 0.0:
        canonical = tuple(reversed(raw))
        area = -area
        orientation = "ccw"
    return PreparedBoundary(
        points=canonical,
        signed_area=area,
        perimeter=_perimeter(canonical),
        removed_duplicate_endpoint=removed,
        orientation=orientation,
    )


class ProductionQEngine:
    """Boundary-only production Q engine.

    The object stores boundary samples and generated QJets only.  It never
    stores a dense Q, DtN, Helmholtz, or quadrature matrix.
    """

    def __init__(
        self,
        points: Iterable[Iterable[float]],
        *,
        config: ProductionQConfig | None = None,
        ensure_ccw: bool = True,
    ) -> None:
        self.config = config or ProductionQConfig()
        self.boundary = prepare_boundary(points, ensure_ccw=ensure_ccw)
        self.points = self.boundary.points
        self._qjet = build_harmonic_moment_corrected_planar_qjet(
            self.points,
            correction_order=self.config.correction_order,
            moment_degree=self.config.harmonic_moment_degree,
            zeta_tail_degree=self.config.zeta_tail_degree,
        )

    @property
    def n(self) -> int:
        return len(self.points)

    def apply_dtn(self, values: Iterable[complex]) -> PullbackPDEEvaluation:
        """Apply the production Laplace DtN operator."""

        return self._qjet.apply_dtn(tuple(values))

    def solve(
        self,
        problem: str,
        values: Iterable[complex],
        **parameters: float,
    ) -> PullbackPDEEvaluation:
        """Solve a boundary-only PDE using production Q defaults.

        Supported problem names are ``laplace_dtn``, ``heat``, ``poisson``,
        ``helmholtz``, ``wave``, and ``helmholtz_dtn``.  The ``helmholtz_dtn``
        path builds the Helmholtz-specific plane-wave repayment layer.
        """

        vector = tuple(values)
        if problem == "helmholtz_dtn":
            wavenumber = float(parameters.get("wavenumber", 1.0))
            qjet = build_helmholtz_moment_corrected_planar_qjet(
                self.points,
                wavenumber,
                correction_order=self.config.correction_order,
                moment_degree=self.config.harmonic_moment_degree,
                zeta_tail_degree=self.config.zeta_tail_degree,
                plane_wave_count=int(
                    parameters.get("plane_wave_count", self.config.helmholtz_plane_wave_count)
                ),
            )
            return qjet.solve_boundary_problem(problem, vector, **parameters)
        return self._qjet.solve_boundary_problem(problem, vector, **parameters)

    def integrate_log_layer(
        self,
        density_samples: Iterable[complex],
        target: Iterable[float],
        *,
        level_count: int = 3,
    ) -> MultipoleZetaQEvaluation:
        """Evaluate the near-boundary logarithmic layer by multipole/zeta Q."""

        return integrate_log_layer(
            self.points,
            density_samples,
            target,
            config=self.config,
            level_count=level_count,
        )

    def spectral_signature(self) -> dict[str, object]:
        """Return the Q-spectrum error-channel diagnostic."""

        return dict(q_spectral_error_signature(self.points).stats)

    def stats(self) -> dict[str, object]:
        """Report production settings and storage/cost accounting."""

        q_stats = dict(self._qjet._stats())
        q_stats.update(
            {
                "public_api": "gulati_quadrature.ProductionQEngine",
                "boundary_samples": self.n,
                "perimeter": self.boundary.perimeter,
                "signed_area": self.boundary.signed_area,
                "orientation": self.boundary.orientation,
                "dense_q_matrix_stored": False,
                "storage_big_o": "O(n + n r)",
                "single_dtn_apply_big_o": "O(n^2 + n r)",
                "circle_pullback_apply_big_o": "O(n log n)",
                "multipole_log_layer_big_o": "O(n p + T(p log n + near))",
            }
        )
        return q_stats


def build_engine(
    points: Iterable[Iterable[float]],
    *,
    config: ProductionQConfig | None = None,
    ensure_ccw: bool = True,
) -> ProductionQEngine:
    """Build the production boundary-only Q engine for an arbitrary domain."""

    return ProductionQEngine(points, config=config, ensure_ccw=ensure_ccw)


def solve_pde(
    points: Iterable[Iterable[float]],
    values: Iterable[complex],
    problem: str,
    *,
    config: ProductionQConfig | None = None,
    **parameters: float,
) -> PullbackPDEEvaluation:
    """One-shot boundary PDE solve through the production Q engine."""

    return build_engine(points, config=config).solve(problem, values, **parameters)


def integrate_log_layer(
    points: Iterable[Iterable[float]],
    density_samples: Iterable[complex],
    target: Iterable[float],
    *,
    config: ProductionQConfig | None = None,
    level_count: int = 3,
) -> MultipoleZetaQEvaluation:
    """One-shot near-boundary log-layer quadrature by multipole/zeta Q."""

    cfg = config or ProductionQConfig()
    prepared = prepare_boundary(points)
    density = tuple(complex(value) for value in density_samples)
    if len(density) != prepared.n:
        raise ValueError("density_samples length must match boundary sample count")
    level_points, level_density = _nested_levels(prepared.points, density, level_count=level_count)
    return log_layer_multipole_zeta_q_borrow_compute_repay(
        level_points,
        level_density,
        target,
        order=cfg.multipole_order,
        leaf_size=cfg.multipole_leaf_size,
        theta=cfg.multipole_theta,
    )


def scale_phase_point(theta: float, log_radius: float = 0.0) -> complex:
    """Return ``V = exp(rho + i theta) = exp(i q)``.

    The scale-phase coordinate is ``q = theta - i rho`` with
    ``rho = log |V|``.  On the unit circle ``rho = 0`` and ``q`` is the usual
    angle.  Off the unit slice, magnitude is carried in the imaginary part of
    the same coordinate.
    """

    radius = math.exp(float(log_radius))
    angle = float(theta)
    return complex(radius * math.cos(angle), radius * math.sin(angle))


def scale_phase_chord_squared(
    theta_i: float,
    log_radius_i: float,
    theta_j: float,
    log_radius_j: float,
) -> float:
    """Evaluate ``|V_i - V_j|^2`` from scale-phase coordinates.

    This is the stable real form of the exact identity

    ``|V_i - V_j|^2 = 4 exp(rho_i+rho_j) |sin((q_i-q_j)/2)|^2``

    where ``q = theta - i rho`` and ``V = exp(i q)``.  Equivalently,

    ``|V_i - V_j|^2 = 2 exp(rho_i+rho_j) (cosh(rho_i-rho_j)-cos(theta_i-theta_j))``.
    """

    dtheta = float(theta_i) - float(theta_j)
    drho = float(log_radius_i) - float(log_radius_j)
    scale = math.exp(float(log_radius_i) + float(log_radius_j))
    return 2.0 * scale * (math.cosh(drho) - math.cos(dtheta))


def inverse_square_chord_from_scale_phase(
    theta_i: float,
    log_radius_i: float,
    theta_j: float,
    log_radius_j: float,
) -> float:
    """Return the inverse-square chord kernel in scale-phase coordinates."""

    chord2 = scale_phase_chord_squared(theta_i, log_radius_i, theta_j, log_radius_j)
    if chord2 <= 0.0:
        raise ValueError("coincident scale-phase points produce a singular chord kernel")
    return 1.0 / chord2


def cycle_certificate(n: int, radius: float = 1.0) -> dict[str, object]:
    """Exact regular-cycle arithmetic checksum without dense matrix storage."""

    if n < 3:
        raise ValueError("n must be at least 3")
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    scaled_factorial_square = math.factorial(n - 1) ** 2
    log_pseudo_det = 2.0 * math.lgamma(n) - (n - 1) * math.log(2.0 * radius * radius)
    return {
        "n": n,
        "radius": float(radius),
        "dense_q_matrix_stored": False,
        "eigenvalues": [m * (n - m) / (2.0 * radius * radius) for m in range(n)],
        "trace": n * (n * n - 1.0) / (12.0 * radius * radius),
        "pseudo_determinant_scaled": scaled_factorial_square,
        "pseudo_determinant_log": log_pseudo_det,
        "cofactor_scaled_by_n": scaled_factorial_square,
        "cofactor_log": log_pseudo_det - math.log(n),
        "discriminant_log": n * math.log(n) + n * (n - 1) * math.log(radius),
        "nullity": 1,
        "checksum": "det'(Q_n) * (2R^2)^(n-1) = ((n-1)!)^2",
    }


def star_boundary(
    n: int,
    *,
    radius: float = 1.0,
    cos: Sequence[float] = (0.16, -0.05, 0.035),
    sin: Sequence[float] = (0.0, 0.06),
) -> tuple[Point, ...]:
    """Generate a smooth star-shaped boundary for examples and tests."""

    if n < 3:
        raise ValueError("n must be at least 3")
    points: list[Point] = []
    for index in range(n):
        theta = 2.0 * math.pi * index / n
        radial = radius
        for mode, coefficient in enumerate(cos, start=1):
            radial += coefficient * math.cos(mode * theta)
        for mode, coefficient in enumerate(sin, start=1):
            radial += coefficient * math.sin(mode * theta)
        if radial <= 0.0:
            raise ValueError("star boundary radial function must stay positive")
        points.append((radial * math.cos(theta), radial * math.sin(theta)))
    return tuple(points)


def cosine_trace(n: int, mode: int, *, phase: float = 0.0) -> tuple[float, ...]:
    """Cosine trace samples on the boundary index circle."""

    if n < 1:
        raise ValueError("n must be positive")
    return tuple(math.cos(2.0 * math.pi * mode * index / n + phase) for index in range(n))


def _nested_levels(
    points: tuple[Point, ...],
    density: tuple[complex, ...],
    *,
    level_count: int,
) -> tuple[tuple[tuple[Point, ...], ...], tuple[tuple[complex, ...], ...]]:
    if level_count < 3:
        raise ValueError("multipole/zeta Q requires at least three levels")
    if len(points) < 2 ** (level_count - 1) * 3:
        raise ValueError("not enough samples for the requested nested levels")
    strides = tuple(2 ** power for power in range(level_count - 1, -1, -1))
    return (
        tuple(points[::stride] for stride in strides),
        tuple(density[::stride] for stride in strides),
    )


def _signed_area(points: Sequence[Point]) -> float:
    total = 0.0
    for index, point in enumerate(points):
        nxt = points[(index + 1) % len(points)]
        total += point[0] * nxt[1] - nxt[0] * point[1]
    return 0.5 * total


def _perimeter(points: Sequence[Point]) -> float:
    return sum(
        _distance(point, points[(index + 1) % len(points)])
        for index, point in enumerate(points)
    )


def _distance(left: Point, right: Point) -> float:
    return math.hypot(left[0] - right[0], left[1] - right[1])


def _reject_duplicate_samples(points: Sequence[Point], tolerance: float) -> None:
    for i, point in enumerate(points):
        for j in range(i + 1, len(points)):
            if _distance(point, points[j]) <= tolerance:
                raise ValueError("duplicate boundary samples produce singular Q weights")
