"""Exact transparent closure for autonomous cylindrical shell tails.

After a periodic angular FFT, a nearest-shell discretization has one scalar
recurrence per Fourier mode,

    sigma <- d - u**2 / sigma.

The stable fixed point is the exact Schur pivot of the semi-infinite tail.  If
``w + 1/w = d/u`` and ``|w| < 1``, the four useful normalizations are

    pivot       = u / w,
    self energy = u * w,
    flux DtN    = u * (1 - w),
    generator   = -log(w).

They are deliberately kept separate in this module.  The fixed-point pivot is
an exact algebraic closure of the shell tail; it is not, by itself, a claim
about continuum discretization error on an arbitrary surface.

Only the project's scalar primitives and foundational radix-two FFT are used.
The implementation stores O(n_theta) generated symbols and never forms a
dense shell or boundary matrix.
"""

from inverse_shape.quadrature import (
    HALF_PI,
    PI,
    BorrowComputeRepayLedger,
    QJetFFTPlan,
    _abs,
    _atan,
    _clean_scalar,
    _finite,
    _floor,
    _hypot,
    _is_power_of_two,
    _log,
    _sin,
    _sqrt,
)

PHI = 0.5 * (1.0 + _sqrt(5.0))
GOLDEN_MULTIPLIER = PHI**-4


class TransparentTailBranchError(ValueError):
    """Raised when no uniquely decaying tail branch has been specified."""


def _atan2(y_value, x_value):
    y = float(y_value)
    x = float(x_value)
    if x > 0.0:
        return _atan(y / x)
    if x < 0.0:
        return _atan(y / x) + (PI if y >= 0.0 else -PI)
    if y > 0.0:
        return HALF_PI
    if y < 0.0:
        return -HALF_PI
    raise ValueError("the argument of zero is undefined")


def _complex_sqrt(value):
    """Principal complex square root using the internal scalar kernel."""

    number = complex(value)
    if not _finite(number.real) or not _finite(number.imag):
        raise ValueError("complex square root requires a finite value")
    if number.imag == 0.0:
        if number.real >= 0.0:
            return complex(_sqrt(number.real), 0.0)
        return complex(0.0, _sqrt(-number.real))
    radius = _hypot(number.real, number.imag)
    real_part = _sqrt(max(0.0, 0.5 * (radius + number.real)))
    imag_part = _sqrt(max(0.0, 0.5 * (radius - number.real)))
    if number.imag < 0.0:
        imag_part = -imag_part
    return complex(real_part, imag_part)


def _complex_log(value):
    number = complex(value)
    radius = _hypot(number.real, number.imag)
    if radius <= 0.0:
        raise ValueError("complex logarithm requires a nonzero value")
    return complex(_log(radius), _atan2(number.imag, number.real))


def _real_asinh(value):
    number = float(value)
    return _log(number + _sqrt(number * number + 1.0))


def _ceil(value):
    floor = _floor(value)
    return floor if floor == value else floor + 1


def _signed_mode(index, size):
    return index if index <= size // 2 else index - size


def _relative_l2(reference, candidate):
    numerator = sum(
        _abs(complex(left) - complex(right)) ** 2
        for left, right in zip(reference, candidate, strict=True)
    )
    denominator = sum(_abs(complex(value)) ** 2 for value in reference)
    return _sqrt(numerator / max(denominator, 1.0e-300))


def pde_spectral_shift(problem, parameter=0.0, damping=0.0):
    """Return the dimensionless shell shift for a boundary PDE resolvent.

    ``heat`` and ``wave`` here mean resolvent-domain closures.  A time-domain
    method may reuse these symbols at each rational or Laplace quadrature
    parameter; the function does not claim that a single static multiplier is
    the complete time-domain transparent condition.
    """

    name = str(problem).strip().lower().replace("-", "_")
    value = complex(parameter)
    loss = float(damping)
    if loss < 0.0 or not _finite(loss):
        raise ValueError("damping must be finite and nonnegative")
    if name in ("laplace", "poisson"):
        if _abs(value) > 0.0:
            raise ValueError("Laplace/Poisson uses parameter=0")
        return 0.0 + 0.0j
    if name in ("screened_poisson", "yukawa"):
        if _abs(value.imag) > 0.0 or value.real < 0.0:
            raise ValueError("screening parameter must be real and nonnegative")
        return complex(value.real * value.real, 0.0)
    if name in ("heat", "heat_resolvent"):
        if value.real <= 0.0:
            raise ValueError("a heat resolvent parameter must have positive real part")
        return value
    if name in ("helmholtz", "helmholtz_resolvent"):
        if _abs(value.imag) > 0.0 or value.real < 0.0:
            raise ValueError("Helmholtz wavenumber must be real and nonnegative")
        if loss <= 0.0:
            raise TransparentTailBranchError(
                "Helmholtz modes on the propagating band require positive "
                "limiting-absorption damping"
            )
        return complex(-(value.real * value.real), loss)
    if name in ("wave", "wave_resolvent"):
        if _abs(value.imag) > 0.0 or value.real < 0.0:
            raise ValueError("wave frequency must be real and nonnegative")
        if loss <= 0.0:
            raise TransparentTailBranchError(
                "the causal wave resolvent requires a positive Laplace damping"
            )
        laplace_frequency = complex(loss, value.real)
        return laplace_frequency * laplace_frequency
    raise ValueError(f"unsupported transparent-tail PDE problem: {problem}")


def _stable_joukowski_root(ratio, tolerance):
    ratio = complex(ratio)
    discriminant = _complex_sqrt(ratio * ratio - 4.0)
    plus = ratio + discriminant
    minus = ratio - discriminant
    denominator = plus if _abs(plus) >= _abs(minus) else minus
    if _abs(denominator) <= 1.0e-300:
        raise TransparentTailBranchError("degenerate Joukowski root")
    large_root = 0.5 * denominator
    root = 1.0 / large_root
    modulus = _abs(root)
    if modulus < 1.0 - tolerance:
        return root, False
    if (
        _abs(ratio.imag) <= tolerance
        and _abs(_abs(ratio.real) - 2.0) <= tolerance
    ):
        return complex(1.0 if ratio.real >= 0.0 else -1.0, 0.0), True
    if modulus <= 1.0 + 32.0 * tolerance:
        raise TransparentTailBranchError(
            "no uniquely decaying root: add a positive resolvent/radiation damping"
        )
    raise TransparentTailBranchError("failed to select the stable Joukowski branch")


class TransparentTailMode:
    """Generated scalar data for one angular Fourier mode."""

    def __init__(self, index, signed_mode, diagonal, coupling, root, marginal):
        self.index = int(index)
        self.signed_mode = int(signed_mode)
        self.diagonal = complex(diagonal)
        self.coupling = float(coupling)
        self.root = complex(root)
        self.marginal = bool(marginal)
        self.pivot = self.coupling / self.root
        self.repelling_pivot = self.coupling * self.root
        self.self_energy = self.coupling * self.root
        self.flux = self.coupling * (1.0 - self.root)
        self.koenigs_multiplier = self.root * self.root
        self.generator = -_complex_log(self.root)

    @property
    def fixed_point_residual(self):
        mapped = self.diagonal - self.coupling * self.coupling / self.pivot
        return _abs(self.pivot - mapped)

    @property
    def root_residual(self):
        ratio = self.diagonal / self.coupling
        return _abs(self.root + 1.0 / self.root - ratio)

    def symbol(self, quantity):
        name = str(quantity).strip().lower().replace("-", "_")
        if name in ("pivot", "schur_pivot", "sigma"):
            return self.pivot
        if name in ("self_energy", "selfenergy"):
            return self.self_energy
        if name in ("flux", "dtn", "boundary_dtn"):
            return self.flux
        if name in ("generator", "half_laplacian", "log_pivot"):
            return self.generator
        if name in ("root", "decay"):
            return self.root
        raise ValueError(f"unknown transparent-tail symbol: {quantity}")

    def as_dict(self):
        return {
            "index": self.index,
            "signed_mode": self.signed_mode,
            "diagonal": self.diagonal,
            "coupling": self.coupling,
            "root": self.root,
            "pivot": self.pivot,
            "repelling_pivot": self.repelling_pivot,
            "self_energy": self.self_energy,
            "flux": self.flux,
            "generator": self.generator,
            "koenigs_multiplier": self.koenigs_multiplier,
            "marginal": self.marginal,
            "fixed_point_residual": self.fixed_point_residual,
            "root_residual": self.root_residual,
        }


class TransparentTailEvaluation:
    """One matrix-free transparent-tail application and its ledger."""

    def __init__(self, values, quantity, ledger, stats):
        self.values = tuple(values)
        self.quantity = str(quantity)
        self.ledger = ledger
        self.stats = dict(stats)
        self.method = "exact_fixed_point_transparent_tail"


class CylindricalTransparentDtN:
    """Exact spectral cap for a uniform autonomous cylindrical shell tail."""

    def __init__(
        self,
        n_theta,
        spectral_shift=0.0,
        coupling=1.0,
        angular_ratio=1.0,
        branch_tolerance=2.0e-12,
    ):
        self.n_theta = int(n_theta)
        self.spectral_shift = complex(spectral_shift)
        self.coupling = float(coupling)
        self.angular_ratio = float(angular_ratio)
        self.branch_tolerance = float(branch_tolerance)
        if self.n_theta < 4 or not _is_power_of_two(self.n_theta):
            raise ValueError("n_theta must be a radix-two size of at least four")
        if self.coupling <= 0.0 or not _finite(self.coupling):
            raise ValueError("coupling must be positive and finite")
        if self.angular_ratio <= 0.0 or not _finite(self.angular_ratio):
            raise ValueError("angular_ratio must be positive and finite")
        if (
            not _finite(self.spectral_shift.real)
            or not _finite(self.spectral_shift.imag)
        ):
            raise ValueError("spectral_shift must be finite")
        if self.branch_tolerance <= 0.0 or not _finite(self.branch_tolerance):
            raise ValueError("branch_tolerance must be positive and finite")
        self.fft_plan = QJetFFTPlan(self.n_theta)
        modes = []
        for index in range(self.n_theta):
            signed = _signed_mode(index, self.n_theta)
            sine = _sin(PI * signed / self.n_theta)
            diagonal = (
                2.0 * self.coupling
                + 4.0 * self.coupling * self.angular_ratio * sine * sine
                + self.spectral_shift
            )
            root, marginal = _stable_joukowski_root(
                diagonal / self.coupling,
                self.branch_tolerance,
            )
            modes.append(
                TransparentTailMode(
                    index,
                    signed,
                    diagonal,
                    self.coupling,
                    root,
                    marginal,
                )
            )
        self.modes = tuple(modes)

    @classmethod
    def for_problem(
        cls,
        n_theta,
        problem,
        parameter=0.0,
        damping=0.0,
        coupling=1.0,
        angular_ratio=1.0,
    ):
        return cls(
            n_theta,
            spectral_shift=pde_spectral_shift(
                problem,
                parameter=parameter,
                damping=damping,
            ),
            coupling=coupling,
            angular_ratio=angular_ratio,
        )

    def _vector(self, values, name="values"):
        vector = tuple(complex(value) for value in values)
        if len(vector) != self.n_theta:
            raise ValueError(f"{name} must contain {self.n_theta} entries")
        if any(not _finite(value.real) or not _finite(value.imag) for value in vector):
            raise ValueError(f"{name} must contain finite entries")
        return vector

    def symbols(self, quantity="pivot"):
        return tuple(mode.symbol(quantity) for mode in self.modes)

    def apply(self, values, quantity="pivot"):
        vector = self._vector(values)
        transformed = self.fft_plan.fft(vector)
        symbols = self.symbols(quantity)
        multiplied = tuple(
            symbol * coefficient
            for symbol, coefficient in zip(symbols, transformed, strict=True)
        )
        output = tuple(_clean_scalar(value) for value in self.fft_plan.ifft(multiplied))
        residual = max(mode.fixed_point_residual for mode in self.modes)
        ledger = BorrowComputeRepayLedger(
            borrowed=("periodic angular trace", "autonomous semi-infinite shell tail"),
            computed=("one custom QJet FFT", f"diagonal {quantity} multiplier", "one inverse FFT"),
            repaid=("exact stable Riccati fixed point", "omitted shell depth"),
            residuals=(residual,),
            residual_norm=residual,
            status="balanced" if residual <= 2.0e-12 else "failed",
            notes=(
                "The ledger certifies the autonomous tail Schur closure. "
                "It does not certify a preceding CAD surface discretization."
            ),
        )
        return TransparentTailEvaluation(output, quantity, ledger, self.stats())

    def apply_boundary_dtn(self, values):
        return self.apply(values, quantity="flux")

    def apply_generator(self, values):
        return self.apply(values, quantity="generator")

    def finite_dirichlet_pivots(self, depth):
        shell_depth = int(depth)
        if shell_depth < 1:
            raise ValueError("depth must be positive")
        pivots = [mode.diagonal for mode in self.modes]
        square = self.coupling * self.coupling
        for _shell in range(1, shell_depth):
            for index, mode in enumerate(self.modes):
                if _abs(pivots[index]) <= 1.0e-300:
                    raise ZeroDivisionError(
                        f"finite Dirichlet pivot vanished in mode {mode.signed_mode}"
                    )
                pivots[index] = mode.diagonal - square / pivots[index]
        return tuple(pivots)

    def finite_dirichlet_symbols(self, depth, quantity="flux"):
        pivots = self.finite_dirichlet_pivots(depth)
        name = str(quantity).strip().lower().replace("-", "_")
        if name in ("pivot", "schur_pivot", "sigma"):
            return pivots
        if name in ("self_energy", "selfenergy"):
            return tuple(self.coupling * self.coupling / pivot for pivot in pivots)
        if name in ("flux", "dtn", "boundary_dtn"):
            return tuple(
                self.coupling - self.coupling * self.coupling / pivot
                for pivot in pivots
            )
        if name in ("generator", "half_laplacian", "log_pivot"):
            return tuple(_complex_log(pivot / self.coupling) for pivot in pivots)
        raise ValueError(f"unknown finite-tail symbol: {quantity}")

    def apply_finite_dirichlet(self, values, depth, quantity="flux"):
        symbols = self.finite_dirichlet_symbols(depth, quantity=quantity)
        return self.apply_mode_symbols(values, symbols)

    def apply_mode_symbols(self, values, symbols):
        """Apply one already compiled diagonal symbol by FFT."""

        vector = self._vector(values)
        diagonal = tuple(complex(value) for value in symbols)
        if len(diagonal) != self.n_theta:
            raise ValueError(f"symbols must contain {self.n_theta} entries")
        if any(
            not _finite(value.real) or not _finite(value.imag)
            for value in diagonal
        ):
            raise ValueError("symbols must contain finite entries")
        transformed = self.fft_plan.fft(vector)
        output = self.fft_plan.ifft(
            tuple(
                symbol * coefficient
                for symbol, coefficient in zip(diagonal, transformed, strict=True)
            )
        )
        return tuple(_clean_scalar(value) for value in output)

    def solve_direct_dirichlet_shells(self, values, depth):
        """Solve the L-shell tridiagonal tail and return interface flux.

        Modes are processed one at a time, so this reference solve stores O(L)
        shell work rather than an ``n_theta`` by ``L`` array.
        """

        shell_depth = int(depth)
        if shell_depth < 1:
            raise ValueError("depth must be positive")
        transformed = self.fft_plan.fft(self._vector(values))
        flux_modes = []
        u_value = self.coupling
        for mode, boundary_value in zip(self.modes, transformed, strict=True):
            upper = [0.0 + 0.0j] * shell_depth
            right = [0.0 + 0.0j] * shell_depth
            denominator = mode.diagonal
            if _abs(denominator) <= 1.0e-300:
                raise ZeroDivisionError("singular direct shell factor")
            if shell_depth > 1:
                upper[0] = -u_value / denominator
            right[0] = u_value * boundary_value / denominator
            for shell in range(1, shell_depth):
                denominator = mode.diagonal + u_value * upper[shell - 1]
                if _abs(denominator) <= 1.0e-300:
                    raise ZeroDivisionError("singular direct shell factor")
                if shell + 1 < shell_depth:
                    upper[shell] = -u_value / denominator
                right[shell] = u_value * right[shell - 1] / denominator
            shell_values = [0.0 + 0.0j] * shell_depth
            shell_values[-1] = right[-1]
            for shell in range(shell_depth - 2, -1, -1):
                shell_values[shell] = right[shell] - upper[shell] * shell_values[shell + 1]
            flux_modes.append(u_value * (boundary_value - shell_values[0]))
        return tuple(
            _clean_scalar(value) for value in self.fft_plan.ifft(flux_modes)
        )

    def cross_ratio(self, mode_index, pivot):
        mode = self.modes[int(mode_index) % self.n_theta]
        if mode.marginal:
            raise ValueError(
                "the repeated fixed point has a parabolic, not cross-ratio, coordinate"
            )
        value = complex(pivot)
        return (value - mode.pivot) / (value - mode.repelling_pivot)

    def cross_ratio_certificate(self, mode_index, depth):
        shell_depth = int(depth)
        if shell_depth < 1:
            raise ValueError("depth must be positive")
        index = int(mode_index) % self.n_theta
        mode = self.modes[index]
        actual = self.finite_dirichlet_pivots(shell_depth)[index]
        if mode.marginal:
            expected = mode.pivot * (shell_depth + 1.0) / shell_depth
            return {
                "mode": mode.signed_mode,
                "depth": shell_depth,
                "marginal": True,
                "finite_pivot": actual,
                "predicted_pivot": expected,
                "linearization_residual": _abs(actual - expected),
                "pivot_error": _abs(actual - mode.pivot),
                "pivot_error_bound": self.coupling / shell_depth,
                "flux_error": self.coupling / (shell_depth + 1.0),
                "contraction": 1.0,
            }
        initial_cross_ratio = self.cross_ratio(index, mode.diagonal)
        predicted_cross_ratio = (
            mode.koenigs_multiplier ** (shell_depth - 1)
        ) * initial_cross_ratio
        actual_cross_ratio = self.cross_ratio(index, actual)
        predicted = (
            mode.pivot - predicted_cross_ratio * mode.repelling_pivot
        ) / (1.0 - predicted_cross_ratio)
        cross_ratio_size = _abs(predicted_cross_ratio)
        gap = _abs(mode.pivot - mode.repelling_pivot)
        bound = (
            gap * cross_ratio_size / max(1.0 - cross_ratio_size, 1.0e-300)
        )
        finite_flux = self.coupling - self.coupling * self.coupling / actual
        return {
            "mode": mode.signed_mode,
            "depth": shell_depth,
            "marginal": False,
            "finite_pivot": actual,
            "predicted_pivot": predicted,
            "actual_cross_ratio": actual_cross_ratio,
            "predicted_cross_ratio": predicted_cross_ratio,
            "linearization_residual": _abs(actual_cross_ratio - predicted_cross_ratio),
            "pivot_reconstruction_residual": _abs(actual - predicted),
            "pivot_error": _abs(actual - mode.pivot),
            "pivot_error_bound": bound,
            "flux_error": _abs(finite_flux - mode.flux),
            "contraction": _abs(mode.koenigs_multiplier),
        }

    def required_depth(self, mode_index, tolerance):
        target = float(tolerance)
        if target <= 0.0 or not _finite(target):
            raise ValueError("tolerance must be positive and finite")
        index = int(mode_index) % self.n_theta
        mode = self.modes[index]
        if mode.marginal:
            return max(1, _ceil(self.coupling / target - 1.0))
        initial = _abs(self.cross_ratio(index, mode.diagonal))
        gap = _abs(mode.pivot - mode.repelling_pivot)
        threshold = target / (gap + target)
        if initial <= threshold:
            return 1
        multiplier = _abs(mode.koenigs_multiplier)
        estimate = 1 + _ceil(_log(threshold / initial) / _log(multiplier))
        depth = max(1, estimate)
        while self.cross_ratio_certificate(index, depth)["pivot_error_bound"] > target:
            depth += 1
        while depth > 1 and self.cross_ratio_certificate(index, depth - 1)[
            "pivot_error_bound"
        ] <= target:
            depth -= 1
        return depth

    def perturbed_transition_certificate(
        self,
        mode_index,
        diagonal_perturbations,
        coupling_perturbations=None,
    ):
        """Close a finite nonautonomous transition by the exact tail cap.

        Perturbations are ordered from the retained interface toward the
        autonomous tail. The reported bound is a posteriori: it uses the
        computed nonzero pivots to certify every local Möbius Lipschitz factor.
        """

        index = int(mode_index) % self.n_theta
        mode = self.modes[index]
        diagonal = tuple(complex(value) for value in diagonal_perturbations)
        if coupling_perturbations is None:
            coupling = (0.0 + 0.0j,) * len(diagonal)
        else:
            coupling = tuple(complex(value) for value in coupling_perturbations)
        if len(coupling) != len(diagonal):
            raise ValueError("diagonal and coupling perturbations must have equal length")
        if any(
            not _finite(value.real) or not _finite(value.imag)
            for value in diagonal + coupling
        ):
            raise ValueError("transition perturbations must be finite")

        pivot = mode.pivot
        bound = 0.0
        local_rows = []
        for shell in range(len(diagonal) - 1, -1, -1):
            if _abs(pivot) <= 1.0e-300:
                raise ZeroDivisionError("perturbed transition pivot vanished")
            shell_coupling = self.coupling + coupling[shell]
            next_pivot = (
                mode.diagonal
                + diagonal[shell]
                - shell_coupling * shell_coupling / pivot
            )
            local_lipschitz = (
                _abs(shell_coupling) ** 2
                / (_abs(pivot) * _abs(mode.pivot))
            )
            local_defect = (
                _abs(diagonal[shell])
                + _abs(shell_coupling * shell_coupling - self.coupling**2)
                / _abs(mode.pivot)
            )
            bound = local_defect + local_lipschitz * bound
            local_rows.append(
                {
                    "shell": shell,
                    "local_lipschitz": local_lipschitz,
                    "local_defect": local_defect,
                    "propagated_bound": bound,
                    "pivot_modulus": _abs(next_pivot),
                }
            )
            pivot = next_pivot
        local_rows.reverse()
        return {
            "mode": mode.signed_mode,
            "transition_shells": len(diagonal),
            "perturbed_pivot": pivot,
            "autonomous_pivot": mode.pivot,
            "actual_pivot_error": _abs(pivot - mode.pivot),
            "certified_pivot_error_bound": bound,
            "maximum_local_lipschitz": max(
                (row["local_lipschitz"] for row in local_rows),
                default=0.0,
            ),
            "summed_local_defect": sum(
                row["local_defect"] for row in local_rows
            ),
            "terminal_cap_exact": True,
            "local_rows": tuple(local_rows),
        }

    def cylinder_identity_residual(self):
        if _abs(self.spectral_shift) > 0.0 or _abs(self.angular_ratio - 1.0) > 1.0e-15:
            raise ValueError("the closed cylinder identity requires zero shift and angular_ratio=1")
        residual = 0.0
        for mode in self.modes:
            expected = 2.0 * _real_asinh(
                _abs(_sin(PI * mode.signed_mode / self.n_theta))
            )
            residual = max(residual, _abs(mode.generator - expected))
        return residual

    def stats(self):
        return {
            "n_theta": self.n_theta,
            "spectral_shift": self.spectral_shift,
            "coupling": self.coupling,
            "angular_ratio": self.angular_ratio,
            "stable_modes": sum(not mode.marginal for mode in self.modes),
            "marginal_modes": sum(mode.marginal for mode in self.modes),
            "stored_mode_symbols": 8 * self.n_theta,
            "stored_fft_twiddles": self.fft_plan.stored_twiddles,
            "dense_shell_matrix_stored": False,
            "dense_boundary_matrix_stored": False,
            "apply_time_big_o": "O(N_theta log N_theta)",
            "auxiliary_storage_big_o": "O(N_theta)",
            "tail_depth_dependence": "none",
            "maximum_fixed_point_residual": max(
                mode.fixed_point_residual for mode in self.modes
            ),
            "maximum_root_residual": max(mode.root_residual for mode in self.modes),
        }


def fibonacci(index):
    value = int(index)
    if value < 0:
        raise ValueError("Fibonacci index must be nonnegative")
    previous = 0
    current = 1
    for _ in range(value):
        previous, current = current, previous + current
    return previous


def golden_tail_certificate(depth):
    """Exact Fibonacci ledger for ``d/u=3`` and ``u=1``."""

    shell_depth = int(depth)
    if shell_depth < 1:
        raise ValueError("depth must be positive")
    numerator = fibonacci(2 * shell_depth + 2)
    denominator = fibonacci(2 * shell_depth)
    rational_pivot = numerator / denominator
    recurrence_pivot = 3.0
    for _ in range(1, shell_depth):
        recurrence_pivot = 3.0 - 1.0 / recurrence_pivot
    fixed_point = PHI * PHI
    exact_error_law = PHI ** (-2 * shell_depth) / denominator
    return {
        "depth": shell_depth,
        "numerator": numerator,
        "denominator": denominator,
        "rational_pivot": rational_pivot,
        "recurrence_pivot": recurrence_pivot,
        "fixed_point": fixed_point,
        "stable_root": PHI**-2,
        "koenigs_multiplier": GOLDEN_MULTIPLIER,
        "pivot_rational_residual": _abs(recurrence_pivot - rational_pivot),
        "error_law_residual": _abs(
            (rational_pivot - fixed_point) - exact_error_law
        ),
        "absolute_pivot_error": _abs(rational_pivot - fixed_point),
    }


def residue_class_sectors(n_theta, bandwidth):
    """Partition signed Fourier labels by ``k mod b``.

    The partition is an invariant operator decomposition when coefficient
    modes lie in ``b Z`` and products are de-aliased. Raw cyclic wrap on an
    FFT grid need not preserve these sectors when ``b`` does not divide the
    transform size.
    """

    size = int(n_theta)
    modulus = int(bandwidth)
    if size < 1 or modulus < 1:
        raise ValueError("n_theta and bandwidth must be positive")
    return tuple(
        tuple(
            index
            for index in range(size)
            if _signed_mode(index, size) % modulus == residue
        )
        for residue in range(modulus)
    )


__all__ = [
    "CylindricalTransparentDtN",
    "GOLDEN_MULTIPLIER",
    "PHI",
    "TransparentTailBranchError",
    "TransparentTailEvaluation",
    "TransparentTailMode",
    "fibonacci",
    "golden_tail_certificate",
    "pde_spectral_shift",
    "residue_class_sectors",
]
