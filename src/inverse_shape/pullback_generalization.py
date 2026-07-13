"""Matrix-free tests of Riemann and cosecant pullback generalizations.

The one-dimensional cosecant kernel is the exact periodic normal form of the
inverse-square singularity.  On a closed curve of length ``L`` its physical
principal weights are

    (pi / L)^2 csc^2(pi (s_i - s_j) / L).

The corresponding graph action is a scaled regular-cycle QJet and therefore
costs ``O(n log n)`` with ``O(n)`` storage.  Geometry away from the diagonal is
not generally translation invariant.  This module keeps that distinction
explicit and provides:

* the exact cosecant principal QJet plus sparse local repayment;
* the best lag-averaged circulant proxy to test a global Riemann/circulant
  claim under favorable conditions;
* streamed diagnostics that never materialize a pair matrix; and
* the cosecant principal preconditioner obtained after azimuthal reduction of
  an axisymmetric three-dimensional inverse-cube surface kernel.

The numerical core uses the project's foundational FFT.  It imports no NumPy,
SciPy, ``math``, or ``cmath`` and never stores a dense boundary matrix.
"""

from inverse_shape.quadrature import (
    PI,
    TAU,
    BorrowComputeRepayLedger,
    CycleQJet,
    _abs,
    _clean_scalar,
    _fft,
    _finite,
    _ifft,
    _sin,
    _sqrt,
)


def _as_complex_points(points):
    converted = []
    for point in points:
        value = (
            point
            if isinstance(point, complex)
            else complex(float(point[0]), float(point[1]))
        )
        if not _finite(value.real) or not _finite(value.imag):
            raise ValueError("curve points must be finite")
        converted.append(value)
    if len(converted) < 4:
        raise ValueError("a periodic pullback requires at least four points")
    for index, point in enumerate(converted):
        if _abs(point - converted[(index + 1) % len(converted)]) <= 0.0:
            raise ValueError("adjacent curve points must be distinct")
    return tuple(converted)


def _as_complex_vector(values, length, name="values"):
    vector = tuple(complex(value) for value in values)
    if len(vector) != length:
        raise ValueError(f"{name} length must match the pullback size")
    return vector


def _distance_squared(left, right):
    difference = left - right
    return difference.real * difference.real + difference.imag * difference.imag


def _polygonal_perimeter(points):
    return sum(
        _abs(points[(index + 1) % len(points)] - point)
        for index, point in enumerate(points)
    )


def _relative_norm(reference, candidate):
    numerator = sum(
        _abs(left - right) ** 2
        for left, right in zip(reference, candidate, strict=True)
    )
    denominator = sum(_abs(value) ** 2 for value in reference)
    return _sqrt(numerator / max(denominator, 1.0e-300))


class PeriodicCurveThreeJet:
    """One arclength-coordinate three-jet stored as four complex scalars."""

    def __init__(self, coordinate, point, first, second, third):
        self.coordinate = float(coordinate)
        self.point = complex(point)
        self.first = complex(first)
        self.second = complex(second)
        self.third = complex(third)

    @property
    def unit_tangent(self):
        magnitude = _abs(self.first)
        if magnitude <= 1.0e-15:
            return 0.0 + 0.0j
        return self.first / magnitude


class PeriodicCurveSamples:
    """Equal-arclength samples and their sparse three-jet generator."""

    def __init__(self, points, period_length=None):
        self.points = _as_complex_points(points)
        if period_length is None:
            period_length = _polygonal_perimeter(self.points)
        self.period_length = float(period_length)
        if self.period_length <= 0.0 or not _finite(self.period_length):
            raise ValueError("period_length must be positive and finite")
        self.step = self.period_length / len(self.points)
        self.three_jets = self._build_three_jets()

    @property
    def n(self):
        return len(self.points)

    @property
    def segment_anisotropy(self):
        lengths = tuple(
            _abs(self.points[(index + 1) % self.n] - self.points[index])
            for index in range(self.n)
        )
        return max(lengths) / min(lengths)

    def _build_three_jets(self):
        h = self.step
        jets = []
        for index, point in enumerate(self.points):
            previous = self.points[(index - 1) % self.n]
            following = self.points[(index + 1) % self.n]
            previous_two = self.points[(index - 2) % self.n]
            following_two = self.points[(index + 2) % self.n]
            first = (following - previous) / (2.0 * h)
            second = (following - 2.0 * point + previous) / (h * h)
            third = (
                following_two
                - 2.0 * following
                + 2.0 * previous
                - previous_two
            ) / (2.0 * h * h * h)
            jets.append(PeriodicCurveThreeJet(index * h, point, first, second, third))
        return tuple(jets)


def equal_arclength_samples(parametric_curve, n, oversample_factor=32, phase=0.5):
    """Sample a periodic parametric curve without an external numerical library."""

    count = int(n)
    if count < 4:
        raise ValueError("n must be at least four")
    oversample = max(512, int(oversample_factor) * count)
    dense = tuple(
        complex(parametric_curve(TAU * index / oversample))
        for index in range(oversample)
    )
    lengths = tuple(
        _abs(dense[(index + 1) % oversample] - dense[index])
        for index in range(oversample)
    )
    if any(length <= 0.0 for length in lengths):
        raise ValueError("the parametric curve contains a collapsed sampled edge")
    total = sum(lengths)
    targets = tuple(((index + float(phase)) % count) * total / count for index in range(count))
    points = []
    edge = 0
    accumulated = 0.0
    for target in targets:
        while edge + 1 < oversample and accumulated + lengths[edge] < target:
            accumulated += lengths[edge]
            edge += 1
        fraction = (target - accumulated) / lengths[edge]
        left = dense[edge]
        right = dense[(edge + 1) % oversample]
        points.append(left + fraction * (right - left))
    return PeriodicCurveSamples(points, total)


class CosecantPullbackQJet:
    """Exact periodic principal channel plus optional sparse local repayment."""

    def __init__(self, samples):
        if not isinstance(samples, PeriodicCurveSamples):
            samples = PeriodicCurveSamples(samples)
        self.samples = samples
        self.n = samples.n
        self.period_length = samples.period_length
        self.cycle = CycleQJet(self.n)
        self.scale = (TAU / self.period_length) ** 2

    def weight(self, lag):
        index = int(lag) % self.n
        if index == 0:
            return 0.0
        sine = _sin(PI * index / self.n)
        return self.scale / (4.0 * sine * sine)

    def apply(self, values):
        vector = _as_complex_vector(values, self.n)
        raw = self.cycle.apply(vector)
        return tuple(_clean_scalar(self.scale * complex(value)) for value in raw)

    def apply_repaid(self, values, bandwidth=2):
        """Replace near-lag cosecant edges by exact physical chord edges."""

        vector = _as_complex_vector(values, self.n)
        width = int(bandwidth)
        if width < 0 or width >= self.n // 2:
            raise ValueError("bandwidth must satisfy 0 <= bandwidth < n/2")
        output = [complex(value) for value in self.apply(vector)]
        for lag in range(1, width + 1):
            principal = self.weight(lag)
            for left in range(self.n):
                right = (left + lag) % self.n
                physical = 1.0 / _distance_squared(
                    self.samples.points[left],
                    self.samples.points[right],
                )
                residual = physical - principal
                difference = vector[left] - vector[right]
                output[left] += residual * difference
                output[right] -= residual * difference
        return tuple(_clean_scalar(value) for value in output)

    def evaluate(self, values, bandwidth=2):
        result = self.apply_repaid(values, bandwidth=bandwidth)
        constant = self.apply_repaid((1.0,) * self.n, bandwidth=bandwidth)
        residual = max(_abs(complex(value)) for value in constant)
        ledger = BorrowComputeRepayLedger(
            borrowed=(
                "periodic csc^2 principal symbol generated by the cycle QJet",
                "equal-arclength boundary three-jets",
            ),
            computed=("two-FFT cosecant principal action",),
            repaid=(
                f"{bandwidth} exact physical chord edge bands",
                "graph row-sum nullspace",
            ),
            residuals=(("constant_residual", residual),),
            residual_norm=residual,
            status="borrowed_repaid",
            notes="The uncomputed smooth far remainder is not claimed to vanish.",
        )
        return result, ledger

    def stats(self, bandwidth=2):
        width = int(bandwidth)
        return {
            "n": self.n,
            "period_length": self.period_length,
            "stored_three_jets": self.n,
            "stored_three_jet_scalars": 5 * self.n,
            "sparse_repayment_edges": width * self.n,
            "stored_dense_matrix": False,
            "stored_pair_table": False,
            "apply_complexity": "O(n log n + bandwidth*n)",
            "storage_complexity": "O(n + bandwidth*n)",
            "role": "exact universal principal channel, not the complete geometry remainder",
        }


class LagAveragedCirculantQJet:
    """Best lag-average proxy for a claimed globally circulant chord kernel.

    The ``O(n^2)`` setup is intentionally generous: every physical pair is
    streamed once into its lag average.  If this proxy is inaccurate, no
    geometry-independent lag-only kernel can recover the same sampled chord
    operator.  Only the resulting ``O(n)`` row and FFT symbol are retained.
    """

    def __init__(self, samples):
        if not isinstance(samples, PeriodicCurveSamples):
            samples = PeriodicCurveSamples(samples)
        self.samples = samples
        self.n = samples.n
        row = [0.0 for _ in range(self.n)]
        for lag in range(1, self.n):
            total = 0.0
            for left in range(self.n):
                right = (left + lag) % self.n
                total += 1.0 / _distance_squared(
                    samples.points[left],
                    samples.points[right],
                )
            row[lag] = total / self.n
        self.adjacency_row = tuple(row)
        self.row_sum = sum(row)
        self.adjacency_symbol = tuple(_fft(row))

    def weight(self, lag):
        return self.adjacency_row[int(lag) % self.n]

    def apply(self, values):
        vector = _as_complex_vector(values, self.n)
        transformed = _fft(vector)
        adjacency = _ifft(
            tuple(
                coefficient * symbol
                for coefficient, symbol in zip(transformed, self.adjacency_symbol, strict=True)
            )
        )
        return tuple(
            _clean_scalar(self.row_sum * vector[index] - complex(adjacency[index]))
            for index in range(self.n)
        )

    def stats(self):
        return {
            "n": self.n,
            "setup_complexity": "O(n^2) streamed",
            "apply_complexity": "O(n log n)",
            "storage_complexity": "O(n)",
            "stored_dense_matrix": False,
            "stored_pair_table": False,
            "role": "optimistic best lag-only projection, not an exact prime-form operator",
        }


def apply_physical_chord_qjet(samples, values):
    """Stream the exact physical inverse-square graph action without a matrix."""

    if not isinstance(samples, PeriodicCurveSamples):
        samples = PeriodicCurveSamples(samples)
    vector = _as_complex_vector(values, samples.n)
    output = [0.0 + 0.0j for _ in range(samples.n)]
    for left in range(samples.n):
        for right in range(left + 1, samples.n):
            weight = 1.0 / _distance_squared(samples.points[left], samples.points[right])
            difference = weight * (vector[left] - vector[right])
            output[left] += difference
            output[right] -= difference
    return tuple(_clean_scalar(value) for value in output)


def _normalized_prime_form_weight(left_jet, right_jet):
    """Return the circle-normalized mixed log-chord Hessian weight."""

    tangent_left = left_jet.unit_tangent
    tangent_right = right_jet.unit_tangent
    if tangent_left == 0.0 or tangent_right == 0.0:
        return 0.0
    difference = left_jet.point - right_jet.point
    return float((tangent_left * tangent_right / (difference * difference)).real)


def streamed_pullback_diagnostics(samples, cosecant=None, lag_average=None, local_band=3):
    """Compare both pullbacks by pair streaming with ``O(n)`` retained state."""

    if not isinstance(samples, PeriodicCurveSamples):
        samples = PeriodicCurveSamples(samples)
    if cosecant is None:
        cosecant = CosecantPullbackQJet(samples)
    if lag_average is None:
        lag_average = LagAveragedCirculantQJet(samples)
    if cosecant.n != samples.n or lag_average.n != samples.n:
        raise ValueError("diagnostic pullbacks must use the same curve samples")

    physical_square = 0.0
    cosecant_square = 0.0
    lag_square = 0.0
    cosecant_far_square = 0.0
    lag_far_square = 0.0
    physical_far_square = 0.0
    local_cosecant_square = 0.0
    prime_far_square = 0.0
    prime_negative = 0
    far_count = 0
    width = int(local_band)
    for left in range(samples.n):
        for right in range(left + 1, samples.n):
            lag = (right - left) % samples.n
            cyclic_lag = min(lag, samples.n - lag)
            physical = 1.0 / _distance_squared(samples.points[left], samples.points[right])
            csc = cosecant.weight(lag)
            lagged = lag_average.weight(lag)
            csc_residual = physical - csc
            lag_residual = physical - lagged
            physical_square += physical * physical
            cosecant_square += csc_residual * csc_residual
            lag_square += lag_residual * lag_residual
            if cyclic_lag <= width:
                local_cosecant_square += csc_residual * csc_residual
            else:
                physical_far_square += physical * physical
                cosecant_far_square += csc_residual * csc_residual
                lag_far_square += lag_residual * lag_residual
                prime = _normalized_prime_form_weight(
                    samples.three_jets[left],
                    samples.three_jets[right],
                )
                difference = prime - physical
                prime_far_square += difference * difference
                if prime < 0.0:
                    prime_negative += 1
                far_count += 1

    physical_denominator = max(physical_square, 1.0e-300)
    far_denominator = max(physical_far_square, 1.0e-300)
    return {
        "n": samples.n,
        "segment_anisotropy": samples.segment_anisotropy,
        "cosecant_kernel_relative_residual": _sqrt(
            cosecant_square / physical_denominator
        ),
        "lag_average_kernel_relative_residual": _sqrt(
            lag_square / physical_denominator
        ),
        "cosecant_far_relative_residual": _sqrt(
            cosecant_far_square / far_denominator
        ),
        "lag_average_far_relative_residual": _sqrt(
            lag_far_square / far_denominator
        ),
        "cosecant_residual_local_fraction": _sqrt(
            local_cosecant_square / max(cosecant_square, 1.0e-300)
        ),
        "prime_form_far_relative_defect": _sqrt(
            prime_far_square / far_denominator
        ),
        "prime_form_far_negative_fraction": prime_negative / max(far_count, 1),
        "stored_dense_matrix": False,
    }


class AxisymmetricMeridionalCosecantQJet:
    """Cosecant principal preconditioner for a periodic surface meridian.

    Azimuthal reduction of ``|X-Y|^-3`` gives

        r' integral |X-X'|^-3 dtheta' ~ 2 / |s-s'|^2.

    This class applies that universal channel by one meridional FFT.  Optional
    sparse bands replace nearby principal edges by the exact reduced physical
    mode coupling.  It is a preconditioner/principal operator, not the complete
    far-geometry correction.
    """

    def __init__(self, surface_qjet, mode=0, bandwidth=0):
        if not surface_qjet.meridian_periodic:
            raise ValueError("the cosecant meridional pullback requires a periodic meridian")
        self.surface = surface_qjet
        self.n = surface_qjet.n_rings
        self.mode = int(mode)
        self.bandwidth = int(bandwidth)
        if self.bandwidth < 0 or self.bandwidth >= self.n // 2:
            raise ValueError("bandwidth must satisfy 0 <= bandwidth < n/2")
        average_step = sum(surface_qjet.meridional_weights) / self.n
        variation = max(
            _abs(step - average_step)
            for step in surface_qjet.meridional_weights
        )
        if variation > 1.0e-10 * average_step:
            raise ValueError("the current meridional FFT requires equal-arclength weights")
        self.step = average_step
        self.period_length = self.n * self.step
        self.cycle = CycleQJet(self.n)
        self.scale = (
            2.0
            * surface_qjet.normalization
            * self.step
            * (TAU / self.period_length) ** 2
        )
        samples = PeriodicCurveSamples(
            tuple(
                complex(radius, z_value)
                for radius, z_value in zip(
                    surface_qjet.radii,
                    surface_qjet.z_values,
                    strict=True,
                )
            ),
            self.period_length,
        )
        self.three_jets = samples.three_jets
        self.local_edges = self._build_local_edges()

    def _unit_cycle_weight(self, lag):
        sine = _sin(PI * lag / self.n)
        return 1.0 / (4.0 * sine * sine)

    def _build_local_edges(self):
        edges = []
        for lag in range(1, self.bandwidth + 1):
            principal = self.scale * self._unit_cycle_weight(lag)
            for left in range(self.n):
                right = (left + lag) % self.n
                left_physical = (
                    self.surface.normalization
                    * self.surface.meridional_weights[right]
                    * self.surface.reduced_meridional_kernel(left, right, self.mode)
                )
                right_physical = (
                    self.surface.normalization
                    * self.surface.meridional_weights[left]
                    * self.surface.reduced_meridional_kernel(right, left, self.mode)
                )
                edges.append(
                    (
                        left,
                        right,
                        left_physical - principal,
                        right_physical - principal,
                    )
                )
        return tuple(edges)

    def apply(self, amplitudes):
        vector = _as_complex_vector(amplitudes, self.n, "amplitudes")
        principal = self.cycle.apply(vector)
        output = [self.scale * complex(value) for value in principal]
        for left, right, left_residual, right_residual in self.local_edges:
            difference = vector[left] - vector[right]
            output[left] += left_residual * difference
            output[right] -= right_residual * difference
        return tuple(_clean_scalar(value) for value in output)

    def relative_error(self, amplitudes):
        reference = self.surface.apply_azimuthal_mode(amplitudes, self.mode)
        return _relative_norm(reference, self.apply(amplitudes))

    def stats(self):
        return {
            "n_meridian": self.n,
            "mode": self.mode,
            "bandwidth": self.bandwidth,
            "stored_three_jets": self.n,
            "stored_sparse_edges": len(self.local_edges),
            "stored_dense_matrix": False,
            "stored_pair_table": False,
            "apply_complexity": "O(n log n + bandwidth*n)",
            "storage_complexity": "O(n + bandwidth*n)",
            "role": "axisymmetric meridional principal preconditioner",
        }


__all__ = [
    "AxisymmetricMeridionalCosecantQJet",
    "CosecantPullbackQJet",
    "LagAveragedCirculantQJet",
    "PeriodicCurveSamples",
    "PeriodicCurveThreeJet",
    "apply_physical_chord_qjet",
    "equal_arclength_samples",
    "streamed_pullback_diagnostics",
]
