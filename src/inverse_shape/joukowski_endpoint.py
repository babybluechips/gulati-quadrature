"""Static Joukowski endpoint calculus without dense pair matrices.

For the exterior Joukowski map

    z(theta) = scale * (R exp(i theta) + R^-1 exp(-i theta)),

the image is an ellipse with semi-axes

    a = scale * (R + R^-1),
    b = scale * (R - R^-1).

Its inverse-square chord kernel factors exactly into a cycle singularity and
a smooth sum-coordinate factor.  The latter has a geometric Fourier series,
so the graph action is compiled into a short sequence of
diagonal--cycle-FFT--diagonal channels.  At the golden slice
``R=phi^2`` the channel ratio is ``phi^-4``.

The module imports only the project's foundational scalar and FFT kernels.
It stores one cycle symbol and O(n) vectors; no n-by-n distance or operator
matrix is formed.
"""

from inverse_shape.quadrature import (
    PI,
    TAU,
    BorrowComputeRepayLedger,
    _abs,
    _clean_scalar,
    _cos,
    _exp,
    _fft,
    _fft_precise,
    _finite,
    _ifft,
    _ifft_precise,
    _is_power_of_two,
    _log,
    _sin,
    _sqrt,
)

PHI = 0.5 * (1.0 + _sqrt(5.0))
GOLDEN_MU = 2.0 * _log(PHI)


def _complex_vector(values, length, name="values"):
    vector = tuple(complex(value) for value in values)
    if len(vector) != length:
        raise ValueError(f"{name} must contain {length} entries")
    return vector


def _relative_l2(reference, candidate):
    numerator = sum(
        _abs(complex(left) - complex(right)) ** 2
        for left, right in zip(reference, candidate, strict=True)
    )
    denominator = sum(_abs(complex(value)) ** 2 for value in reference)
    return _sqrt(numerator / max(denominator, 1.0e-300))


def _next_power_of_two(value):
    result = 1
    while result < value:
        result <<= 1
    return result


def _complex_rows(values, n_rows, n_columns, name="values"):
    rows = tuple(tuple(complex(value) for value in row) for row in values)
    if len(rows) != n_rows:
        raise ValueError(f"{name} must contain {n_rows} rows")
    if any(len(row) != n_columns for row in rows):
        raise ValueError(f"each {name} row must contain {n_columns} entries")
    return rows


def _fft2(rows):
    row_count = len(rows)
    if row_count == 0:
        return tuple()
    column_count = len(rows[0])
    row_transforms = [list(_fft(row)) for row in rows]
    output = [
        [0.0 + 0.0j for _ in range(column_count)]
        for _ in range(row_count)
    ]
    for column in range(column_count):
        transformed = _fft(
            [row_transforms[row][column] for row in range(row_count)]
        )
        for row in range(row_count):
            output[row][column] = transformed[row]
    return tuple(tuple(row) for row in output)


def _ifft2(rows):
    row_count = len(rows)
    if row_count == 0:
        return tuple()
    column_count = len(rows[0])
    column_inverted = [
        [0.0 + 0.0j for _ in range(column_count)]
        for _ in range(row_count)
    ]
    for column in range(column_count):
        transformed = _ifft([rows[row][column] for row in range(row_count)])
        for row in range(row_count):
            column_inverted[row][column] = transformed[row]
    return tuple(tuple(_ifft(row)) for row in column_inverted)


def _sinh(value):
    x = float(value)
    if _abs(x) < 0.25:
        xx = x * x
        term = x
        total = x
        for order in range(1, 12):
            term *= xx / ((2 * order) * (2 * order + 1))
            total += term
        return total
    return 0.5 * (_exp(x) - _exp(-x))


class JoukowskiMapQJet:
    """Value/three-jet generator for a fixed exterior Joukowski slice."""

    def __init__(self, scale, mu):
        self.scale = float(scale)
        self.mu = float(mu)
        if self.scale <= 0.0 or not _finite(self.scale):
            raise ValueError("scale must be positive and finite")
        if self.mu <= 0.0 or not _finite(self.mu):
            raise ValueError("mu must be positive and finite")
        self.radius = _exp(self.mu)
        inverse = 1.0 / self.radius
        self.axis_a = self.scale * (self.radius + inverse)
        self.axis_b = self.scale * (self.radius - inverse)

    @property
    def modulation_ratio(self):
        return _exp(-2.0 * self.mu)

    @property
    def eccentricity(self):
        return 2.0 * self.scale / self.axis_a

    @property
    def is_golden(self):
        return _abs(self.mu - GOLDEN_MU) <= 4.0e-15

    def point(self, theta):
        angle = float(theta)
        return (
            self.axis_a * _cos(angle),
            self.axis_b * _sin(angle),
        )

    def theta_jet(self, theta):
        angle = float(theta)
        return (
            self.point(angle),
            (
                -self.axis_a * _sin(angle),
                self.axis_b * _cos(angle),
            ),
            (
                -self.axis_a * _cos(angle),
                -self.axis_b * _sin(angle),
            ),
            (
                self.axis_a * _sin(angle),
                -self.axis_b * _cos(angle),
            ),
        )

    def speed(self, theta):
        derivative = self.theta_jet(theta)[1]
        return _sqrt(derivative[0] ** 2 + derivative[1] ** 2)

    def chord_squared(self, theta_i, theta_j):
        point_i = self.point(theta_i)
        point_j = self.point(theta_j)
        return (
            (point_i[0] - point_j[0]) ** 2
            + (point_i[1] - point_j[1]) ** 2
        )

    def factored_chord_squared(self, theta_i, theta_j):
        delta = float(theta_i) - float(theta_j)
        sigma = float(theta_i) + float(theta_j)
        amplitude = 0.5 * (
            self.axis_a * self.axis_a + self.axis_b * self.axis_b
        ) - 0.5 * (
            self.axis_a * self.axis_a - self.axis_b * self.axis_b
        ) * _cos(sigma)
        return 4.0 * _sin(0.5 * delta) ** 2 * amplitude

    def certificate(self):
        quotient_floor = 1.0 - self.modulation_ratio
        return {
            "scale": self.scale,
            "mu": self.mu,
            "radius": self.radius,
            "axis_a": self.axis_a,
            "axis_b": self.axis_b,
            "eccentricity": self.eccentricity,
            "modulation_ratio": self.modulation_ratio,
            "joukowski_quotient_floor": quotient_floor,
            "distance_to_critical_circle": self.radius - 1.0,
            "is_golden": self.is_golden,
        }


class StaticJoukowskiEvaluation:
    def __init__(self, values, ledger, stats):
        self.values = values
        self.ledger = ledger
        self.stats = dict(stats)
        self.method = "static_joukowski_modulated_cycle"


class StaticJoukowskiEllipseQJet:
    """Inverse-square ellipse graph compiled into modulated cycle FFTs."""

    def __init__(
        self,
        map_qjet,
        n_theta,
        tolerance=2.0e-16,
        maximum_channel=256,
    ):
        if not isinstance(map_qjet, JoukowskiMapQJet):
            raise TypeError("map_qjet must be a JoukowskiMapQJet")
        self.map_qjet = map_qjet
        self.n_theta = int(n_theta)
        self.tolerance = float(tolerance)
        self.maximum_channel = int(maximum_channel)
        if self.n_theta < 8 or not _is_power_of_two(self.n_theta):
            raise ValueError("n_theta must be a radix-two size of at least eight")
        if self.tolerance <= 0.0 or not _finite(self.tolerance):
            raise ValueError("tolerance must be positive and finite")
        if self.maximum_channel < 0:
            raise ValueError("maximum_channel must be nonnegative")
        self.theta_step = TAU / self.n_theta
        self.thetas = tuple(
            self.theta_step * index for index in range(self.n_theta)
        )
        self.points = tuple(self.map_qjet.point(theta) for theta in self.thetas)
        self.weights = tuple(
            self.theta_step * self.map_qjet.speed(theta)
            for theta in self.thetas
        )
        cycle_kernel = [0.0] * self.n_theta
        for offset in range(1, self.n_theta):
            sine = _sin(PI * offset / self.n_theta)
            cycle_kernel[offset] = 1.0 / (4.0 * sine * sine)
        self.cycle_symbol = tuple(_fft_precise(cycle_kernel))
        self.channel_radius = self._select_channel_radius()
        ratio = self.map_qjet.modulation_ratio
        normalizer = self.map_qjet.axis_a * self.map_qjet.axis_b
        self.channels = tuple(
            (
                mode,
                ratio ** _abs(mode) / normalizer,
            )
            for mode in range(-self.channel_radius, self.channel_radius + 1)
        )
        self.channel_phases = tuple(
            (
                mode,
                tuple(
                    complex(_cos(mode * theta), _sin(mode * theta))
                    for theta in self.thetas
                ),
            )
            for mode, _coefficient in self.channels
        )
        self.cycle_eigenvalues = tuple(
            mode * (self.n_theta - mode) / 2.0
            for mode in range(self.n_theta)
        )
        self.default_weight_graphs = self._compile_weight_graphs(self.weights)
        self._custom_weight_key = None
        self._custom_weight_graphs = None
        self.last_apply_stats = {}

    @property
    def n_nodes(self):
        return self.n_theta

    @property
    def dense_entries_avoided(self):
        return self.n_theta * self.n_theta

    def _select_channel_radius(self):
        ratio = self.map_qjet.modulation_ratio
        if ratio == 0.0:
            return 0
        radius = 0
        while radius < self.maximum_channel:
            tail = 2.0 * ratio ** (radius + 1) / (1.0 - ratio)
            if tail <= self.tolerance:
                return radius
            radius += 1
        raise ValueError(
            "the Joukowski quotient is too close to its cusp; "
            "use the Mellin endpoint chart"
        )

    def quotient_tail_bound(self):
        ratio = self.map_qjet.modulation_ratio
        return (
            2.0
            * ratio ** (self.channel_radius + 1)
            / (
                self.map_qjet.axis_a
                * self.map_qjet.axis_b
                * (1.0 - ratio)
            )
        )

    def _cycle_convolve(self, values):
        transformed = _fft_precise(values)
        product = tuple(
            transformed[index] * self.cycle_symbol[index]
            for index in range(self.n_theta)
        )
        return tuple(_ifft_precise(product))

    def _cycle_graph(self, values):
        transformed = _fft_precise(values)
        product = tuple(
            transformed[index] * self.cycle_eigenvalues[index]
            for index in range(self.n_theta)
        )
        return tuple(_ifft_precise(product))

    def _compile_weight_graphs(self, weights):
        phases_by_mode = dict(self.channel_phases)
        return tuple(
            self._cycle_graph(
                tuple(
                    phases_by_mode[mode][index] * weights[index]
                    for index in range(self.n_theta)
                )
            )
            for mode, _coefficient in self.channels
        )

    def _potential(self, sources):
        source = _complex_vector(sources, self.n_theta, "sources")
        output = [0.0 + 0.0j for _ in range(self.n_theta)]
        phases_by_mode = dict(self.channel_phases)
        for mode, coefficient in self.channels:
            phases = phases_by_mode[mode]
            modulated = tuple(
                phases[index] * source[index]
                for index in range(self.n_theta)
            )
            convolved = self._cycle_convolve(modulated)
            for index in range(self.n_theta):
                output[index] += coefficient * phases[index] * convolved[index]
        return tuple(_clean_scalar(value) for value in output)

    def apply_fields(self, fields):
        return self.apply_fields_with_weights(fields, self.weights)

    def apply_fields_with_weights(self, fields, weights):
        rows = tuple(
            _complex_vector(field, self.n_theta, "field") for field in fields
        )
        if not rows:
            raise ValueError("at least one field is required")
        parsed_weights = _complex_vector(weights, self.n_theta, "weights")
        if any(_abs(value.imag) > 1.0e-15 for value in parsed_weights):
            raise ValueError("source weights must be real")
        source_weights = tuple(float(value.real) for value in parsed_weights)
        if any(weight <= 0.0 or not _finite(weight) for weight in source_weights):
            raise ValueError("source weights must be positive and finite")
        if source_weights == self.weights:
            weight_graphs = self.default_weight_graphs
        elif source_weights == self._custom_weight_key:
            weight_graphs = self._custom_weight_graphs
        else:
            weight_graphs = self._compile_weight_graphs(source_weights)
            self._custom_weight_key = source_weights
            self._custom_weight_graphs = weight_graphs
        output = [
            [0.0 + 0.0j for _ in range(self.n_theta)]
            for _ in rows
        ]
        compensation = [
            [0.0 + 0.0j for _ in range(self.n_theta)]
            for _ in rows
        ]
        phases_by_mode = dict(self.channel_phases)
        for channel_index, (mode, coefficient) in enumerate(self.channels):
            phases = phases_by_mode[mode]
            weighted_phase = tuple(
                phases[index] * source_weights[index]
                for index in range(self.n_theta)
            )
            weight_graph = weight_graphs[channel_index]
            for field_index, row in enumerate(rows):
                graph_product = self._cycle_graph(
                    tuple(
                        weighted_phase[index] * row[index]
                        for index in range(self.n_theta)
                    )
                )
                for index in range(self.n_theta):
                    contribution = (
                        coefficient
                        * phases[index]
                        * (
                            graph_product[index]
                            - row[index] * weight_graph[index]
                        )
                    )
                    corrected = contribution - compensation[field_index][index]
                    updated = output[field_index][index] + corrected
                    compensation[field_index][index] = (
                        updated - output[field_index][index]
                    ) - corrected
                    output[field_index][index] = updated
        self.last_apply_stats = {
            "method": "static_cycle_graph_commutator",
            "cycle_fft_channels": len(self.channels),
            "constant_channel_cancelled_symbolically": True,
            "stored_cycle_symbol_entries": self.n_theta,
            "stored_dense_matrix": False,
        }
        return tuple(
            tuple(_clean_scalar(value) for value in row) for row in output
        )

    def apply_with_weights(self, values, weights):
        return self.apply_fields_with_weights((values,), weights)[0]

    def apply(self, values):
        return self.apply_fields((values,))[0]

    def direct_apply(self, values):
        row = _complex_vector(values, self.n_theta)
        output = [0.0 + 0.0j for _ in range(self.n_theta)]
        pair_count = 0
        for left in range(self.n_theta):
            for right in range(left + 1, self.n_theta):
                distance_squared = self.map_qjet.chord_squared(
                    self.thetas[left],
                    self.thetas[right],
                )
                if distance_squared <= 1.0e-30:
                    raise ValueError("distinct Joukowski nodes collide")
                kernel = 1.0 / distance_squared
                difference = row[left] - row[right]
                output[left] += self.weights[right] * kernel * difference
                output[right] -= self.weights[left] * kernel * difference
                pair_count += 1
        self.last_apply_stats = {
            "method": "direct_pair_stream_reference",
            "direct_pairs": pair_count,
            "stored_dense_matrix": False,
        }
        return tuple(_clean_scalar(value) for value in output)

    def direct_apply_factored(self, values):
        """Compensated O(n^2) audit using the exact factored chord."""

        row = _complex_vector(values, self.n_theta)
        output = [0.0 + 0.0j for _ in range(self.n_theta)]
        compensation = [0.0 + 0.0j for _ in range(self.n_theta)]

        def accumulate(index, contribution):
            corrected = contribution - compensation[index]
            updated = output[index] + corrected
            compensation[index] = (updated - output[index]) - corrected
            output[index] = updated

        for left in range(self.n_theta):
            for right in range(left + 1, self.n_theta):
                distance_squared = self.map_qjet.factored_chord_squared(
                    self.thetas[left],
                    self.thetas[right],
                )
                kernel = 1.0 / distance_squared
                difference = row[left] - row[right]
                accumulate(
                    left,
                    self.weights[right] * kernel * difference,
                )
                accumulate(
                    right,
                    -self.weights[left] * kernel * difference,
                )
        self.last_apply_stats = {
            "method": "compensated_factored_pair_stream_reference",
            "direct_pairs": self.n_theta * (self.n_theta - 1) // 2,
            "stored_dense_matrix": False,
        }
        return tuple(_clean_scalar(value) for value in output)

    def weighted_inner(self, left, right):
        left_values = _complex_vector(left, self.n_theta, "left")
        right_values = _complex_vector(right, self.n_theta, "right")
        return _clean_scalar(
            sum(
                self.weights[index]
                * left_values[index].conjugate()
                * right_values[index]
                for index in range(self.n_theta)
            )
        )

    def constant_residual(self):
        result = self.apply((1.0,) * self.n_theta)
        return max(_abs(complex(value)) for value in result)

    def factorization_residual(self):
        residual = 0.0
        for left in range(self.n_theta):
            for right in range(left):
                direct = self.map_qjet.chord_squared(
                    self.thetas[left],
                    self.thetas[right],
                )
                factored = self.map_qjet.factored_chord_squared(
                    self.thetas[left],
                    self.thetas[right],
                )
                residual = max(
                    residual,
                    _abs(direct - factored) / max(_abs(direct), 1.0e-300),
                )
        return residual

    def direct_relative_error(self, values):
        direct = self.direct_apply_factored(values)
        static = self.apply(values)
        return _relative_l2(direct, static)

    def stats(self):
        result = {
            **self.map_qjet.certificate(),
            "n_theta": self.n_theta,
            "channel_radius": self.channel_radius,
            "cycle_fft_channels": len(self.channels),
            "quotient_tail_bound": self.quotient_tail_bound(),
            "stored_map_three_jets": 1,
            "generated_boundary_entries": self.n_theta,
            "stored_cycle_symbol_entries": self.n_theta,
            "stored_weight_graph_entries": len(self.channels) * self.n_theta,
            "dense_entries_avoided": self.dense_entries_avoided,
            "stored_dense_distance_matrix": False,
            "stored_dense_operator_matrix": False,
            "apply_complexity": "O(L n log n), L fixed by Joukowski tail tolerance",
            "storage_complexity": "O(L n)",
            "fft_kernel": "project foundational QJet radix-two FFT",
        }
        result.update(self.last_apply_stats)
        return result

    def evaluate(self, values):
        result = self.apply(values)
        residual = self.constant_residual()
        ledger = BorrowComputeRepayLedger(
            borrowed=(
                "exterior Joukowski circle with mu>0",
                "inverse-square cycle singularity",
                "geometric sum-coordinate Fourier channel",
            ),
            computed=(
                "static diagonal--cycle-FFT--diagonal channels",
                "physical arclength-weighted graph action",
            ),
            repaid=(
                "finite channel tail bounded before application",
                "diagonal omission and exact graph row sum",
                "constant nullspace",
                "no dense pair matrix",
            ),
            residuals=(
                ("constant_residual", residual),
                ("joukowski_tail_bound", self.quotient_tail_bound()),
            ),
            residual_norm=max(residual, self.quotient_tail_bound()),
            status="borrowed_repaid",
            notes=(
                "When mu approaches zero the map approaches its critical "
                "circle and this chart must hand off to a Mellin cusp channel."
            ),
        )
        return StaticJoukowskiEvaluation(result, ledger, self.stats())


class StaticJoukowskiAnnulusQJet:
    """Two-dimensional Joukowski chart compiled into static FFT channels.

    The chart is the exterior elliptic annulus

        V = exp(rho + i theta),
        X = scale * (V + V^-1),

    sampled at uniform cell centers in ``rho`` and uniformly in ``theta``.
    The base ``|V_i-V_j|^-2`` action is a diagonally conjugated
    Toeplitz--circulant convolution.  The smooth Joukowski quotient is a
    finite, tolerance-certified sum of separable diagonal modulations.
    """

    def __init__(
        self,
        scale,
        rho_start,
        rho_stop,
        n_scale,
        n_theta,
        tolerance=2.0e-15,
        maximum_channel=96,
        kernel_power=2.0,
        normalization=None,
    ):
        self.scale = float(scale)
        self.rho_start = float(rho_start)
        self.rho_stop = float(rho_stop)
        self.n_scale = int(n_scale)
        self.n_theta = int(n_theta)
        self.tolerance = float(tolerance)
        self.maximum_channel = int(maximum_channel)
        self.kernel_power = float(kernel_power)
        self.kernel_half_power = 0.5 * self.kernel_power
        self.normalization = (
            1.0 / TAU
            if normalization is None and self.kernel_power == 3.0
            else 1.0
            if normalization is None
            else float(normalization)
        )
        if self.scale <= 0.0 or not _finite(self.scale):
            raise ValueError("scale must be positive and finite")
        if self.rho_start <= 0.0 or self.rho_stop <= self.rho_start:
            raise ValueError("the exterior Joukowski chart requires 0 < rho_start < rho_stop")
        if self.n_scale < 2:
            raise ValueError("n_scale must be at least two")
        if self.n_theta < 8 or not _is_power_of_two(self.n_theta):
            raise ValueError("n_theta must be a radix-two size of at least eight")
        if self.tolerance <= 0.0 or not _finite(self.tolerance):
            raise ValueError("tolerance must be positive and finite")
        if self.maximum_channel < 0:
            raise ValueError("maximum_channel must be nonnegative")
        if self.kernel_power not in (2.0, 3.0):
            raise ValueError("kernel_power must be 2 or 3")
        if not _finite(self.normalization):
            raise ValueError("normalization must be finite")
        self.rho_step = (
            self.rho_stop - self.rho_start
        ) / self.n_scale
        self.theta_step = TAU / self.n_theta
        self.rhos = tuple(
            self.rho_start + (index + 0.5) * self.rho_step
            for index in range(self.n_scale)
        )
        self.radii = tuple(_exp(rho) for rho in self.rhos)
        self.thetas = tuple(
            self.theta_step * index for index in range(self.n_theta)
        )
        self.pad_scale = _next_power_of_two(2 * self.n_scale - 1)
        self.points = tuple(
            tuple(self._point(rho, theta) for theta in self.thetas)
            for rho in self.rhos
        )
        self.weights = tuple(
            tuple(self._area_weight(rho, theta) for theta in self.thetas)
            for rho in self.rhos
        )
        kernel = [
            [0.0 for _ in range(self.n_theta)]
            for _ in range(self.pad_scale)
        ]
        for delta_index in range(-(self.n_scale - 1), self.n_scale):
            row_index = delta_index % self.pad_scale
            delta_rho = delta_index * self.rho_step
            radial = _sinh(0.5 * delta_rho) ** 2
            for phase_index in range(self.n_theta):
                angular = _sin(PI * phase_index / self.n_theta) ** 2
                denominator = 4.0 * (radial + angular)
                kernel[row_index][phase_index] = (
                    0.0
                    if denominator == 0.0
                    else denominator ** (-self.kernel_half_power)
                )
        self.base_symbol = _fft2(tuple(tuple(row) for row in kernel))
        (
            self.channel_radius,
            self.channel_coefficients,
            self._quotient_tail_bound,
        ) = self._compile_channel_ladder()
        self.channels = tuple(
            (
                left_power,
                right_power,
                self.channel_coefficients[left_power]
                * self.channel_coefficients[right_power],
            )
            for left_power in range(self.channel_radius + 1)
            for right_power in range(self.channel_radius + 1)
        )
        self.row_sum = self._physical_potential(self.weights)
        self.last_apply_stats = {}

    @property
    def n_nodes(self):
        return self.n_scale * self.n_theta

    @property
    def dense_entries_avoided(self):
        return self.n_nodes * self.n_nodes

    @property
    def maximum_quotient_ratio(self):
        return _exp(-2.0 * self.rhos[0])

    def _compile_channel_ladder(self):
        ratio = self.maximum_quotient_ratio
        coefficient = 1.0
        coefficients = []
        full = (1.0 - ratio) ** (-self.kernel_half_power)
        for radius in range(self.maximum_channel + 1):
            coefficients.append(coefficient)
            next_coefficient = coefficient * (
                self.kernel_half_power + radius
            ) / (radius + 1)
            next_term = next_coefficient * ratio ** (radius + 1)
            next_ratio = ratio * (
                self.kernel_half_power + radius + 1
            ) / (radius + 2)
            one_dimensional_tail = next_term / max(
                1.0 - next_ratio,
                1.0e-15,
            )
            tail = (
                2.0 * full * one_dimensional_tail
                + one_dimensional_tail * one_dimensional_tail
            )
            if tail <= self.tolerance:
                return radius, tuple(coefficients), tail
            coefficient = next_coefficient
        raise ValueError(
            "the annulus approaches the Joukowski critical circle; "
            "compile a Mellin cusp chart"
        )

    def quotient_tail_bound(self):
        return self._quotient_tail_bound

    def _point(self, rho, theta):
        radius = _exp(rho)
        inverse = 1.0 / radius
        return (
            self.scale * (radius + inverse) * _cos(theta),
            self.scale * (radius - inverse) * _sin(theta),
            0.0,
        )

    def _area_weight(self, rho, theta):
        radius = _exp(rho)
        inverse = 1.0 / radius
        jacobian = self.scale * self.scale * (
            radius * radius
            + inverse * inverse
            - 2.0 * _cos(2.0 * theta)
        )
        return jacobian * self.rho_step * self.theta_step

    def _base_potential(self, sources):
        rows = _complex_rows(
            sources,
            self.n_scale,
            self.n_theta,
            "sources",
        )
        padded = [
            [
                rows[row][column]
                / self.radii[row] ** self.kernel_half_power
                for column in range(self.n_theta)
            ]
            for row in range(self.n_scale)
        ]
        padded.extend(
            [0.0 + 0.0j for _ in range(self.n_theta)]
            for _ in range(self.pad_scale - self.n_scale)
        )
        transformed = _fft2(tuple(tuple(row) for row in padded))
        product = tuple(
            tuple(
                transformed[row][column] * self.base_symbol[row][column]
                for column in range(self.n_theta)
            )
            for row in range(self.pad_scale)
        )
        convolved = _ifft2(product)
        return tuple(
            tuple(
                convolved[row][column] / self.radii[row]
                ** self.kernel_half_power
                for column in range(self.n_theta)
            )
            for row in range(self.n_scale)
        )

    def _physical_potential(self, sources):
        rows = _complex_rows(
            sources,
            self.n_scale,
            self.n_theta,
            "sources",
        )
        output = [
            [0.0 + 0.0j for _ in range(self.n_theta)]
            for _ in range(self.n_scale)
        ]
        for radial_power in range(2 * self.channel_radius + 1):
            padded = [
                [
                    _exp(
                        -(radial_power + self.kernel_half_power)
                        * self.rhos[row]
                    )
                    * rows[row][column]
                    for column in range(self.n_theta)
                ]
                for row in range(self.n_scale)
            ]
            padded.extend(
                [0.0 + 0.0j for _ in range(self.n_theta)]
                for _ in range(self.pad_scale - self.n_scale)
            )
            transformed = _fft2(tuple(tuple(row) for row in padded))
            first_left = max(0, radial_power - self.channel_radius)
            last_left = min(self.channel_radius, radial_power)
            product = [
                [0.0 + 0.0j for _ in range(self.n_theta)]
                for _ in range(self.pad_scale)
            ]
            for radial_frequency in range(self.pad_scale):
                for angular_frequency in range(self.n_theta):
                    total = 0.0 + 0.0j
                    for left_power in range(first_left, last_left + 1):
                        right_power = radial_power - left_power
                        angular_mode = right_power - left_power
                        coefficient = (
                            self.channel_coefficients[left_power]
                            * self.channel_coefficients[right_power]
                        )
                        kernel_frequency = (
                            angular_frequency - angular_mode
                        ) % self.n_theta
                        source_frequency = (
                            angular_frequency - 2 * angular_mode
                        ) % self.n_theta
                        total += (
                            coefficient
                            * self.base_symbol[radial_frequency][kernel_frequency]
                            * transformed[radial_frequency][source_frequency]
                        )
                    product[radial_frequency][angular_frequency] = total
            potential = _ifft2(tuple(tuple(row) for row in product))
            for row in range(self.n_scale):
                target_modulation = _exp(
                    -(radial_power + self.kernel_half_power)
                    * self.rhos[row]
                )
                for column in range(self.n_theta):
                    output[row][column] += (
                        target_modulation * potential[row][column]
                    )
        return tuple(
            tuple(
                _clean_scalar(value / self.scale**self.kernel_power)
                for value in row
            )
            for row in output
        )

    def apply(self, values):
        rows = _complex_rows(
            values,
            self.n_scale,
            self.n_theta,
        )
        weighted = tuple(
            tuple(
                self.weights[row][column] * rows[row][column]
                for column in range(self.n_theta)
            )
            for row in range(self.n_scale)
        )
        potential = self._physical_potential(weighted)
        self.last_apply_stats = {
            "method": "static_joukowski_2d_endpoint_convolution",
            "modulation_channels": len(self.channels),
            "grouped_transform_channels": 2 * self.channel_radius + 1,
            "stored_dense_matrix": False,
        }
        return tuple(
            tuple(
                _clean_scalar(
                    self.normalization
                    * (
                        rows[row][column]
                        * complex(self.row_sum[row][column])
                        - complex(potential[row][column])
                    )
                )
                for column in range(self.n_theta)
            )
            for row in range(self.n_scale)
        )

    def direct_apply(self, values):
        rows = _complex_rows(
            values,
            self.n_scale,
            self.n_theta,
        )
        output = [
            [0.0 + 0.0j for _ in range(self.n_theta)]
            for _ in range(self.n_scale)
        ]
        nodes = tuple(
            (row, column)
            for row in range(self.n_scale)
            for column in range(self.n_theta)
        )
        pair_count = 0
        for left in range(len(nodes)):
            row_i, column_i = nodes[left]
            point_i = self.points[row_i][column_i]
            value_i = rows[row_i][column_i]
            for right in range(left + 1, len(nodes)):
                row_j, column_j = nodes[right]
                point_j = self.points[row_j][column_j]
                distance_squared = sum(
                    (point_i[axis] - point_j[axis]) ** 2
                    for axis in range(3)
                )
                if distance_squared <= 1.0e-30:
                    raise ValueError("distinct Joukowski annulus nodes collide")
                kernel = distance_squared ** (-self.kernel_half_power)
                difference = value_i - rows[row_j][column_j]
                output[row_i][column_i] += (
                    self.normalization
                    * self.weights[row_j][column_j]
                    * kernel
                    * difference
                )
                output[row_j][column_j] -= (
                    self.normalization
                    * self.weights[row_i][column_i]
                    * kernel
                    * difference
                )
                pair_count += 1
        self.last_apply_stats = {
            "method": "direct_pair_stream_reference",
            "direct_pairs": pair_count,
            "stored_dense_matrix": False,
        }
        return tuple(
            tuple(_clean_scalar(value) for value in row) for row in output
        )

    def constant_residual(self):
        result = self.apply(
            tuple(
                tuple(1.0 for _ in range(self.n_theta))
                for _ in range(self.n_scale)
            )
        )
        return max(_abs(complex(value)) for row in result for value in row)

    def direct_relative_error(self, values):
        direct = self.direct_apply(values)
        static = self.apply(values)
        return _relative_l2(
            tuple(value for row in direct for value in row),
            tuple(value for row in static for value in row),
        )

    def stats(self):
        result = {
            "scale": self.scale,
            "rho_start": self.rho_start,
            "rho_stop": self.rho_stop,
            "n_scale": self.n_scale,
            "n_theta": self.n_theta,
            "n_nodes": self.n_nodes,
            "pad_scale": self.pad_scale,
            "maximum_quotient_ratio": self.maximum_quotient_ratio,
            "kernel_power": self.kernel_power,
            "normalization": self.normalization,
            "channel_radius": self.channel_radius,
            "modulation_channels": len(self.channels),
            "grouped_transform_channels": 2 * self.channel_radius + 1,
            "quotient_tail_bound": self.quotient_tail_bound(),
            "stored_base_symbol_entries": self.pad_scale * self.n_theta,
            "dense_entries_avoided": self.dense_entries_avoided,
            "stored_dense_distance_matrix": False,
            "stored_dense_operator_matrix": False,
            "apply_complexity": (
                "O(L N log N + L^2 N), L fixed by kernel power and critical distance"
            ),
            "storage_complexity": "O(N)",
            "fft_kernel": "project foundational QJet radix-two FFT",
        }
        result.update(self.last_apply_stats)
        return result

    def evaluate(self, values):
        result = self.apply(values)
        residual = self.constant_residual()
        ledger = BorrowComputeRepayLedger(
            borrowed=(
                "uniform exterior Joukowski scale-phase chart",
                "Toeplitz radial difference and circulant angular difference",
                "binomial sum-coordinate quotient channels",
            ),
            computed=(
                "grouped static two-dimensional QJet FFT channels",
                f"matrix-free inverse-distance power {self.kernel_power:g}",
            ),
            repaid=(
                "finite-strip zero padding",
                "finite Joukowski quotient tail certificate",
                "physical conformal area weights",
                "graph row sum and constant nullspace",
            ),
            residuals=(
                ("constant_residual", residual),
                ("joukowski_tail_bound", self.quotient_tail_bound()),
            ),
            residual_norm=max(residual, self.quotient_tail_bound()),
            status="borrowed_repaid",
            notes=(
                "The p=3 channel is the normalized discrete principal action; "
                "continuum DtN still requires tangent-cell and lower-order repayment."
            ),
        )
        return StaticJoukowskiEvaluation(result, ledger, self.stats())


_BERNOULLI_EVEN_OVER_FACTORIAL = (
    1.0 / 12.0,
    -1.0 / 720.0,
    1.0 / 30240.0,
    -1.0 / 1209600.0,
    1.0 / 47900160.0,
    -691.0 / 1307674368000.0,
    1.0 / 74724249600.0,
    -3617.0 / 10670622842880000.0,
)


def hurwitz_zeta_euler_maclaurin(
    s,
    phase,
    cutoff=32,
    corrections=7,
):
    """Evaluate Hurwitz zeta and return a first-omitted-term estimate.

    This branch-free evaluator is intended for the real or moderately complex
    Mellin exponents generated by endpoint pencils.  It does not use a special
    function library.  The returned estimate is an asymptotic next-term
    diagnostic, not a global interval-arithmetic proof.
    """

    exponent = complex(s)
    beta = float(phase)
    terms = int(cutoff)
    orders = int(corrections)
    if not _finite(exponent.real) or not _finite(exponent.imag):
        raise ValueError("s must be finite")
    if _abs(exponent - 1.0) <= 1.0e-14:
        raise ValueError("Hurwitz zeta has a pole at s=1")
    if beta <= 0.0 or not _finite(beta):
        raise ValueError("phase must be positive and finite")
    if terms < 4:
        raise ValueError("cutoff must be at least four")
    if orders < 0 or orders >= len(_BERNOULLI_EVEN_OVER_FACTORIAL):
        raise ValueError("corrections exceed the built-in Bernoulli ladder")
    total = sum((beta + index) ** (-exponent) for index in range(terms))
    endpoint = beta + terms
    total += endpoint ** (1.0 - exponent) / (exponent - 1.0)
    total += 0.5 * endpoint ** (-exponent)

    def correction(order):
        rising = 1.0 + 0.0j
        for offset in range(2 * order - 1):
            rising *= exponent + offset
        return (
            _BERNOULLI_EVEN_OVER_FACTORIAL[order - 1]
            * rising
            * endpoint ** (-exponent - 2 * order + 1)
        )

    for order in range(1, orders + 1):
        total += correction(order)
    estimate = _abs(correction(orders + 1))
    return _clean_scalar(total), estimate


class MellinEndpointChannel:
    """One statically evaluated Kondratiev/Mellin endpoint channel."""

    def __init__(
        self,
        exponent,
        amplitude,
        phase=0.5,
        label="endpoint",
    ):
        self.exponent = float(exponent)
        self.amplitude = complex(amplitude)
        self.phase = float(phase)
        self.label = str(label)
        if self.exponent <= 0.0 or not _finite(self.exponent):
            raise ValueError("a Mellin endpoint exponent must be positive")
        if not _finite(self.amplitude.real) or not _finite(self.amplitude.imag):
            raise ValueError("endpoint amplitude must be finite")
        if self.phase <= 0.0 or not _finite(self.phase):
            raise ValueError("endpoint phase must be positive and finite")

    @property
    def zeta_argument(self):
        return 1.0 - self.exponent

    def evaluate(self, step, cutoff=32, corrections=7):
        h = float(step)
        if h <= 0.0 or not _finite(h):
            raise ValueError("endpoint step must be positive and finite")
        zeta_value, zeta_error = hurwitz_zeta_euler_maclaurin(
            self.zeta_argument,
            self.phase,
            cutoff,
            corrections,
        )
        scale = h**self.exponent
        value = self.amplitude * scale * complex(zeta_value)
        error = _abs(self.amplitude) * scale * zeta_error
        return _clean_scalar(value), error

    def certificate(self, step, cutoff=32, corrections=7):
        value, error = self.evaluate(step, cutoff, corrections)
        return {
            "label": self.label,
            "exponent": self.exponent,
            "phase": self.phase,
            "zeta_argument": self.zeta_argument,
            "value": value,
            "next_term_estimate": error,
            "grid_refinement_iterations": 0,
        }


class StaticMellinEndpointRepayment:
    """Finite endpoint ledger evaluated without refining the physical grid."""

    def __init__(self, channels):
        self.channels = tuple(channels)
        if not self.channels:
            raise ValueError("at least one Mellin endpoint channel is required")
        if any(not isinstance(channel, MellinEndpointChannel) for channel in self.channels):
            raise TypeError("channels must be MellinEndpointChannel instances")

    def evaluate(self, step, cutoff=32, corrections=7):
        values = []
        errors = []
        certificates = []
        for channel in self.channels:
            value, error = channel.evaluate(step, cutoff, corrections)
            values.append(complex(value))
            errors.append(error)
            certificates.append(
                channel.certificate(step, cutoff, corrections)
            )
        total = _clean_scalar(sum(values))
        return {
            "value": total,
            "next_term_estimate": sum(errors),
            "channels": tuple(certificates),
            "channel_count": len(self.channels),
            "grid_refinement_iterations": 0,
            "stored_dense_matrix": False,
            "complexity": "O(number of endpoint channels)",
        }

    def repay(self, borrowed_value, step, cutoff=32, corrections=7):
        result = self.evaluate(step, cutoff, corrections)
        return _clean_scalar(complex(borrowed_value) + complex(result["value"])), result


def golden_joukowski_map_qjet(scale=1.0):
    return JoukowskiMapQJet(scale, GOLDEN_MU)


def golden_joukowski_ellipse_qjet(
    n_theta,
    scale=1.0,
    tolerance=2.0e-16,
):
    return StaticJoukowskiEllipseQJet(
        golden_joukowski_map_qjet(scale),
        n_theta,
        tolerance,
    )


__all__ = [
    "GOLDEN_MU",
    "PHI",
    "JoukowskiMapQJet",
    "MellinEndpointChannel",
    "StaticJoukowskiAnnulusQJet",
    "StaticJoukowskiEllipseQJet",
    "StaticJoukowskiEvaluation",
    "StaticMellinEndpointRepayment",
    "golden_joukowski_ellipse_qjet",
    "golden_joukowski_map_qjet",
    "hurwitz_zeta_euler_maclaurin",
]
