"""O(N log N) scale-phase convolution QJets for exact normal-form surfaces.

The module contains no external numerical dependency.  It reuses the project's
foundational QJet FFT and retains only:

* one generated two-dimensional convolution symbol;
* diagonal scale factors;
* one meridional three-jet per chart line; and
* O(N) work buffers during application.

No dense N-by-N boundary matrix or pair table is stored.

Three exact normal forms are provided:

* a cylinder, translation invariant in ``(z, theta)``;
* a cone, diagonally conjugate to convolution in ``(rho=log r, theta)``;
* a stereographic sphere strip, diagonally conjugate to convolution in
  ``(eta, theta)``.

The Koenigs/tetration cone factory uses a linearizing coordinate ``h`` with
``xi(h)=xi_0 omega^h``.  In that coordinate, scale and phase are affine in
height, so the kernel is a sheared two-dimensional convolution.  This is the
precise setting in which tetration contributes to complexity: it supplies a
translation coordinate.  A general tetration map is not automatically a
convolution outside its single-valued linearization domain.
"""

from inverse_shape.axisymmetric3d import (
    build_axisymmetric_surface_qjet,
    scale_phase_distance_squared,
)
from inverse_shape.quadrature import (
    PI,
    TAU,
    BorrowComputeRepayLedger,
    _abs,
    _clean_scalar,
    _cos,
    _exp,
    _fft,
    _ifft,
    _log,
    _sin,
    _sqrt,
)


def _next_power_of_two(value):
    result = 1
    while result < value:
        result <<= 1
    return result


def _complex_rows(values, n_rows, n_columns, name="values"):
    rows = tuple(tuple(complex(value) for value in row) for row in values)
    if len(rows) != n_rows:
        raise ValueError(f"{name} must contain one row per scale sample")
    if any(len(row) != n_columns for row in rows):
        raise ValueError(f"each {name} row must contain n_theta samples")
    return rows


def _fft2(rows):
    row_count = len(rows)
    if row_count == 0:
        return tuple()
    column_count = len(rows[0])
    row_transforms = [list(_fft(row)) for row in rows]
    output = [[0.0 + 0.0j for _ in range(column_count)] for _ in range(row_count)]
    for column in range(column_count):
        transformed = _fft([row_transforms[row][column] for row in range(row_count)])
        for row in range(row_count):
            output[row][column] = transformed[row]
    return tuple(tuple(row) for row in output)


def _ifft2(rows):
    row_count = len(rows)
    if row_count == 0:
        return tuple()
    column_count = len(rows[0])
    column_inverted = [[0.0 + 0.0j for _ in range(column_count)] for _ in range(row_count)]
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


def _cosh(value):
    x = float(value)
    if _abs(x) < 0.25:
        xx = x * x
        term = 1.0
        total = 1.0
        for order in range(1, 12):
            term *= xx / ((2 * order - 1) * (2 * order))
            total += term
        return total
    return 0.5 * (_exp(x) + _exp(-x))


class MeridionalThreeJet:
    """Sparse value/derivative generator for one scale line."""

    def __init__(
        self,
        coordinate,
        radius,
        z_value,
        phase,
        radius_derivatives,
        z_derivatives,
        phase_derivatives=(0.0, 0.0, 0.0),
    ):
        self.coordinate = float(coordinate)
        self.radius = float(radius)
        self.z_value = float(z_value)
        self.phase = float(phase)
        self.radius_derivatives = tuple(float(value) for value in radius_derivatives)
        self.z_derivatives = tuple(float(value) for value in z_derivatives)
        self.phase_derivatives = tuple(float(value) for value in phase_derivatives)
        if len(self.radius_derivatives) != 3 or len(self.z_derivatives) != 3:
            raise ValueError("a meridional three-jet requires derivatives 1, 2, and 3")
        if len(self.phase_derivatives) != 3:
            raise ValueError("phase_derivatives must contain derivatives 1, 2, and 3")

    def as_tuple(self):
        return (
            self.coordinate,
            self.radius,
            self.z_value,
            self.phase,
            *self.radius_derivatives,
            *self.z_derivatives,
            *self.phase_derivatives,
        )


class ScalePhaseConvolutionEvaluation:
    def __init__(self, values, ledger, stats):
        self.values = values
        self.ledger = ledger
        self.stats = dict(stats)
        self.method = "scale_phase_2d_fft_qjet"


class ScalePhaseConvolutionQJet:
    """Toeplitz-circulant graph action generated by one 2D FFT symbol."""

    def __init__(
        self,
        n_scale,
        n_theta,
        target_factors,
        source_factors,
        node_area_weights,
        radii,
        z_values,
        phase_offsets,
        three_jets,
        kernel_generator,
        chart_name,
        normalization=1.0 / (2.0 * PI),
    ):
        self.n_scale = int(n_scale)
        self.n_theta = int(n_theta)
        if self.n_scale < 2 or self.n_theta < 4:
            raise ValueError("the scale-phase grid requires at least 2 by 4 samples")
        self.target_factors = tuple(float(value) for value in target_factors)
        self.source_factors = tuple(float(value) for value in source_factors)
        self.node_area_weights = tuple(float(value) for value in node_area_weights)
        self.radii = tuple(float(value) for value in radii)
        self.z_values = tuple(float(value) for value in z_values)
        self.phase_offsets = tuple(float(value) for value in phase_offsets)
        self.three_jets = tuple(three_jets)
        sequences = (
            self.target_factors,
            self.source_factors,
            self.node_area_weights,
            self.radii,
            self.z_values,
            self.phase_offsets,
            self.three_jets,
        )
        if any(len(sequence) != self.n_scale for sequence in sequences):
            raise ValueError("all meridional generators must have n_scale entries")
        if any(radius <= 0.0 for radius in self.radii):
            raise ValueError("all scale-phase radii must be positive")
        self.chart_name = str(chart_name)
        self.normalization = float(normalization)
        self.theta_step = TAU / self.n_theta
        self.pad_scale = _next_power_of_two(2 * self.n_scale - 1)
        kernel = [
            [0.0 for _ in range(self.n_theta)] for _ in range(self.pad_scale)
        ]
        for delta_index in range(-(self.n_scale - 1), self.n_scale):
            row_index = delta_index % self.pad_scale
            for phase_index in range(self.n_theta):
                value = float(kernel_generator(delta_index, phase_index))
                if delta_index == 0 and phase_index == 0:
                    value = 0.0
                if value < 0.0:
                    raise ValueError("the generated principal kernel must be nonnegative")
                kernel[row_index][phase_index] = value
        self.kernel_symbol = _fft2(tuple(tuple(row) for row in kernel))
        source_grid = tuple(
            tuple(self.source_factors[index] for _ in range(self.n_theta))
            for index in range(self.n_scale)
        )
        self.row_sum = self._convolve(source_grid)
        coordinate_step = _abs(
            self.three_jets[1].coordinate - self.three_jets[0].coordinate
        )
        meridional_weights = tuple(
            _sqrt(
                jet.radius_derivatives[0] ** 2 + jet.z_derivatives[0] ** 2
            )
            * coordinate_step
            for jet in self.three_jets
        )
        self.geometry_qjet = build_axisymmetric_surface_qjet(
            self.radii,
            self.z_values,
            meridional_weights,
            self.n_theta,
            kernel_power=3.0,
        )

    @property
    def n_nodes(self):
        return self.n_scale * self.n_theta

    @property
    def dense_entries_avoided(self):
        return self.n_nodes * self.n_nodes

    @property
    def generated_symbol_entries(self):
        return self.pad_scale * self.n_theta

    def theta(self, index, scale_index=0):
        return self.theta_step * (int(index) % self.n_theta) + self.phase_offsets[scale_index]

    def _convolve(self, values):
        rows = _complex_rows(values, self.n_scale, self.n_theta)
        padded = [list(row) for row in rows]
        padded.extend(
            [0.0 + 0.0j for _ in range(self.n_theta)]
            for _ in range(self.pad_scale - self.n_scale)
        )
        transformed = _fft2(tuple(tuple(row) for row in padded))
        product = tuple(
            tuple(
                transformed[row][column] * self.kernel_symbol[row][column]
                for column in range(self.n_theta)
            )
            for row in range(self.pad_scale)
        )
        convolved = _ifft2(product)
        return tuple(
            tuple(_clean_scalar(value) for value in convolved[row])
            for row in range(self.n_scale)
        )

    def apply(self, values):
        rows = _complex_rows(values, self.n_scale, self.n_theta)
        weighted = tuple(
            tuple(self.source_factors[row] * value for value in rows[row])
            for row in range(self.n_scale)
        )
        convolved = self._convolve(weighted)
        return tuple(
            tuple(
                _clean_scalar(
                    self.normalization
                    * self.target_factors[row]
                    * (
                        rows[row][column] * complex(self.row_sum[row][column])
                        - complex(convolved[row][column])
                    )
                )
                for column in range(self.n_theta)
            )
            for row in range(self.n_scale)
        )

    def _phase_shift(self, values, shifts):
        rows = _complex_rows(values, self.n_scale, self.n_theta)
        output = []
        for row, shift in zip(rows, shifts, strict=True):
            transformed = _fft(row)
            shifted = []
            for index, coefficient in enumerate(transformed):
                signed_mode = (
                    index if index <= self.n_theta // 2 else index - self.n_theta
                )
                angle = signed_mode * shift
                multiplier = complex(_cos(angle), _sin(angle))
                shifted.append(multiplier * coefficient)
            output.append(tuple(_clean_scalar(value) for value in _ifft(shifted)))
        return tuple(output)

    def apply_repaid(self, values):
        """Fast principal action plus O(N) sparse tangent-cell repayment."""

        rows = _complex_rows(values, self.n_scale, self.n_theta)
        raw = self.apply(rows)
        unsheared = self._phase_shift(
            rows,
            tuple(-phase for phase in self.phase_offsets),
        )
        zero = tuple(
            tuple(0.0 for _ in range(self.n_theta)) for _ in range(self.n_scale)
        )
        correction_unsheared = self.geometry_qjet.repay_tangent_cell(
            unsheared,
            zero,
        )
        correction = self._phase_shift(correction_unsheared, self.phase_offsets)
        return tuple(
            tuple(
                _clean_scalar(complex(raw[row][column]) + complex(correction[row][column]))
                for column in range(self.n_theta)
            )
            for row in range(self.n_scale)
        )

    def direct_apply(self, values):
        """Independent small-grid reference using the physical scale-phase chord."""

        rows = _complex_rows(values, self.n_scale, self.n_theta)
        output = [
            [0.0 + 0.0j for _ in range(self.n_theta)] for _ in range(self.n_scale)
        ]
        nodes = tuple(
            (scale, phase)
            for scale in range(self.n_scale)
            for phase in range(self.n_theta)
        )
        for left in range(len(nodes)):
            scale_i, phase_i = nodes[left]
            value_i = rows[scale_i][phase_i]
            for right in range(left + 1, len(nodes)):
                scale_j, phase_j = nodes[right]
                d2 = scale_phase_distance_squared(
                    self.radii[scale_i],
                    self.z_values[scale_i],
                    self.theta(phase_i, scale_i),
                    self.radii[scale_j],
                    self.z_values[scale_j],
                    self.theta(phase_j, scale_j),
                )
                kernel = d2 ** -1.5
                difference = value_i - rows[scale_j][phase_j]
                output[scale_i][phase_i] += (
                    self.normalization
                    * self.node_area_weights[scale_j]
                    * kernel
                    * difference
                )
                output[scale_j][phase_j] -= (
                    self.normalization
                    * self.node_area_weights[scale_i]
                    * kernel
                    * difference
                )
        return tuple(
            tuple(_clean_scalar(value) for value in row) for row in output
        )

    def weighted_inner(self, left, right):
        left_rows = _complex_rows(left, self.n_scale, self.n_theta, "left")
        right_rows = _complex_rows(right, self.n_scale, self.n_theta, "right")
        total = 0.0 + 0.0j
        for row in range(self.n_scale):
            weight = self.node_area_weights[row]
            for column in range(self.n_theta):
                total += weight * left_rows[row][column].conjugate() * right_rows[row][column]
        return _clean_scalar(total)

    def constant_residual(self):
        constant = tuple(
            tuple(1.0 for _ in range(self.n_theta)) for _ in range(self.n_scale)
        )
        applied = self.apply(constant)
        return max(_abs(value) for row in applied for value in row)

    def stats(self):
        return {
            "chart": self.chart_name,
            "n_scale": self.n_scale,
            "n_theta": self.n_theta,
            "n_nodes": self.n_nodes,
            "pad_scale": self.pad_scale,
            "generated_symbol_entries": self.generated_symbol_entries,
            "stored_three_jets": len(self.three_jets),
            "stored_three_jet_scalars": 13 * len(self.three_jets),
            "dense_entries_avoided": self.dense_entries_avoided,
            "stored_dense_surface_matrix": False,
            "stored_pair_kernel_table": False,
            "apply_complexity": "O(N log N)",
            "storage_complexity": "O(N)",
            "fft_kernel": "project foundational QJet radix-two FFT",
            "local_repayment": "O(N) positive sparse edge form from meridional three-jets",
        }

    def evaluate(self, values):
        result = self.apply_repaid(values)
        residual = self.constant_residual()
        ledger = BorrowComputeRepayLedger(
            borrowed=(
                f"{self.chart_name} scale-phase normal form",
                "Toeplitz scale coordinate and circulant phase coordinate",
                "sparse meridional three-jet geometry",
            ),
            computed=(
                "two-dimensional zero-padded QJet FFT convolution",
                "diagonal-convolution-diagonal graph action",
            ),
            repaid=(
                "physical source-area factor",
                "target conformal-scale factor",
                "finite-strip row sum and constant nullspace",
                "O(N) positive tangent-cell edge correction generated from sparse three-jets",
            ),
            residuals=(("constant_mode_residual", residual),),
            residual_norm=residual,
            status="borrowed_repaid",
            notes=(
                "The stored symbol has O(N) entries. The physical N-by-N matrix and "
                "the pair-kernel table are never formed."
            ),
        )
        return ScalePhaseConvolutionEvaluation(result, ledger, self.stats())


def cylinder_convolution_qjet(radius, z_start, z_stop, n_scale, n_theta):
    radius = float(radius)
    if radius <= 0.0:
        raise ValueError("cylinder radius must be positive")
    dz = (float(z_stop) - float(z_start)) / int(n_scale)
    if dz == 0.0:
        raise ValueError("cylinder length must be nonzero")
    cell = _abs(dz)
    theta_step = TAU / int(n_theta)
    z_values = tuple(float(z_start) + (index + 0.5) * dz for index in range(n_scale))
    radii = tuple(radius for _ in range(n_scale))
    phases = tuple(0.0 for _ in range(n_scale))
    areas = tuple(radius * cell * theta_step for _ in range(n_scale))
    jets = tuple(
        MeridionalThreeJet(
            z_value,
            radius,
            z_value,
            0.0,
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
        )
        for z_value in z_values
    )

    def kernel(delta_index, phase_index):
        delta_z = delta_index * dz
        phase = phase_index * theta_step
        d2 = delta_z * delta_z + 4.0 * radius * radius * _sin(0.5 * phase) ** 2
        return 0.0 if d2 == 0.0 else d2 ** -1.5

    return ScalePhaseConvolutionQJet(
        n_scale,
        n_theta,
        tuple(1.0 for _ in range(n_scale)),
        areas,
        areas,
        radii,
        z_values,
        phases,
        jets,
        kernel,
        "cylinder(z,theta)",
    )


def cone_convolution_qjet(
    slope,
    rho_start,
    rho_stop,
    n_scale,
    n_theta,
    z_offset=0.0,
):
    slope = float(slope)
    drho = (float(rho_stop) - float(rho_start)) / int(n_scale)
    if drho == 0.0:
        raise ValueError("cone log-radius interval must be nonzero")
    cell = _abs(drho)
    theta_step = TAU / int(n_theta)
    metric = _sqrt(1.0 + slope * slope)
    rhos = tuple(float(rho_start) + (index + 0.5) * drho for index in range(n_scale))
    radii = tuple(_exp(rho) for rho in rhos)
    z_values = tuple(float(z_offset) + slope * radius for radius in radii)
    phases = tuple(0.0 for _ in range(n_scale))
    areas = tuple(metric * radius * radius * cell * theta_step for radius in radii)
    target = tuple(radius ** -1.5 for radius in radii)
    source = tuple(
        metric * radius**0.5 * cell * theta_step / 8.0 for radius in radii
    )
    jets = tuple(
        MeridionalThreeJet(
            rho,
            radius,
            z_value,
            0.0,
            (radius, radius, radius),
            (slope * radius, slope * radius, slope * radius),
        )
        for rho, radius, z_value in zip(rhos, radii, z_values, strict=True)
    )

    def kernel(delta_index, phase_index):
        delta_rho = delta_index * drho
        phase = phase_index * theta_step
        normal = (
            (1.0 + slope * slope) * _sinh(0.5 * delta_rho) ** 2
            + _sin(0.5 * phase) ** 2
        )
        return 0.0 if normal == 0.0 else normal ** -1.5

    return ScalePhaseConvolutionQJet(
        n_scale,
        n_theta,
        target,
        source,
        areas,
        radii,
        z_values,
        phases,
        jets,
        kernel,
        "cone(rho=log r,theta)",
    )


def sphere_stereographic_convolution_qjet(
    radius,
    eta_start,
    eta_stop,
    n_scale,
    n_theta,
):
    radius = float(radius)
    if radius <= 0.0:
        raise ValueError("sphere radius must be positive")
    deta = (float(eta_stop) - float(eta_start)) / int(n_scale)
    if deta == 0.0:
        raise ValueError("sphere eta interval must be nonzero")
    cell = _abs(deta)
    theta_step = TAU / int(n_theta)
    etas = tuple(float(eta_start) + (index + 0.5) * deta for index in range(n_scale))
    radii = tuple(radius / _cosh(eta) for eta in etas)
    z_values = tuple(-radius * _sinh(eta) / _cosh(eta) for eta in etas)
    phases = tuple(0.0 for _ in range(n_scale))
    areas = tuple(ring_radius * ring_radius * cell * theta_step for ring_radius in radii)
    chord_factor = tuple(_sqrt(2.0) * ring_radius for ring_radius in radii)
    target = tuple(value ** -1.5 for value in chord_factor)
    source = tuple(
        area * value ** -1.5
        for area, value in zip(areas, chord_factor, strict=True)
    )
    jets = []
    for eta, ring_radius, z_value in zip(etas, radii, z_values, strict=True):
        normalized_r = ring_radius / radius
        normalized_z = z_value / radius
        radius_first = ring_radius * normalized_z
        radius_second = ring_radius * (1.0 - 2.0 * normalized_r * normalized_r)
        radius_third = radius_first * (1.0 - 6.0 * normalized_r * normalized_r)
        z_first = -radius * normalized_r * normalized_r
        z_second = -2.0 * radius * normalized_r * normalized_r * normalized_z
        z_third = 2.0 * radius * normalized_r * normalized_r * (
            normalized_r * normalized_r - 2.0 * normalized_z * normalized_z
        )
        jets.append(
            MeridionalThreeJet(
                eta,
                ring_radius,
                z_value,
                0.0,
                (radius_first, radius_second, radius_third),
                (z_first, z_second, z_third),
            )
        )

    def kernel(delta_index, phase_index):
        delta_eta = delta_index * deta
        phase = phase_index * theta_step
        normal = 2.0 * (
            _sinh(0.5 * delta_eta) ** 2 + _sin(0.5 * phase) ** 2
        )
        return 0.0 if normal == 0.0 else normal ** -1.5

    return ScalePhaseConvolutionQJet(
        n_scale,
        n_theta,
        target,
        source,
        areas,
        radii,
        z_values,
        phases,
        tuple(jets),
        kernel,
        "sphere(stereographic eta,theta)",
    )


def koenigs_tetration_cone_qjet(
    log_abs_multiplier,
    phase_increment,
    height_start,
    height_stop,
    seed_radius,
    cone_slope,
    n_scale,
    n_theta,
    z_offset=0.0,
):
    """Conic orbit in a Schroeder/Koenigs tetration linearization coordinate.

    ``xi(h)=xi_0*omega^h`` with
    ``log(omega)=log_abs_multiplier+i*phase_increment``.  The physical cone is
    embedded linearly in ``|xi|``; this assumption is what makes height a
    translation coordinate and the kernel Toeplitz.
    """

    alpha = float(log_abs_multiplier)
    beta = float(phase_increment)
    if alpha == 0.0:
        raise ValueError("the tetration cone requires nonzero radial multiplier")
    dh = (float(height_stop) - float(height_start)) / int(n_scale)
    if dh == 0.0:
        raise ValueError("height interval must be nonzero")
    cell = _abs(dh)
    theta_step = TAU / int(n_theta)
    metric = _sqrt(1.0 + float(cone_slope) ** 2)
    heights = tuple(
        float(height_start) + (index + 0.5) * dh for index in range(n_scale)
    )
    rhos = tuple(_log(float(seed_radius)) + alpha * height for height in heights)
    radii = tuple(_exp(rho) for rho in rhos)
    z_values = tuple(float(z_offset) + float(cone_slope) * radius for radius in radii)
    phases = tuple(beta * height for height in heights)
    areas = tuple(
        _abs(alpha) * metric * radius * radius * cell * theta_step for radius in radii
    )
    target = tuple(radius ** -1.5 for radius in radii)
    source = tuple(
        _abs(alpha) * metric * radius**0.5 * cell * theta_step / 8.0
        for radius in radii
    )
    jets = tuple(
        MeridionalThreeJet(
            height,
            radius,
            z_value,
            phase,
            (alpha * radius, alpha * alpha * radius, alpha**3 * radius),
            (
                float(cone_slope) * alpha * radius,
                float(cone_slope) * alpha * alpha * radius,
                float(cone_slope) * alpha**3 * radius,
            ),
            (beta, 0.0, 0.0),
        )
        for height, radius, z_value, phase in zip(
            heights,
            radii,
            z_values,
            phases,
            strict=True,
        )
    )

    def kernel(delta_index, phase_index):
        delta_height = delta_index * dh
        delta_rho = alpha * delta_height
        phase = phase_index * theta_step + beta * delta_height
        normal = (
            (1.0 + float(cone_slope) ** 2) * _sinh(0.5 * delta_rho) ** 2
            + _sin(0.5 * phase) ** 2
        )
        return 0.0 if normal == 0.0 else normal ** -1.5

    return ScalePhaseConvolutionQJet(
        n_scale,
        n_theta,
        target,
        source,
        areas,
        radii,
        z_values,
        phases,
        jets,
        kernel,
        "Koenigs-tetration cone(h,theta)",
    )


__all__ = [
    "MeridionalThreeJet",
    "ScalePhaseConvolutionEvaluation",
    "ScalePhaseConvolutionQJet",
    "cone_convolution_qjet",
    "cylinder_convolution_qjet",
    "koenigs_tetration_cone_qjet",
    "sphere_stereographic_convolution_qjet",
]
