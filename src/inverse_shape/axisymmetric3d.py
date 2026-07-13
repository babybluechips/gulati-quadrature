"""Matrix-free scale-phase QJet calculus for surfaces of revolution.

The numerical core uses the project's foundational QJet FFT and does not use
NumPy, SciPy, ``math``, or ``cmath``.  A surface is represented by meridional
ring generators ``(r_i, z_i, ds_i)`` and an equispaced azimuthal phase grid.
No dense surface matrix and no table of pair kernels is retained.

For

    X_i(theta) = (r_i cos(theta), r_i sin(theta), z_i),
    rho_i = log(r_i),

the exact cylinder/conic normal form is

    |X_i(theta)-X_j(phi)|^2
      = (z_i-z_j)^2
        + 2 exp(rho_i+rho_j)
          (cosh(rho_i-rho_j)-cos(theta-phi)).

The implementation evaluates its cancellation-safe equivalent

    (z_i-z_j)^2 + (r_i-r_j)^2
      + 4 r_i r_j sin^2((theta-phi)/2).

On a two-dimensional boundary in R^3, the half-Laplacian/DtN principal
kernel has order ``|X-Y|^-3``.  Azimuthal reduction of that kernel has the
one-dimensional inverse-square meridional singularity.  ``kernel_power=2``
is supported as a diagnostic control, not as the 3D DtN normalization.
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
    _finite,
    _ifft,
    _log,
    _sin,
    _sqrt,
)


def _complex_rows(values, n_rings, n_theta, name="values"):
    rows = tuple(tuple(complex(value) for value in row) for row in values)
    if len(rows) != n_rings:
        raise ValueError(f"{name} must contain one row per meridional ring")
    if any(len(row) != n_theta for row in rows):
        raise ValueError(f"each {name} row must contain n_theta samples")
    return rows


def _complex_vector(values, length, name="values"):
    vector = tuple(complex(value) for value in values)
    if len(vector) != length:
        raise ValueError(f"{name} length must match the meridional ring count")
    return vector


def _stable_sinh_half(value):
    x = 0.5 * float(value)
    if _abs(x) < 0.125:
        xx = x * x
        term = x
        total = x
        for order in range(1, 10):
            term *= xx / ((2 * order) * (2 * order + 1))
            total += term
        return total
    return 0.5 * (_exp(x) - _exp(-x))


def scale_phase_distance_squared(radius_i, z_i, theta_i, radius_j, z_j, theta_j):
    """Return the stable exact cylinder/conic scale-phase chord square."""

    ri = float(radius_i)
    rj = float(radius_j)
    if ri <= 0.0 or rj <= 0.0:
        raise ValueError("scale-phase radii must be positive")
    dz = float(z_i) - float(z_j)
    dr = ri - rj
    half_phase = 0.5 * (float(theta_i) - float(theta_j))
    phase_chord = _sin(half_phase)
    return dz * dz + dr * dr + 4.0 * ri * rj * phase_chord * phase_chord


def hyperbolic_scale_phase_distance_squared(
    radius_i,
    z_i,
    theta_i,
    radius_j,
    z_j,
    theta_j,
):
    """Evaluate the same chord square through ``rho=log(r)`` explicitly."""

    ri = float(radius_i)
    rj = float(radius_j)
    if ri <= 0.0 or rj <= 0.0:
        raise ValueError("scale-phase radii must be positive")
    rho_i = _log(ri)
    rho_j = _log(rj)
    sinh_half = _stable_sinh_half(rho_i - rho_j)
    sin_half = _sin(0.5 * (float(theta_i) - float(theta_j)))
    dz = float(z_i) - float(z_j)
    return dz * dz + 4.0 * ri * rj * (
        sinh_half * sinh_half + sin_half * sin_half
    )


def cartesian_distance_squared(radius_i, z_i, theta_i, radius_j, z_j, theta_j):
    """Cartesian reference for the scale-phase identity."""

    xi = float(radius_i) * _cos(float(theta_i))
    yi = float(radius_i) * _sin(float(theta_i))
    xj = float(radius_j) * _cos(float(theta_j))
    yj = float(radius_j) * _sin(float(theta_j))
    dx = xi - xj
    dy = yi - yj
    dz = float(z_i) - float(z_j)
    return dx * dx + dy * dy + dz * dz


class AxisymmetricQEvaluation:
    """Result and audit metadata for one streamed surface application."""

    def __init__(self, values, ledger, stats):
        self.values = values
        self.ledger = ledger
        self.stats = dict(stats)
        self.method = "axisymmetric_scale_phase_qjet"


class AxisymmetricSurfaceQJet:
    """Streamed angular QJet for an axisymmetric surface.

    ``meridional_weights[i]`` is the quadrature measure ``ds`` carried by the
    profile sample.  The area represented by one azimuthal node is therefore
    ``r_i * ds_i * 2*pi/n_theta``.
    """

    def __init__(
        self,
        radii,
        z_values,
        meridional_weights,
        n_theta,
        kernel_power=3.0,
        normalization=None,
        meridian_periodic=False,
        meridian_poles=False,
    ):
        self.radii = tuple(float(value) for value in radii)
        self.z_values = tuple(float(value) for value in z_values)
        self.meridional_weights = tuple(float(value) for value in meridional_weights)
        if len(self.radii) < 2:
            raise ValueError("at least two meridional rings are required")
        if len(self.z_values) != len(self.radii):
            raise ValueError("z_values length must match radii")
        if len(self.meridional_weights) != len(self.radii):
            raise ValueError("meridional_weights length must match radii")
        if any(value <= 0.0 or not _finite(value) for value in self.radii):
            raise ValueError("all midpoint-sampled radii must be positive and finite")
        if any(value <= 0.0 or not _finite(value) for value in self.meridional_weights):
            raise ValueError("meridional weights must be positive and finite")
        if any(not _finite(value) for value in self.z_values):
            raise ValueError("z coordinates must be finite")
        self.n_rings = len(self.radii)
        self.n_theta = int(n_theta)
        if self.n_theta < 4:
            raise ValueError("n_theta must be at least four")
        self.kernel_power = float(kernel_power)
        if self.kernel_power <= 0.0:
            raise ValueError("kernel_power must be positive")
        if normalization is None:
            normalization = 1.0 / (2.0 * PI)
        self.normalization = float(normalization)
        if not _finite(self.normalization):
            raise ValueError("normalization must be finite")
        self.theta_step = TAU / self.n_theta
        self.meridian_periodic = bool(meridian_periodic)
        self.meridian_poles = bool(meridian_poles)
        if self.meridian_periodic and self.meridian_poles:
            raise ValueError("the meridian cannot be both periodic and pole-closed")
        self.rhos = tuple(_log(radius) for radius in self.radii)
        self.node_area_weights = tuple(
            radius * ds * self.theta_step
            for radius, ds in zip(self.radii, self.meridional_weights, strict=True)
        )

    @property
    def n_nodes(self):
        return self.n_rings * self.n_theta

    @property
    def surface_area(self):
        return self.n_theta * sum(self.node_area_weights)

    @property
    def dense_entries_avoided(self):
        return self.n_nodes * self.n_nodes

    def theta(self, index):
        return self.theta_step * (int(index) % self.n_theta)

    def cartesian_point(self, ring_index, theta_index):
        radius = self.radii[ring_index]
        phase = self.theta(theta_index)
        return (
            radius * _cos(phase),
            radius * _sin(phase),
            self.z_values[ring_index],
        )

    def scale_phase_point(self, ring_index, theta_index):
        return (
            self.rhos[ring_index],
            self.theta(theta_index),
            self.z_values[ring_index],
        )

    def _kernel_samples(self, ring_i, ring_j):
        same_ring = ring_i == ring_j
        values = [0.0 for _ in range(self.n_theta)]
        stop = self.n_theta // 2
        for shift in range(stop + 1):
            if same_ring and shift == 0:
                value = 0.0
            else:
                phase = self.theta_step * shift
                d2 = scale_phase_distance_squared(
                    self.radii[ring_i],
                    self.z_values[ring_i],
                    0.0,
                    self.radii[ring_j],
                    self.z_values[ring_j],
                    phase,
                )
                if d2 <= 0.0:
                    raise ValueError("distinct surface samples collide in scale-phase coordinates")
                value = d2 ** (-0.5 * self.kernel_power)
            values[shift] = value
            if shift:
                values[-shift] = value
        return values

    def _kernel_spectrum(self, ring_i, ring_j):
        transformed = _fft(self._kernel_samples(ring_i, ring_j))
        return tuple(float(complex(value).real) for value in transformed)

    def angular_kernel_coefficient(self, ring_i, ring_j, mode=0):
        """Return ``int K(theta)e^{-im theta} dtheta`` by the QJet FFT."""

        if not 0 <= ring_i < self.n_rings or not 0 <= ring_j < self.n_rings:
            raise ValueError("ring index out of range")
        spectrum = self._kernel_spectrum(ring_i, ring_j)
        return self.theta_step * spectrum[int(mode) % self.n_theta]

    def reduced_meridional_kernel(self, ring_i, ring_j, mode=0):
        """Include the source-ring radius in the angularly reduced kernel."""

        return self.radii[ring_j] * self.angular_kernel_coefficient(
            ring_i,
            ring_j,
            mode,
        )

    def apply(self, values):
        rows = _complex_rows(values, self.n_rings, self.n_theta)
        transformed = [list(_fft(row)) for row in rows]
        output = [
            [0.0 + 0.0j for _ in range(self.n_theta)] for _ in range(self.n_rings)
        ]

        for ring_i in range(self.n_rings):
            spectrum = self._kernel_spectrum(ring_i, ring_i)
            row_sum = spectrum[0]
            area_i = self.node_area_weights[ring_i]
            for mode in range(self.n_theta):
                output[ring_i][mode] += (
                    area_i
                    * (row_sum - spectrum[mode])
                    * transformed[ring_i][mode]
                )

            for ring_j in range(ring_i + 1, self.n_rings):
                spectrum = self._kernel_spectrum(ring_i, ring_j)
                row_sum = spectrum[0]
                area_j = self.node_area_weights[ring_j]
                for mode in range(self.n_theta):
                    kernel_mode = spectrum[mode]
                    value_i = transformed[ring_i][mode]
                    value_j = transformed[ring_j][mode]
                    output[ring_i][mode] += area_j * (
                        row_sum * value_i - kernel_mode * value_j
                    )
                    output[ring_j][mode] += area_i * (
                        row_sum * value_j - kernel_mode * value_i
                    )

        return tuple(
            tuple(
                _clean_scalar(self.normalization * value)
                for value in _ifft(output_row)
            )
            for output_row in output
        )

    @staticmethod
    def _asinh(value):
        x = float(value)
        return _log(x + _sqrt(x * x + 1.0))

    def _cell_integrals(self, ring):
        half_meridian = 0.5 * self.meridional_weights[ring]
        half_azimuth = 0.5 * self.radii[ring] * self.theta_step
        integral_meridian = 4.0 * half_azimuth * self._asinh(
            half_meridian / half_azimuth
        )
        integral_azimuth = 4.0 * half_meridian * self._asinh(
            half_azimuth / half_meridian
        )
        return integral_meridian, integral_azimuth

    def repay_tangent_cell(self, values, raw_values):
        """Add the local-cell correction as a positive sparse edge form."""

        if _abs(self.kernel_power - 3.0) > 1.0e-14:
            raise ValueError("tangent-cell repayment is defined for kernel_power=3")
        rows = _complex_rows(values, self.n_rings, self.n_theta)
        raw = _complex_rows(raw_values, self.n_rings, self.n_theta, "raw_values")
        output = [list(row) for row in raw]
        integrals = tuple(self._cell_integrals(ring) for ring in range(self.n_rings))

        for ring in range(self.n_rings):
            mass = self.node_area_weights[ring]
            azimuth_step = self.radii[ring] * self.theta_step
            conductance = (
                0.5
                * self.normalization
                * mass
                * integrals[ring][1]
                / (azimuth_step * azimuth_step)
            )
            coefficient = conductance / mass
            for phase in range(self.n_theta):
                following = (phase + 1) % self.n_theta
                difference = rows[ring][phase] - rows[ring][following]
                output[ring][phase] += coefficient * difference
                output[ring][following] -= coefficient * difference

        edge_count = self.n_rings if self.meridian_periodic else self.n_rings - 1
        for ring in range(edge_count):
            following_ring = (ring + 1) % self.n_rings
            mass = self.node_area_weights[ring]
            following_mass = self.node_area_weights[following_ring]
            meridian_step = 0.5 * (
                self.meridional_weights[ring]
                + self.meridional_weights[following_ring]
            )
            conductance = (
                0.25
                * self.normalization
                * (
                    mass * integrals[ring][0]
                    + following_mass * integrals[following_ring][0]
                )
                / (meridian_step * meridian_step)
            )
            for phase in range(self.n_theta):
                difference = rows[ring][phase] - rows[following_ring][phase]
                output[ring][phase] += conductance / mass * difference
                output[following_ring][phase] -= (
                    conductance / following_mass * difference
                )
        return tuple(
            tuple(_clean_scalar(value) for value in row) for row in output
        )

    def apply_repaid(self, values):
        """Apply Q and repay the analytically integrated local tangent cell."""

        rows = _complex_rows(values, self.n_rings, self.n_theta)
        return self.repay_tangent_cell(rows, self.apply(rows))

    def apply_azimuthal_modes(self, mode_amplitudes):
        """Apply several separated ``a_i exp(i*m*theta)`` fields in one pass."""

        requested = []
        for signed_mode, amplitudes in mode_amplitudes:
            mode = int(signed_mode)
            index = mode % self.n_theta
            requested.append((mode, index, _complex_vector(amplitudes, self.n_rings)))
        if not requested:
            return tuple()

        output = [
            [0.0 + 0.0j for _ in range(self.n_rings)] for _ in range(len(requested))
        ]
        for ring_i in range(self.n_rings):
            spectrum = self._kernel_spectrum(ring_i, ring_i)
            row_sum = spectrum[0]
            area_i = self.node_area_weights[ring_i]
            for request_index, (_, mode_index, amplitudes) in enumerate(requested):
                output[request_index][ring_i] += (
                    area_i
                    * (row_sum - spectrum[mode_index])
                    * amplitudes[ring_i]
                )

            for ring_j in range(ring_i + 1, self.n_rings):
                spectrum = self._kernel_spectrum(ring_i, ring_j)
                row_sum = spectrum[0]
                area_j = self.node_area_weights[ring_j]
                for request_index, (_, mode_index, amplitudes) in enumerate(requested):
                    kernel_mode = spectrum[mode_index]
                    value_i = amplitudes[ring_i]
                    value_j = amplitudes[ring_j]
                    output[request_index][ring_i] += area_j * (
                        row_sum * value_i - kernel_mode * value_j
                    )
                    output[request_index][ring_j] += area_i * (
                        row_sum * value_j - kernel_mode * value_i
                    )

        return tuple(
            (
                mode,
                tuple(_clean_scalar(self.normalization * value) for value in values),
            )
            for (mode, _, _), values in zip(requested, output, strict=True)
        )

    def apply_azimuthal_modes_repaid(self, mode_amplitudes):
        """Separated-mode application with the local tangent-cell repayment."""

        if _abs(self.kernel_power - 3.0) > 1.0e-14:
            raise ValueError("tangent-cell repayment is defined for kernel_power=3")
        requested = tuple(
            (int(mode), _complex_vector(values, self.n_rings))
            for mode, values in mode_amplitudes
        )
        raw = self.apply_azimuthal_modes(requested)
        integrals = tuple(self._cell_integrals(ring) for ring in range(self.n_rings))
        repaid = []
        for (mode, amplitudes), (_, output_values) in zip(requested, raw, strict=True):
            values = [complex(value) for value in output_values]
            for ring in range(self.n_rings):
                mass = self.node_area_weights[ring]
                azimuth_step = self.radii[ring] * self.theta_step
                conductance = (
                    0.5
                    * self.normalization
                    * mass
                    * integrals[ring][1]
                    / (azimuth_step * azimuth_step)
                )
                values[ring] += (
                    conductance
                    / mass
                    * (2.0 - 2.0 * _cos(mode * self.theta_step))
                    * amplitudes[ring]
                )

            edge_count = self.n_rings if self.meridian_periodic else self.n_rings - 1
            for ring in range(edge_count):
                following_ring = (ring + 1) % self.n_rings
                mass = self.node_area_weights[ring]
                following_mass = self.node_area_weights[following_ring]
                meridian_step = 0.5 * (
                    self.meridional_weights[ring]
                    + self.meridional_weights[following_ring]
                )
                conductance = (
                    0.25
                    * self.normalization
                    * (
                        mass * integrals[ring][0]
                        + following_mass * integrals[following_ring][0]
                    )
                    / (meridian_step * meridian_step)
                )
                difference = amplitudes[ring] - amplitudes[following_ring]
                values[ring] += conductance / mass * difference
                values[following_ring] -= conductance / following_mass * difference
            repaid.append((mode, tuple(_clean_scalar(value) for value in values)))
        return tuple(repaid)

    def apply_azimuthal_mode(self, amplitudes, mode=0):
        return self.apply_azimuthal_modes(((mode, amplitudes),))[0][1]

    def apply_azimuthal_mode_repaid(self, amplitudes, mode=0):
        return self.apply_azimuthal_modes_repaid(((mode, amplitudes),))[0][1]

    def direct_apply(self, values):
        """Small-problem reference; streams point pairs and stores no matrix."""

        rows = _complex_rows(values, self.n_rings, self.n_theta)
        output = [
            [0.0 + 0.0j for _ in range(self.n_theta)] for _ in range(self.n_rings)
        ]
        nodes = tuple(
            (ring, phase)
            for ring in range(self.n_rings)
            for phase in range(self.n_theta)
        )
        for left in range(len(nodes)):
            ring_i, phase_i = nodes[left]
            value_i = rows[ring_i][phase_i]
            for right in range(left + 1, len(nodes)):
                ring_j, phase_j = nodes[right]
                d2 = scale_phase_distance_squared(
                    self.radii[ring_i],
                    self.z_values[ring_i],
                    self.theta(phase_i),
                    self.radii[ring_j],
                    self.z_values[ring_j],
                    self.theta(phase_j),
                )
                if d2 <= 0.0:
                    raise ValueError("distinct surface nodes collide")
                kernel = d2 ** (-0.5 * self.kernel_power)
                difference = value_i - rows[ring_j][phase_j]
                output[ring_i][phase_i] += (
                    self.normalization
                    * self.node_area_weights[ring_j]
                    * kernel
                    * difference
                )
                output[ring_j][phase_j] -= (
                    self.normalization
                    * self.node_area_weights[ring_i]
                    * kernel
                    * difference
                )
        return tuple(
            tuple(_clean_scalar(value) for value in row) for row in output
        )

    def weighted_inner(self, left, right):
        left_rows = _complex_rows(left, self.n_rings, self.n_theta, "left")
        right_rows = _complex_rows(right, self.n_rings, self.n_theta, "right")
        total = 0.0 + 0.0j
        for ring in range(self.n_rings):
            area = self.node_area_weights[ring]
            for phase in range(self.n_theta):
                total += area * left_rows[ring][phase].conjugate() * right_rows[ring][phase]
        return _clean_scalar(total)

    def energy(self, values):
        applied = self.apply(values)
        return float(complex(self.weighted_inner(values, applied)).real)

    def constant_residual(self):
        constant = tuple(
            tuple(1.0 for _ in range(self.n_theta)) for _ in range(self.n_rings)
        )
        applied = self.apply(constant)
        return max(_abs(value) for row in applied for value in row)

    def stats(self):
        fft_levels = 0
        size = self.n_theta
        while size > 1 and size % 2 == 0:
            fft_levels += 1
            size //= 2
        radix_two = size == 1
        angular_work = (
            self.n_theta * max(fft_levels, 1)
            if radix_two
            else self.n_theta * self.n_theta
        )
        pair_count = self.n_rings * (self.n_rings + 1) // 2
        return {
            "n_rings": self.n_rings,
            "n_theta": self.n_theta,
            "n_nodes": self.n_nodes,
            "kernel_power": self.kernel_power,
            "normalization": self.normalization,
            "surface_area": self.surface_area,
            "radix_two_fft": radix_two,
            "meridian_periodic": self.meridian_periodic,
            "meridian_poles": self.meridian_poles,
            "streamed_ring_pairs": pair_count,
            "estimated_work_units": pair_count * (angular_work + self.n_theta),
            "asymptotic_apply_cost": "O(n_s^2 n_theta log n_theta)",
            "asymptotic_storage": "O(n_s n_theta)",
            "stored_dense_surface_matrix": False,
            "stored_pair_kernel_table": False,
            "dense_entries_avoided": self.dense_entries_avoided,
            "normal_form": "dz^2 + 2 exp(rho_i+rho_j)(cosh(d_rho)-cos(d_theta))",
        }

    def evaluate(self, values):
        result = self.apply_repaid(values)
        residual = self.constant_residual()
        ledger = BorrowComputeRepayLedger(
            borrowed=(
                "q=theta-i*rho scale-phase coordinates on each meridional ring",
                "periodic azimuthal phase for the foundational QJet FFT",
            ),
            computed=(
                "streamed angular convolution of the full 3D chord kernel",
                "ring-pair modal action without matrix assembly",
            ),
            repaid=(
                "exact cylindrical area Jacobian r*ds*dtheta",
                "graph row-sum diagonal and constant nullspace",
                "physical normalization 1/(2*pi) for the sphere DtN principal kernel",
                "analytic tangent-cell integral contracted with local surface Hessian jets",
            ),
            residuals=(("constant_mode_residual", residual),),
            residual_norm=residual,
            status="borrowed_repaid",
            notes=(
                "The collision node is omitted in principal-value form and its leading "
                "tangent-cell contribution is repaid explicitly. Higher-order continuum "
                "error remains visible in refinement tables."
            ),
        )
        return AxisymmetricQEvaluation(result, ledger, self.stats())


def build_axisymmetric_surface_qjet(
    radii,
    z_values,
    meridional_weights,
    n_theta,
    kernel_power=3.0,
    normalization=None,
    meridian_periodic=False,
    meridian_poles=False,
):
    return AxisymmetricSurfaceQJet(
        radii,
        z_values,
        meridional_weights,
        n_theta,
        kernel_power=kernel_power,
        normalization=normalization,
        meridian_periodic=meridian_periodic,
        meridian_poles=meridian_poles,
    )


def spheroid_qjet(
    equatorial_radius,
    polar_radius,
    n_meridian,
    n_theta,
    kernel_power=3.0,
):
    """Midpoint surface-of-revolution QJet for a sphere or spheroid."""

    a = float(equatorial_radius)
    c = float(polar_radius)
    if a <= 0.0 or c <= 0.0:
        raise ValueError("spheroid radii must be positive")
    count = int(n_meridian)
    if count < 2:
        raise ValueError("n_meridian must be at least two")
    du = PI / count
    radii = []
    z_values = []
    weights = []
    for index in range(count):
        u = (index + 0.5) * du
        sin_u = _sin(u)
        cos_u = _cos(u)
        radii.append(a * sin_u)
        z_values.append(c * cos_u)
        speed = _sqrt((a * cos_u) ** 2 + (c * sin_u) ** 2)
        weights.append(speed * du)
    return AxisymmetricSurfaceQJet(
        radii,
        z_values,
        weights,
        n_theta,
        kernel_power=kernel_power,
        meridian_periodic=False,
        meridian_poles=True,
    )


def radial_profile_qjet(
    base_radius,
    cosine_coefficients,
    n_meridian,
    n_theta,
    kernel_power=3.0,
):
    """Closed star-shaped surface with ``R(u)=base+sum c_k cos(k*u)``."""

    base = float(base_radius)
    coefficients = tuple(float(value) for value in cosine_coefficients)
    count = int(n_meridian)
    if base <= sum(_abs(value) for value in coefficients):
        raise ValueError("radial profile must remain strictly positive")
    du = PI / count
    radii = []
    z_values = []
    weights = []
    for index in range(count):
        u = (index + 0.5) * du
        radius = base
        derivative = 0.0
        for mode, coefficient in enumerate(coefficients, start=1):
            radius += coefficient * _cos(mode * u)
            derivative -= mode * coefficient * _sin(mode * u)
        sin_u = _sin(u)
        cos_u = _cos(u)
        radii.append(radius * sin_u)
        z_values.append(radius * cos_u)
        dr_du = derivative * sin_u + radius * cos_u
        dz_du = derivative * cos_u - radius * sin_u
        weights.append(_sqrt(dr_du * dr_du + dz_du * dz_du) * du)
    return AxisymmetricSurfaceQJet(
        radii,
        z_values,
        weights,
        n_theta,
        kernel_power=kernel_power,
        meridian_periodic=False,
        meridian_poles=True,
    )


def torus_qjet(
    major_radius,
    minor_radius,
    n_meridian,
    n_theta,
    kernel_power=3.0,
):
    """Closed genus-one axisymmetric surface."""

    major = float(major_radius)
    minor = float(minor_radius)
    if minor <= 0.0 or major <= minor:
        raise ValueError("torus requires major_radius > minor_radius > 0")
    count = int(n_meridian)
    du = TAU / count
    radii = []
    z_values = []
    weights = []
    for index in range(count):
        u = (index + 0.5) * du
        radii.append(major + minor * _cos(u))
        z_values.append(minor * _sin(u))
        weights.append(minor * du)
    return AxisymmetricSurfaceQJet(
        radii,
        z_values,
        weights,
        n_theta,
        kernel_power=kernel_power,
        meridian_periodic=True,
        meridian_poles=False,
    )


def conic_qjet(
    radius_start,
    radius_stop,
    z_start,
    z_stop,
    n_meridian,
    n_theta,
    kernel_power=3.0,
):
    """Open cylindrical/conical lateral surface sampled at profile midpoints."""

    r0 = float(radius_start)
    r1 = float(radius_stop)
    z0 = float(z_start)
    z1 = float(z_stop)
    if r0 < 0.0 or r1 < 0.0 or max(r0, r1) <= 0.0:
        raise ValueError("conic endpoint radii must be nonnegative and not both zero")
    count = int(n_meridian)
    dt = 1.0 / count
    dr = r1 - r0
    dz = z1 - z0
    ds = _sqrt(dr * dr + dz * dz) * dt
    radii = []
    z_values = []
    weights = []
    for index in range(count):
        t = (index + 0.5) * dt
        radii.append(r0 + dr * t)
        z_values.append(z0 + dz * t)
        weights.append(ds)
    return AxisymmetricSurfaceQJet(
        radii,
        z_values,
        weights,
        n_theta,
        kernel_power=kernel_power,
        meridian_periodic=False,
        meridian_poles=False,
    )


__all__ = [
    "AxisymmetricQEvaluation",
    "AxisymmetricSurfaceQJet",
    "build_axisymmetric_surface_qjet",
    "cartesian_distance_squared",
    "conic_qjet",
    "hyperbolic_scale_phase_distance_squared",
    "radial_profile_qjet",
    "scale_phase_distance_squared",
    "spheroid_qjet",
    "torus_qjet",
]
