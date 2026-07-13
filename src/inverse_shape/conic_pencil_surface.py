"""Matrix-free conic-pencil surface geometry and QJet shape calculus.

A surface is represented as a bundle of moving planar conics

    X(u, theta) = c(u)
                  + a(u) cos(theta) e1(u)
                  + b(u) sin(theta) e2(u),

where the orthonormal frame is transported by a unit quaternion.  The retained
state is one fixed-size three-jet per conic slice.  Surface points, normals,
area weights, and hierarchical moments are generated only inside an operator
application.  No dense distance or operator matrix is stored.

Two kernels have separate roles:

* ``|X-Y|^-2`` is the discriminant/inverse-square shape metric used by the
  reduced shape Hessian ``J^* Q J``;
* ``(2*pi)^-1 |X-Y|^-3`` is the flat principal surface kernel of the three-
  dimensional DtN operator.  The named ``apply_dtn_principal`` method includes
  this normalization.

The tree path is a quadrupole Barnes-Hut evaluation with exact leaf repayment.
It has ``O(N)`` storage and targets ``O(N log N)`` fixed-accuracy work under
bounded geometry; this is not a worst-case guarantee.  The streamed direct
path is retained as an independent ``O(N^2)`` matrix-free reference.
"""

from inverse_shape.joukowski_endpoint import (
    JoukowskiMapQJet,
    StaticJoukowskiEllipseQJet,
)
from inverse_shape.quadrature import (
    PI,
    TAU,
    BorrowComputeRepayLedger,
    _abs,
    _clean_scalar,
    _cos,
    _exp,
    _fft_precise,
    _finite,
    _ifft_precise,
    _log,
    _sin,
    _sqrt,
)


def _vadd(left, right):
    return (
        left[0] + right[0],
        left[1] + right[1],
        left[2] + right[2],
    )


def _vsub(left, right):
    return (
        left[0] - right[0],
        left[1] - right[1],
        left[2] - right[2],
    )


def _vscale(scale, value):
    return (scale * value[0], scale * value[1], scale * value[2])


def _vdot(left, right):
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _vcross(left, right):
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _vnorm(value):
    return _sqrt(max(_vdot(value, value), 0.0))


def _vnormalize(value):
    norm = _vnorm(value)
    if norm <= 1.0e-15:
        raise ValueError("cannot normalize a zero vector")
    return _vscale(1.0 / norm, value)


def _as_vector3(value, name="vector"):
    result = tuple(float(component) for component in value)
    if len(result) != 3 or any(not _finite(component) for component in result):
        raise ValueError(f"{name} must contain three finite components")
    return result


def _qdot(left, right):
    return sum(a * b for a, b in zip(left, right, strict=True))


def _qnormalize(value):
    quaternion = tuple(float(component) for component in value)
    if len(quaternion) != 4:
        raise ValueError("a quaternion must contain four components")
    norm = _sqrt(sum(component * component for component in quaternion))
    if norm <= 1.0e-15:
        raise ValueError("a frame quaternion must be nonzero")
    return tuple(component / norm for component in quaternion)


def _qconjugate(value):
    return (value[0], -value[1], -value[2], -value[3])


def _qmul(left, right):
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


def _qrotate(quaternion, vector):
    pure = (0.0, vector[0], vector[1], vector[2])
    rotated = _qmul(_qmul(quaternion, pure), _qconjugate(quaternion))
    return (rotated[1], rotated[2], rotated[3])


def _qaxis_angle(axis, angle):
    direction = _vnormalize(axis)
    half = 0.5 * float(angle)
    sine = _sin(half)
    return _qnormalize(
        (
            _cos(half),
            sine * direction[0],
            sine * direction[1],
            sine * direction[2],
        )
    )


def _qincrement(rotation):
    angle = _vnorm(rotation)
    if angle <= 1.0e-12:
        return _qnormalize(
            (
                1.0,
                0.5 * rotation[0],
                0.5 * rotation[1],
                0.5 * rotation[2],
            )
        )
    return _qaxis_angle(_vscale(1.0 / angle, rotation), angle)


def _qangular_velocity(quaternion, derivative):
    product = _qmul(derivative, _qconjugate(quaternion))
    return (2.0 * product[1], 2.0 * product[2], 2.0 * product[3])


def _qfrom_frame(first, second, third):
    """Return the rotor whose matrix columns are the supplied frame."""

    m00, m10, m20 = first
    m01, m11, m21 = second
    m02, m12, m22 = third
    trace = m00 + m11 + m22
    if trace > 0.0:
        scale = 2.0 * _sqrt(trace + 1.0)
        quaternion = (
            0.25 * scale,
            (m21 - m12) / scale,
            (m02 - m20) / scale,
            (m10 - m01) / scale,
        )
    elif m00 > m11 and m00 > m22:
        scale = 2.0 * _sqrt(1.0 + m00 - m11 - m22)
        quaternion = (
            (m21 - m12) / scale,
            0.25 * scale,
            (m01 + m10) / scale,
            (m02 + m20) / scale,
        )
    elif m11 > m22:
        scale = 2.0 * _sqrt(1.0 + m11 - m00 - m22)
        quaternion = (
            (m02 - m20) / scale,
            (m01 + m10) / scale,
            0.25 * scale,
            (m12 + m21) / scale,
        )
    else:
        scale = 2.0 * _sqrt(1.0 + m22 - m00 - m11)
        quaternion = (
            (m10 - m01) / scale,
            (m02 + m20) / scale,
            (m12 + m21) / scale,
            0.25 * scale,
        )
    return _qnormalize(quaternion)


def _mat3_add(left, right, left_scale=1.0, right_scale=1.0):
    return tuple(
        tuple(
            left_scale * left[row][column]
            + right_scale * right[row][column]
            for column in range(3)
        )
        for row in range(3)
    )


def _mat3_mul(left, right):
    return tuple(
        tuple(
            sum(left[row][inner] * right[inner][column] for inner in range(3))
            for column in range(3)
        )
        for row in range(3)
    )


def _mat3_det(matrix):
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def _mat3_inverse(matrix):
    determinant = _mat3_det(matrix)
    if _abs(determinant) <= 1.0e-15:
        raise ValueError("the conic pencil crosses a degenerate member")
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    inverse = (
        (e * i - f * h, c * h - b * i, b * f - c * e),
        (f * g - d * i, a * i - c * g, c * d - a * f),
        (d * h - e * g, b * g - a * h, a * e - b * d),
    )
    return tuple(
        tuple(value / determinant for value in row) for row in inverse
    )


def _lagrange_derivative_weights(offsets, order):
    factorial = (1.0, 1.0, 2.0, 6.0)[order]
    weights = []
    for node_index, node in enumerate(offsets):
        polynomial = [1.0]
        denominator = 1.0
        for other_index, other in enumerate(offsets):
            if other_index == node_index:
                continue
            next_polynomial = [0.0] * (len(polynomial) + 1)
            for degree, coefficient in enumerate(polynomial):
                next_polynomial[degree] -= other * coefficient
                next_polynomial[degree + 1] += coefficient
            polynomial = next_polynomial
            denominator *= node - other
        weights.append(factorial * polynomial[order] / denominator)
    return tuple(weights)


def _finite_difference(values, index, step, order, periodic):
    count = len(values)
    if order not in (1, 2, 3):
        raise ValueError("finite-difference order must be 1, 2, or 3")
    stencil_size = min(5, count)
    if periodic:
        first_offset = -(stencil_size // 2)
        offsets = tuple(first_offset + local for local in range(stencil_size))
        samples = tuple(values[(index + offset) % count] for offset in offsets)
    else:
        start = min(max(index - stencil_size // 2, 0), count - stencil_size)
        locations = tuple(start + local for local in range(stencil_size))
        offsets = tuple(location - index for location in locations)
        samples = tuple(values[location] for location in locations)
    coefficients = _lagrange_derivative_weights(offsets, order)
    scale = step**order
    first = samples[0]
    if isinstance(first, tuple):
        return tuple(
            sum(
                coefficient * sample[component]
                for coefficient, sample in zip(
                    coefficients,
                    samples,
                    strict=True,
                )
            )
            / scale
            for component in range(len(first))
        )
    return sum(
        coefficient * sample
        for coefficient, sample in zip(coefficients, samples, strict=True)
    ) / scale


def _frame_finite_difference(values, index, step, order, periodic):
    pivot = values[index]
    locally_aligned = tuple(
        value
        if _qdot(pivot, value) >= 0.0
        else tuple(-component for component in value)
        for value in values
    )
    return _finite_difference(
        locally_aligned,
        index,
        step,
        order,
        periodic,
    )


class ConicPencilSliceQJet:
    """Fixed-size value/three-jet generator for one moving conic slice."""

    def __init__(
        self,
        coordinate,
        center_jet,
        frame_jet,
        log_axis_jet,
    ):
        self.coordinate = float(coordinate)
        self.center_jet = tuple(tuple(value) for value in center_jet)
        self.frame_jet = tuple(tuple(value) for value in frame_jet)
        self.log_axis_jet = tuple(tuple(value) for value in log_axis_jet)

    @property
    def center(self):
        return self.center_jet[0]

    @property
    def quaternion(self):
        return self.frame_jet[0]

    @property
    def axes(self):
        return (
            _exp(self.log_axis_jet[0][0]),
            _exp(self.log_axis_jet[0][1]),
        )

    @property
    def conic_matrix(self):
        axis_a, axis_b = self.axes
        return (
            (1.0 / (axis_a * axis_a), 0.0, 0.0),
            (0.0, 1.0 / (axis_b * axis_b), 0.0),
            (0.0, 0.0, -1.0),
        )


class GeneratedConicSurfaceNodes:
    """Ephemeral O(N) work buffer generated from sparse slice jets."""

    def __init__(
        self,
        points,
        weights,
        normals,
        basis_a,
        basis_b,
        slice_indices,
        phase_indices,
    ):
        self.points = tuple(points)
        self.weights = tuple(float(value) for value in weights)
        self.normals = tuple(normals)
        self.basis_a = tuple(basis_a)
        self.basis_b = tuple(basis_b)
        self.slice_indices = tuple(slice_indices)
        self.phase_indices = tuple(phase_indices)

    @property
    def n(self):
        return len(self.points)


class _TreeNode:
    def __init__(self, indices, points, leaf_size):
        minimum = [min(points[index][axis] for index in indices) for axis in range(3)]
        maximum = [max(points[index][axis] for index in indices) for axis in range(3)]
        self.center = tuple(0.5 * (minimum[axis] + maximum[axis]) for axis in range(3))
        self.radius = max(_vnorm(_vsub(points[index], self.center)) for index in indices)
        self.children = tuple()
        self.indices = tuple(indices)
        self.moments = tuple()
        if len(indices) > leaf_size:
            spans = tuple(maximum[axis] - minimum[axis] for axis in range(3))
            axis = max(range(3), key=lambda item: spans[item])
            split = 0.5 * (minimum[axis] + maximum[axis])
            left = [index for index in indices if points[index][axis] <= split]
            right = [index for index in indices if points[index][axis] > split]
            if not left or not right:
                ordered = sorted(indices, key=lambda item: points[item][axis])
                middle = len(ordered) // 2
                left = ordered[:middle]
                right = ordered[middle:]
            self.children = (
                _TreeNode(left, points, leaf_size),
                _TreeNode(right, points, leaf_size),
            )
            self.indices = tuple()

    @property
    def is_leaf(self):
        return not self.children


def _zero_moment():
    return (0.0 + 0.0j, (0.0 + 0.0j,) * 3, (0.0 + 0.0j,) * 6)


def _moment_add(left, right):
    return (
        left[0] + right[0],
        tuple(left[1][axis] + right[1][axis] for axis in range(3)),
        tuple(left[2][axis] + right[2][axis] for axis in range(6)),
    )


def _moment_translate(moment, displacement):
    mass, first, second = moment
    dx, dy, dz = displacement
    mx, my, mz = first
    xx, xy, xz, yy, yz, zz = second
    return (
        mass,
        (
            mx + mass * dx,
            my + mass * dy,
            mz + mass * dz,
        ),
        (
            xx + 2.0 * dx * mx + mass * dx * dx,
            xy + dx * my + dy * mx + mass * dx * dy,
            xz + dx * mz + dz * mx + mass * dx * dz,
            yy + 2.0 * dy * my + mass * dy * dy,
            yz + dy * mz + dz * my + mass * dy * dz,
            zz + 2.0 * dz * mz + mass * dz * dz,
        ),
    )


def _build_tree_moments(node, points, charges):
    channel_count = len(charges)
    if node.is_leaf:
        moments = [_zero_moment() for _ in range(channel_count)]
        for index in node.indices:
            displacement = _vsub(points[index], node.center)
            dx, dy, dz = displacement
            for channel in range(channel_count):
                charge = charges[channel][index]
                contribution = (
                    charge,
                    (charge * dx, charge * dy, charge * dz),
                    (
                        charge * dx * dx,
                        charge * dx * dy,
                        charge * dx * dz,
                        charge * dy * dy,
                        charge * dy * dz,
                        charge * dz * dz,
                    ),
                )
                moments[channel] = _moment_add(moments[channel], contribution)
        node.moments = tuple(moments)
        return node.moments

    moments = [_zero_moment() for _ in range(channel_count)]
    for child in node.children:
        child_moments = _build_tree_moments(child, points, charges)
        displacement = _vsub(child.center, node.center)
        for channel in range(channel_count):
            translated = _moment_translate(child_moments[channel], displacement)
            moments[channel] = _moment_add(moments[channel], translated)
    node.moments = tuple(moments)
    return node.moments


def _moment_potential(moment, displacement, kernel_power):
    mass, first, second = moment
    radius_squared = _vdot(displacement, displacement)
    if radius_squared <= 1.0e-30:
        raise ValueError("far-field expansion evaluated at its center")
    radius_power = radius_squared ** (-0.5 * kernel_power)
    dot_first = sum(displacement[axis] * first[axis] for axis in range(3))
    xx, xy, xz, yy, yz, zz = second
    trace_second = xx + yy + zz
    x, y, z = displacement
    radial_second = (
        x * x * xx
        + 2.0 * x * y * xy
        + 2.0 * x * z * xz
        + y * y * yy
        + 2.0 * y * z * yz
        + z * z * zz
    )
    power = float(kernel_power)
    return radius_power * (
        mass
        + power * dot_first / radius_squared
        - 0.5 * power * trace_second / radius_squared
        + 0.5
        * power
        * (power + 2.0)
        * radial_second
        / (radius_squared * radius_squared)
    )


def _tree_potentials(
    node,
    target_index,
    points,
    charges,
    kernel_power,
    opening,
    counters,
):
    target = points[target_index]
    displacement = _vsub(target, node.center)
    distance = _vnorm(displacement)
    if distance > 0.0 and node.radius / distance < opening:
        counters["accepted_blocks"] += 1
        return tuple(
            _moment_potential(moment, displacement, kernel_power)
            for moment in node.moments
        )

    if node.is_leaf:
        result = [0.0 + 0.0j for _ in charges]
        for source_index in node.indices:
            if source_index == target_index:
                continue
            difference = _vsub(target, points[source_index])
            distance_squared = _vdot(difference, difference)
            if distance_squared <= 1.0e-28:
                raise ValueError("distinct conic surface nodes collide")
            kernel = distance_squared ** (-0.5 * kernel_power)
            counters["direct_pairs"] += 1
            for channel in range(len(charges)):
                result[channel] += charges[channel][source_index] * kernel
        return tuple(result)

    result = [0.0 + 0.0j for _ in charges]
    for child in node.children:
        contribution = _tree_potentials(
            child,
            target_index,
            points,
            charges,
            kernel_power,
            opening,
            counters,
        )
        for channel in range(len(charges)):
            result[channel] += contribution[channel]
    return tuple(result)


class ConicPencilSurfaceEvaluation:
    def __init__(self, values, ledger, stats):
        self.values = values
        self.ledger = ledger
        self.stats = stats


class ConicPencilSurfaceQJet:
    """Sparse conic-bundle generator with matrix-free surface operators."""

    parameter_count_per_slice = 8
    stored_scalars_per_slice = 36

    def __init__(self, centers, quaternions, axes, n_theta, periodic=False):
        centers = tuple(_as_vector3(center, "center") for center in centers)
        quaternions = tuple(_qnormalize(value) for value in quaternions)
        axes = tuple(tuple(float(component) for component in pair) for pair in axes)
        if len(centers) < 4:
            raise ValueError("a conic pencil surface requires at least four slices")
        if len(quaternions) != len(centers) or len(axes) != len(centers):
            raise ValueError("center, frame, and axis counts must match")
        if any(
            len(pair) != 2
            or pair[0] <= 0.0
            or pair[1] <= 0.0
            or not _finite(pair[0])
            or not _finite(pair[1])
            for pair in axes
        ):
            raise ValueError("each conic requires two positive finite axes")
        aligned = [quaternions[0]]
        for quaternion in quaternions[1:]:
            aligned.append(
                quaternion
                if _qdot(aligned[-1], quaternion) >= 0.0
                else tuple(-value for value in quaternion)
            )
        self.n_slices = len(centers)
        self.n_theta = int(n_theta)
        if self.n_theta < 8:
            raise ValueError("n_theta must be at least eight")
        self.periodic = bool(periodic)
        self.coordinate_step = (
            1.0 / self.n_slices if self.periodic else 1.0 / (self.n_slices - 1)
        )
        log_axes = tuple((_log(pair[0]), _log(pair[1])) for pair in axes)
        self.slice_jets = tuple(
            ConicPencilSliceQJet(
                index * self.coordinate_step,
                (
                    centers[index],
                    _finite_difference(
                        centers,
                        index,
                        self.coordinate_step,
                        1,
                        self.periodic,
                    ),
                    _finite_difference(
                        centers,
                        index,
                        self.coordinate_step,
                        2,
                        self.periodic,
                    ),
                    _finite_difference(
                        centers,
                        index,
                        self.coordinate_step,
                        3,
                        self.periodic,
                    ),
                ),
                (
                    aligned[index],
                    _frame_finite_difference(
                        tuple(aligned),
                        index,
                        self.coordinate_step,
                        1,
                        self.periodic,
                    ),
                    _frame_finite_difference(
                        tuple(aligned),
                        index,
                        self.coordinate_step,
                        2,
                        self.periodic,
                    ),
                    _frame_finite_difference(
                        tuple(aligned),
                        index,
                        self.coordinate_step,
                        3,
                        self.periodic,
                    ),
                ),
                (
                    log_axes[index],
                    _finite_difference(
                        log_axes,
                        index,
                        self.coordinate_step,
                        1,
                        self.periodic,
                    ),
                    _finite_difference(
                        log_axes,
                        index,
                        self.coordinate_step,
                        2,
                        self.periodic,
                    ),
                    _finite_difference(
                        log_axes,
                        index,
                        self.coordinate_step,
                        3,
                        self.periodic,
                    ),
                ),
            )
            for index in range(self.n_slices)
        )
        self._same_slice_joukowski_operators = {}
        self._same_slice_cycle_weight_graphs = {}
        self.last_apply_stats = {}

    @property
    def n_nodes(self):
        return self.n_slices * self.n_theta

    @property
    def parameter_count(self):
        return self.parameter_count_per_slice * self.n_slices

    def _phase(self, phase_index):
        return TAU * phase_index / self.n_theta

    def generate_nodes(self):
        points = []
        basis_a = []
        basis_b = []
        slice_indices = []
        phase_indices = []
        theta_derivatives = []
        coordinate_derivatives = []
        for slice_index, jet in enumerate(self.slice_jets):
            axis_a, axis_b = jet.axes
            frame_a = _qrotate(jet.quaternion, (1.0, 0.0, 0.0))
            frame_b = _qrotate(jet.quaternion, (0.0, 1.0, 0.0))
            angular_velocity = _qangular_velocity(
                jet.quaternion,
                jet.frame_jet[1],
            )
            for phase_index in range(self.n_theta):
                phase = self._phase(phase_index)
                component_a = _vscale(axis_a * _cos(phase), frame_a)
                component_b = _vscale(axis_b * _sin(phase), frame_b)
                radial = _vadd(component_a, component_b)
                points.append(_vadd(jet.center, radial))
                basis_a.append(component_a)
                basis_b.append(component_b)
                slice_indices.append(slice_index)
                phase_indices.append(phase_index)
                theta_derivatives.append(
                    _vadd(
                        _vscale(-axis_a * _sin(phase), frame_a),
                        _vscale(axis_b * _cos(phase), frame_b),
                    )
                )
                coordinate_derivatives.append(
                    _vadd(
                        jet.center_jet[1],
                        _vadd(
                            _vadd(
                                _vscale(jet.log_axis_jet[1][0], component_a),
                                _vscale(jet.log_axis_jet[1][1], component_b),
                            ),
                            _vcross(angular_velocity, radial),
                        ),
                    )
                )

        weights = []
        normals = []
        theta_step = TAU / self.n_theta
        for index in range(len(points)):
            slice_index = slice_indices[index]
            endpoint_factor = (
                1.0
                if self.periodic or 0 < slice_index < self.n_slices - 1
                else 0.5
            )
            area_vector = _vcross(
                coordinate_derivatives[index],
                theta_derivatives[index],
            )
            area_density = _vnorm(area_vector)
            if area_density <= 1.0e-14:
                raise ValueError("the conic pencil surface has a degenerate tangent cell")
            weights.append(
                endpoint_factor
                * area_density
                * self.coordinate_step
                * theta_step
            )
            normals.append(_vscale(1.0 / area_density, area_vector))
        return GeneratedConicSurfaceNodes(
            points,
            weights,
            normals,
            basis_a,
            basis_b,
            slice_indices,
            phase_indices,
        )

    def _as_fields(self, fields):
        rows = tuple(tuple(complex(value) for value in field) for field in fields)
        if not rows or any(len(row) != self.n_nodes for row in rows):
            raise ValueError("each field must contain one value per surface node")
        return rows

    def apply_fields_direct(self, fields, kernel_power=2.0):
        kernel_power = float(kernel_power)
        if kernel_power <= 0.0 or not _finite(kernel_power):
            raise ValueError("kernel_power must be positive and finite")
        rows = self._as_fields(fields)
        nodes = self.generate_nodes()
        output = [
            [0.0 + 0.0j for _ in range(self.n_nodes)] for _ in rows
        ]
        pair_count = 0
        for left in range(self.n_nodes):
            for right in range(left + 1, self.n_nodes):
                difference = _vsub(nodes.points[left], nodes.points[right])
                distance_squared = _vdot(difference, difference)
                if distance_squared <= 1.0e-28:
                    raise ValueError("distinct conic surface nodes collide")
                kernel = distance_squared ** (-0.5 * kernel_power)
                pair_count += 1
                for channel, field in enumerate(rows):
                    delta = field[left] - field[right]
                    output[channel][left] += nodes.weights[right] * kernel * delta
                    output[channel][right] -= nodes.weights[left] * kernel * delta
        self.last_apply_stats = {
            "method": "direct_pair_stream",
            "direct_pairs": pair_count,
            "accepted_blocks": 0,
            "generated_node_entries": self.n_nodes,
            "stored_dense_matrix": False,
        }
        return tuple(
            tuple(_clean_scalar(value) for value in row) for row in output
        )

    def apply_fields_tree(
        self,
        fields,
        kernel_power=2.0,
        opening=0.42,
        leaf_size=12,
    ):
        kernel_power = float(kernel_power)
        opening = float(opening)
        leaf_size = int(leaf_size)
        if kernel_power <= 0.0 or not _finite(kernel_power):
            raise ValueError("kernel_power must be positive and finite")
        if opening <= 0.0 or opening >= 1.0 or not _finite(opening):
            raise ValueError("opening must lie strictly between zero and one")
        if leaf_size < 1:
            raise ValueError("leaf_size must be positive")
        rows = self._as_fields(fields)
        nodes = self.generate_nodes()
        root = _TreeNode(tuple(range(self.n_nodes)), nodes.points, leaf_size)
        charges = [tuple(nodes.weights)]
        charges.extend(
            tuple(
                nodes.weights[index] * field[index]
                for index in range(self.n_nodes)
            )
            for field in rows
        )
        charges = tuple(charges)
        _build_tree_moments(root, nodes.points, charges)
        output = [
            [0.0 + 0.0j for _ in range(self.n_nodes)] for _ in rows
        ]
        counters = {"direct_pairs": 0, "accepted_blocks": 0}
        for target_index in range(self.n_nodes):
            potentials = _tree_potentials(
                root,
                target_index,
                nodes.points,
                charges,
                kernel_power,
                opening,
                counters,
            )
            for channel, field in enumerate(rows):
                output[channel][target_index] = (
                    field[target_index] * potentials[0]
                    - potentials[channel + 1]
                )
        self.last_apply_stats = {
            "method": "quadrupole_tree_with_exact_leaves",
            "direct_pairs": counters["direct_pairs"],
            "accepted_blocks": counters["accepted_blocks"],
            "generated_node_entries": self.n_nodes,
            "stored_dense_matrix": False,
            "opening": opening,
            "leaf_size": leaf_size,
        }
        return tuple(
            tuple(_clean_scalar(value) for value in row) for row in output
        )

    def apply(
        self,
        values,
        kernel_power=2.0,
        method="tree",
        opening=0.42,
        leaf_size=12,
    ):
        fields = (tuple(values),)
        if method == "direct":
            return self.apply_fields_direct(fields, kernel_power)[0]
        if method == "tree":
            return self.apply_fields_tree(
                fields,
                kernel_power,
                opening,
                leaf_size,
            )[0]
        raise ValueError("method must be 'direct' or 'tree'")

    def apply_inverse_square_metric(
        self,
        values,
        method="tree",
        opening=0.42,
        leaf_size=12,
    ):
        """Apply the unnormalized inverse-square surface shape metric."""

        return self.apply(values, 2.0, method, opening, leaf_size)

    def apply_dtn_principal(
        self,
        values,
        method="tree",
        opening=0.42,
        leaf_size=12,
    ):
        """Apply the normalized flat 3D DtN principal surface action.

        This evaluates

            (2*pi)^-1 sum_j w_j (f_i-f_j) / |X_i-X_j|^3.

        It is the discretized off-diagonal principal channel.  A convergent
        high-order continuum quadrature also needs tangent-cell repayment and
        the geometry-dependent lower-order operator.
        """

        raw = self.apply(values, 3.0, method, opening, leaf_size)
        return tuple(_clean_scalar(value / TAU) for value in raw)

    def apply_same_slice_joukowski_fields(
        self,
        fields,
        tolerance=2.0e-16,
    ):
        """Apply exact inverse-square interactions within each conic slice.

        This is the static local singular channel.  Cross-slice interactions
        are intentionally excluded and must be supplied by a scale-phase
        chart or a separate far operator.
        """

        rows = self._as_fields(fields)
        nodes = self.generate_nodes()
        output = [
            [0.0 + 0.0j for _ in range(self.n_nodes)]
            for _ in rows
        ]
        total_channels = 0
        maximum_tail = 0.0
        for slice_index, jet in enumerate(self.slice_jets):
            axis_a, axis_b = jet.axes
            start = slice_index * self.n_theta
            stop = start + self.n_theta
            slice_fields = tuple(row[start:stop] for row in rows)
            slice_weights = nodes.weights[start:stop]
            scale = max(axis_a, axis_b)
            if _abs(axis_a - axis_b) <= 4.0e-15 * scale:
                eigenvalues = tuple(
                    mode * (self.n_theta - mode) / 2.0
                    for mode in range(self.n_theta)
                )

                def cycle_graph(values):
                    transformed = _fft_precise(values)
                    return tuple(
                        _ifft_precise(
                            tuple(
                                transformed[mode] * eigenvalues[mode]
                                for mode in range(self.n_theta)
                            )
                        )
                    )

                cycle_key = (slice_index, tuple(slice_weights))
                weight_graph = self._same_slice_cycle_weight_graphs.get(
                    cycle_key
                )
                if weight_graph is None:
                    weight_graph = cycle_graph(slice_weights)
                    self._same_slice_cycle_weight_graphs[cycle_key] = weight_graph
                applied_rows = []
                inverse_radius_squared = 1.0 / (axis_a * axis_a)
                for row in slice_fields:
                    graph_product = cycle_graph(
                        tuple(
                            slice_weights[index] * row[index]
                            for index in range(self.n_theta)
                        )
                    )
                    applied_rows.append(
                        tuple(
                            inverse_radius_squared
                            * (
                                graph_product[index]
                                - row[index] * weight_graph[index]
                            )
                            for index in range(self.n_theta)
                        )
                    )
                applied = tuple(applied_rows)
                total_channels += 1
            else:
                transpose_axes = axis_b > axis_a
                major = max(axis_a, axis_b)
                minor = min(axis_a, axis_b)
                focal = _sqrt(major * major - minor * minor)
                map_scale = 0.5 * focal
                mu = 0.5 * _log((major + minor) / (major - minor))
                operator_key = (slice_index, float(tolerance), major, minor)
                operator = self._same_slice_joukowski_operators.get(
                    operator_key
                )
                if operator is None:
                    operator = StaticJoukowskiEllipseQJet(
                        JoukowskiMapQJet(map_scale, mu),
                        self.n_theta,
                        tolerance=tolerance,
                    )
                    self._same_slice_joukowski_operators[operator_key] = operator
                if transpose_axes and self.n_theta % 4 == 0:
                    quarter = self.n_theta // 4
                    canonical_fields = tuple(
                        tuple(
                            row[(index + quarter) % self.n_theta]
                            for index in range(self.n_theta)
                        )
                        for row in slice_fields
                    )
                    canonical_weights = tuple(
                        slice_weights[(index + quarter) % self.n_theta]
                        for index in range(self.n_theta)
                    )
                    canonical = operator.apply_fields_with_weights(
                        canonical_fields,
                        canonical_weights,
                    )
                    applied = tuple(
                        tuple(
                            row[(index - quarter) % self.n_theta]
                            for index in range(self.n_theta)
                        )
                        for row in canonical
                    )
                elif transpose_axes:
                    applied_rows = [
                        [0.0 + 0.0j for _ in range(self.n_theta)]
                        for _ in rows
                    ]
                    for left in range(self.n_theta):
                        for right in range(left + 1, self.n_theta):
                            distance_squared = sum(
                                (
                                    nodes.points[start + left][axis]
                                    - nodes.points[start + right][axis]
                                )
                                ** 2
                                for axis in range(3)
                            )
                            kernel = 1.0 / distance_squared
                            for channel, row in enumerate(slice_fields):
                                delta = row[left] - row[right]
                                applied_rows[channel][left] += (
                                    slice_weights[right] * kernel * delta
                                )
                                applied_rows[channel][right] -= (
                                    slice_weights[left] * kernel * delta
                                )
                    applied = tuple(tuple(row) for row in applied_rows)
                else:
                    applied = operator.apply_fields_with_weights(
                        slice_fields,
                        slice_weights,
                    )
                total_channels += len(operator.channels)
                maximum_tail = max(
                    maximum_tail,
                    operator.quotient_tail_bound(),
                )
            for channel in range(len(rows)):
                output[channel][start:stop] = applied[channel]
        self.last_apply_stats = {
            "method": "static_same_slice_joukowski",
            "compiled_slice_count": self.n_slices,
            "cycle_fft_channels": total_channels,
            "maximum_joukowski_tail_bound": maximum_tail,
            "cross_slice_interactions_included": False,
            "stored_dense_matrix": False,
        }
        return tuple(
            tuple(_clean_scalar(value) for value in row) for row in output
        )

    def apply_same_slice_joukowski(self, values, tolerance=2.0e-16):
        return self.apply_same_slice_joukowski_fields(
            (values,),
            tolerance,
        )[0]

    def evaluate(
        self,
        values,
        kernel_power=2.0,
        method="tree",
        opening=0.42,
        leaf_size=12,
    ):
        result = self.apply(
            values,
            kernel_power,
            method,
            opening,
            leaf_size,
        )
        constant = self.apply(
            (1.0,) * self.n_nodes,
            kernel_power,
            method,
            opening,
            leaf_size,
        )
        residual = max(_abs(complex(value)) for value in constant)
        ledger = BorrowComputeRepayLedger(
            borrowed=(
                "moving conic value/three-jets",
                "SU(2) frame rotors and local conic pencil certificates",
                "quadrupole far blocks generated from O(N) surface work buffers",
            ),
            computed=(
                f"matrix-free |X-Y|^-{float(kernel_power):g} surface action",
                "surface nodes and area weights generated then released",
            ),
            repaid=(
                "exact leaf interactions",
                "graph difference form and constant nullspace",
                "no dense distance or operator matrix",
            ),
            residuals=(("constant_residual", residual),),
            residual_norm=residual,
            status="borrowed_repaid",
            notes=(
                "Tree complexity is fixed-accuracy and geometry-dependent; "
                "the direct pair stream remains the independent reference."
            ),
        )
        return ConicPencilSurfaceEvaluation(result, ledger, self.stats())

    def jvp(self, parameter_directions, nodes=None):
        directions = tuple(
            tuple(float(value) for value in row) for row in parameter_directions
        )
        if len(directions) != self.n_slices or any(
            len(row) != self.parameter_count_per_slice for row in directions
        ):
            raise ValueError("each slice direction must contain eight parameters")
        if nodes is None:
            nodes = self.generate_nodes()
        displacements = []
        for index in range(self.n_nodes):
            row = directions[nodes.slice_indices[index]]
            translation = (row[0], row[1], row[2])
            axis_part = _vadd(
                _vscale(row[3], nodes.basis_a[index]),
                _vscale(row[4], nodes.basis_b[index]),
            )
            radial = _vadd(nodes.basis_a[index], nodes.basis_b[index])
            rotation = _vcross((row[5], row[6], row[7]), radial)
            displacements.append(
                _vadd(translation, _vadd(axis_part, rotation))
            )
        return tuple(displacements)

    def vjp(self, node_forces, nodes=None):
        forces = tuple(_as_vector3(force, "node force") for force in node_forces)
        if len(forces) != self.n_nodes:
            raise ValueError("one force vector is required per surface node")
        if nodes is None:
            nodes = self.generate_nodes()
        result = [
            [0.0 for _ in range(self.parameter_count_per_slice)]
            for _ in range(self.n_slices)
        ]
        for index, force in enumerate(forces):
            slice_index = nodes.slice_indices[index]
            weight = nodes.weights[index]
            weighted = _vscale(weight, force)
            result[slice_index][0] += weighted[0]
            result[slice_index][1] += weighted[1]
            result[slice_index][2] += weighted[2]
            result[slice_index][3] += _vdot(weighted, nodes.basis_a[index])
            result[slice_index][4] += _vdot(weighted, nodes.basis_b[index])
            radial = _vadd(nodes.basis_a[index], nodes.basis_b[index])
            torque = _vcross(radial, weighted)
            result[slice_index][5] += torque[0]
            result[slice_index][6] += torque[1]
            result[slice_index][7] += torque[2]
        return tuple(tuple(row) for row in result)

    def shape_hessian_apply(
        self,
        parameter_directions,
        method="tree",
        opening=0.42,
        leaf_size=12,
        ridge=1.0e-6,
        shell_smoothness=0.0,
    ):
        directions = tuple(
            tuple(float(value) for value in row)
            for row in parameter_directions
        )
        nodes = self.generate_nodes()
        displacements = self.jvp(directions, nodes)
        coordinate_fields = tuple(
            tuple(displacement[axis] for displacement in displacements)
            for axis in range(3)
        )
        if method == "direct":
            applied_fields = self.apply_fields_direct(coordinate_fields, 2.0)
        elif method == "tree":
            applied_fields = self.apply_fields_tree(
                coordinate_fields,
                2.0,
                opening,
                leaf_size,
            )
        else:
            raise ValueError("method must be 'direct' or 'tree'")
        node_forces = tuple(
            tuple(float(complex(applied_fields[axis][index]).real) for axis in range(3))
            for index in range(self.n_nodes)
        )
        reduced = [list(row) for row in self.vjp(node_forces, nodes)]
        for slice_index in range(self.n_slices):
            for parameter in range(self.parameter_count_per_slice):
                reduced[slice_index][parameter] += float(ridge) * directions[
                    slice_index
                ][parameter]
        if shell_smoothness:
            edge_count = self.n_slices if self.periodic else self.n_slices - 1
            for left in range(edge_count):
                right = (left + 1) % self.n_slices
                for parameter in range(self.parameter_count_per_slice):
                    difference = (
                        directions[left][parameter] - directions[right][parameter]
                    )
                    reduced[left][parameter] += shell_smoothness * difference
                    reduced[right][parameter] -= shell_smoothness * difference
        return tuple(tuple(row) for row in reduced)

    @staticmethod
    def parameter_inner(left, right):
        return sum(
            a * b
            for left_row, right_row in zip(left, right, strict=True)
            for a, b in zip(left_row, right_row, strict=True)
        )

    def shape_energy(self, parameter_directions, **kwargs):
        applied = self.shape_hessian_apply(parameter_directions, **kwargs)
        return self.parameter_inner(parameter_directions, applied)

    def solve_shape_load(
        self,
        load,
        method="direct",
        opening=0.42,
        leaf_size=12,
        ridge=1.0e-3,
        shell_smoothness=1.0e-2,
        iterations=32,
        tolerance=1.0e-8,
    ):
        rhs = tuple(tuple(float(value) for value in row) for row in load)
        if len(rhs) != self.n_slices or any(
            len(row) != self.parameter_count_per_slice for row in rhs
        ):
            raise ValueError("shape load must contain eight values per slice")
        solution = tuple(
            (0.0,) * self.parameter_count_per_slice
            for _ in range(self.n_slices)
        )
        residual = rhs
        direction = residual
        residual_square = self.parameter_inner(residual, residual)
        initial = _sqrt(max(residual_square, 0.0))
        completed = 0
        for iteration in range(int(iterations)):
            applied = self.shape_hessian_apply(
                direction,
                method=method,
                opening=opening,
                leaf_size=leaf_size,
                ridge=ridge,
                shell_smoothness=shell_smoothness,
            )
            denominator = self.parameter_inner(direction, applied)
            if denominator <= 1.0e-30:
                break
            alpha = residual_square / denominator
            solution = tuple(
                tuple(
                    solution[row][column] + alpha * direction[row][column]
                    for column in range(self.parameter_count_per_slice)
                )
                for row in range(self.n_slices)
            )
            next_residual = tuple(
                tuple(
                    residual[row][column] - alpha * applied[row][column]
                    for column in range(self.parameter_count_per_slice)
                )
                for row in range(self.n_slices)
            )
            next_square = self.parameter_inner(next_residual, next_residual)
            completed = iteration + 1
            if _sqrt(max(next_square, 0.0)) <= tolerance * max(
                initial,
                1.0e-300,
            ):
                residual = next_residual
                residual_square = next_square
                break
            beta = next_square / max(residual_square, 1.0e-300)
            direction = tuple(
                tuple(
                    next_residual[row][column]
                    + beta * direction[row][column]
                    for column in range(self.parameter_count_per_slice)
                )
                for row in range(self.n_slices)
            )
            residual = next_residual
            residual_square = next_square
        return solution, {
            "iterations": completed,
            "initial_residual": initial,
            "final_residual": _sqrt(max(residual_square, 0.0)),
            "relative_residual": _sqrt(max(residual_square, 0.0))
            / max(initial, 1.0e-300),
            "stored_dense_reduced_hessian": False,
        }

    def deformed(self, parameter_directions, step=1.0):
        directions = tuple(
            tuple(float(value) for value in row)
            for row in parameter_directions
        )
        centers = []
        quaternions = []
        axes = []
        for jet, row in zip(self.slice_jets, directions, strict=True):
            centers.append(
                _vadd(jet.center, _vscale(float(step), (row[0], row[1], row[2])))
            )
            rotation = _vscale(float(step), (row[5], row[6], row[7]))
            quaternions.append(
                _qnormalize(_qmul(_qincrement(rotation), jet.quaternion))
            )
            axis_a, axis_b = jet.axes
            axes.append(
                (
                    axis_a * _exp(float(step) * row[3]),
                    axis_b * _exp(float(step) * row[4]),
                )
            )
        return ConicPencilSurfaceQJet(
            centers,
            quaternions,
            axes,
            self.n_theta,
            self.periodic,
        )

    def pencil_certificates(self, parameter=0.5):
        edge_count = self.n_slices if self.periodic else self.n_slices - 1
        return tuple(
            conic_pencil_certificate(
                self.slice_jets[left],
                self.slice_jets[(left + 1) % self.n_slices],
                parameter,
            )
            for left in range(edge_count)
        )

    def stats(self):
        stats = {
            "n_slices": self.n_slices,
            "n_theta": self.n_theta,
            "n_nodes": self.n_nodes,
            "periodic": self.periodic,
            "stored_slice_three_jets": self.n_slices,
            "stored_geometry_scalars": self.stored_scalars_per_slice
            * self.n_slices,
            "generated_surface_entries_per_apply": self.n_nodes,
            "stored_dense_distance_matrix": False,
            "stored_dense_operator_matrix": False,
            "stored_dense_reduced_hessian": False,
            "direct_reference_cost": "O(N^2) streamed, O(N) storage",
            "tree_apply_cost": (
                "targets O(N log N) fixed accuracy under bounded geometry; "
                "O(N) storage"
            ),
            "shape_hessian": "J^* Q_inverse_square J generated by JVP/Q/VJP",
        }
        stats.update(self.last_apply_stats)
        return stats


def conic_pencil_certificate(left_jet, right_jet, parameter=0.5):
    """Return the determinant and log-determinant curvature of a 3x3 pencil."""

    value = float(parameter)
    left = left_jet.conic_matrix
    right = right_jet.conic_matrix
    delta = _mat3_add(right, left, 1.0, -1.0)
    pencil = _mat3_add(left, delta, 1.0, value)
    determinant = _mat3_det(pencil)
    inverse = _mat3_inverse(pencil)
    product = _mat3_mul(inverse, delta)
    square = _mat3_mul(product, product)
    curvature = square[0][0] + square[1][1] + square[2][2]
    return {
        "parameter": value,
        "determinant": determinant,
        "minus_second_log_det": curvature,
        "degenerate": _abs(determinant) <= 1.0e-12,
        "role": "local conic-chart certificate, not a surface distance matrix",
    }


def straight_conic_tube_qjet(
    length,
    axis_a,
    axis_b,
    n_slices,
    n_theta,
):
    count = int(n_slices)
    centers = tuple(
        (0.0, 0.0, float(length) * (index / (count - 1) - 0.5))
        for index in range(count)
    )
    quaternions = ((1.0, 0.0, 0.0, 0.0),) * count
    axes = ((float(axis_a), float(axis_b)),) * count
    return ConicPencilSurfaceQJet(centers, quaternions, axes, n_theta)


def tapered_conic_tube_qjet(
    length,
    axis_a_start,
    axis_a_stop,
    axis_b_start,
    axis_b_stop,
    n_slices,
    n_theta,
):
    count = int(n_slices)
    centers = []
    axes = []
    for index in range(count):
        fraction = index / (count - 1)
        centers.append((0.0, 0.0, float(length) * (fraction - 0.5)))
        axes.append(
            (
                axis_a_start + fraction * (axis_a_stop - axis_a_start),
                axis_b_start + fraction * (axis_b_stop - axis_b_start),
            )
        )
    return ConicPencilSurfaceQJet(
        centers,
        ((1.0, 0.0, 0.0, 0.0),) * count,
        axes,
        n_theta,
    )


def bent_conic_tube_qjet(
    bend_radius,
    bend_angle,
    axis_a,
    axis_b,
    n_slices,
    n_theta,
):
    count = int(n_slices)
    centers = []
    quaternions = []
    for index in range(count):
        fraction = index / (count - 1)
        angle = float(bend_angle) * (fraction - 0.5)
        centers.append(
            (
                float(bend_radius) * (1.0 - _cos(angle)),
                0.0,
                float(bend_radius) * _sin(angle),
            )
        )
        quaternions.append(_qaxis_angle((0.0, 1.0, 0.0), angle))
    return ConicPencilSurfaceQJet(
        centers,
        quaternions,
        ((float(axis_a), float(axis_b)),) * count,
        n_theta,
    )


def twisted_ellipse_tube_qjet(
    length,
    axis_a,
    axis_b,
    total_twist,
    n_slices,
    n_theta,
):
    count = int(n_slices)
    centers = []
    quaternions = []
    for index in range(count):
        fraction = index / (count - 1)
        centers.append((0.0, 0.0, float(length) * (fraction - 0.5)))
        quaternions.append(
            _qaxis_angle(
                (0.0, 0.0, 1.0),
                float(total_twist) * (fraction - 0.5),
            )
        )
    return ConicPencilSurfaceQJet(
        centers,
        quaternions,
        ((float(axis_a), float(axis_b)),) * count,
        n_theta,
    )


def toroidal_conic_bundle_qjet(
    major_radius,
    axis_a,
    axis_b,
    n_slices,
    n_theta,
):
    count = int(n_slices)
    centers = []
    quaternions = []
    for index in range(count):
        angle = TAU * index / count
        radial = (_cos(angle), _sin(angle), 0.0)
        vertical = (0.0, 0.0, 1.0)
        tangent = (-_sin(angle), _cos(angle), 0.0)
        centers.append(_vscale(float(major_radius), radial))
        quaternions.append(_qfrom_frame(radial, vertical, tangent))
    return ConicPencilSurfaceQJet(
        centers,
        quaternions,
        ((float(axis_a), float(axis_b)),) * count,
        n_theta,
        periodic=True,
    )


def aircraft_conic_bundle_qjet(length, n_slices, n_theta):
    """Tapered, bent, twisted elliptical body represented only by conic jets."""

    count = int(n_slices)
    centers = []
    quaternions = []
    axes = []
    for index in range(count):
        fraction = index / (count - 1)
        envelope = _sin(PI * fraction)
        z_value = float(length) * (fraction - 0.5)
        bend = 0.08 * _sin(TAU * fraction)
        centers.append((bend, 0.0, z_value))
        bend_rotor = _qaxis_angle(
            (0.0, 1.0, 0.0),
            0.08 * TAU / float(length) * _cos(TAU * fraction),
        )
        twist_rotor = _qaxis_angle(
            (0.0, 0.0, 1.0),
            0.35 * (fraction - 0.5),
        )
        quaternions.append(_qnormalize(_qmul(bend_rotor, twist_rotor)))
        axes.append((0.16 + 0.92 * envelope, 0.10 + 0.34 * envelope))
    return ConicPencilSurfaceQJet(
        centers,
        quaternions,
        axes,
        n_theta,
    )


__all__ = [
    "ConicPencilSliceQJet",
    "ConicPencilSurfaceEvaluation",
    "ConicPencilSurfaceQJet",
    "GeneratedConicSurfaceNodes",
    "aircraft_conic_bundle_qjet",
    "bent_conic_tube_qjet",
    "conic_pencil_certificate",
    "straight_conic_tube_qjet",
    "tapered_conic_tube_qjet",
    "toroidal_conic_bundle_qjet",
    "twisted_ellipse_tube_qjet",
]
