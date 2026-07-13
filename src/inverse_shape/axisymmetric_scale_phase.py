"""Exact angular reduction for arbitrary nonperiodic surfaces of revolution.

For a meridian ``(r(u), z(u))`` and equispaced azimuths, every cross-ring
inverse-square kernel has a closed Fourier series controlled by the
hyperbolic distance between the two meridian points. The finite angular grid
is recovered exactly by summing all alias rungs in closed form. A fixed-rank
nested interpolation pass applies the remaining one-dimensional mode
operators without a dense surface matrix.
"""

from inverse_shape.quadrature import (
    TAU,
    BorrowComputeRepayLedger,
    _abs,
    _clean_scalar,
    _fft_precise,
    _finite,
    _ifft_precise,
    _is_power_of_two,
    _sqrt,
)
from inverse_shape.scale_phase_cauchy import (
    ExactCauchyCrossBlock,
    StaticCauchyCrossBlock,
    _ScaleRange,
    _cardinal_values,
    _chebyshev_grid,
    _interpolation_nodes,
    _interpolation_tail,
    _vector,
)


_COMPILE_VISIT_FACTOR = 64
_STATIC_BLOCK_FACTOR = 16
_EXACT_PAIR_FACTOR = 64


def _meridian_point(value):
    point = tuple(float(component) for component in value)
    if len(point) != 2:
        raise ValueError("a meridian evaluator must return (radius, height)")
    if any(not _finite(component) for component in point):
        raise ValueError("meridian values must be finite")
    if point[0] <= 0.0:
        raise ValueError("axisymmetric radius must remain positive")
    return point


def _hyperbolic_data(left, right):
    radius_left, height_left = left
    radius_right, height_right = right
    radial_delta = radius_left - radius_right
    height_delta = height_left - height_right
    scaled = _sqrt(
        (radial_delta * radial_delta + height_delta * height_delta)
        / (4.0 * radius_left * radius_right)
    )
    if scaled <= 1.0e-15:
        raise ValueError("distinct meridian samples generate coincident rings")
    root = _sqrt(1.0 + scaled * scaled)
    half_decay = 1.0 / (root + scaled)
    decay = half_decay * half_decay
    one_minus_decay = 2.0 * scaled * half_decay
    denominator = 4.0 * radius_left * radius_right * scaled * root
    return decay, one_minus_decay, denominator


def _power_two_geometric_sum(value, count):
    power = float(value)
    total = 1.0
    length = 1
    while length < count:
        total *= 1.0 + power
        power *= power
        length *= 2
    return power, total


def _aliased_mode_coefficient(left, right, mode, n_theta):
    decay, one_minus_decay, denominator = _hyperbolic_data(left, right)
    phase_count = int(n_theta)
    mode_index = abs(int(mode)) % phase_count
    mode_value = min(mode_index, phase_count - mode_index)
    cycle_decay, geometric_sum = _power_two_geometric_sum(
        decay,
        phase_count,
    )
    alias_denominator = one_minus_decay * geometric_sum
    if alias_denominator <= 0.0:
        raise ValueError("unresolved coincident cross-ring alias channel")
    if mode_value == 0:
        numerator = 1.0 + cycle_decay
    else:
        numerator = decay**mode_value + decay ** (
            phase_count - mode_value
        )
    return numerator / (denominator * alias_denominator)


def _jet4(value, name):
    result = tuple(float(component) for component in value)
    if len(result) != 4 or any(not _finite(component) for component in result):
        raise ValueError(f"{name} must contain four finite jet coefficients")
    return result


def _solve_four(matrix, right_hand_side):
    rows = [
        [float(value) for value in row] + [float(right_hand_side[index])]
        for index, row in enumerate(matrix)
    ]
    for pivot in range(4):
        selected = max(range(pivot, 4), key=lambda row: _abs(rows[row][pivot]))
        if _abs(rows[selected][pivot]) <= 1.0e-30:
            raise ValueError("singular endpoint-jet interpolation system")
        rows[pivot], rows[selected] = rows[selected], rows[pivot]
        scale = rows[pivot][pivot]
        for column in range(pivot, 5):
            rows[pivot][column] /= scale
        for row in range(4):
            if row == pivot:
                continue
            multiplier = rows[row][pivot]
            for column in range(pivot, 5):
                rows[row][column] -= multiplier * rows[pivot][column]
    return tuple(rows[index][4] for index in range(4))


def _septic_hermite_coefficients(left, right, step):
    h = float(step)
    coefficients = [
        left[0],
        h * left[1],
        0.5 * h * h * left[2],
        h**3 * left[3] / 6.0,
    ]
    right_targets = (
        right[0] - sum(coefficients),
        h * right[1]
        - sum(index * coefficients[index] for index in range(1, 4)),
        h * h * right[2]
        - sum(
            index * (index - 1) * coefficients[index]
            for index in range(2, 4)
        ),
        h**3 * right[3]
        - sum(
            index
            * (index - 1)
            * (index - 2)
            * coefficients[index]
            for index in range(3, 4)
        ),
    )
    matrix = tuple(
        tuple(
            1.0
            if derivative == 0
            else _falling_factorial(degree, derivative)
            for degree in range(4, 8)
        )
        for derivative in range(4)
    )
    coefficients.extend(_solve_four(matrix, right_targets))
    return tuple(coefficients)


def _falling_factorial(value, order):
    result = 1.0
    for offset in range(order):
        result *= value - offset
    return result


class MeridianThreeJetSpline:
    """Piecewise septic meridian generated from endpoint three-jets."""

    def __init__(self, coordinates, radius_jets, height_jets):
        self.coordinates = tuple(float(value) for value in coordinates)
        self.radius_jets = tuple(
            _jet4(value, "radius jet") for value in radius_jets
        )
        self.height_jets = tuple(
            _jet4(value, "height jet") for value in height_jets
        )
        count = len(self.coordinates)
        if count < 2:
            raise ValueError("a meridian spline requires at least two nodes")
        if len(self.radius_jets) != count or len(self.height_jets) != count:
            raise ValueError("one radius and height jet is required per node")
        if any(
            self.coordinates[index + 1] <= self.coordinates[index]
            for index in range(count - 1)
        ):
            raise ValueError("meridian coordinates must be strictly increasing")
        self.radius_coefficients = []
        self.height_coefficients = []
        for index in range(count - 1):
            step = self.coordinates[index + 1] - self.coordinates[index]
            self.radius_coefficients.append(
                _septic_hermite_coefficients(
                    self.radius_jets[index],
                    self.radius_jets[index + 1],
                    step,
                )
            )
            self.height_coefficients.append(
                _septic_hermite_coefficients(
                    self.height_jets[index],
                    self.height_jets[index + 1],
                    step,
                )
            )
        self.radius_coefficients = tuple(self.radius_coefficients)
        self.height_coefficients = tuple(self.height_coefficients)

    def _interval(self, coordinate):
        value = float(coordinate)
        if value <= self.coordinates[0]:
            return 0, 0.0
        if value >= self.coordinates[-1]:
            return len(self.coordinates) - 2, 1.0
        low = 0
        high = len(self.coordinates) - 1
        while high - low > 1:
            middle = (low + high) // 2
            if value < self.coordinates[middle]:
                high = middle
            else:
                low = middle
        step = self.coordinates[low + 1] - self.coordinates[low]
        return low, (value - self.coordinates[low]) / step

    @staticmethod
    def _evaluate(coefficients, parameter):
        result = 0.0
        for coefficient in reversed(coefficients):
            result = result * parameter + coefficient
        return result

    @staticmethod
    def _evaluate_derivative(coefficients, parameter, order):
        result = 0.0
        for degree in range(len(coefficients) - 1, order - 1, -1):
            factor = 1.0
            for offset in range(order):
                factor *= degree - offset
            result = result * parameter + factor * coefficients[degree]
        return result

    def __call__(self, coordinate):
        interval, parameter = self._interval(coordinate)
        radius = self._evaluate(
            self.radius_coefficients[interval],
            parameter,
        )
        height = self._evaluate(
            self.height_coefficients[interval],
            parameter,
        )
        return _meridian_point((radius, height))

    def evaluate_jet(self, coordinate):
        interval, parameter = self._interval(coordinate)
        step = self.coordinates[interval + 1] - self.coordinates[interval]
        radius = tuple(
            self._evaluate_derivative(
                self.radius_coefficients[interval],
                parameter,
                order,
            )
            / step**order
            for order in range(4)
        )
        height = tuple(
            self._evaluate_derivative(
                self.height_coefficients[interval],
                parameter,
                order,
            )
            / step**order
            for order in range(4)
        )
        return radius, height

    def stats(self):
        return {
            "nodes": len(self.coordinates),
            "stored_geometry_scalars": 16 * (len(self.coordinates) - 1),
            "polynomial_degree": 7,
            "source_jet_order": 3,
            "storage_complexity": "O(n_scale)",
            "stored_dense_matrix": False,
        }


class StaticAxisymmetricModePlan:
    """Nested fixed-rank transform for all angular cross-ring modes."""

    def __init__(
        self,
        coordinates,
        meridian,
        n_theta,
        tolerance=2.0e-14,
        expansion_order=32,
        leaf_size=8,
    ):
        self.coordinates = tuple(float(value) for value in coordinates)
        self.n = len(self.coordinates)
        self.meridian = meridian
        self.n_theta = int(n_theta)
        self.tolerance = float(tolerance)
        self.expansion_order = int(expansion_order)
        self.leaf_size = int(leaf_size)
        if self.n < 2:
            raise ValueError("at least two meridian coordinates are required")
        if any(not _finite(value) for value in self.coordinates):
            raise ValueError("meridian coordinates must be finite")
        if any(
            self.coordinates[index + 1] <= self.coordinates[index]
            for index in range(self.n - 1)
        ):
            raise ValueError("meridian coordinates must be strictly increasing")
        if self.n_theta < 4 or not _is_power_of_two(self.n_theta):
            raise ValueError("n_theta must be a radix-two size of at least four")
        if self.expansion_order < 2 or self.leaf_size < 2:
            raise ValueError("expansion_order and leaf_size are too small")
        self.samples = tuple(
            _meridian_point(meridian(value)) for value in self.coordinates
        )
        self.chebyshev_nodes, self.barycentric_weights = _chebyshev_grid(
            self.expansion_order
        )
        self.root = _ScaleRange(
            0,
            self.n,
            self.coordinates,
            self.leaf_size,
        )
        self.compile_visit_budget = _COMPILE_VISIT_FACTOR * self.n
        self.static_block_budget = _STATIC_BLOCK_FACTOR * self.n
        self.exact_pair_budget = _EXACT_PAIR_FACTOR * self.n
        self.compile_pair_visits = 0
        self.preorder = []
        self.postorder = []
        self.leaves = []
        self._index_tree(self.root)
        self.preorder = tuple(self.preorder)
        self.postorder = tuple(self.postorder)
        self.leaves = tuple(self.leaves)
        self.compressed_blocks = []
        self.exact_blocks = []
        self._compile_self(self.root)
        self.compressed_blocks = tuple(self.compressed_blocks)
        self.exact_blocks = tuple(self.exact_blocks)
        self.compiled_exact_pairs = sum(
            block.pair_count for block in self.exact_blocks
        )
        self.compiled_block_count = (
            len(self.compressed_blocks) + len(self.exact_blocks)
        )
        if self.compiled_block_count > self.static_block_budget:
            raise RuntimeError(
                "axisymmetric plan exceeded its linear block budget; "
                "subdivide the generating chart"
            )
        if self.compiled_exact_pairs > self.exact_pair_budget:
            raise RuntimeError(
                "axisymmetric plan exceeded its linear exact-pair budget; "
                "subdivide the generating chart"
            )
        represented = sum(
            block.pair_count for block in self.compressed_blocks
        ) + sum(block.pair_count for block in self.exact_blocks)
        expected = self.n * (self.n - 1) // 2
        if represented != expected:
            raise RuntimeError("axisymmetric plan failed its pair partition")
        self.pair_partition_residual = represented - expected
        self._compile_static_transforms()
        self.last_apply_stats = {}

    def _index_tree(self, node):
        self.preorder.append(node)
        if node.is_leaf:
            self.leaves.append(node)
        else:
            for child in node.children:
                self._index_tree(child)
        self.postorder.append(node)

    def _compile_self(self, node):
        if node.is_leaf:
            if node.n_nodes > 1:
                self.exact_blocks.append(ExactCauchyCrossBlock(node, node))
            return
        left, right = node.children
        self._compile_self(left)
        self._compile_pair(left, right)
        self._compile_self(right)

    def _compile_pair(self, left, right):
        self.compile_pair_visits += 1
        if self.compile_pair_visits > self.compile_visit_budget:
            raise RuntimeError(
                "axisymmetric compilation exceeded its linear visit budget; "
                "no quadratic fallback is permitted"
            )
        tail = _interpolation_tail(left, right, self.expansion_order)
        if tail <= self.tolerance:
            self.compressed_blocks.append(
                StaticCauchyCrossBlock(
                    left,
                    right,
                    self.expansion_order,
                    tail,
                )
            )
            return
        if left.is_leaf and right.is_leaf:
            self.exact_blocks.append(ExactCauchyCrossBlock(left, right))
            return
        split_left = not left.is_leaf and (
            right.is_leaf
            or left.diameter > right.diameter
            or (
                left.diameter == right.diameter
                and left.n_nodes >= right.n_nodes
            )
        )
        if split_left:
            for child in left.children:
                self._compile_pair(child, right)
            return
        for child in right.children:
            self._compile_pair(left, child)

    def _compile_static_transforms(self):
        self._interpolation_coordinate_cache = {
            node: _interpolation_nodes(node, self.chebyshev_nodes)
            for node in self.preorder
        }
        self._interpolation_geometry_cache = {
            node: tuple(
                _meridian_point(self.meridian(value))
                for value in self._interpolation_coordinate_cache[node]
            )
            for node in self.preorder
        }
        self._transfer_cache = {
            (node, child): tuple(
                _cardinal_values(
                    value,
                    node,
                    self.chebyshev_nodes,
                    self.barycentric_weights,
                )
                for value in self._interpolation_coordinate_cache[child]
            )
            for node in self.preorder
            if not node.is_leaf
            for child in node.children
        }
        self._leaf_basis_cache = {
            node: tuple(
                _cardinal_values(
                    self.coordinates[index],
                    node,
                    self.chebyshev_nodes,
                    self.barycentric_weights,
                )
                for index in range(node.first, node.last)
            )
            for node in self.leaves
        }
        self._mode_kernel_cache = {}
        for mode in range(self.n_theta // 2 + 1):
            for block in self.compressed_blocks:
                left_geometry = self._interpolation_geometry_cache[block.left]
                right_geometry = self._interpolation_geometry_cache[block.right]
                self._mode_kernel_cache[(mode, block)] = tuple(
                    tuple(
                        _aliased_mode_coefficient(
                            left,
                            right,
                            mode,
                            self.n_theta,
                        )
                        for right in right_geometry
                    )
                    for left in left_geometry
                )

    def _upward_moments(self, row):
        moments = {}
        work = 0
        for node in self.postorder:
            if node.is_leaf:
                values = [0.0 + 0.0j for _ in range(self.expansion_order)]
                for local, index in enumerate(range(node.first, node.last)):
                    for coefficient, basis in enumerate(
                        self._leaf_basis_cache[node][local]
                    ):
                        values[coefficient] += basis * row[index]
                        work += 1
                moments[node] = values
                continue
            values = [0.0 + 0.0j for _ in range(self.expansion_order)]
            for child in node.children:
                rows = self._transfer_cache[(node, child)]
                for child_index, interpolation_row in enumerate(rows):
                    child_value = moments[child][child_index]
                    for parent_index, basis in enumerate(interpolation_row):
                        values[parent_index] += basis * child_value
                        work += 1
            moments[node] = values
        return moments, work

    def apply_mode(self, values, mode):
        row = _vector(values, self.n, "values")
        mode_value = min(
            abs(int(mode)) % self.n_theta,
            self.n_theta - abs(int(mode)) % self.n_theta,
        )
        moments, upward_work = self._upward_moments(row)
        local_values = {
            node: [0.0 + 0.0j for _ in range(self.expansion_order)]
            for node in self.preorder
        }
        interaction_work = 0
        for block in self.compressed_blocks:
            tile = self._mode_kernel_cache[(mode_value, block)]
            for left_index, kernel_row in enumerate(tile):
                for right_index, kernel in enumerate(kernel_row):
                    local_values[block.left][left_index] += (
                        kernel * moments[block.right][right_index]
                    )
                    local_values[block.right][right_index] += (
                        kernel * moments[block.left][left_index]
                    )
                    interaction_work += 2
        downward_work = 0
        for node in self.preorder:
            if node.is_leaf:
                continue
            for child in node.children:
                for child_index, interpolation_row in enumerate(
                    self._transfer_cache[(node, child)]
                ):
                    for parent_index, basis in enumerate(interpolation_row):
                        local_values[child][child_index] += (
                            basis * local_values[node][parent_index]
                        )
                        downward_work += 1
        output = [0.0 + 0.0j for _ in range(self.n)]
        compensation = [0.0 + 0.0j for _ in range(self.n)]

        def accumulate(index, contribution):
            corrected = contribution - compensation[index]
            updated = output[index] + corrected
            compensation[index] = (updated - output[index]) - corrected
            output[index] = updated

        leaf_work = 0
        for node in self.leaves:
            for local, index in enumerate(range(node.first, node.last)):
                contribution = sum(
                    basis * local_values[node][coefficient]
                    for coefficient, basis in enumerate(
                        self._leaf_basis_cache[node][local]
                    )
                )
                leaf_work += self.expansion_order
                accumulate(index, contribution)
        exact_pairs = 0
        for block in self.exact_blocks:
            same = block.left is block.right
            for left in range(block.left.first, block.left.last):
                right_start = left + 1 if same else block.right.first
                for right in range(right_start, block.right.last):
                    kernel = _aliased_mode_coefficient(
                        self.samples[left],
                        self.samples[right],
                        mode_value,
                        self.n_theta,
                    )
                    accumulate(left, kernel * row[right])
                    accumulate(right, kernel * row[left])
                    exact_pairs += 1
        self.last_apply_stats = {
            "mode": mode_value,
            "upward_work": upward_work,
            "interaction_work": interaction_work,
            "downward_work": downward_work,
            "leaf_work": leaf_work,
            "exact_pairs": exact_pairs,
            "stored_dense_matrix": False,
        }
        return tuple(_clean_scalar(value) for value in output)

    def stats(self):
        compressed_pairs = sum(
            block.pair_count for block in self.compressed_blocks
        )
        exact_pairs = sum(block.pair_count for block in self.exact_blocks)
        result = {
            "n_scale": self.n,
            "n_theta": self.n_theta,
            "expansion_order": self.expansion_order,
            "compressed_blocks": len(self.compressed_blocks),
            "exact_blocks": len(self.exact_blocks),
            "compressed_pair_fraction": compressed_pairs
            / max(compressed_pairs + exact_pairs, 1),
            "pair_partition_residual": self.pair_partition_residual,
            "stored_mode_tiles": len(self._mode_kernel_cache),
            "stored_mode_tile_entries": (
                len(self._mode_kernel_cache)
                * self.expansion_order
                * self.expansion_order
            ),
            "compile_pair_visits": self.compile_pair_visits,
            "compile_visit_budget": self.compile_visit_budget,
            "compiled_block_count": self.compiled_block_count,
            "static_block_budget": self.static_block_budget,
            "compiled_exact_pairs": self.compiled_exact_pairs,
            "exact_pair_budget": self.exact_pair_budget,
            "stored_dense_matrix": False,
            "quadratic_fallback": False,
            "reference_oracle_in_production_object": False,
            "compile_complexity": "O(p^2 N) with fixed p and hard budget",
            "apply_complexity": "O(p^2 n_scale) per angular mode",
            "storage_complexity": "O(p^2 n_scale n_theta) with fixed p",
        }
        result.update(self.last_apply_stats)
        return result


class AxisymmetricScalePhaseEvaluation:
    def __init__(self, values, ledger, stats):
        self.values = values
        self.ledger = ledger
        self.stats = stats


class AxisymmetricScalePhaseQJet:
    """Finite-angle inverse-square graph on a general surface of revolution."""

    def __init__(
        self,
        coordinates,
        meridian,
        n_theta,
        meridional_weights,
        tolerance=2.0e-14,
        expansion_order=32,
        leaf_size=8,
        normalization=1.0,
    ):
        self.coordinates = tuple(float(value) for value in coordinates)
        self.n_scale = len(self.coordinates)
        self.n_theta = int(n_theta)
        self.meridional_weights = tuple(
            float(value) for value in meridional_weights
        )
        if len(self.meridional_weights) != self.n_scale:
            raise ValueError("one meridional weight is required per ring")
        if any(
            value <= 0.0 or not _finite(value)
            for value in self.meridional_weights
        ):
            raise ValueError("meridional weights must be positive and finite")
        self.normalization = float(normalization)
        self.theta_step = TAU / self.n_theta
        self.plan = StaticAxisymmetricModePlan(
            self.coordinates,
            meridian,
            self.n_theta,
            tolerance,
            expansion_order,
            leaf_size,
        )
        cross = self.plan.apply_mode(self.meridional_weights, 0)
        self.cross_row_sum = tuple(TAU * complex(value) for value in cross)
        self.last_apply_stats = {}

    @property
    def n_nodes(self):
        return self.n_scale * self.n_theta

    def _rows(self, values):
        rows = tuple(tuple(complex(value) for value in row) for row in values)
        if len(rows) != self.n_scale:
            raise ValueError("values must contain one row per ring")
        if any(len(row) != self.n_theta for row in rows):
            raise ValueError("each ring must contain n_theta values")
        return rows

    def apply(self, values):
        rows = self._rows(values)
        transformed = tuple(tuple(_fft_precise(row)) for row in rows)
        output_modes = [
            [0.0 + 0.0j for _ in range(self.n_theta)]
            for _ in range(self.n_scale)
        ]
        for angular_index in range(self.n_theta):
            mode = min(angular_index, self.n_theta - angular_index)
            source = tuple(
                self.meridional_weights[scale]
                * transformed[scale][angular_index]
                for scale in range(self.n_scale)
            )
            potential = self.plan.apply_mode(source, mode)
            cycle_eigenvalue = mode * (self.n_theta - mode) / 2.0
            for scale in range(self.n_scale):
                radius = self.plan.samples[scale][0]
                same_scale = (
                    self.meridional_weights[scale]
                    * self.theta_step
                    * cycle_eigenvalue
                    / (radius * radius)
                )
                output_modes[scale][angular_index] = self.normalization * (
                    (self.cross_row_sum[scale] + same_scale)
                    * transformed[scale][angular_index]
                    - TAU * complex(potential[scale])
                )
        result = tuple(
            tuple(_clean_scalar(value) for value in _ifft_precise(row))
            for row in output_modes
        )
        self.last_apply_stats = {
            "method": "axisymmetric_hyperbolic_alias_qjet",
            "angular_fft_rows": 2 * self.n_scale,
            "stored_dense_matrix": False,
        }
        return result

    def constant_residual(self):
        constant = tuple(
            (1.0,) * self.n_theta for _ in range(self.n_scale)
        )
        applied = self.apply(constant)
        return max(_abs(complex(value)) for row in applied for value in row)

    def weighted_inner(self, left, right):
        left_rows = self._rows(left)
        right_rows = self._rows(right)
        total = 0.0 + 0.0j
        compensation = 0.0 + 0.0j
        for scale in range(self.n_scale):
            weight = self.meridional_weights[scale] * self.theta_step
            for phase in range(self.n_theta):
                contribution = (
                    weight
                    * left_rows[scale][phase].conjugate()
                    * right_rows[scale][phase]
                )
                corrected = contribution - compensation
                updated = total + corrected
                compensation = (updated - total) - corrected
                total = updated
        return _clean_scalar(total)

    def stats(self):
        result = {
            "n_scale": self.n_scale,
            "n_theta": self.n_theta,
            "n_nodes": self.n_nodes,
            "angular_alias_repayment": "closed form, all alias rungs",
            "same_ring_channel": "exact finite-cycle spectrum",
            "mode_plan": self.plan.stats(),
            "stored_dense_distance_matrix": False,
            "stored_dense_operator_matrix": False,
            "quadratic_fallback": False,
            "reference_oracle_in_production_object": False,
            "apply_complexity": "O(p^2 N + N log n_theta) with fixed p",
            "storage_complexity": "O(p^2 N) with fixed p",
        }
        if hasattr(self, "meridian_three_jet_spline"):
            result["meridian_geometry"] = (
                self.meridian_three_jet_spline.stats()
            )
        result.update(self.last_apply_stats)
        return result

    def evaluate(self, values):
        result = self.apply(values)
        residual = self.constant_residual()
        ledger = BorrowComputeRepayLedger(
            borrowed=(
                "surface-of-revolution meridian",
                "hyperbolic upper-half-plane distance",
                "fixed nested meridian transforms",
            ),
            computed=(
                "angular QJet FFT",
                "fixed-rank meridian mode transforms",
            ),
            repaid=(
                "all finite-angle alias rungs in closed form",
                "exact neighboring meridian blocks",
                "same-ring finite-cycle spectrum",
                "row-sum constant channel",
            ),
            residuals=(("constant_residual", residual),),
            residual_norm=residual,
            status="borrowed_repaid",
            notes=(
                "No pair table is stored. Arbitrary nonperiodic smooth "
                "meridians are accepted; axis crossings require endpoint "
                "channels."
            ),
        )
        return AxisymmetricScalePhaseEvaluation(result, ledger, self.stats())


def axisymmetric_qjet_from_three_jets(
    coordinates,
    radius_jets,
    height_jets,
    n_theta,
    meridional_weights,
    **options,
):
    spline = MeridianThreeJetSpline(
        coordinates,
        radius_jets,
        height_jets,
    )
    qjet = AxisymmetricScalePhaseQJet(
        coordinates,
        spline,
        n_theta,
        meridional_weights,
        **options,
    )
    qjet.meridian_three_jet_spline = spline
    return qjet


__all__ = [
    "AxisymmetricScalePhaseEvaluation",
    "AxisymmetricScalePhaseQJet",
    "MeridianThreeJetSpline",
    "StaticAxisymmetricModePlan",
    "axisymmetric_qjet_from_three_jets",
]
