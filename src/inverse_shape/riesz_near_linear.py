"""Hard-no-quadratic Riesz QJet for arbitrary points in three dimensions.

The operator is

    (Q_p f)_i = sum_{j != i} w_j (f_i-f_j) / |X_i-X_j|**p.

A fair-split source tree partitions the source set for each target. A
separated node is represented by a fixed-order Gegenbauer expansion in
normalized Cartesian moments. The directed treecode is averaged with its
exact algebraic transpose, so the approximate kernel remains symmetric. Only
unresolved leaves are evaluated point by point. Target plans are streamed;
there is no rejected-block pair stream, adaptive numerical rank, or dense
distance or operator matrix.

For fixed kernel tolerance, maximum expansion order, leaf size, dimension,
and floating-point coordinate depth, storage is O(N), tree construction is
O(N log^2 N) with the current fair-split sorts, and application is
O(N log N).  Runtime work guards fail closed if an implementation regression
violates the compiled near-linear budget; they never switch to O(N^2).
"""

from inverse_shape.quadrature import (
    BorrowComputeRepayLedger,
    _abs,
    _clean_scalar,
    _finite,
    _sqrt,
)


def _point(value):
    row = tuple(float(component) for component in value)
    if len(row) != 3:
        raise ValueError("each Riesz point must have three coordinates")
    if any(not _finite(component) for component in row):
        raise ValueError("Riesz coordinates must be finite")
    return row


def _vsub(left, right):
    return (
        left[0] - right[0],
        left[1] - right[1],
        left[2] - right[2],
    )


def _vdot(left, right):
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _vnorm(value):
    return _sqrt(max(_vdot(value, value), 0.0))


def _binomial(n, k):
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    value = 1
    for offset in range(1, k + 1):
        value = value * (n - k + offset) // offset
    return value


class NearLinearContractError(RuntimeError):
    """Raised instead of entering an unbudgeted quadratic fallback."""


class RieszMonomialBasis:
    """Total-degree monomials and normalized child-to-parent translations."""

    def __init__(self, maximum_order):
        self.maximum_order = int(maximum_order)
        if self.maximum_order < 0:
            raise ValueError("maximum Riesz order must be nonnegative")
        indices = []
        by_degree = []
        count_through = []
        for degree in range(self.maximum_order + 1):
            rows = []
            for x_order in range(degree + 1):
                for y_order in range(degree - x_order + 1):
                    z_order = degree - x_order - y_order
                    monomial = (x_order, y_order, z_order)
                    indices.append(monomial)
                    rows.append(monomial)
            by_degree.append(tuple(rows))
            count_through.append(len(indices))
        self.indices = tuple(indices)
        self.by_degree = tuple(by_degree)
        self.count_through = tuple(count_through)
        self.lookup = {monomial: index for index, monomial in enumerate(indices)}
        self.degrees = tuple(sum(monomial) for monomial in self.indices)
        degree_ranges = []
        first = 0
        for last in self.count_through:
            degree_ranges.append((first, last))
            first = last
        self.degree_ranges = tuple(degree_ranges)
        linear_shifts = []
        quadratic_shifts = []
        for monomial in self.indices:
            degree = sum(monomial)
            if degree < self.maximum_order:
                linear_shifts.append(
                    tuple(
                        self.lookup[
                            tuple(
                                monomial[coordinate]
                                + (1 if coordinate == axis else 0)
                                for coordinate in range(3)
                            )
                        ]
                        for axis in range(3)
                    )
                )
            else:
                linear_shifts.append(tuple())
            if degree + 2 <= self.maximum_order:
                quadratic_shifts.append(
                    tuple(
                        self.lookup[
                            tuple(
                                monomial[coordinate]
                                + (2 if coordinate == axis else 0)
                                for coordinate in range(3)
                            )
                        ]
                        for axis in range(3)
                    )
                )
            else:
                quadratic_shifts.append(tuple())
        self.linear_shifts = tuple(linear_shifts)
        self.quadratic_shifts = tuple(quadratic_shifts)
        plans = []
        for alpha in self.indices:
            terms = []
            for bx in range(alpha[0] + 1):
                for by in range(alpha[1] + 1):
                    for bz in range(alpha[2] + 1):
                        beta = (bx, by, bz)
                        gamma = (
                            alpha[0] - bx,
                            alpha[1] - by,
                            alpha[2] - bz,
                        )
                        coefficient = (
                            _binomial(alpha[0], bx)
                            * _binomial(alpha[1], by)
                            * _binomial(alpha[2], bz)
                        )
                        terms.append(
                            (
                                self.lookup[beta],
                                gamma,
                                bx + by + bz,
                                coefficient,
                            )
                        )
            plans.append(tuple(terms))
        self.translation_plans = tuple(plans)
        self.translation_term_count = sum(len(plan) for plan in plans)

    def count(self, order=None):
        degree = self.maximum_order if order is None else int(order)
        if degree < 0 or degree > self.maximum_order:
            raise ValueError("requested order is outside the monomial basis")
        return self.count_through[degree]

    def monomials(self, value, order=None):
        degree = self.maximum_order if order is None else int(order)
        count = self.count(degree)
        x_powers = [1.0]
        y_powers = [1.0]
        z_powers = [1.0]
        for _index in range(degree):
            x_powers.append(x_powers[-1] * value[0])
            y_powers.append(y_powers[-1] * value[1])
            z_powers.append(z_powers[-1] * value[2])
        return tuple(
            x_powers[monomial[0]]
            * y_powers[monomial[1]]
            * z_powers[monomial[2]]
            for monomial in self.indices[:count]
        )

    def translate(self, child_moments, child, parent):
        if parent.radius <= 0.0:
            return tuple(child_moments)
        displacement = tuple(
            (child.center[axis] - parent.center[axis]) / parent.radius
            for axis in range(3)
        )
        scale = child.radius / parent.radius
        x_powers = [1.0]
        y_powers = [1.0]
        z_powers = [1.0]
        scale_powers = [1.0]
        for _index in range(self.maximum_order):
            x_powers.append(x_powers[-1] * displacement[0])
            y_powers.append(y_powers[-1] * displacement[1])
            z_powers.append(z_powers[-1] * displacement[2])
            scale_powers.append(scale_powers[-1] * scale)
        output = []
        for plan in self.translation_plans:
            total = 0.0 + 0.0j
            for beta_index, gamma, beta_degree, coefficient in plan:
                total += (
                    coefficient
                    * x_powers[gamma[0]]
                    * y_powers[gamma[1]]
                    * z_powers[gamma[2]]
                    * scale_powers[beta_degree]
                    * child_moments[beta_index]
                )
            output.append(total)
        return tuple(output)


class _RieszTreeNode:
    def __init__(self, order, first, last, points, leaf_size, depth=0):
        self._order = order
        self.first = int(first)
        self.last = int(last)
        self.depth = int(depth)
        self.children = tuple()
        minimum = tuple(
            min(points[order[position]][axis] for position in range(first, last))
            for axis in range(3)
        )
        maximum = tuple(
            max(points[order[position]][axis] for position in range(first, last))
            for axis in range(3)
        )
        self.center = tuple(
            0.5 * (minimum[axis] + maximum[axis]) for axis in range(3)
        )
        self.radius = max(
            _vnorm(_vsub(points[order[position]], self.center))
            for position in range(first, last)
        )
        self.id = -1
        self.weight_moments = tuple()
        self.absolute_weight = 0.0
        if self.n_nodes > leaf_size:
            spans = tuple(maximum[axis] - minimum[axis] for axis in range(3))
            axis = max(range(3), key=lambda item: spans[item])
            ordered = sorted(
                order[first:last],
                key=lambda index: (points[index][axis], index),
            )
            order[first:last] = ordered
            middle = first + self.n_nodes // 2
            self.children = (
                _RieszTreeNode(
                    order,
                    first,
                    middle,
                    points,
                    leaf_size,
                    depth + 1,
                ),
                _RieszTreeNode(
                    order,
                    middle,
                    last,
                    points,
                    leaf_size,
                    depth + 1,
                ),
            )

    @property
    def n_nodes(self):
        return self.last - self.first

    @property
    def is_leaf(self):
        return not self.children

    def indices(self):
        return self._order[self.first : self.last]


class AnalyticRieszBlock:
    """One symmetric low-rank block oriented source-to-target."""

    def __init__(self, source, target, order, relative_tail_bound):
        self.source = source
        self.target = target
        self.order = int(order)
        self.relative_tail_bound = float(relative_tail_bound)

    @property
    def pair_count(self):
        return self.source.n_nodes * self.target.n_nodes


class ExactRieszCrossBlock:
    def __init__(self, left, right):
        self.left = left
        self.right = right

    @property
    def pair_count(self):
        return self.left.n_nodes * self.right.n_nodes


class ExactRieszSelfBlock:
    def __init__(self, node):
        self.node = node

    @property
    def pair_count(self):
        return self.node.n_nodes * (self.node.n_nodes - 1) // 2


class NearLinearRieszEvaluation:
    def __init__(self, values, compression_inf_bound, ledger, stats):
        self.values = tuple(values)
        self.compression_inf_bound = float(compression_inf_bound)
        self.ledger = ledger
        self.stats = dict(stats)


class ProductionRieszQJet:
    """Production symmetric WSPD Riesz graph with fixed-order source jets."""

    def __init__(
        self,
        points,
        weights,
        kernel_power=2.0,
        tolerance=5.0e-13,
        maximum_order=16,
        leaf_size=8,
        work_budget_factor=96,
    ):
        self.points = tuple(_point(value) for value in points)
        self.n = len(self.points)
        if self.n < 2:
            raise ValueError("a Riesz QJet requires at least two nodes")
        if len(set(self.points)) != self.n:
            raise ValueError("distinct Riesz nodes may not coincide")
        self.weights = tuple(float(value) for value in weights)
        if len(self.weights) != self.n:
            raise ValueError("weights must contain one value per Riesz node")
        if any(value <= 0.0 or not _finite(value) for value in self.weights):
            raise ValueError("Riesz weights must be positive and finite")
        self.kernel_power = float(kernel_power)
        self.tolerance = float(tolerance)
        self.maximum_order = int(maximum_order)
        self.leaf_size = int(leaf_size)
        self.work_budget_factor = int(work_budget_factor)
        if self.kernel_power <= 0.0 or not _finite(self.kernel_power):
            raise ValueError("kernel_power must be positive and finite")
        if self.tolerance <= 0.0 or not _finite(self.tolerance):
            raise ValueError("tolerance must be positive and finite")
        if self.maximum_order < 1:
            raise ValueError("maximum_order must be positive")
        if self.leaf_size < 1 or self.work_budget_factor < 1:
            raise ValueError("leaf_size and work_budget_factor must be positive")

        self.basis = RieszMonomialBasis(self.maximum_order)
        self._gegenbauer_at_one = [1.0]
        for degree in range(1, self.maximum_order + 2):
            self._gegenbauer_at_one.append(
                self._gegenbauer_at_one[-1]
                * (self.kernel_power + degree - 1.0)
                / degree
            )
        self._order = list(range(self.n))
        self.root = _RieszTreeNode(
            self._order,
            0,
            self.n,
            self.points,
            self.leaf_size,
        )
        self.nodes = []
        self._collect_postorder(self.root)
        self.nodes = tuple(self.nodes)
        self.maximum_depth = max(node.depth for node in self.nodes)
        self.analytic_blocks = []
        self.exact_cross_blocks = []
        self.exact_self_blocks = []
        self._compile_self(self.root)
        self.analytic_blocks = tuple(self.analytic_blocks)
        self.exact_cross_blocks = tuple(self.exact_cross_blocks)
        self.exact_self_blocks = tuple(self.exact_self_blocks)
        represented = (
            sum(block.pair_count for block in self.analytic_blocks)
            + sum(block.pair_count for block in self.exact_cross_blocks)
            + sum(block.pair_count for block in self.exact_self_blocks)
        )
        self.pair_count = self.n * (self.n - 1) // 2
        self.partition_residual = represented - self.pair_count
        if self.partition_residual:
            raise RuntimeError("Riesz tree failed its exact unordered-pair partition")
        self.exact_pair_count = (
            sum(block.pair_count for block in self.exact_cross_blocks)
            + sum(block.pair_count for block in self.exact_self_blocks)
        )
        self.analytic_pair_count = sum(
            block.pair_count for block in self.analytic_blocks
        )
        self.analytic_apply_units = sum(
            self.basis.count(block.order)
            * (block.source.n_nodes + block.target.n_nodes)
            for block in self.analytic_blocks
        )
        logarithmic_size = max(1, (self.n + 1).bit_length())
        self.direct_pair_budget = (
            self.work_budget_factor
            * self.n
            * self.leaf_size
            * (self.maximum_depth + 1)
        )
        self.analytic_apply_budget = (
            self.work_budget_factor
            * self.n
            * logarithmic_size
            * self.basis.count()
        )
        if self.exact_pair_count > self.direct_pair_budget:
            raise NearLinearContractError(
                "near-field pair budget exceeded; quadratic fallback is forbidden"
            )
        if self.analytic_apply_units > self.analytic_apply_budget:
            raise NearLinearContractError(
                "analytic block work budget exceeded; quadratic fallback is forbidden"
            )
        self._build_static_weight_moments()
        self.row_kernel_error = self._build_row_kernel_error()
        self.last_apply_stats = {}

    @classmethod
    def from_triangle_mesh(cls, vertices, triangles, **options):
        from inverse_shape.arbitrary_surface import triangle_lumped_vertex_weights

        points = tuple(_point(value) for value in vertices)
        weights = triangle_lumped_vertex_weights(points, triangles)
        return cls(points, weights, **options)

    def _collect_postorder(self, node):
        for child in node.children:
            self._collect_postorder(child)
        node.id = len(self.nodes)
        self.nodes.append(node)

    def _tail_relative(self, order, rho):
        ratio_value = float(rho)
        if ratio_value < 0.0 or ratio_value >= 1.0:
            return float("inf")
        first_degree = int(order) + 1
        first = (
            self._gegenbauer_at_one[first_degree]
            * ratio_value**first_degree
        )
        ratio_bound = ratio_value * max(
            1.0,
            (self.kernel_power + first_degree) / (first_degree + 1.0),
        )
        if ratio_bound >= 1.0:
            return float("inf")
        absolute_series_tail = first / (1.0 - ratio_bound)
        return absolute_series_tail * (1.0 + ratio_value) ** self.kernel_power

    def _required_order(self, source, target):
        center_distance = _vnorm(_vsub(target.center, source.center))
        minimum_target_distance = center_distance - target.radius
        if minimum_target_distance <= 0.0:
            return None
        if source.radius == 0.0:
            return 0, 0.0
        rho = source.radius / minimum_target_distance
        if rho >= 1.0:
            return None
        for order in range(self.maximum_order + 1):
            bound = self._tail_relative(order, rho)
            if bound <= self.tolerance:
                return order, bound
        return None

    def _analytic_candidate(self, left, right):
        candidates = []
        for source, target in ((left, right), (right, left)):
            result = self._required_order(source, target)
            if result is None:
                continue
            order, bound = result
            work = self.basis.count(order) * (source.n_nodes + target.n_nodes)
            candidates.append((work, order, source.first, source, target, bound))
        if not candidates:
            return None
        _work, order, _first, source, target, bound = min(candidates)
        return AnalyticRieszBlock(source, target, order, bound)

    def _compile_self(self, node):
        if node.is_leaf:
            if node.n_nodes > 1:
                self.exact_self_blocks.append(ExactRieszSelfBlock(node))
            return
        left, right = node.children
        self._compile_self(left)
        self._compile_pair(left, right)
        self._compile_self(right)

    def _compile_pair(self, left, right):
        candidate = self._analytic_candidate(left, right)
        if candidate is not None:
            self.analytic_blocks.append(candidate)
            return
        if left.is_leaf and right.is_leaf:
            self.exact_cross_blocks.append(ExactRieszCrossBlock(left, right))
            return
        split_left = not left.is_leaf and (
            right.is_leaf
            or left.radius > right.radius
            or (
                left.radius == right.radius
                and left.n_nodes >= right.n_nodes
            )
        )
        if split_left:
            for child in left.children:
                self._compile_pair(child, right)
            return
        for child in right.children:
            self._compile_pair(left, child)

    def _leaf_moments(self, node, residues):
        output = [0.0 + 0.0j for _ in range(self.basis.count())]
        for index in node.indices():
            if node.radius > 0.0:
                normalized = tuple(
                    (self.points[index][axis] - node.center[axis]) / node.radius
                    for axis in range(3)
                )
            else:
                normalized = (0.0, 0.0, 0.0)
            monomials = self.basis.monomials(normalized)
            residue = residues[index]
            for basis_index, monomial in enumerate(monomials):
                output[basis_index] += residue * monomial
        return tuple(output)

    def _build_moments(self, residues):
        moments = [tuple() for _ in self.nodes]
        for node in self.nodes:
            if node.is_leaf:
                moments[node.id] = self._leaf_moments(node, residues)
                continue
            output = [0.0 + 0.0j for _ in range(self.basis.count())]
            for child in node.children:
                translated = self.basis.translate(
                    moments[child.id],
                    child,
                    node,
                )
                for index, value in enumerate(translated):
                    output[index] += value
            moments[node.id] = tuple(output)
        return tuple(moments)

    def _build_static_weight_moments(self):
        residues = tuple(complex(value) for value in self.weights)
        moments = self._build_moments(residues)
        for node in self.nodes:
            node.weight_moments = moments[node.id]
            node.absolute_weight = sum(self.weights[index] for index in node.indices())

    def _expansion_coefficients(self, point, source, order):
        difference = _vsub(point, source.center)
        distance = _vnorm(difference)
        if distance <= source.radius:
            raise RuntimeError("accepted Riesz expansion contains its target")
        rho = source.radius / distance
        direction = tuple(value / distance for value in difference)
        count = self.basis.count(order)
        coefficients = [0.0 for _ in range(count)]
        scale = distance ** (-self.kernel_power)
        coefficients[0] = scale
        if order >= 1:
            degree_one = (
                self.basis.lookup[(1, 0, 0)],
                self.basis.lookup[(0, 1, 0)],
                self.basis.lookup[(0, 0, 1)],
            )
            for axis in range(3):
                coefficients[degree_one[axis]] = (
                    scale * self.kernel_power * rho * direction[axis]
                )
        half_power = 0.5 * self.kernel_power
        for degree in range(2, order + 1):
            linear_scale = 2.0 * (degree + half_power - 1.0) * rho / degree
            quadratic_scale = (
                (degree + self.kernel_power - 2.0)
                * rho
                * rho
                / degree
            )
            previous_first, previous_last = self.basis.degree_ranges[degree - 1]
            for source_index in range(previous_first, previous_last):
                coefficient = coefficients[source_index]
                targets = self.basis.linear_shifts[source_index]
                for axis in range(3):
                    coefficients[targets[axis]] += (
                        linear_scale * direction[axis] * coefficient
                    )
            earlier_first, earlier_last = self.basis.degree_ranges[degree - 2]
            for source_index in range(earlier_first, earlier_last):
                coefficient = coefficients[source_index]
                targets = self.basis.quadratic_shifts[source_index]
                for axis in range(3):
                    coefficients[targets[axis]] -= (
                        quadratic_scale * coefficient
                    )
        absolute_tail = (
            distance ** (-self.kernel_power)
            * self._tail_relative(order, rho)
            / (1.0 + rho) ** self.kernel_power
        )
        return tuple(coefficients), absolute_tail

    def _normalized_monomials(self, point, node, order):
        if node.radius > 0.0:
            normalized = tuple(
                (point[axis] - node.center[axis]) / node.radius
                for axis in range(3)
            )
        else:
            normalized = (0.0, 0.0, 0.0)
        return self.basis.monomials(normalized, order)

    @staticmethod
    def _dot(left, right):
        return sum(
            left[index] * right[index]
            for index in range(min(len(left), len(right)))
        )

    def _apply_potential(self, residues, moments):
        output = [0.0 + 0.0j for _ in range(self.n)]
        for block in self.analytic_blocks:
            count = self.basis.count(block.order)
            source_moments = moments[block.source.id][:count]
            adjoint = [0.0 + 0.0j for _ in range(count)]
            for target_index in block.target.indices():
                coefficients, _tail = self._expansion_coefficients(
                    self.points[target_index],
                    block.source,
                    block.order,
                )
                output[target_index] += self._dot(
                    coefficients,
                    source_moments,
                )
                residue = residues[target_index]
                for index, coefficient in enumerate(coefficients):
                    adjoint[index] += residue * coefficient
            for source_index in block.source.indices():
                monomials = self._normalized_monomials(
                    self.points[source_index],
                    block.source,
                    block.order,
                )
                output[source_index] += self._dot(monomials, adjoint)
        for block in self.exact_cross_blocks:
            for left in block.left.indices():
                for right in block.right.indices():
                    kernel = self._kernel(left, right)
                    output[left] += kernel * residues[right]
                    output[right] += kernel * residues[left]
        for block in self.exact_self_blocks:
            indices = block.node.indices()
            for left_offset, left in enumerate(indices):
                for right in indices[left_offset + 1 :]:
                    kernel = self._kernel(left, right)
                    output[left] += kernel * residues[right]
                    output[right] += kernel * residues[left]
        return tuple(output)

    def _apply_graph_centered(self, values, field_moments):
        """Apply graph differences blockwise without subtracting global potentials."""

        output = [0.0 + 0.0j for _ in range(self.n)]
        compensation = [0.0 + 0.0j for _ in range(self.n)]

        def accumulate(index, contribution):
            corrected = contribution - compensation[index]
            updated = output[index] + corrected
            compensation[index] = (updated - output[index]) - corrected
            output[index] = updated

        for block in self.analytic_blocks:
            count = self.basis.count(block.order)
            weight_moments = block.source.weight_moments[:count]
            dynamic_moments = field_moments[block.source.id][:count]
            source_weight = weight_moments[0].real
            source_center = dynamic_moments[0] / source_weight
            shifted_moments = tuple(
                dynamic_moments[index]
                - source_center * weight_moments[index]
                for index in range(count)
            )

            target_indices = block.target.indices()
            target_weight = sum(self.weights[index] for index in target_indices)
            target_center = sum(
                self.weights[index] * values[index] for index in target_indices
            ) / target_weight
            adjoint_weight = [0.0 + 0.0j for _ in range(count)]
            adjoint_field = [0.0 + 0.0j for _ in range(count)]
            for target_index in target_indices:
                coefficients, _tail = self._expansion_coefficients(
                    self.points[target_index],
                    block.source,
                    block.order,
                )
                kernel_weight = self._dot(coefficients, weight_moments)
                centered_field = self._dot(coefficients, shifted_moments)
                accumulate(
                    target_index,
                    (values[target_index] - source_center) * kernel_weight
                    - centered_field,
                )
                weight = self.weights[target_index]
                centered_residue = weight * (
                    values[target_index] - target_center
                )
                for index, coefficient in enumerate(coefficients):
                    adjoint_weight[index] += weight * coefficient
                    adjoint_field[index] += centered_residue * coefficient

            for source_index in block.source.indices():
                monomials = self._normalized_monomials(
                    self.points[source_index],
                    block.source,
                    block.order,
                )
                kernel_weight = self._dot(monomials, adjoint_weight)
                centered_field = self._dot(monomials, adjoint_field)
                accumulate(
                    source_index,
                    (values[source_index] - target_center) * kernel_weight
                    - centered_field,
                )

        for block in self.exact_cross_blocks:
            for left in block.left.indices():
                for right in block.right.indices():
                    kernel = self._kernel(left, right)
                    difference = values[left] - values[right]
                    accumulate(left, self.weights[right] * kernel * difference)
                    accumulate(right, -self.weights[left] * kernel * difference)
        for block in self.exact_self_blocks:
            indices = block.node.indices()
            for left_offset, left in enumerate(indices):
                for right in indices[left_offset + 1 :]:
                    kernel = self._kernel(left, right)
                    difference = values[left] - values[right]
                    accumulate(left, self.weights[right] * kernel * difference)
                    accumulate(right, -self.weights[left] * kernel * difference)
        return tuple(output)

    def _build_row_kernel_error(self):
        output = [0.0 for _ in range(self.n)]
        for block in self.analytic_blocks:
            target_weighted_tail = 0.0
            for target_index in block.target.indices():
                _coefficients, tail = self._expansion_coefficients(
                    self.points[target_index],
                    block.source,
                    block.order,
                )
                output[target_index] += tail * block.source.absolute_weight
                target_weighted_tail += self.weights[target_index] * tail
            for source_index in block.source.indices():
                output[source_index] += target_weighted_tail
        return tuple(output)

    def _kernel(self, left, right):
        difference = _vsub(self.points[left], self.points[right])
        distance_squared = _vdot(difference, difference)
        if distance_squared <= 0.0:
            raise ValueError("distinct Riesz nodes are numerically colliding")
        return distance_squared ** (-0.5 * self.kernel_power)

    def apply(self, values):
        row = tuple(complex(value) for value in values)
        if len(row) != self.n:
            raise ValueError("values must contain one entry per Riesz node")
        if any(not _finite(value.real) or not _finite(value.imag) for value in row):
            raise ValueError("Riesz values must be finite")
        if all(value == row[0] for value in row):
            self.last_apply_stats = {
                "constant_shortcut": True,
                "dynamic_moment_entries": 0,
            }
            return tuple(0.0 for _ in row)
        residues = tuple(
            self.weights[index] * row[index] for index in range(self.n)
        )
        dynamic_moments = self._build_moments(residues)
        graph_values = self._apply_graph_centered(row, dynamic_moments)
        result = tuple(
            _clean_scalar(value) for value in graph_values
        )
        self.last_apply_stats = {
            "constant_shortcut": False,
            "dynamic_moment_entries": len(self.nodes) * self.basis.count(),
        }
        return result

    def apply_fields(self, fields):
        return tuple(self.apply(values) for values in fields)

    def compression_inf_bound(self, values):
        row = tuple(complex(value) for value in values)
        if len(row) != self.n:
            raise ValueError("values must contain one entry per Riesz node")
        maximum = max(_abs(value) for value in row)
        return max(
            (_abs(row[index]) + maximum) * self.row_kernel_error[index]
            for index in range(self.n)
        )

    def constant_residual(self):
        previous_stats = dict(self.last_apply_stats)
        try:
            return max(_abs(value) for value in self.apply((1.0,) * self.n))
        finally:
            self.last_apply_stats = previous_stats

    def stats(self):
        maximum_tail = max(
            (block.relative_tail_bound for block in self.analytic_blocks),
            default=0.0,
        )
        maximum_block_order = max(
            (block.order for block in self.analytic_blocks),
            default=0,
        )
        result = {
            "method": "fixed_order_symmetric_gegenbauer_riesz_wspd",
            "nodes": self.n,
            "kernel_power": self.kernel_power,
            "tolerance": self.tolerance,
            "maximum_order": self.maximum_order,
            "maximum_block_order": maximum_block_order,
            "maximum_block_rank": self.basis.count(maximum_block_order),
            "leaf_size": self.leaf_size,
            "tree_nodes": len(self.nodes),
            "maximum_tree_depth": self.maximum_depth,
            "analytic_blocks": len(self.analytic_blocks),
            "exact_cross_blocks": len(self.exact_cross_blocks),
            "exact_self_blocks": len(self.exact_self_blocks),
            "analytic_pairs": self.analytic_pair_count,
            "near_field_pairs": self.exact_pair_count,
            "analytic_pair_fraction": self.analytic_pair_count / self.pair_count,
            "near_field_pair_fraction": self.exact_pair_count / self.pair_count,
            "maximum_analytic_relative_tail": maximum_tail,
            "pair_partition_residual": self.partition_residual,
            "basis_size": self.basis.count(),
            "translation_terms_per_tree_edge": self.basis.translation_term_count,
            "persistent_moment_entries": len(self.nodes) * self.basis.count(),
            "analytic_apply_units": self.analytic_apply_units,
            "analytic_apply_budget": self.analytic_apply_budget,
            "near_field_pair_budget": self.direct_pair_budget,
            "hard_no_quadratic_contract": True,
            "quadratic_fallback": False,
            "adaptive_rank": 0,
            "rank_growth_with_n": False,
            "temporary_pair_table_entries": 0,
            "stored_dense_distance_matrix": False,
            "stored_dense_operator_matrix": False,
            "compile_complexity": "O(N log^2 N) for fixed order in 3D",
            "apply_complexity": "O(N log N) for fixed order in 3D",
            "storage_complexity": "O(N) for fixed order in 3D",
            "error_certificate": "analytic Gegenbauer tail plus exact near field",
        }
        result.update(self.last_apply_stats)
        return result

    def evaluate(self, values):
        result = self.apply(values)
        compression_bound = self.compression_inf_bound(values)
        constant_residual = self.constant_residual()
        stats = self.stats()
        ledger = BorrowComputeRepayLedger(
            borrowed=(
                "fair-split WSPD pair partition",
                "fixed-order normalized Cartesian moments",
                "Gegenbauer expansion of the Riesz kernel",
            ),
            computed=(
                "symmetric forward/transpose analytic block action",
                "dynamic weighted-field moment tree",
                "blockwise centered graph differences",
            ),
            repaid=(
                "all adjacent leaf interactions exactly",
                "local constant graph channel in every WSPD block",
                "every far block by an analytic tail-certified expansion",
            ),
            residuals=(
                ("compression_inf_bound", compression_bound),
                ("constant_residual", constant_residual),
                ("pair_partition_residual", float(self.partition_residual)),
            ),
            residual_norm=max(
                compression_bound,
                constant_residual,
                _abs(float(self.partition_residual)),
            ),
            status="borrowed_repaid",
            notes=(
                "The backend has no quadratic fallback. Expansion order is "
                "bounded independently of N; near-field leaf work and WSPD "
                "block work are protected by explicit near-linear budgets."
            ),
        )
        return NearLinearRieszEvaluation(
            result,
            compression_bound,
            ledger,
            stats,
        )


class TargetRieszInteraction:
    """One target-to-source-node Gegenbauer interaction."""

    def __init__(self, source, order, relative_tail_bound):
        self.source = source
        self.order = int(order)
        self.relative_tail_bound = float(relative_tail_bound)


class TargetTreeDiagnosticRieszQJet:
    """Diagnostic target treecode retained for implementation comparisons."""

    def __init__(
        self,
        points,
        weights,
        kernel_power=2.0,
        tolerance=5.0e-13,
        maximum_order=16,
        leaf_size=8,
        work_budget_factor=96,
    ):
        self.points = tuple(_point(value) for value in points)
        self.n = len(self.points)
        if self.n < 2:
            raise ValueError("a Riesz QJet requires at least two nodes")
        if len(set(self.points)) != self.n:
            raise ValueError("distinct Riesz nodes may not coincide")
        self.weights = tuple(float(value) for value in weights)
        if len(self.weights) != self.n:
            raise ValueError("weights must contain one value per Riesz node")
        if any(value <= 0.0 or not _finite(value) for value in self.weights):
            raise ValueError("Riesz weights must be positive and finite")
        self.kernel_power = float(kernel_power)
        self.tolerance = float(tolerance)
        self.maximum_order = int(maximum_order)
        self.leaf_size = int(leaf_size)
        self.work_budget_factor = int(work_budget_factor)
        if self.kernel_power <= 0.0 or not _finite(self.kernel_power):
            raise ValueError("kernel_power must be positive and finite")
        if self.tolerance <= 0.0 or not _finite(self.tolerance):
            raise ValueError("tolerance must be positive and finite")
        if self.maximum_order < 1:
            raise ValueError("maximum_order must be positive")
        if self.leaf_size < 1 or self.work_budget_factor < 1:
            raise ValueError("leaf_size and work_budget_factor must be positive")

        self.basis = RieszMonomialBasis(self.maximum_order)
        self._gegenbauer_at_one = [1.0]
        for degree in range(1, self.maximum_order + 2):
            self._gegenbauer_at_one.append(
                self._gegenbauer_at_one[-1]
                * (self.kernel_power + degree - 1.0)
                / degree
            )
        self._order = list(range(self.n))
        self.root = _RieszTreeNode(
            self._order,
            0,
            self.n,
            self.points,
            self.leaf_size,
        )
        self.nodes = []
        self._collect_postorder(self.root)
        self.nodes = tuple(self.nodes)
        self._position = [0 for _ in range(self.n)]
        for position, index in enumerate(self._order):
            self._position[index] = position
        self.maximum_depth = max(node.depth for node in self.nodes)
        logarithmic_size = max(1, (self.n + 1).bit_length())
        self.target_plan_visit_budget = (
            self.work_budget_factor * self.n * logarithmic_size
        )
        self.target_plan_visits = 0
        self.direct_plan_penalty = max(
            16,
            self.basis.count(),
        )
        self.analytic_interaction_count = 0
        self.directed_near_pairs = 0
        self.analytic_forward_units = 0
        self.maximum_analytic_relative_tail = 0.0
        self.maximum_block_order = 0
        adjoint_orders = [-1 for _ in self.nodes]
        for target_index in range(self.n):
            (
                target_analytic,
                target_direct,
                _plan_cost,
                plan_visits,
            ) = self._target_plan(
                target_index,
                self.root,
            )
            self.target_plan_visits += plan_visits
            represented = sum(
                interaction.source.n_nodes for interaction in target_analytic
            ) + len(target_direct)
            if represented != self.n - 1:
                raise RuntimeError(
                    "target treecode failed its exact ordered-pair partition"
                )
            self.analytic_interaction_count += len(target_analytic)
            self.directed_near_pairs += len(target_direct)
            for interaction in target_analytic:
                self.analytic_forward_units += self.basis.count(
                    interaction.order
                )
                self.maximum_analytic_relative_tail = max(
                    self.maximum_analytic_relative_tail,
                    interaction.relative_tail_bound,
                )
                self.maximum_block_order = max(
                    self.maximum_block_order,
                    interaction.order,
                )
                adjoint_orders[interaction.source.id] = max(
                    adjoint_orders[interaction.source.id],
                    interaction.order,
                )
        self.adjoint_orders = tuple(adjoint_orders)
        self.adjoint_evaluation_units = sum(
            node.n_nodes * self.basis.count(order)
            for node, order in zip(self.nodes, self.adjoint_orders, strict=True)
            if order >= 0
        )
        self.direct_pair_budget = (
            self.work_budget_factor
            * self.n
            * self.leaf_size
            * (self.maximum_depth + 1)
        )
        self.analytic_apply_budget = (
            self.work_budget_factor
            * self.n
            * logarithmic_size
            * self.basis.count()
        )
        if self.directed_near_pairs > self.direct_pair_budget:
            raise NearLinearContractError(
                "near-field target budget exceeded; quadratic fallback is forbidden"
            )
        if self.target_plan_visits > self.target_plan_visit_budget:
            raise NearLinearContractError(
                "target traversal budget exceeded; quadratic fallback is forbidden"
            )
        if (
            self.analytic_forward_units + self.adjoint_evaluation_units
            > self.analytic_apply_budget
        ):
            raise NearLinearContractError(
                "analytic target budget exceeded; quadratic fallback is forbidden"
            )
        self._build_static_weight_moments()
        self.row_kernel_error = self._build_row_kernel_error()
        self.weight_potential = self._apply_potential(
            tuple(complex(value) for value in self.weights),
            tuple(node.weight_moments for node in self.nodes),
        )
        self.last_apply_stats = {}

    @classmethod
    def from_triangle_mesh(cls, vertices, triangles, **options):
        from inverse_shape.arbitrary_surface import triangle_lumped_vertex_weights

        points = tuple(_point(value) for value in vertices)
        weights = triangle_lumped_vertex_weights(points, triangles)
        return cls(points, weights, **options)

    def _collect_postorder(self, node):
        for child in node.children:
            self._collect_postorder(child)
        node.id = len(self.nodes)
        self.nodes.append(node)

    def _contains(self, node, point_index):
        position = self._position[point_index]
        return node.first <= position < node.last

    def _tail_relative(self, order, rho):
        ratio_value = float(rho)
        if ratio_value < 0.0 or ratio_value >= 1.0:
            return float("inf")
        first_degree = int(order) + 1
        first = (
            self._gegenbauer_at_one[first_degree]
            * ratio_value**first_degree
        )
        ratio_bound = ratio_value * max(
            1.0,
            (self.kernel_power + first_degree) / (first_degree + 1.0),
        )
        if ratio_bound >= 1.0:
            return float("inf")
        return (
            first
            / (1.0 - ratio_bound)
            * (1.0 + ratio_value) ** self.kernel_power
        )

    def _required_order_point(self, source, target_point):
        distance = _vnorm(_vsub(target_point, source.center))
        if distance <= 0.0:
            return None
        if source.radius == 0.0:
            return 0, 0.0
        rho = source.radius / distance
        if rho >= 1.0:
            return None
        for order in range(self.maximum_order + 1):
            bound = self._tail_relative(order, rho)
            if bound <= self.tolerance:
                return order, bound
        return None

    def _target_plan(self, target_index, node):
        if self._contains(node, target_index):
            if node.is_leaf:
                direct = tuple(
                    index for index in node.indices() if index != target_index
                )
                return (
                    tuple(),
                    direct,
                    self.direct_plan_penalty * len(direct),
                    1,
                )
            analytic = []
            direct = []
            cost = 0
            visits = 1
            for child in node.children:
                (
                    child_analytic,
                    child_direct,
                    child_cost,
                    child_visits,
                ) = self._target_plan(target_index, child)
                analytic.extend(child_analytic)
                direct.extend(child_direct)
                cost += child_cost
                visits += child_visits
            return tuple(analytic), tuple(direct), cost, visits
        result = self._required_order_point(node, self.points[target_index])
        if result is not None:
            order, bound = result
            return (
                (TargetRieszInteraction(node, order, bound),),
                tuple(),
                self.basis.count(order),
                1,
            )
        if node.is_leaf:
            direct = tuple(node.indices())
            return (
                tuple(),
                direct,
                self.direct_plan_penalty * len(direct),
                1,
            )
        analytic = []
        direct = []
        child_cost = 0
        visits = 1
        for child in node.children:
            (
                child_analytic,
                child_direct,
                cost,
                child_visits,
            ) = self._target_plan(target_index, child)
            analytic.extend(child_analytic)
            direct.extend(child_direct)
            child_cost += cost
            visits += child_visits
        return tuple(analytic), tuple(direct), child_cost, visits

    def _leaf_moments(self, node, residues):
        output = [0.0 + 0.0j for _ in range(self.basis.count())]
        for index in node.indices():
            if node.radius > 0.0:
                normalized = tuple(
                    (self.points[index][axis] - node.center[axis]) / node.radius
                    for axis in range(3)
                )
            else:
                normalized = (0.0, 0.0, 0.0)
            monomials = self.basis.monomials(normalized)
            residue = residues[index]
            for basis_index, monomial in enumerate(monomials):
                output[basis_index] += residue * monomial
        return tuple(output)

    def _build_moments(self, residues):
        moments = [tuple() for _ in self.nodes]
        for node in self.nodes:
            if node.is_leaf:
                moments[node.id] = self._leaf_moments(node, residues)
                continue
            output = [0.0 + 0.0j for _ in range(self.basis.count())]
            for child in node.children:
                translated = self.basis.translate(
                    moments[child.id],
                    child,
                    node,
                )
                for index, value in enumerate(translated):
                    output[index] += value
            moments[node.id] = tuple(output)
        return tuple(moments)

    def _build_static_weight_moments(self):
        moments = self._build_moments(
            tuple(complex(value) for value in self.weights)
        )
        for node in self.nodes:
            node.weight_moments = moments[node.id]
            node.absolute_weight = sum(self.weights[index] for index in node.indices())

    def _expansion_coefficients(self, point, source, order):
        difference = _vsub(point, source.center)
        distance = _vnorm(difference)
        if distance <= source.radius:
            raise RuntimeError("accepted Riesz expansion contains its target")
        rho = source.radius / distance
        direction = tuple(value / distance for value in difference)
        count = self.basis.count(order)
        coefficients = [0.0 for _ in range(count)]
        scale = distance ** (-self.kernel_power)
        coefficients[0] = scale
        if order >= 1:
            degree_one = (
                self.basis.lookup[(1, 0, 0)],
                self.basis.lookup[(0, 1, 0)],
                self.basis.lookup[(0, 0, 1)],
            )
            for axis in range(3):
                coefficients[degree_one[axis]] = (
                    scale * self.kernel_power * rho * direction[axis]
                )
        half_power = 0.5 * self.kernel_power
        for degree in range(2, order + 1):
            linear_scale = 2.0 * (degree + half_power - 1.0) * rho / degree
            quadratic_scale = (
                (degree + self.kernel_power - 2.0)
                * rho
                * rho
                / degree
            )
            previous_first, previous_last = self.basis.degree_ranges[degree - 1]
            for source_index in range(previous_first, previous_last):
                coefficient = coefficients[source_index]
                targets = self.basis.linear_shifts[source_index]
                for axis in range(3):
                    coefficients[targets[axis]] += (
                        linear_scale * direction[axis] * coefficient
                    )
            earlier_first, earlier_last = self.basis.degree_ranges[degree - 2]
            for source_index in range(earlier_first, earlier_last):
                coefficient = coefficients[source_index]
                targets = self.basis.quadratic_shifts[source_index]
                for axis in range(3):
                    coefficients[targets[axis]] -= (
                        quadratic_scale * coefficient
                    )
        absolute_tail = (
            scale
            * self._tail_relative(order, rho)
            / (1.0 + rho) ** self.kernel_power
        )
        return tuple(coefficients), absolute_tail

    def _normalized_monomials(self, point, node, order):
        if node.radius > 0.0:
            normalized = tuple(
                (point[axis] - node.center[axis]) / node.radius
                for axis in range(3)
            )
        else:
            normalized = (0.0, 0.0, 0.0)
        return self.basis.monomials(normalized, order)

    @staticmethod
    def _dot(left, right):
        return sum(
            left[index] * right[index]
            for index in range(min(len(left), len(right)))
        )

    def _kernel(self, left, right):
        difference = _vsub(self.points[left], self.points[right])
        distance_squared = _vdot(difference, difference)
        if distance_squared <= 0.0:
            raise ValueError("distinct Riesz nodes are numerically colliding")
        return distance_squared ** (-0.5 * self.kernel_power)

    def _apply_potential(self, residues, moments):
        forward = [0.0 + 0.0j for _ in range(self.n)]
        transpose = [0.0 + 0.0j for _ in range(self.n)]
        adjoint = [
            (
                [0.0 + 0.0j for _ in range(self.basis.count(order))]
                if order >= 0
                else None
            )
            for order in self.adjoint_orders
        ]
        for target_index in range(self.n):
            (
                interactions,
                direct_sources,
                _plan_cost,
                _plan_visits,
            ) = self._target_plan(target_index, self.root)
            target_residue = residues[target_index]
            for interaction in interactions:
                coefficients, _tail = self._expansion_coefficients(
                    self.points[target_index],
                    interaction.source,
                    interaction.order,
                )
                source_moments = moments[interaction.source.id][
                    : len(coefficients)
                ]
                forward[target_index] += self._dot(
                    coefficients,
                    source_moments,
                )
                row = adjoint[interaction.source.id]
                for index, coefficient in enumerate(coefficients):
                    row[index] += target_residue * coefficient
            for source_index in direct_sources:
                kernel = self._kernel(target_index, source_index)
                forward[target_index] += kernel * residues[source_index]
                transpose[source_index] += kernel * target_residue
        for node, order in zip(self.nodes, self.adjoint_orders, strict=True):
            if order < 0:
                continue
            coefficients = adjoint[node.id]
            for source_index in node.indices():
                monomials = self._normalized_monomials(
                    self.points[source_index],
                    node,
                    order,
                )
                transpose[source_index] += self._dot(monomials, coefficients)
        return tuple(
            0.5 * (forward[index] + transpose[index])
            for index in range(self.n)
        )

    def _build_row_kernel_error(self):
        forward_error = [0.0 for _ in range(self.n)]
        adjoint_error = [0.0 for _ in self.nodes]
        for target_index in range(self.n):
            (
                interactions,
                _direct_sources,
                _plan_cost,
                _plan_visits,
            ) = self._target_plan(target_index, self.root)
            for interaction in interactions:
                _coefficients, tail = self._expansion_coefficients(
                    self.points[target_index],
                    interaction.source,
                    interaction.order,
                )
                forward_error[target_index] += (
                    tail * interaction.source.absolute_weight
                )
                adjoint_error[interaction.source.id] += (
                    self.weights[target_index] * tail
                )
        transpose_error = [0.0 for _ in range(self.n)]
        for node in self.nodes:
            value = adjoint_error[node.id]
            if value == 0.0:
                continue
            for source_index in node.indices():
                transpose_error[source_index] += value
        return tuple(
            0.5 * (forward_error[index] + transpose_error[index])
            for index in range(self.n)
        )

    def apply(self, values):
        row = tuple(complex(value) for value in values)
        if len(row) != self.n:
            raise ValueError("values must contain one entry per Riesz node")
        if any(not _finite(value.real) or not _finite(value.imag) for value in row):
            raise ValueError("Riesz values must be finite")
        if all(value == row[0] for value in row):
            self.last_apply_stats = {
                "constant_shortcut": True,
                "dynamic_moment_entries": 0,
            }
            return tuple(0.0 for _ in row)
        residues = tuple(
            self.weights[index] * row[index] for index in range(self.n)
        )
        dynamic_moments = self._build_moments(residues)
        field_potential = self._apply_potential(residues, dynamic_moments)
        result = tuple(
            _clean_scalar(
                row[index] * self.weight_potential[index]
                - field_potential[index]
            )
            for index in range(self.n)
        )
        self.last_apply_stats = {
            "constant_shortcut": False,
            "dynamic_moment_entries": len(self.nodes) * self.basis.count(),
        }
        return result

    def apply_fields(self, fields):
        return tuple(self.apply(values) for values in fields)

    def compression_inf_bound(self, values):
        row = tuple(complex(value) for value in values)
        if len(row) != self.n:
            raise ValueError("values must contain one entry per Riesz node")
        maximum = max(_abs(value) for value in row)
        return max(
            (_abs(row[index]) + maximum) * self.row_kernel_error[index]
            for index in range(self.n)
        )

    def constant_residual(self):
        return max(_abs(value) for value in self.apply((1.0,) * self.n))

    def stats(self):
        result = {
            "method": "fixed_order_symmetric_adjoint_riesz_treecode",
            "nodes": self.n,
            "kernel_power": self.kernel_power,
            "tolerance": self.tolerance,
            "maximum_order": self.maximum_order,
            "maximum_block_order": self.maximum_block_order,
            "maximum_block_rank": self.basis.count(self.maximum_block_order),
            "leaf_size": self.leaf_size,
            "tree_nodes": len(self.nodes),
            "maximum_tree_depth": self.maximum_depth,
            "analytic_blocks": self.analytic_interaction_count,
            "analytic_target_interactions": self.analytic_interaction_count,
            "directed_near_pairs": self.directed_near_pairs,
            "near_field_pairs": self.directed_near_pairs // 2,
            "analytic_pair_fraction": 1.0
            - self.directed_near_pairs / (self.n * (self.n - 1)),
            "near_field_pair_fraction": self.directed_near_pairs
            / (self.n * (self.n - 1)),
            "maximum_analytic_relative_tail": (
                self.maximum_analytic_relative_tail
            ),
            "pair_partition_residual": 0,
            "basis_size": self.basis.count(),
            "translation_terms_per_tree_edge": self.basis.translation_term_count,
            "persistent_moment_entries": len(self.nodes) * self.basis.count(),
            "analytic_apply_units": (
                self.analytic_forward_units + self.adjoint_evaluation_units
            ),
            "analytic_forward_units": self.analytic_forward_units,
            "adjoint_evaluation_units": self.adjoint_evaluation_units,
            "analytic_apply_budget": self.analytic_apply_budget,
            "near_field_pair_budget": self.direct_pair_budget,
            "target_plan_visits": self.target_plan_visits,
            "target_plan_visit_budget": self.target_plan_visit_budget,
            "hard_no_quadratic_contract": True,
            "quadratic_fallback": False,
            "adaptive_rank": 0,
            "rank_growth_with_n": False,
            "temporary_pair_table_entries": 0,
            "stored_dense_distance_matrix": False,
            "stored_dense_operator_matrix": False,
            "compile_complexity": "O(N log N) streamed plans plus O(N) moments",
            "apply_complexity": "O(N log N) for fixed order in 3D",
            "storage_complexity": "O(N) fixed-order tree moments",
            "error_certificate": "analytic Gegenbauer tail plus exact near field",
        }
        result.update(self.last_apply_stats)
        return result

    def evaluate(self, values):
        result = self.apply(values)
        compression_bound = self.compression_inf_bound(values)
        constant_residual = self.constant_residual()
        stats = self.stats()
        ledger = BorrowComputeRepayLedger(
            borrowed=(
                "fair-split source tree",
                "fixed-order normalized Cartesian moments",
                "streamed target interaction plans",
            ),
            computed=(
                "Gegenbauer forward treecode",
                "exact algebraic adjoint treecode",
            ),
            repaid=(
                "all adjacent leaf interactions exactly",
                "constant graph channel exactly",
                "far interactions by analytic tail-certified expansions",
                "weighted symmetry by averaging the map with its adjoint",
            ),
            residuals=(
                ("compression_inf_bound", compression_bound),
                ("constant_residual", constant_residual),
                ("pair_partition_residual", 0.0),
            ),
            residual_norm=max(compression_bound, constant_residual),
            status="borrowed_repaid",
            notes=(
                "No quadratic fallback exists. Expansion order is bounded "
                "independently of N and both directed near work and analytic "
                "target/adjoint work are guarded by near-linear budgets."
            ),
        )
        return NearLinearRieszEvaluation(
            result,
            compression_bound,
            ledger,
            stats,
        )


BlockWSPDRieszQJet = ProductionRieszQJet
NearLinearRieszQJet = ProductionRieszQJet


__all__ = [
    "AnalyticRieszBlock",
    "BlockWSPDRieszQJet",
    "ExactRieszCrossBlock",
    "ExactRieszSelfBlock",
    "NearLinearContractError",
    "NearLinearRieszEvaluation",
    "NearLinearRieszQJet",
    "ProductionRieszQJet",
    "RieszMonomialBasis",
    "TargetRieszInteraction",
    "TargetTreeDiagnosticRieszQJet",
]
