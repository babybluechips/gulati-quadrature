"""Static scale-phase inverse-square calculus without pair trees.

Angular Fourier transformation converts every cross-scale mode into

    exp(-|m| |rho_i-rho_j|) / |exp(2 rho_i)-exp(2 rho_j)|.

The remaining ordered Cauchy transform is compiled on one-dimensional scale
intervals. A fixed-order nested Chebyshev pass uses upward moments, sparse
cross-cluster interactions, and downward local expansions. Unresolved
neighboring intervals are retained as exact endpoint blocks. The rank never
adapts, no pair values or interaction matrices are stored, and one static plan
is reused for every angular mode.

The class implements the continuum angular Fourier operator. It is not the
aliased finite-angle pair sum. Quadratic references live only in
``inverse_shape.testing.reference_pairwise`` and are never imported here.
"""

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
    _is_power_of_two,
    _log,
    _sqrt,
)


_COMPILE_VISIT_FACTOR = 64
_STATIC_BLOCK_FACTOR = 16
_EXACT_PAIR_FACTOR = 64


def _vector(values, length, name):
    row = tuple(complex(value) for value in values)
    if len(row) != length:
        raise ValueError(f"{name} must contain {length} entries")
    if any(not _finite(value.real) or not _finite(value.imag) for value in row):
        raise ValueError(f"{name} must contain finite entries")
    return row


def _decay(nonnegative_value):
    value = max(float(nonnegative_value), 0.0)
    if value >= 744.0:
        return 0.0
    return _exp(-value)


class _ScaleRange:
    def __init__(self, first, last, coordinates, leaf_size):
        self.first = int(first)
        self.last = int(last)
        self.minimum = coordinates[self.first]
        self.maximum = coordinates[self.last - 1]
        self.center = 0.5 * (self.minimum + self.maximum)
        self.half_width = 0.5 * (self.maximum - self.minimum)
        self.children = tuple()
        if self.n_nodes > leaf_size:
            middle = (self.first + self.last) // 2
            self.children = (
                _ScaleRange(self.first, middle, coordinates, leaf_size),
                _ScaleRange(middle, self.last, coordinates, leaf_size),
            )

    @property
    def n_nodes(self):
        return self.last - self.first

    @property
    def diameter(self):
        return self.maximum - self.minimum

    @property
    def is_leaf(self):
        return not self.children


def _chebyshev_grid(order):
    nodes = []
    weights = []
    for index in range(order):
        angle = PI * (index + 0.5) / order
        nodes.append(_cos(angle))
        sign = -1.0 if index % 2 else 1.0
        weights.append(sign * _sqrt(max(0.0, 1.0 - nodes[-1] ** 2)))
    return tuple(nodes), tuple(weights)


def _interpolation_nodes(node, chebyshev_nodes):
    return tuple(
        node.center + node.half_width * value for value in chebyshev_nodes
    )


def _cardinal_values(value, node, chebyshev_nodes, barycentric_weights):
    reference = (value - node.center) / node.half_width
    for index, interpolation_node in enumerate(chebyshev_nodes):
        if _abs(reference - interpolation_node) <= 4.0e-15:
            output = [0.0 for _ in chebyshev_nodes]
            output[index] = 1.0
            return tuple(output)
    numerators = tuple(
        weight / (reference - interpolation_node)
        for interpolation_node, weight in zip(
            chebyshev_nodes,
            barycentric_weights,
            strict=True,
        )
    )
    denominator = sum(numerators)
    return tuple(value / denominator for value in numerators)


def _interpolation_tail(left, right, order):
    distance = right.center - left.center
    spread = left.half_width + right.half_width
    if distance <= spread:
        return float("inf")
    ratio = spread / distance
    lebesgue = 1.0 + 2.0 * _log(float(order)) / PI
    geometric_tail = (
        ratio**order * (1.0 + ratio) / max(1.0 - ratio, 1.0e-300)
    )
    return (1.0 + lebesgue * lebesgue) * geometric_tail


class StaticCauchyCrossBlock:
    def __init__(self, left, right, rank, tail_bound):
        self.left = left
        self.right = right
        self.rank = int(rank)
        self.tail_bound = float(tail_bound)

    @property
    def pair_count(self):
        return self.left.n_nodes * self.right.n_nodes

class ExactCauchyCrossBlock:
    def __init__(self, left, right):
        self.left = left
        self.right = right

    @property
    def pair_count(self):
        if self.left is self.right:
            return self.left.n_nodes * (self.left.n_nodes - 1) // 2
        return self.left.n_nodes * self.right.n_nodes


class StaticTriangularCauchyPlan:
    """Fixed-rank nested Cauchy plan for sorted positive nodes."""

    def __init__(
        self,
        nodes,
        tolerance=2.0e-14,
        expansion_order=32,
        leaf_size=8,
    ):
        self.nodes = tuple(float(value) for value in nodes)
        self.rhos = tuple(0.5 * _log(value) for value in self.nodes)
        self.n = len(self.nodes)
        self.tolerance = float(tolerance)
        self.expansion_order = int(expansion_order)
        self.leaf_size = int(leaf_size)
        if self.n < 2:
            raise ValueError("a Cauchy plan requires at least two nodes")
        if any(
            not _finite(value) or value <= 0.0 for value in self.nodes
        ):
            raise ValueError("Cauchy nodes must be positive and finite")
        if any(
            self.nodes[index + 1] <= self.nodes[index]
            for index in range(self.n - 1)
        ):
            raise ValueError("Cauchy nodes must be strictly increasing")
        if self.tolerance <= 0.0 or not _finite(self.tolerance):
            raise ValueError("tolerance must be positive and finite")
        if self.expansion_order < 2 or self.leaf_size < 2:
            raise ValueError("expansion_order and leaf_size are too small")
        self.chebyshev_nodes, self.barycentric_weights = _chebyshev_grid(
            self.expansion_order
        )
        self.root = _ScaleRange(0, self.n, self.rhos, self.leaf_size)
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
                "static Cauchy plan exceeded its linear block budget; "
                "subdivide the generating chart"
            )
        if self.compiled_exact_pairs > self.exact_pair_budget:
            raise RuntimeError(
                "static Cauchy plan exceeded its linear exact-pair budget; "
                "subdivide the generating chart"
            )
        represented = sum(
            block.pair_count for block in self.compressed_blocks
        ) + sum(block.pair_count for block in self.exact_blocks)
        expected = self.n * (self.n - 1) // 2
        if represented != expected:
            raise RuntimeError("Cauchy plan failed its pair partition")
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

    def _compile_static_transforms(self):
        self._interpolation_node_cache = {
            node: _interpolation_nodes(node, self.chebyshev_nodes)
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
                for value in self._interpolation_node_cache[child]
            )
            for node in self.preorder
            if not node.is_leaf
            for child in node.children
        }
        self._leaf_basis_cache = {
            node: tuple(
                _cardinal_values(
                    self.rhos[index],
                    node,
                    self.chebyshev_nodes,
                    self.barycentric_weights,
                )
                for index in range(node.first, node.last)
            )
            for node in self.leaves
        }
        self._interaction_kernel_cache = {}
        for block in self.compressed_blocks:
            left_rhos = self._interpolation_node_cache[block.left]
            right_rhos = self._interpolation_node_cache[block.right]
            self._interaction_kernel_cache[block] = tuple(
                tuple(
                    1.0
                    / (
                        _exp(2.0 * right_rho)
                        - _exp(2.0 * left_rho)
                    )
                    for right_rho in right_rhos
                )
                for left_rho in left_rhos
            )

    def _compile_self(self, node):
        if node.is_leaf:
            if node.n_nodes > 1:
                self.exact_blocks.append(ExactCauchyCrossBlock(node, node))
            return
        left, right = node.children
        self._compile_self(left)
        self._compile_pair(left, right)
        self._compile_self(right)

    def _candidate(self, left, right):
        tail = _interpolation_tail(left, right, self.expansion_order)
        if tail > self.tolerance:
            return None
        return StaticCauchyCrossBlock(
            left,
            right,
            self.expansion_order,
            tail,
        )

    def _compile_pair(self, left, right):
        self.compile_pair_visits += 1
        if self.compile_pair_visits > self.compile_visit_budget:
            raise RuntimeError(
                "static Cauchy compilation exceeded its linear visit budget; "
                "no quadratic fallback is permitted"
            )
        candidate = self._candidate(left, right)
        if candidate is not None:
            self.compressed_blocks.append(candidate)
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

    def _transfer_rows(self, parent, child):
        return self._transfer_cache[(parent, child)]

    def _leaf_moments(self, node, row, rho_values, mode_value):
        plus = [0.0 + 0.0j for _ in range(self.expansion_order)]
        minus = [0.0 + 0.0j for _ in range(self.expansion_order)]
        for local, index in enumerate(range(node.first, node.last)):
            basis = self._leaf_basis_cache[node][local]
            plus_scale = _decay(
                mode_value * (node.maximum - rho_values[index])
            )
            minus_scale = _decay(
                mode_value * (rho_values[index] - node.minimum)
            )
            for coefficient, basis_value in enumerate(basis):
                plus[coefficient] += basis_value * plus_scale * row[index]
                minus[coefficient] += basis_value * minus_scale * row[index]
        return plus, minus

    def _upward_moments(self, row, rho_values, mode_value):
        plus = {}
        minus = {}
        transfer_work = 0
        for node in self.postorder:
            if node.is_leaf:
                plus[node], minus[node] = self._leaf_moments(
                    node,
                    row,
                    rho_values,
                    mode_value,
                )
                continue
            node_plus = [0.0 + 0.0j for _ in range(self.expansion_order)]
            node_minus = [0.0 + 0.0j for _ in range(self.expansion_order)]
            for child in node.children:
                rows = self._transfer_rows(node, child)
                plus_scale = _decay(
                    mode_value
                    * (node.maximum - child.maximum)
                )
                minus_scale = _decay(
                    mode_value
                    * (child.minimum - node.minimum)
                )
                for child_index, interpolation_row in enumerate(rows):
                    child_plus = plus[child][child_index] * plus_scale
                    child_minus = minus[child][child_index] * minus_scale
                    for parent_index, basis_value in enumerate(
                        interpolation_row
                    ):
                        node_plus[parent_index] += basis_value * child_plus
                        node_minus[parent_index] += basis_value * child_minus
                        transfer_work += 2
            plus[node] = node_plus
            minus[node] = node_minus
        return plus, minus, transfer_work

    def apply_exponential_mode(self, values, rhos, mode):
        row = _vector(values, self.n, "values")
        rho_values = tuple(float(value) for value in rhos)
        if len(rho_values) != self.n:
            raise ValueError("rhos must contain one entry per Cauchy node")
        mode_value = abs(int(mode))
        output = [0.0 + 0.0j for _ in range(self.n)]
        compensation = [0.0 + 0.0j for _ in range(self.n)]

        def accumulate(index, contribution):
            corrected = contribution - compensation[index]
            updated = output[index] + corrected
            compensation[index] = (updated - output[index]) - corrected
            output[index] = updated

        moments_plus, moments_minus, transfer_work = self._upward_moments(
            row,
            rho_values,
            mode_value,
        )
        local_plus = {
            node: [0.0 + 0.0j for _ in range(self.expansion_order)]
            for node in self.preorder
        }
        local_minus = {
            node: [0.0 + 0.0j for _ in range(self.expansion_order)]
            for node in self.preorder
        }
        interaction_work = 0
        for block in self.compressed_blocks:
            attenuation = _decay(
                mode_value
                * (
                    block.right.minimum - block.left.maximum
                )
            )
            kernel_tile = self._interaction_kernel_cache[block]
            for left_index, kernel_row in enumerate(kernel_tile):
                for right_index, kernel_value in enumerate(kernel_row):
                    kernel = attenuation * kernel_value
                    local_plus[block.left][left_index] += (
                        kernel * moments_minus[block.right][right_index]
                    )
                    local_minus[block.right][right_index] += (
                        kernel * moments_plus[block.left][left_index]
                    )
                    interaction_work += 2
        downward_work = 0
        for node in self.preorder:
            if node.is_leaf:
                continue
            for child in node.children:
                rows = self._transfer_rows(node, child)
                plus_scale = _decay(
                    mode_value
                    * (node.maximum - child.maximum)
                )
                minus_scale = _decay(
                    mode_value
                    * (child.minimum - node.minimum)
                )
                for child_index, interpolation_row in enumerate(rows):
                    plus_value = 0.0 + 0.0j
                    minus_value = 0.0 + 0.0j
                    for parent_index, basis_value in enumerate(
                        interpolation_row
                    ):
                        plus_value += (
                            basis_value * local_plus[node][parent_index]
                        )
                        minus_value += (
                            basis_value * local_minus[node][parent_index]
                        )
                        downward_work += 2
                    local_plus[child][child_index] += plus_scale * plus_value
                    local_minus[child][child_index] += minus_scale * minus_value
        leaf_work = 0
        for node in self.leaves:
            for local, index in enumerate(range(node.first, node.last)):
                basis = self._leaf_basis_cache[node][local]
                plus_scale = _decay(
                    mode_value * (node.maximum - rho_values[index])
                )
                minus_scale = _decay(
                    mode_value * (rho_values[index] - node.minimum)
                )
                contribution = 0.0 + 0.0j
                for coefficient, basis_value in enumerate(basis):
                    contribution += basis_value * (
                        plus_scale * local_plus[node][coefficient]
                        + minus_scale * local_minus[node][coefficient]
                    )
                    leaf_work += 2
                accumulate(index, contribution)
        exact_pairs = 0
        for block in self.exact_blocks:
            same = block.left is block.right
            for left in range(block.left.first, block.left.last):
                right_start = left + 1 if same else block.right.first
                for right in range(right_start, block.right.last):
                    kernel = _decay(
                        mode_value * _abs(rho_values[right] - rho_values[left])
                    ) / (self.nodes[right] - self.nodes[left])
                    accumulate(left, kernel * row[right])
                    accumulate(right, kernel * row[left])
                    exact_pairs += 1
        self.last_apply_stats = {
            "mode": mode_value,
            "factor_work": (
                transfer_work
                + interaction_work
                + downward_work
                + leaf_work
            ),
            "upward_transfer_work": transfer_work,
            "interaction_work": interaction_work,
            "downward_work": downward_work,
            "leaf_evaluation_work": leaf_work,
            "exact_pairs": exact_pairs,
            "maximum_temporary_interpolation_entries": (
                self.expansion_order * self.expansion_order
            ),
            "stored_dense_matrix": False,
        }
        return tuple(_clean_scalar(value) for value in output)

    def block_max_relative_error(self, block):
        kernel_tile = self._interaction_kernel_cache[block]
        maximum = 0.0
        for left in range(block.left.first, block.left.last):
            left_basis = _cardinal_values(
                self.rhos[left],
                block.left,
                self.chebyshev_nodes,
                self.barycentric_weights,
            )
            for right in range(block.right.first, block.right.last):
                right_basis = _cardinal_values(
                    self.rhos[right],
                    block.right,
                    self.chebyshev_nodes,
                    self.barycentric_weights,
                )
                approximate = 0.0
                for left_index, kernel_row in enumerate(kernel_tile):
                    for right_index, kernel_value in enumerate(kernel_row):
                        approximate += (
                            left_basis[left_index]
                            * right_basis[right_index]
                            * kernel_value
                        )
                exact = 1.0 / (self.nodes[right] - self.nodes[left])
                maximum = max(maximum, _abs(approximate - exact) / exact)
        return maximum

    def stats(self):
        compressed_pairs = sum(
            block.pair_count for block in self.compressed_blocks
        )
        exact_pairs = sum(block.pair_count for block in self.exact_blocks)
        result = {
            "nodes": self.n,
            "tolerance": self.tolerance,
            "expansion_order": self.expansion_order,
            "leaf_size": self.leaf_size,
            "compressed_blocks": len(self.compressed_blocks),
            "exact_blocks": len(self.exact_blocks),
            "compressed_pair_fraction": compressed_pairs
            / max(compressed_pairs + exact_pairs, 1),
            "exact_pair_fraction": exact_pairs
            / max(compressed_pairs + exact_pairs, 1),
            "cluster_records": len(self.preorder),
            "leaf_records": len(self.leaves),
            "stored_factor_entries": 0,
            "stored_interaction_matrices": len(self.compressed_blocks),
            "stored_static_transform_entries": (
                len(self.preorder) * self.expansion_order
                + (len(self.preorder) - 1)
                * self.expansion_order
                * self.expansion_order
                + self.n * self.expansion_order
                + len(self.compressed_blocks)
                * self.expansion_order
                * self.expansion_order
            ),
            "stored_block_records": (
                len(self.compressed_blocks) + len(self.exact_blocks)
            ),
            "compile_pair_visits": self.compile_pair_visits,
            "compile_visit_budget": self.compile_visit_budget,
            "compiled_block_count": self.compiled_block_count,
            "static_block_budget": self.static_block_budget,
            "compiled_exact_pairs": self.compiled_exact_pairs,
            "exact_pair_budget": self.exact_pair_budget,
            "maximum_tail_bound": max(
                (block.tail_bound for block in self.compressed_blocks),
                default=0.0,
            ),
            "pair_partition_residual": self.pair_partition_residual,
            "temporary_pair_table_entries": 0,
            "stored_dense_matrix": False,
            "quadratic_fallback": False,
            "reference_oracle_in_production_object": False,
            "compile_complexity": "O(p^2 N) with fixed p and hard budget",
            "apply_complexity": "O(p^2 N) with fixed p",
            "persistent_storage_complexity": "O(N)",
            "working_storage_complexity": "O(p N) with fixed p",
            "storage_complexity": "O(N) with fixed p",
        }
        result.update(self.last_apply_stats)
        return result


class ScalePhaseCauchyEvaluation:
    def __init__(self, values, ledger, stats):
        self.values = values
        self.ledger = ledger
        self.stats = stats


class ScalePhaseCauchyQJet:
    """Inverse-square graph in nonuniform exponential scale-phase coordinates."""

    def __init__(
        self,
        rhos,
        n_theta,
        meridional_weights,
        tolerance=2.0e-14,
        expansion_order=32,
        leaf_size=8,
        normalization=1.0,
    ):
        self.rhos = tuple(float(value) for value in rhos)
        self.n_scale = len(self.rhos)
        self.n_theta = int(n_theta)
        if self.n_scale < 2:
            raise ValueError("at least two scale lines are required")
        if self.n_theta < 4 or not _is_power_of_two(self.n_theta):
            raise ValueError("n_theta must be a radix-two size of at least four")
        if any(
            self.rhos[index + 1] <= self.rhos[index]
            for index in range(self.n_scale - 1)
        ):
            raise ValueError("rhos must be strictly increasing")
        self.x_nodes = tuple(_exp(2.0 * value) for value in self.rhos)
        self.meridional_weights = tuple(
            float(value) for value in meridional_weights
        )
        if len(self.meridional_weights) != self.n_scale:
            raise ValueError("one meridional weight is required per scale line")
        if any(
            value <= 0.0 or not _finite(value)
            for value in self.meridional_weights
        ):
            raise ValueError("meridional weights must be positive and finite")
        self.normalization = float(normalization)
        self.theta_step = TAU / self.n_theta
        self.plan = StaticTriangularCauchyPlan(
            self.x_nodes,
            tolerance=tolerance,
            expansion_order=expansion_order,
            leaf_size=leaf_size,
        )
        cross = self.plan.apply_exponential_mode(
            self.meridional_weights,
            self.rhos,
            0,
        )
        self.cross_row_sum = tuple(TAU * complex(value) for value in cross)
        self.last_apply_stats = {}

    @property
    def n_nodes(self):
        return self.n_scale * self.n_theta

    def _rows(self, values):
        rows = tuple(tuple(complex(value) for value in row) for row in values)
        if len(rows) != self.n_scale:
            raise ValueError("values must contain one row per scale line")
        if any(len(row) != self.n_theta for row in rows):
            raise ValueError("each scale row must contain n_theta values")
        return rows

    def apply(self, values):
        rows = self._rows(values)
        transformed = tuple(tuple(_fft_precise(row)) for row in rows)
        output_modes = [
            [0.0 + 0.0j for _ in range(self.n_theta)]
            for _ in range(self.n_scale)
        ]
        total_factor_work = 0
        total_exact_pairs = 0
        for angular_index in range(self.n_theta):
            mode = min(angular_index, self.n_theta - angular_index)
            source = tuple(
                self.meridional_weights[scale]
                * transformed[scale][angular_index]
                for scale in range(self.n_scale)
            )
            potential = self.plan.apply_exponential_mode(
                source,
                self.rhos,
                mode,
            )
            plan_stats = self.plan.last_apply_stats
            total_factor_work += plan_stats["factor_work"]
            total_exact_pairs += plan_stats["exact_pairs"]
            cycle_eigenvalue = mode * (self.n_theta - mode) / 2.0
            for scale in range(self.n_scale):
                same_scale = (
                    self.meridional_weights[scale]
                    * self.theta_step
                    * cycle_eigenvalue
                    / self.x_nodes[scale]
                )
                output_modes[scale][angular_index] = self.normalization * (
                    (
                        self.cross_row_sum[scale]
                        + same_scale
                    )
                    * transformed[scale][angular_index]
                    - TAU * complex(potential[scale])
                )
        result = tuple(
            tuple(_clean_scalar(value) for value in _ifft_precise(row))
            for row in output_modes
        )
        self.last_apply_stats = {
            "method": "scale_phase_angular_fft_triangular_cauchy",
            "angular_fft_rows": 2 * self.n_scale,
            "total_cauchy_factor_work": total_factor_work,
            "total_exact_scale_pairs": total_exact_pairs,
            "stored_dense_matrix": False,
        }
        return result

    def constant_residual(self):
        constant = tuple(
            (1.0,) * self.n_theta for _ in range(self.n_scale)
        )
        applied = self.apply(constant)
        return max(_abs(complex(value)) for row in applied for value in row)

    def stats(self):
        result = {
            "n_scale": self.n_scale,
            "n_theta": self.n_theta,
            "n_nodes": self.n_nodes,
            "angular_operator": "exact continuum Fourier coefficients",
            "same_scale_channel": "exact finite-cycle spectrum",
            "cauchy_plan": self.plan.stats(),
            "stored_dense_distance_matrix": False,
            "stored_dense_operator_matrix": False,
            "quadratic_fallback": False,
            "reference_oracle_in_production_object": False,
            "apply_complexity": "O(p^2 N + N log n_theta) with fixed p",
            "storage_complexity": "O(N) with fixed p",
        }
        result.update(self.last_apply_stats)
        return result

    def evaluate(self, values):
        result = self.apply(values)
        residual = self.constant_residual()
        stats = self.stats()
        ledger = BorrowComputeRepayLedger(
            borrowed=(
                "nonuniform exponential scale-phase normal form",
                "exact angular Fourier coefficients",
                "fixed-order triangular Cauchy plan",
            ),
            computed=(
                "angular QJet FFT",
                "one-sided exponentially conjugated Cauchy transforms",
            ),
            repaid=(
                "exact neighboring scale blocks",
                "same-scale finite-cycle spectrum",
                "row-sum constant channel",
            ),
            residuals=(
                ("constant_residual", residual),
                (
                    "maximum_cauchy_tail_bound",
                    stats["cauchy_plan"]["maximum_tail_bound"],
                ),
            ),
            residual_norm=max(
                residual,
                stats["cauchy_plan"]["maximum_tail_bound"],
            ),
            status="borrowed_repaid",
            notes=(
                "The plan is exact on retained neighboring blocks and uses a "
                "fixed, analytically bounded Chebyshev tail on separated "
                "one-dimensional scale intervals."
            ),
        )
        return ScalePhaseCauchyEvaluation(result, ledger, stats)


__all__ = [
    "ExactCauchyCrossBlock",
    "ScalePhaseCauchyEvaluation",
    "ScalePhaseCauchyQJet",
    "StaticCauchyCrossBlock",
    "StaticTriangularCauchyPlan",
]
