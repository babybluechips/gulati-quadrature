"""Arbitrary-surface QJets and the exhaustive diagnostic reference backend.

The public ``CertifiedArbitrarySurfaceQJet`` is imported from
``riesz_near_linear``. It uses fixed-width source QJets on a symmetric
well-separated pair tree. The older target-tree and exhaustive ACA hierarchies
remain explicitly named for diagnostics only. They are not fallbacks.
"""

from inverse_shape.quadrature import (
    BorrowComputeRepayLedger,
    _abs,
    _clean_scalar,
    _finite,
    _sqrt,
)
from inverse_shape.riesz_near_linear import ProductionRieszQJet


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


def _vcross(left, right):
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _point(value):
    row = tuple(float(component) for component in value)
    if len(row) != 3:
        raise ValueError("each surface point must have three coordinates")
    if any(not _finite(component) for component in row):
        raise ValueError("surface coordinates must be finite")
    return row


def triangle_lumped_vertex_weights(vertices, triangles):
    """Return one-third triangle-area weights for an arbitrary 3D mesh."""

    points = tuple(_point(value) for value in vertices)
    if len(points) < 3:
        raise ValueError("a triangle mesh requires at least three vertices")
    weights = [0.0 for _ in points]
    face_count = 0
    for value in triangles:
        face = tuple(int(index) for index in value)
        if len(face) != 3 or len(set(face)) != 3:
            raise ValueError("each mesh face must contain three distinct vertices")
        if any(index < 0 or index >= len(points) for index in face):
            raise ValueError("triangle vertex index is out of range")
        left = _vsub(points[face[1]], points[face[0]])
        right = _vsub(points[face[2]], points[face[0]])
        area = 0.5 * _vnorm(_vcross(left, right))
        if area <= 1.0e-30:
            raise ValueError("degenerate mesh triangles are not supported")
        share = area / 3.0
        for index in face:
            weights[index] += share
        face_count += 1
    if face_count == 0:
        raise ValueError("a triangle mesh requires at least one face")
    if any(value <= 0.0 for value in weights):
        raise ValueError("every mesh vertex must belong to a nondegenerate face")
    return tuple(weights)


def _even_indices(count, wanted):
    wanted = min(int(wanted), int(count))
    if wanted <= 0:
        return tuple()
    if wanted == count:
        return tuple(range(count))
    if wanted == 1:
        return (count // 2,)
    denominator = wanted - 1
    return tuple(
        (index * (count - 1) + denominator // 2) // denominator
        for index in range(wanted)
    )


def _relative_l2(reference, candidate):
    numerator = sum(
        _abs(complex(left) - complex(right)) ** 2
        for left, right in zip(reference, candidate, strict=True)
    )
    denominator = sum(_abs(complex(value)) ** 2 for value in reference)
    return _sqrt(numerator / max(denominator, 1.0e-300))


def inverse_square_sums_from_log_discriminant_laplacians(
    laplacians,
    dimension=3,
):
    """Recover inverse-square sums from peeled log-discriminant jets.

    If ``L_i(x)=sum_{j != i} a_j log(|x-x_j|^2)``, this consumes
    ``Delta L_i(x_i)`` and returns ``sum_{j != i} a_j/|x_i-x_j|^2``.
    """

    dimension_value = int(dimension)
    if dimension_value <= 2:
        raise ValueError("the Euclidean Laplacian identity requires dimension > 2")
    scale = 1.0 / (2.0 * (dimension_value - 2))
    output = tuple(_clean_scalar(scale * complex(value)) for value in laplacians)
    if any(
        not _finite(complex(value).real) or not _finite(complex(value).imag)
        for value in output
    ):
        raise ValueError("log-discriminant Laplacians must be finite")
    return output


class GeneratedEuclideanDiscriminantQJet:
    """O(N) graph contraction from generated peeled Laplacian jets."""

    def __init__(self, n, dimension=3):
        self.n = int(n)
        self.dimension = int(dimension)
        if self.n < 1:
            raise ValueError("at least one generated jet is required")
        if self.dimension <= 2:
            raise ValueError("dimension must exceed two")
        self.last_apply_stats = {}

    def _vector(self, values, name):
        row = tuple(complex(value) for value in values)
        if len(row) != self.n:
            raise ValueError(f"{name} must contain {self.n} entries")
        if any(
            not _finite(value.real) or not _finite(value.imag) for value in row
        ):
            raise ValueError(f"{name} must contain finite entries")
        return row

    def apply(
        self,
        values,
        weight_log_laplacians,
        weighted_field_log_laplacians,
    ):
        row = self._vector(values, "values")
        weight_sums = inverse_square_sums_from_log_discriminant_laplacians(
            self._vector(weight_log_laplacians, "weight_log_laplacians"),
            self.dimension,
        )
        field_sums = inverse_square_sums_from_log_discriminant_laplacians(
            self._vector(
                weighted_field_log_laplacians,
                "weighted_field_log_laplacians",
            ),
            self.dimension,
        )
        result = tuple(
            _clean_scalar(
                row[index] * complex(weight_sums[index])
                - complex(field_sums[index])
            )
            for index in range(self.n)
        )
        self.last_apply_stats = {
            "method": "generated_euclidean_log_discriminant_laplacian",
            "nodes": self.n,
            "dimension": self.dimension,
            "jet_contractions": 2 * self.n,
            "adaptive_rank": 0,
            "stored_dense_matrix": False,
        }
        return result

    def stats(self):
        result = {
            "nodes": self.n,
            "dimension": self.dimension,
            "persistent_complex_entries": 0,
            "apply_complexity": "O(N) after peeled-jet generation",
            "generator_included": False,
            "adaptive_rank": 0,
            "stored_dense_distance_matrix": False,
            "stored_dense_operator_matrix": False,
        }
        result.update(self.last_apply_stats)
        return result


class _SurfaceTreeNode:
    """Balanced spatial node storing only a shared-order interval."""

    def __init__(self, order, first, last, points, leaf_size):
        self._order = order
        self.first = int(first)
        self.last = int(last)
        self.children = tuple()
        minimum = [
            min(points[order[position]][axis] for position in range(first, last))
            for axis in range(3)
        ]
        maximum = [
            max(points[order[position]][axis] for position in range(first, last))
            for axis in range(3)
        ]
        self.center = tuple(
            0.5 * (minimum[axis] + maximum[axis]) for axis in range(3)
        )
        self.radius = max(
            _vnorm(_vsub(points[order[position]], self.center))
            for position in range(first, last)
        )
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
                _SurfaceTreeNode(order, first, middle, points, leaf_size),
                _SurfaceTreeNode(order, middle, last, points, leaf_size),
            )

    @property
    def n_nodes(self):
        return self.last - self.first

    @property
    def is_leaf(self):
        return not self.children

    def global_index(self, local_index):
        return self._order[self.first + int(local_index)]


class CertifiedSurfaceCrossBlock:
    def __init__(
        self,
        left,
        right,
        left_factors,
        right_factors,
        maximum_relative_residual,
        maximum_absolute_residual,
    ):
        self.left = left
        self.right = right
        self.left_factors = tuple(left_factors)
        self.right_factors = tuple(right_factors)
        self.maximum_relative_residual = float(maximum_relative_residual)
        self.maximum_absolute_residual = float(maximum_absolute_residual)

    @property
    def rank(self):
        return len(self.left_factors)

    @property
    def pair_count(self):
        return self.left.n_nodes * self.right.n_nodes

    @property
    def stored_factor_entries(self):
        return self.rank * (self.left.n_nodes + self.right.n_nodes)


class ExactSurfaceCrossBlock:
    def __init__(self, left, right):
        self.left = left
        self.right = right

    @property
    def pair_count(self):
        return self.left.n_nodes * self.right.n_nodes


class ExactSurfaceSelfBlock:
    def __init__(self, node):
        self.node = node

    @property
    def pair_count(self):
        return self.node.n_nodes * (self.node.n_nodes - 1) // 2


class ArbitrarySurfaceEvaluation:
    def __init__(self, values, compression_inf_bound, ledger, stats):
        self.values = values
        self.compression_inf_bound = float(compression_inf_bound)
        self.ledger = ledger
        self.stats = stats


class ExhaustiveArbitrarySurfaceQJet:
    """Quadratic-worst-case diagnostic hierarchy; not a production backend."""

    def __init__(
        self,
        points,
        weights,
        kernel_power=2.0,
        tolerance=5.0e-13,
        admissibility=0.45,
        maximum_rank=48,
        leaf_size=8,
        probe_nodes=8,
    ):
        self.points = tuple(_point(value) for value in points)
        self.n = len(self.points)
        if self.n < 2:
            raise ValueError("an arbitrary surface requires at least two nodes")
        if len(set(self.points)) != self.n:
            raise ValueError("distinct surface nodes may not coincide")
        self.weights = tuple(float(value) for value in weights)
        if len(self.weights) != self.n:
            raise ValueError("weights must contain one value per surface node")
        if any(value <= 0.0 or not _finite(value) for value in self.weights):
            raise ValueError("surface weights must be positive and finite")
        self.kernel_power = float(kernel_power)
        self.tolerance = float(tolerance)
        self.admissibility = float(admissibility)
        self.maximum_rank = int(maximum_rank)
        self.leaf_size = int(leaf_size)
        self.probe_nodes = int(probe_nodes)
        if self.kernel_power <= 0.0 or not _finite(self.kernel_power):
            raise ValueError("kernel_power must be positive and finite")
        if self.tolerance <= 0.0 or not _finite(self.tolerance):
            raise ValueError("tolerance must be positive and finite")
        if (
            self.admissibility <= 0.0
            or self.admissibility >= 1.0
            or not _finite(self.admissibility)
        ):
            raise ValueError("admissibility must lie strictly between zero and one")
        if self.maximum_rank < 1 or self.leaf_size < 1:
            raise ValueError("maximum_rank and leaf_size must be positive")
        if self.probe_nodes < 2:
            raise ValueError("probe_nodes must be at least two")

        self._order = list(range(self.n))
        self.root = _SurfaceTreeNode(
            self._order,
            0,
            self.n,
            self.points,
            self.leaf_size,
        )
        self.low_rank_blocks = []
        self.exact_cross_blocks = []
        self.exact_self_blocks = []
        self.row_kernel_error = [0.0 for _ in range(self.n)]
        self._aca_kernel_samples = 0
        self._certification_kernel_samples = 0
        self._rejected_blocks = 0
        self._aca_attempts = []
        self._compile_self(self.root)
        self.low_rank_blocks = tuple(self.low_rank_blocks)
        self.exact_cross_blocks = tuple(self.exact_cross_blocks)
        self.exact_self_blocks = tuple(self.exact_self_blocks)
        represented = (
            sum(block.pair_count for block in self.low_rank_blocks)
            + sum(block.pair_count for block in self.exact_cross_blocks)
            + sum(block.pair_count for block in self.exact_self_blocks)
        )
        expected = self.n * (self.n - 1) // 2
        if represented != expected:
            raise RuntimeError("surface tree failed its exact pair partition")
        self.pair_count = expected
        self.partition_residual = represented - expected
        self.last_apply_stats = {}

    @classmethod
    def from_triangle_mesh(cls, vertices, triangles, **options):
        points = tuple(_point(value) for value in vertices)
        weights = triangle_lumped_vertex_weights(points, triangles)
        return cls(points, weights, **options)

    def _kernel(self, left, right):
        difference = _vsub(self.points[left], self.points[right])
        distance_squared = _vdot(difference, difference)
        if distance_squared <= 1.0e-28:
            raise ValueError("distinct surface nodes are numerically colliding")
        return distance_squared ** (-0.5 * self.kernel_power)

    def _admissible(self, left, right):
        center_distance = _vnorm(_vsub(left.center, right.center))
        gap = center_distance - left.radius - right.radius
        if gap <= 1.0e-30:
            return False
        return max(left.radius, right.radius) / gap <= self.admissibility

    def _compile_self(self, node):
        if node.is_leaf:
            if node.n_nodes > 1:
                self.exact_self_blocks.append(ExactSurfaceSelfBlock(node))
            return
        left, right = node.children
        self._compile_self(left)
        self._compile_pair(left, right)
        self._compile_self(right)

    def _compile_pair(self, left, right):
        if self._admissible(left, right):
            candidate = self._adaptive_cross_block(left, right)
            if candidate is not None:
                self.low_rank_blocks.append(candidate)
                return
            self._rejected_blocks += 1
        if left.is_leaf and right.is_leaf:
            self.exact_cross_blocks.append(ExactSurfaceCrossBlock(left, right))
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

    def _adaptive_cross_block(self, left, right):
        cache = {}
        left_factors = []
        right_factors = []
        used_left = set()

        def kernel(left_local, right_local):
            key = (left_local, right_local)
            value = cache.get(key)
            if value is None:
                value = self._kernel(
                    left.global_index(left_local),
                    right.global_index(right_local),
                )
                cache[key] = value
            return value

        def residual(left_local, right_local):
            value = kernel(left_local, right_local)
            for left_factor, right_factor in zip(
                left_factors,
                right_factors,
                strict=True,
            ):
                value -= left_factor[left_local] * right_factor[right_local]
            return value

        left_probes = _even_indices(left.n_nodes, self.probe_nodes)
        right_probes = _even_indices(right.n_nodes, self.probe_nodes)

        def probe_error(require_unused=False):
            pivot_left = None
            maximum_residual = -1.0
            maximum_kernel = 0.0
            for left_local in left_probes:
                if require_unused and left_local in used_left:
                    continue
                for right_local in right_probes:
                    raw = kernel(left_local, right_local)
                    value = residual(left_local, right_local)
                    magnitude = _abs(value)
                    maximum_kernel = max(maximum_kernel, _abs(raw))
                    if magnitude > maximum_residual:
                        maximum_residual = magnitude
                        pivot_left = left_local
            relative = maximum_residual / max(maximum_kernel, 1.0e-300)
            return relative, pivot_left, maximum_kernel

        economic_rank = (
            left.n_nodes * right.n_nodes - 1
        ) // (left.n_nodes + right.n_nodes)
        rank_limit = min(
            self.maximum_rank,
            left.n_nodes,
            right.n_nodes,
            economic_rank,
        )
        if rank_limit < 1:
            return None
        for _rank in range(rank_limit):
            relative, pivot_left, scale = probe_error(require_unused=True)
            if relative <= 0.25 * self.tolerance:
                break
            if pivot_left is None:
                break
            row = tuple(
                residual(pivot_left, right_local)
                for right_local in range(right.n_nodes)
            )
            pivot_right = max(
                range(right.n_nodes),
                key=lambda index: _abs(row[index]),
            )
            pivot = row[pivot_right]
            if _abs(pivot) <= 8.0e-17 * max(scale, 1.0e-300):
                break
            column = tuple(
                residual(left_local, pivot_right)
                for left_local in range(left.n_nodes)
            )
            left_factors.append(tuple(value / pivot for value in column))
            right_factors.append(row)
            used_left.add(pivot_left)
        self._aca_kernel_samples += len(cache)
        if not left_factors:
            return None
        certificate = self._certify_factors(
            left,
            right,
            tuple(left_factors),
            tuple(right_factors),
        )
        self._aca_attempts.append(
            (
                left.n_nodes,
                right.n_nodes,
                len(left_factors),
                certificate[0],
            )
        )
        if certificate[0] > self.tolerance:
            return None
        relative, absolute, left_errors, right_errors = certificate
        for local, value in enumerate(left_errors):
            self.row_kernel_error[left.global_index(local)] += value
        for local, value in enumerate(right_errors):
            self.row_kernel_error[right.global_index(local)] += value
        return CertifiedSurfaceCrossBlock(
            left,
            right,
            tuple(left_factors),
            tuple(right_factors),
            relative,
            absolute,
        )

    def _certify_factors(self, left, right, left_factors, right_factors):
        maximum_kernel = 0.0
        maximum_residual = 0.0
        left_errors = [0.0 for _ in range(left.n_nodes)]
        right_errors = [0.0 for _ in range(right.n_nodes)]
        for left_local in range(left.n_nodes):
            left_index = left.global_index(left_local)
            for right_local in range(right.n_nodes):
                right_index = right.global_index(right_local)
                raw = self._kernel(left_index, right_index)
                approximate = sum(
                    left_factor[left_local] * right_factor[right_local]
                    for left_factor, right_factor in zip(
                        left_factors,
                        right_factors,
                        strict=True,
                    )
                )
                residual = _abs(raw - approximate)
                maximum_kernel = max(maximum_kernel, _abs(raw))
                maximum_residual = max(maximum_residual, residual)
                left_errors[left_local] += self.weights[right_index] * residual
                right_errors[right_local] += self.weights[left_index] * residual
        self._certification_kernel_samples += left.n_nodes * right.n_nodes
        relative = maximum_residual / max(maximum_kernel, 1.0e-300)
        return relative, maximum_residual, left_errors, right_errors

    def _as_fields(self, fields):
        rows = tuple(tuple(complex(value) for value in row) for row in fields)
        if not rows or any(len(row) != self.n for row in rows):
            raise ValueError("each field must contain one value per surface node")
        if any(
            not _finite(value.real) or not _finite(value.imag)
            for row in rows
            for value in row
        ):
            raise ValueError("surface fields must be finite")
        return rows

    def _empty_accumulators(self, count):
        output = [[0.0 + 0.0j for _ in range(self.n)] for _ in range(count)]
        compensation = [
            [0.0 + 0.0j for _ in range(self.n)] for _ in range(count)
        ]
        return output, compensation

    @staticmethod
    def _accumulate(output, compensation, channel, index, contribution):
        corrected = contribution - compensation[channel][index]
        updated = output[channel][index] + corrected
        compensation[channel][index] = (
            updated - output[channel][index]
        ) - corrected
        output[channel][index] = updated

    def apply_fields(self, fields):
        rows = self._as_fields(fields)
        output, compensation = self._empty_accumulators(len(rows))
        factor_work = 0
        for block in self.low_rank_blocks:
            for left_factor, right_factor in zip(
                block.left_factors,
                block.right_factors,
                strict=True,
            ):
                right_weight = sum(
                    right_factor[local]
                    * self.weights[block.right.global_index(local)]
                    for local in range(block.right.n_nodes)
                )
                left_weight = sum(
                    left_factor[local]
                    * self.weights[block.left.global_index(local)]
                    for local in range(block.left.n_nodes)
                )
                for channel, row in enumerate(rows):
                    right_field = sum(
                        right_factor[local]
                        * self.weights[block.right.global_index(local)]
                        * row[block.right.global_index(local)]
                        for local in range(block.right.n_nodes)
                    )
                    left_field = sum(
                        left_factor[local]
                        * self.weights[block.left.global_index(local)]
                        * row[block.left.global_index(local)]
                        for local in range(block.left.n_nodes)
                    )
                    for local in range(block.left.n_nodes):
                        index = block.left.global_index(local)
                        self._accumulate(
                            output,
                            compensation,
                            channel,
                            index,
                            left_factor[local]
                            * (row[index] * right_weight - right_field),
                        )
                    for local in range(block.right.n_nodes):
                        index = block.right.global_index(local)
                        self._accumulate(
                            output,
                            compensation,
                            channel,
                            index,
                            right_factor[local]
                            * (row[index] * left_weight - left_field),
                        )
                factor_work += block.left.n_nodes + block.right.n_nodes

        exact_pairs = 0
        for block in self.exact_self_blocks:
            for left_local in range(block.node.n_nodes):
                left = block.node.global_index(left_local)
                for right_local in range(left_local + 1, block.node.n_nodes):
                    right = block.node.global_index(right_local)
                    self._apply_exact_pair(
                        left,
                        right,
                        rows,
                        output,
                        compensation,
                    )
                    exact_pairs += 1
        for block in self.exact_cross_blocks:
            for left_local in range(block.left.n_nodes):
                left = block.left.global_index(left_local)
                for right_local in range(block.right.n_nodes):
                    right = block.right.global_index(right_local)
                    self._apply_exact_pair(
                        left,
                        right,
                        rows,
                        output,
                        compensation,
                    )
                    exact_pairs += 1
        self.last_apply_stats = {
            "method": "certified_arbitrary_surface_hierarchy",
            "low_rank_factor_work": factor_work,
            "exact_pairs": exact_pairs,
            "stored_dense_matrix": False,
        }
        return tuple(
            tuple(_clean_scalar(value) for value in row) for row in output
        )

    def _apply_exact_pair(
        self,
        left,
        right,
        rows,
        output,
        compensation,
    ):
        kernel = self._kernel(left, right)
        for channel, row in enumerate(rows):
            delta = row[left] - row[right]
            self._accumulate(
                output,
                compensation,
                channel,
                left,
                self.weights[right] * kernel * delta,
            )
            self._accumulate(
                output,
                compensation,
                channel,
                right,
                -self.weights[left] * kernel * delta,
            )

    def apply(self, values):
        return self.apply_fields((values,))[0]

    def apply_fields_direct(self, fields):
        rows = self._as_fields(fields)
        output, compensation = self._empty_accumulators(len(rows))
        pair_count = 0
        for left in range(self.n):
            for right in range(left + 1, self.n):
                self._apply_exact_pair(
                    left,
                    right,
                    rows,
                    output,
                    compensation,
                )
                pair_count += 1
        self.last_apply_stats = {
            "method": "exact_arbitrary_surface_pair_stream",
            "exact_pairs": pair_count,
            "stored_dense_matrix": False,
        }
        return tuple(
            tuple(_clean_scalar(value) for value in row) for row in output
        )

    def apply_direct(self, values):
        return self.apply_fields_direct((values,))[0]

    def compression_inf_bound(self, values):
        row = tuple(complex(value) for value in values)
        if len(row) != self.n:
            raise ValueError("values must contain one entry per surface node")
        maximum = max(_abs(value) for value in row)
        return max(
            (_abs(row[index]) + maximum) * self.row_kernel_error[index]
            for index in range(self.n)
        )

    def direct_relative_error(self, values):
        direct = self.apply_direct(values)
        candidate = self.apply(values)
        return _relative_l2(direct, candidate)

    def constant_residual(self):
        values = self.apply((1.0,) * self.n)
        return max(_abs(complex(value)) for value in values)

    def stats(self):
        factor_entries = sum(
            block.stored_factor_entries for block in self.low_rank_blocks
        )
        low_rank_pairs = sum(
            block.pair_count for block in self.low_rank_blocks
        )
        exact_pairs = (
            sum(block.pair_count for block in self.exact_cross_blocks)
            + sum(block.pair_count for block in self.exact_self_blocks)
        )
        total_rank = sum(block.rank for block in self.low_rank_blocks)
        result = {
            "nodes": self.n,
            "kernel_power": self.kernel_power,
            "tolerance": self.tolerance,
            "admissibility": self.admissibility,
            "maximum_rank_limit": self.maximum_rank,
            "leaf_size": self.leaf_size,
            "low_rank_blocks": len(self.low_rank_blocks),
            "exact_cross_blocks": len(self.exact_cross_blocks),
            "exact_self_blocks": len(self.exact_self_blocks),
            "rejected_blocks": self._rejected_blocks,
            "total_rank": total_rank,
            "maximum_rank": max(
                (block.rank for block in self.low_rank_blocks),
                default=0,
            ),
            "mean_rank": total_rank / max(len(self.low_rank_blocks), 1),
            "stored_factor_entries": factor_entries,
            "stored_geometry_entries": 4 * self.n,
            "aca_kernel_samples": self._aca_kernel_samples,
            "aca_attempts": len(self._aca_attempts),
            "best_rejected_relative_residual": min(
                (
                    residual
                    for _left, _right, _rank, residual in self._aca_attempts
                    if residual > self.tolerance
                ),
                default=0.0,
            ),
            "certification_kernel_samples": (
                self._certification_kernel_samples
            ),
            "represented_pairs": low_rank_pairs + exact_pairs,
            "pair_partition_residual": self.partition_residual,
            "low_rank_pair_fraction": low_rank_pairs / self.pair_count,
            "exact_pair_fraction": exact_pairs / self.pair_count,
            "maximum_certified_relative_block_residual": max(
                (
                    block.maximum_relative_residual
                    for block in self.low_rank_blocks
                ),
                default=0.0,
            ),
            "maximum_row_kernel_error": max(self.row_kernel_error),
            "temporary_pair_table_entries": 0,
            "stored_dense_distance_matrix": False,
            "stored_dense_operator_matrix": False,
            "topology_assumption": "none",
            "orientation_required": False,
            "parameterization_required": False,
            "compression_certificate": "exhaustive streamed block residual",
            "compile_complexity": (
                "O(N^2 log N) worst case with exhaustive certification"
            ),
            "apply_complexity": (
                "O(sum r_b(m_b+n_b)+P_exact); data dependent, "
                "O(N^2) worst case"
            ),
            "storage_complexity": (
                "O(N+sum r_b(m_b+n_b)+number of terminal blocks)"
            ),
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
                "arbitrary weighted 3D surface nodes",
                "balanced spatial pair partition",
                "fixed-width ACA candidate factors",
            ),
            computed=(
                f"matrix-free |X-Y|^-{self.kernel_power:g} graph action",
                "exhaustive discrete residual certificate",
            ),
            repaid=(
                "all rejected and terminal blocks by exact pair streams",
                "constant graph channel",
                "weighted transpose pairing",
                "unordered pair partition checksum",
            ),
            residuals=(
                ("compression_inf_bound", compression_bound),
                ("constant_residual", constant_residual),
                (
                    "pair_partition_residual",
                    float(self.partition_residual),
                ),
            ),
            residual_norm=max(
                compression_bound,
                constant_residual,
                _abs(float(self.partition_residual)),
            ),
            status="borrowed_repaid",
            notes=(
                "Correctness is topology independent. Compression is "
                "data dependent; exact repayment preserves correctness when "
                "a surface does not admit bounded-rank separated blocks."
            ),
        )
        return ArbitrarySurfaceEvaluation(
            result,
            compression_bound,
            ledger,
            stats,
        )


CertifiedArbitrarySurfaceQJet = ProductionRieszQJet
ArbitrarySurfaceQJet = ProductionRieszQJet


__all__ = [
    "ArbitrarySurfaceEvaluation",
    "ArbitrarySurfaceQJet",
    "CertifiedArbitrarySurfaceQJet",
    "CertifiedSurfaceCrossBlock",
    "ExactSurfaceCrossBlock",
    "ExactSurfaceSelfBlock",
    "ExhaustiveArbitrarySurfaceQJet",
    "GeneratedEuclideanDiscriminantQJet",
    "inverse_square_sums_from_log_discriminant_laplacians",
    "triangle_lumped_vertex_weights",
]
