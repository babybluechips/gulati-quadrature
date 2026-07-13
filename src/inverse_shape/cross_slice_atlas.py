"""Static product-patch atlas for curved cross-slice interactions.

The conic-pencil representation isolates the singular interaction on each
complete conic slice.  This module compiles the remaining interaction between
distinct slices on product patches

    (contiguous slice interval) x (contiguous phase interval).

Separated patch pairs are sampled only through adaptive pivot rows, pivot
columns, and deterministic audit points.  An accepted pair retains factors
``K_AB ~= sum_k u_k v_k^T`` and never a pair matrix.  A pair which does not
meet tolerance within the economic rank ``r(m+n) < mn`` is subdivided.  Only
terminal neighboring patches are repaid by exact streamed pairs.

Both directions of every low-rank graph block use the same factors.  The
constant nullspace and weighted self-adjointness therefore hold algebraically
even before comparison with the direct reference.
"""

from inverse_shape.conic_pencil_surface import ConicPencilSurfaceQJet
from inverse_shape.quadrature import (
    TAU,
    BorrowComputeRepayLedger,
    _abs,
    _clean_scalar,
    _cos,
    _fft_precise,
    _finite,
    _ifft_precise,
    _sin,
    _sqrt,
)


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


def _even_indices(count, wanted):
    count = int(count)
    wanted = min(int(wanted), count)
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


class _SliceRange:
    """Binary slice-range tree used only to enumerate disjoint slice pairs."""

    def __init__(self, first_slice, last_slice):
        self.first_slice = int(first_slice)
        self.last_slice = int(last_slice)
        self.children = tuple()
        if self.last_slice - self.first_slice > 1:
            middle = (self.first_slice + self.last_slice) // 2
            self.children = (
                _SliceRange(self.first_slice, middle),
                _SliceRange(middle, self.last_slice),
            )

    @property
    def is_leaf(self):
        return not self.children


class _AtlasPatch:
    """A rectangular chart patch represented by four integer endpoints."""

    def __init__(
        self,
        first_slice,
        last_slice,
        first_phase,
        last_phase,
        n_theta,
        points,
    ):
        self.first_slice = int(first_slice)
        self.last_slice = int(last_slice)
        self.first_phase = int(first_phase)
        self.last_phase = int(last_phase)
        self.n_theta = int(n_theta)
        self._points = points
        self._children = None
        inverse = 1.0 / self.n_nodes
        self.center = tuple(
            sum(points[index][axis] for index in self.indices()) * inverse
            for axis in range(3)
        )
        self.radius = max(
            _vnorm(_vsub(points[index], self.center))
            for index in self.indices()
        )

    @property
    def n_slices(self):
        return self.last_slice - self.first_slice

    @property
    def n_phases(self):
        return self.last_phase - self.first_phase

    @property
    def n_nodes(self):
        return self.n_slices * self.n_phases

    def global_index(self, local_index):
        local = int(local_index)
        slice_offset = local // self.n_phases
        phase_offset = local - slice_offset * self.n_phases
        return (
            (self.first_slice + slice_offset) * self.n_theta
            + self.first_phase
            + phase_offset
        )

    def indices(self):
        return tuple(self.global_index(local) for local in range(self.n_nodes))

    def probe_locals(self, slice_count, phase_count):
        slices = _even_indices(self.n_slices, slice_count)
        phases = _even_indices(self.n_phases, phase_count)
        return tuple(
            slice_offset * self.n_phases + phase_offset
            for slice_offset in slices
            for phase_offset in phases
        )

    def split(self):
        if self._children is not None:
            return self._children
        candidates = []
        if self.n_slices > 1:
            middle = (self.first_slice + self.last_slice) // 2
            candidates.append(
                (
                    _AtlasPatch(
                        self.first_slice,
                        middle,
                        self.first_phase,
                        self.last_phase,
                        self.n_theta,
                        self._points,
                    ),
                    _AtlasPatch(
                        middle,
                        self.last_slice,
                        self.first_phase,
                        self.last_phase,
                        self.n_theta,
                        self._points,
                    ),
                )
            )
        if self.n_phases > 1:
            middle = (self.first_phase + self.last_phase) // 2
            candidates.append(
                (
                    _AtlasPatch(
                        self.first_slice,
                        self.last_slice,
                        self.first_phase,
                        middle,
                        self.n_theta,
                        self._points,
                    ),
                    _AtlasPatch(
                        self.first_slice,
                        self.last_slice,
                        middle,
                        self.last_phase,
                        self.n_theta,
                        self._points,
                    ),
                )
            )
        if not candidates:
            self._children = tuple()
            return self._children
        self._children = min(
            candidates,
            key=lambda pair: (
                max(pair[0].radius, pair[1].radius),
                pair[0].radius + pair[1].radius,
            ),
        )
        return self._children

    def split_slices(self):
        if self.n_slices <= 1:
            return tuple()
        middle = (self.first_slice + self.last_slice) // 2
        return (
            _AtlasPatch(
                self.first_slice,
                middle,
                self.first_phase,
                self.last_phase,
                self.n_theta,
                self._points,
            ),
            _AtlasPatch(
                middle,
                self.last_slice,
                self.first_phase,
                self.last_phase,
                self.n_theta,
                self._points,
            ),
        )


class StaticPhaseDifferenceChart:
    """One local cross-slice kernel in difference/modulation coordinates."""

    def __init__(
        self,
        left_slice,
        right_slice,
        n_theta,
        points,
        kernel_power,
        tolerance,
        maximum_modes,
    ):
        self.left_slice = int(left_slice)
        self.right_slice = int(right_slice)
        self.n_theta = int(n_theta)
        self.kernel_power = float(kernel_power)
        self.tolerance = float(tolerance)
        self.maximum_modes = min(int(maximum_modes), self.n_theta)
        self.left_start = self.left_slice * self.n_theta
        self.right_start = self.right_slice * self.n_theta
        mode_norms = [0.0 for _ in range(self.n_theta)]
        maximum_kernel = 0.0

        def sample_diagonal(offset):
            nonlocal maximum_kernel
            diagonal = []
            for source_phase in range(self.n_theta):
                target_phase = (source_phase + offset) % self.n_theta
                left = self.left_start + target_phase
                right = self.right_start + source_phase
                difference = _vsub(points[left], points[right])
                distance_squared = _vdot(difference, difference)
                if distance_squared <= 1.0e-28:
                    raise ValueError("distinct local cross-slice nodes collide")
                value = distance_squared ** (-0.5 * self.kernel_power)
                maximum_kernel = max(maximum_kernel, value)
                diagonal.append(value)
            return tuple(diagonal)

        for offset in range(self.n_theta):
            transformed = _fft_precise(sample_diagonal(offset))
            for mode in range(self.n_theta):
                coefficient = transformed[mode] / self.n_theta
                mode_norms[mode] = max(mode_norms[mode], _abs(coefficient))
        mode_norms = tuple(mode_norms)
        ordering = [0]
        for frequency in range(1, (self.n_theta + 1) // 2):
            ordering.append(frequency)
            ordering.append(self.n_theta - frequency)
        if self.n_theta % 2 == 0:
            ordering.append(self.n_theta // 2)
        total_tail = sum(mode_norms)
        selected = []
        target = self.tolerance * max(maximum_kernel, 1.0e-300)
        for mode in ordering:
            selected.append(mode)
            total_tail -= mode_norms[mode]
            if total_tail <= target:
                break
        self.compile_kernel_samples = self.n_theta * self.n_theta
        self.maximum_kernel = maximum_kernel
        self.tail_bound = max(total_tail, 0.0)
        self.direct = (
            len(selected) > self.maximum_modes
            or 3 * len(selected) >= self.n_theta
        )
        if self.direct:
            self.modes = tuple()
            self.phases = tuple()
            self.symbols = tuple()
            self.transpose_symbols = tuple()
        else:
            self.modes = tuple(selected)
            self.phases = tuple(
                tuple(
                    complex(
                        _cos(TAU * mode * phase / self.n_theta),
                        _sin(TAU * mode * phase / self.n_theta),
                    )
                    for phase in range(self.n_theta)
                )
                for mode in self.modes
            )
            kernels = [
                [0.0 + 0.0j for _ in range(self.n_theta)]
                for _ in self.modes
            ]
            for offset in range(self.n_theta):
                transformed = _fft_precise(sample_diagonal(offset))
                for local_mode, mode in enumerate(self.modes):
                    kernels[local_mode][offset] = (
                        transformed[mode] / self.n_theta
                    )
            kernels = tuple(tuple(kernel) for kernel in kernels)
            self.compile_kernel_samples += self.n_theta * self.n_theta
            self.symbols = tuple(
                tuple(_fft_precise(kernel)) for kernel in kernels
            )
            self.transpose_symbols = tuple(
                tuple(
                    _fft_precise(
                        tuple(
                            kernel[(-offset) % self.n_theta]
                            for offset in range(self.n_theta)
                        )
                    )
                )
                for kernel in kernels
            )

    @property
    def pair_count(self):
        return self.n_theta * self.n_theta

    @property
    def stored_channel_entries(self):
        if self.direct:
            return 0
        return 3 * len(self.modes) * self.n_theta

    def _convolve(self, symbol, values):
        transformed = _fft_precise(values)
        return tuple(
            _ifft_precise(
                tuple(
                    transformed[index] * symbol[index]
                    for index in range(self.n_theta)
                )
            )
        )

    def apply_fields(self, rows, weights, kernel):
        left_indices = tuple(
            self.left_start + phase for phase in range(self.n_theta)
        )
        right_indices = tuple(
            self.right_start + phase for phase in range(self.n_theta)
        )
        left_output = [
            [0.0 + 0.0j for _ in range(self.n_theta)]
            for _ in rows
        ]
        right_output = [
            [0.0 + 0.0j for _ in range(self.n_theta)]
            for _ in rows
        ]
        if self.direct:
            for left_local, left in enumerate(left_indices):
                for right_local, right in enumerate(right_indices):
                    value = kernel(left, right)
                    for channel, row in enumerate(rows):
                        delta = row[left] - row[right]
                        left_output[channel][left_local] += (
                            weights[right] * value * delta
                        )
                        right_output[channel][right_local] -= (
                            weights[left] * value * delta
                        )
            return left_output, right_output, self.pair_count

        left_weights = tuple(weights[index] for index in left_indices)
        right_weights = tuple(weights[index] for index in right_indices)
        for channel_index, phases in enumerate(self.phases):
            symbol = self.symbols[channel_index]
            transpose_symbol = self.transpose_symbols[channel_index]
            right_modulated_weights = tuple(
                phases[local] * right_weights[local]
                for local in range(self.n_theta)
            )
            right_weight_potential = self._convolve(
                symbol,
                right_modulated_weights,
            )
            left_weight_potential = self._convolve(
                transpose_symbol,
                left_weights,
            )
            for field_index, row in enumerate(rows):
                right_field_potential = self._convolve(
                    symbol,
                    tuple(
                        right_modulated_weights[local]
                        * row[right_indices[local]]
                        for local in range(self.n_theta)
                    ),
                )
                left_field_potential = self._convolve(
                    transpose_symbol,
                    tuple(
                        left_weights[local] * row[left_indices[local]]
                        for local in range(self.n_theta)
                    ),
                )
                for local in range(self.n_theta):
                    left_output[field_index][local] += (
                        row[left_indices[local]]
                        * right_weight_potential[local]
                        - right_field_potential[local]
                    )
                    right_output[field_index][local] += phases[local] * (
                        row[right_indices[local]]
                        * left_weight_potential[local]
                        - left_field_potential[local]
                    )
        return left_output, right_output, 0


class StaticAtlasCrossBlock:
    """One retained product-patch factorization ``U V^T``."""

    def __init__(
        self,
        left_patch,
        right_patch,
        left_factors,
        right_factors,
        sampled_relative_residual,
        sampled_entries,
    ):
        self.left_patch = left_patch
        self.right_patch = right_patch
        self.left_factors = tuple(tuple(row) for row in left_factors)
        self.right_factors = tuple(tuple(row) for row in right_factors)
        self.sampled_relative_residual = float(sampled_relative_residual)
        self.sampled_entries = int(sampled_entries)

    @property
    def rank(self):
        return len(self.left_factors)

    @property
    def pair_count(self):
        return self.left_patch.n_nodes * self.right_patch.n_nodes

    @property
    def stored_factor_entries(self):
        return self.rank * (
            self.left_patch.n_nodes + self.right_patch.n_nodes
        )


class ExactAtlasPatchPair:
    """Four-endpoint metadata for one exact terminal patch repayment."""

    def __init__(self, left_patch, right_patch):
        self.left_patch = left_patch
        self.right_patch = right_patch

    @property
    def pair_count(self):
        return self.left_patch.n_nodes * self.right_patch.n_nodes


class StaticCrossSliceAtlasEvaluation:
    def __init__(self, values, ledger, stats):
        self.values = values
        self.ledger = ledger
        self.stats = stats


class StaticCrossSliceAtlasQJet:
    """Compile and apply the curved cross-slice residual without dense storage."""

    def __init__(
        self,
        surface,
        kernel_power=2.0,
        tolerance=1.0e-10,
        admissibility=0.3,
        maximum_rank=48,
        leaf_nodes=8,
        local_slice_span=1,
        maximum_phase_modes=12,
        probe_slices=7,
        probe_phases=16,
    ):
        if not isinstance(surface, ConicPencilSurfaceQJet):
            raise TypeError("surface must be a ConicPencilSurfaceQJet")
        self.surface = surface
        self.kernel_power = float(kernel_power)
        self.tolerance = float(tolerance)
        self.admissibility = float(admissibility)
        self.maximum_rank = int(maximum_rank)
        self.leaf_nodes = int(leaf_nodes)
        self.local_slice_span = int(local_slice_span)
        self.maximum_phase_modes = int(maximum_phase_modes)
        self.probe_slices = int(probe_slices)
        self.probe_phases = int(probe_phases)
        if self.kernel_power <= 0.0 or not _finite(self.kernel_power):
            raise ValueError("kernel_power must be positive and finite")
        if self.tolerance <= 0.0 or not _finite(self.tolerance):
            raise ValueError("tolerance must be positive and finite")
        if self.admissibility <= 0.0 or not _finite(self.admissibility):
            raise ValueError("admissibility must be positive and finite")
        if self.maximum_rank < 1 or self.leaf_nodes < 1:
            raise ValueError("maximum_rank and leaf_nodes must be positive")
        if self.local_slice_span < 0 or self.maximum_phase_modes < 1:
            raise ValueError(
                "local_slice_span must be nonnegative and phase modes positive"
            )
        if self.probe_slices < 2 or self.probe_phases < 4:
            raise ValueError("the atlas needs at least 2x4 deterministic probes")

        self.nodes = self.surface.generate_nodes()
        self.local_slice_pairs = self._build_local_slice_pairs()
        self.phase_charts = tuple(
            StaticPhaseDifferenceChart(
                left,
                right,
                self.surface.n_theta,
                self.nodes.points,
                self.kernel_power,
                0.25 * self.tolerance,
                self.maximum_phase_modes,
            )
            for left, right in sorted(self.local_slice_pairs)
        )
        self.low_rank_blocks = []
        self.exact_patch_pairs = []
        self._compile_kernel_samples = 0
        self._rejected_cross_blocks = 0
        self._aca_attempts = []
        self._compile_self(_SliceRange(0, self.surface.n_slices))
        self.low_rank_blocks = tuple(self.low_rank_blocks)
        self.exact_patch_pairs = tuple(self.exact_patch_pairs)
        expected = (
            self.surface.n_slices
            * (self.surface.n_slices - 1)
            // 2
            * self.surface.n_theta
            * self.surface.n_theta
        )
        represented = sum(chart.pair_count for chart in self.phase_charts) + sum(
            block.pair_count for block in self.low_rank_blocks
        ) + sum(block.pair_count for block in self.exact_patch_pairs)
        if represented != expected:
            raise RuntimeError("cross-slice atlas failed its pair partition check")
        self.cross_pair_count = expected
        self.partition_residual = represented - expected
        self.last_apply_stats = {}

    @property
    def n_nodes(self):
        return self.surface.n_nodes

    def _patch(self, slice_range):
        return _AtlasPatch(
            slice_range.first_slice,
            slice_range.last_slice,
            0,
            self.surface.n_theta,
            self.surface.n_theta,
            self.nodes.points,
        )

    def _build_local_slice_pairs(self):
        count = self.surface.n_slices
        span = min(self.local_slice_span, count - 1)
        pairs = set()
        for left in range(count):
            for offset in range(1, span + 1):
                right = left + offset
                if self.surface.periodic:
                    right %= count
                elif right >= count:
                    continue
                pair = (min(left, right), max(left, right))
                if pair[0] != pair[1]:
                    pairs.add(pair)
        return frozenset(pairs)

    def _contains_local_pair(self, left, right):
        for left_slice in range(left.first_slice, left.last_slice):
            for right_slice in range(right.first_slice, right.last_slice):
                pair = (
                    min(left_slice, right_slice),
                    max(left_slice, right_slice),
                )
                if pair in self.local_slice_pairs:
                    return True
        return False

    def _kernel(self, left, right):
        difference = _vsub(self.nodes.points[left], self.nodes.points[right])
        distance_squared = _vdot(difference, difference)
        if distance_squared <= 1.0e-28:
            raise ValueError("distinct cross-slice surface nodes collide")
        return distance_squared ** (-0.5 * self.kernel_power)

    def _admissible(self, left, right):
        distance = _vnorm(_vsub(left.center, right.center))
        gap = distance - left.radius - right.radius
        if gap <= 1.0e-30:
            return False
        return max(left.radius, right.radius) / gap <= self.admissibility

    def _compile_self(self, slice_range):
        if slice_range.is_leaf:
            return
        left, right = slice_range.children
        self._compile_self(left)
        self._compile_patch_pair(self._patch(left), self._patch(right))
        self._compile_self(right)

    def _compile_patch_pair(self, left, right):
        if self._contains_local_pair(left, right):
            if left.n_slices == 1 and right.n_slices == 1:
                return
            if left.n_slices >= right.n_slices and left.n_slices > 1:
                for child in left.split_slices():
                    self._compile_patch_pair(child, right)
                return
            for child in right.split_slices():
                self._compile_patch_pair(left, child)
            return

        if self._admissible(left, right):
            block = self._adaptive_cross_block(left, right)
            if block is not None:
                self.low_rank_blocks.append(block)
                return
            self._rejected_cross_blocks += 1

        left_terminal = left.n_nodes <= self.leaf_nodes
        right_terminal = right.n_nodes <= self.leaf_nodes
        if left_terminal and right_terminal:
            self.exact_patch_pairs.append(ExactAtlasPatchPair(left, right))
            return

        if not left_terminal and (
            right_terminal
            or left.radius > right.radius
            or (
                left.radius == right.radius
                and left.n_nodes >= right.n_nodes
            )
        ):
            children = left.split()
            if not children:
                self.exact_patch_pairs.append(ExactAtlasPatchPair(left, right))
                return
            for child in children:
                self._compile_patch_pair(child, right)
            return

        children = right.split()
        if not children:
            self.exact_patch_pairs.append(ExactAtlasPatchPair(left, right))
            return
        for child in children:
            self._compile_patch_pair(left, child)

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

        left_probes = left.probe_locals(
            self.probe_slices,
            self.probe_phases,
        )
        right_probes = right.probe_locals(
            self.probe_slices,
            self.probe_phases,
        )

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
        sampled_relative_residual = 1.0
        for _rank in range(rank_limit):
            (
                sampled_relative_residual,
                pivot_left,
                maximum_kernel,
            ) = probe_error(require_unused=True)
            if sampled_relative_residual <= self.tolerance:
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
            if _abs(pivot) <= 8.0e-17 * max(maximum_kernel, 1.0e-300):
                break
            column = tuple(
                residual(left_local, pivot_right)
                for left_local in range(left.n_nodes)
            )
            left_factors.append(tuple(value / pivot for value in column))
            right_factors.append(row)
            used_left.add(pivot_left)

        sampled_relative_residual, _pivot, _scale = probe_error()
        self._compile_kernel_samples += len(cache)
        self._aca_attempts.append(
            (
                left.n_nodes,
                right.n_nodes,
                len(left_factors),
                sampled_relative_residual,
                len(cache),
            )
        )
        if sampled_relative_residual > self.tolerance or not left_factors:
            return None
        return StaticAtlasCrossBlock(
            left,
            right,
            tuple(left_factors),
            tuple(right_factors),
            sampled_relative_residual,
            len(cache),
        )

    def _as_fields(self, fields):
        rows = tuple(tuple(complex(value) for value in row) for row in fields)
        if not rows or any(len(row) != self.n_nodes for row in rows):
            raise ValueError("each field must contain one value per surface node")
        return rows

    def _same_slice_direct(self, rows):
        output = [
            [0.0 + 0.0j for _ in range(self.n_nodes)]
            for _ in rows
        ]
        pair_count = 0
        for slice_index in range(self.surface.n_slices):
            start = slice_index * self.surface.n_theta
            stop = start + self.surface.n_theta
            for left in range(start, stop):
                for right in range(left + 1, stop):
                    kernel = self._kernel(left, right)
                    pair_count += 1
                    for channel, row in enumerate(rows):
                        delta = row[left] - row[right]
                        output[channel][left] += (
                            self.nodes.weights[right] * kernel * delta
                        )
                        output[channel][right] -= (
                            self.nodes.weights[left] * kernel * delta
                        )
        return output, pair_count

    def apply_cross_fields(self, fields):
        rows = self._as_fields(fields)
        output = [
            [0.0 + 0.0j for _ in range(self.n_nodes)]
            for _ in rows
        ]
        compensation = [
            [0.0 + 0.0j for _ in range(self.n_nodes)]
            for _ in rows
        ]

        def accumulate(channel, index, contribution):
            corrected = contribution - compensation[channel][index]
            updated = output[channel][index] + corrected
            compensation[channel][index] = (
                updated - output[channel][index]
            ) - corrected
            output[channel][index] = updated

        phase_direct_pairs = 0
        phase_channel_work = 0
        for chart in self.phase_charts:
            left_values, right_values, direct_pairs = chart.apply_fields(
                rows,
                self.nodes.weights,
                self._kernel,
            )
            phase_direct_pairs += direct_pairs
            phase_channel_work += len(chart.modes) * self.surface.n_theta
            for channel in range(len(rows)):
                for local in range(self.surface.n_theta):
                    accumulate(
                        channel,
                        chart.left_start + local,
                        left_values[channel][local],
                    )
                    accumulate(
                        channel,
                        chart.right_start + local,
                        right_values[channel][local],
                    )

        factor_work = 0
        for block in self.low_rank_blocks:
            left_indices = block.left_patch.indices()
            right_indices = block.right_patch.indices()
            for left_factor, right_factor in zip(
                block.left_factors,
                block.right_factors,
                strict=True,
            ):
                right_weight = sum(
                    right_factor[local] * self.nodes.weights[index]
                    for local, index in enumerate(right_indices)
                )
                left_weight = sum(
                    left_factor[local] * self.nodes.weights[index]
                    for local, index in enumerate(left_indices)
                )
                for channel, row in enumerate(rows):
                    right_field = sum(
                        right_factor[local]
                        * self.nodes.weights[index]
                        * row[index]
                        for local, index in enumerate(right_indices)
                    )
                    left_field = sum(
                        left_factor[local]
                        * self.nodes.weights[index]
                        * row[index]
                        for local, index in enumerate(left_indices)
                    )
                    for local, index in enumerate(left_indices):
                        accumulate(
                            channel,
                            index,
                            left_factor[local]
                            * (row[index] * right_weight - right_field),
                        )
                    for local, index in enumerate(right_indices):
                        accumulate(
                            channel,
                            index,
                            right_factor[local]
                            * (row[index] * left_weight - left_field),
                        )
                factor_work += len(left_indices) + len(right_indices)

        direct_pairs = 0
        for block in self.exact_patch_pairs:
            left_indices = block.left_patch.indices()
            right_indices = block.right_patch.indices()
            for left in left_indices:
                for right in right_indices:
                    kernel = self._kernel(left, right)
                    direct_pairs += 1
                    for channel, row in enumerate(rows):
                        delta = row[left] - row[right]
                        accumulate(
                            channel,
                            left,
                            self.nodes.weights[right] * kernel * delta,
                        )
                        accumulate(
                            channel,
                            right,
                            -self.nodes.weights[left] * kernel * delta,
                        )

        self.last_apply_stats = {
            "method": "static_product_patch_cross_atlas",
            "cross_only": True,
            "phase_channel_work": phase_channel_work,
            "phase_direct_pairs": phase_direct_pairs,
            "low_rank_factor_work": factor_work,
            "exact_cross_pairs": direct_pairs,
            "stored_dense_matrix": False,
        }
        return tuple(
            tuple(_clean_scalar(value) for value in row) for row in output
        )

    def apply_cross(self, values):
        return self.apply_cross_fields((values,))[0]

    def apply_fields(self, fields):
        rows = self._as_fields(fields)
        if self.kernel_power == 2.0:
            local = self.surface.apply_same_slice_joukowski_fields(rows)
            local_pair_count = 0
            local_method = "exact_joukowski_or_cycle_fft"
        else:
            local, local_pair_count = self._same_slice_direct(rows)
            local = tuple(tuple(row) for row in local)
            local_method = "exact_streamed_same_slice_repayment"
        cross = self.apply_cross_fields(rows)
        result = tuple(
            tuple(
                _clean_scalar(local[channel][index] + cross[channel][index])
                for index in range(self.n_nodes)
            )
            for channel in range(len(rows))
        )
        cross_stats = dict(self.last_apply_stats)
        self.last_apply_stats = {
            **cross_stats,
            "cross_only": False,
            "local_method": local_method,
            "exact_local_pairs": local_pair_count,
        }
        return result

    def apply(self, values):
        return self.apply_fields((values,))[0]

    def apply_dtn_principal(self, values):
        if self.kernel_power != 3.0:
            raise ValueError("the DtN principal normalization requires kernel_power=3")
        return tuple(
            _clean_scalar(complex(value) / TAU) for value in self.apply(values)
        )

    def direct_relative_error(self, values):
        direct = self.surface.apply(
            values,
            kernel_power=self.kernel_power,
            method="direct",
        )
        return _relative_l2(direct, self.apply(values))

    def constant_residual(self):
        values = self.apply((1.0,) * self.n_nodes)
        return max(_abs(complex(value)) for value in values)

    def stats(self):
        total_rank = sum(block.rank for block in self.low_rank_blocks)
        factor_entries = sum(
            block.stored_factor_entries for block in self.low_rank_blocks
        )
        exact_pairs = sum(
            block.pair_count for block in self.exact_patch_pairs
        )
        phase_pairs = sum(chart.pair_count for chart in self.phase_charts)
        low_rank_pairs = sum(
            block.pair_count for block in self.low_rank_blocks
        )
        maximum_residual = max(
            (
                block.sampled_relative_residual
                for block in self.low_rank_blocks
            ),
            default=0.0,
        )
        result = {
            "n_slices": self.surface.n_slices,
            "n_theta": self.surface.n_theta,
            "n_nodes": self.n_nodes,
            "kernel_power": self.kernel_power,
            "tolerance": self.tolerance,
            "admissibility": self.admissibility,
            "leaf_nodes": self.leaf_nodes,
            "local_slice_span": self.local_slice_span,
            "phase_charts": len(self.phase_charts),
            "phase_fft_charts": sum(
                not chart.direct for chart in self.phase_charts
            ),
            "phase_direct_charts": sum(
                chart.direct for chart in self.phase_charts
            ),
            "phase_channels": sum(
                len(chart.modes) for chart in self.phase_charts
            ),
            "maximum_phase_tail_bound": max(
                (chart.tail_bound for chart in self.phase_charts),
                default=0.0,
            ),
            "stored_phase_channel_entries": sum(
                chart.stored_channel_entries for chart in self.phase_charts
            ),
            "low_rank_blocks": len(self.low_rank_blocks),
            "exact_cross_patch_blocks": len(self.exact_patch_pairs),
            "rejected_cross_blocks": self._rejected_cross_blocks,
            "total_rank": total_rank,
            "maximum_rank": max(
                (block.rank for block in self.low_rank_blocks),
                default=0,
            ),
            "mean_rank": total_rank / max(len(self.low_rank_blocks), 1),
            "stored_factor_entries": factor_entries,
            "stored_geometry_entries": self.n_nodes,
            "compile_kernel_samples": self._compile_kernel_samples
            + sum(
                chart.compile_kernel_samples for chart in self.phase_charts
            ),
            "cross_pair_count": self.cross_pair_count,
            "phase_chart_pairs": phase_pairs,
            "represented_low_rank_pairs": low_rank_pairs,
            "exact_cross_pairs_per_apply": exact_pairs,
            "low_rank_pair_fraction": low_rank_pairs
            / max(self.cross_pair_count, 1),
            "exact_cross_pair_fraction": exact_pairs
            / max(self.cross_pair_count, 1),
            "phase_chart_pair_fraction": phase_pairs
            / max(self.cross_pair_count, 1),
            "cross_pair_partition_residual": self.partition_residual,
            "maximum_sampled_block_residual": maximum_residual,
            "temporary_pair_table_entries": 0,
            "stored_dense_distance_matrix": False,
            "stored_dense_operator_matrix": False,
            "compile_complexity": (
                "O(sum attempts r_b(m_b+n_b)+probe work); no pair matrix"
            ),
            "apply_complexity": (
                "O(sum_b r_b(m_b+n_b)+P_near+N*n_theta local); "
                "O(r N log N+N*n_theta) for bounded rank/neighbors"
            ),
            "storage_complexity": (
                "O(N+L_local*N+sum_b r_b(m_b+n_b))"
            ),
        }
        result.update(self.last_apply_stats)
        return result

    def evaluate(self, values):
        result = self.apply(values)
        residual = self.constant_residual()
        stats = self.stats()
        ledger = BorrowComputeRepayLedger(
            borrowed=(
                "conic-pencil surface value/three-jets",
                "complete same-slice Joukowski or cycle chart",
                "deterministic product-patch endpoint probes",
            ),
            computed=(
                f"static curved cross-slice |X-Y|^-{self.kernel_power:g} atlas",
                "symmetric low-rank weighted graph action",
            ),
            repaid=(
                "exact terminal cross-patch interactions",
                "exact complete-slice singular channel",
                "constant nullspace and weighted transpose pairing",
                "cross-pair partition checksum",
            ),
            residuals=(
                ("constant_residual", residual),
                (
                    "maximum_sampled_block_residual",
                    stats["maximum_sampled_block_residual"],
                ),
                (
                    "cross_pair_partition_residual",
                    float(self.partition_residual),
                ),
            ),
            residual_norm=max(
                residual,
                stats["maximum_sampled_block_residual"],
                _abs(float(self.partition_residual)),
            ),
            status="borrowed_repaid",
            notes=(
                "The sampled ACA residual is an implementation certificate, "
                "not a continuum error bound. Direct matrix-free comparison "
                "is retained as the independent numerical audit."
            ),
        )
        return StaticCrossSliceAtlasEvaluation(result, ledger, stats)


CurvedCrossSliceAtlasQJet = StaticCrossSliceAtlasQJet


__all__ = [
    "CurvedCrossSliceAtlasQJet",
    "ExactAtlasPatchPair",
    "StaticAtlasCrossBlock",
    "StaticCrossSliceAtlasEvaluation",
    "StaticCrossSliceAtlasQJet",
    "StaticPhaseDifferenceChart",
]
