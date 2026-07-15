"""Topology-aware singular-cell and harmonic-moment repayment in three dimensions.

The raw surface graph

    A_N f_i = (2*pi)^-1 sum_{j != i} w_j (f_i-f_j) / |x_i-x_j|^3

omits the singular cell represented by node ``i``.  On a tangent disk of
geodesic radius ``a`` the omitted channel has the exact formal expansion

    -a/4 Delta f - a^3/192 Delta^2 f - a^5/11520 Delta^3 f - ... .

``MeshDifferentialGeometry`` evaluates this expansion with the sparse
cotangent Laplace--Beltrami operator.  The effective radius is recovered from
the barycentric cell area with the Gaussian-curvature inversion

    a_geo = sqrt(A/pi) * (1 + K A/(24*pi) + 3 K^2 A^2/(640*pi^2)).

The remaining smooth/curvature channel is compiled by exact harmonic
reproduction.  Every solid harmonic polynomial ``p`` satisfies

    Lambda_Gamma(p|Gamma) = grad(p).n

on every domain.  A fixed-width local stencil is therefore chosen at each
node so that the corrected operator reproduces these fluxes through a selected
degree.  Application stores only sparse stencils and costs ``O(N s)`` for
fixed stencil width ``s``.  No dense surface matrix or pair table is formed.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from inverse_shape.quadrature import PI, _abs, _cos, _finite, _sin, _sqrt

Point3 = tuple[float, float, float]
ComplexVector = tuple[complex, ...]


def _point(value: Iterable[float]) -> Point3:
    row = tuple(float(component) for component in value)
    if len(row) != 3 or any(not _finite(component) for component in row):
        raise ValueError("surface points must contain three finite coordinates")
    return row


def _sub(left: Point3, right: Point3) -> Point3:
    return (
        left[0] - right[0],
        left[1] - right[1],
        left[2] - right[2],
    )


def _dot(left: Point3, right: Point3) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _cross(left: Point3, right: Point3) -> Point3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _norm(value: Point3) -> float:
    return _sqrt(max(_dot(value, value), 0.0))


def _scale(factor: float, value: Point3) -> Point3:
    return (factor * value[0], factor * value[1], factor * value[2])


def _unit(value: Point3) -> Point3:
    length = _norm(value)
    if length <= 1.0e-30:
        raise ValueError("cannot normalize a zero geometric vector")
    return _scale(1.0 / length, value)


def _atan2(y_value: float, x_value: float) -> float:
    # The foundational quadrature module intentionally avoids importing math.
    # Local mesh compilation is not an FFT kernel, so the standard scalar
    # implementation is appropriate here.
    import math

    return math.atan2(float(y_value), float(x_value))


def _weighted_inner(
    weights: tuple[float, ...],
    left: Iterable[complex],
    right: Iterable[complex],
) -> complex:
    return sum(
        weight * complex(first).conjugate() * complex(second)
        for weight, first, second in zip(weights, left, right, strict=True)
    )


def _weighted_norm(weights: tuple[float, ...], values: Iterable[complex]) -> float:
    return _sqrt(max(_weighted_inner(weights, values, values).real, 0.0))


def _degree_monomials(degree: int) -> tuple[tuple[int, int, int], ...]:
    rows = []
    for x_order in range(degree, -1, -1):
        for y_order in range(degree - x_order, -1, -1):
            z_order = degree - x_order - y_order
            rows.append((x_order, y_order, z_order))
    return tuple(rows)


def _nullspace(matrix: list[list[float]], columns: int) -> tuple[tuple[float, ...], ...]:
    if not matrix:
        return tuple(
            tuple(1.0 if row == column else 0.0 for row in range(columns))
            for column in range(columns)
        )
    rows = [list(row) for row in matrix]
    pivot_columns: list[int] = []
    pivot_row = 0
    tolerance = 1.0e-13
    for column in range(columns):
        selected = max(
            range(pivot_row, len(rows)),
            key=lambda row: _abs(rows[row][column]),
            default=pivot_row,
        )
        if _abs(rows[selected][column]) <= tolerance:
            continue
        rows[pivot_row], rows[selected] = rows[selected], rows[pivot_row]
        pivot = rows[pivot_row][column]
        rows[pivot_row] = [value / pivot for value in rows[pivot_row]]
        for row in range(len(rows)):
            if row == pivot_row:
                continue
            factor = rows[row][column]
            if factor == 0.0:
                continue
            rows[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(rows[row], rows[pivot_row], strict=True)
            ]
        pivot_columns.append(column)
        pivot_row += 1
        if pivot_row == len(rows):
            break
    free_columns = [column for column in range(columns) if column not in pivot_columns]
    basis = []
    for free in free_columns:
        vector = [0.0 for _ in range(columns)]
        vector[free] = 1.0
        for row, pivot in reversed(tuple(enumerate(pivot_columns))):
            vector[pivot] = -sum(
                rows[row][column] * vector[column]
                for column in free_columns
            )
        length = _sqrt(sum(value * value for value in vector))
        basis.append(tuple(value / length for value in vector))
    return tuple(basis)


@dataclass(frozen=True)
class HarmonicPolynomial3D:
    """One homogeneous solid harmonic and its Cartesian gradient."""

    degree: int
    index: int
    terms: tuple[tuple[tuple[int, int, int], float], ...]

    @property
    def name(self) -> str:
        return f"H{self.degree}_{self.index}"

    def value_gradient(self, point: Point3) -> tuple[float, Point3]:
        x_value, y_value, z_value = point
        maximum = self.degree
        x_powers = [1.0]
        y_powers = [1.0]
        z_powers = [1.0]
        for _ in range(maximum):
            x_powers.append(x_powers[-1] * x_value)
            y_powers.append(y_powers[-1] * y_value)
            z_powers.append(z_powers[-1] * z_value)
        value = 0.0
        gradient = [0.0, 0.0, 0.0]
        for exponent, coefficient in self.terms:
            a, b, c = exponent
            value += coefficient * x_powers[a] * y_powers[b] * z_powers[c]
            if a:
                gradient[0] += (
                    coefficient * a * x_powers[a - 1] * y_powers[b] * z_powers[c]
                )
            if b:
                gradient[1] += (
                    coefficient * b * x_powers[a] * y_powers[b - 1] * z_powers[c]
                )
            if c:
                gradient[2] += (
                    coefficient * c * x_powers[a] * y_powers[b] * z_powers[c - 1]
                )
        return value, (gradient[0], gradient[1], gradient[2])


def solid_harmonic_polynomials(maximum_degree: int) -> tuple[HarmonicPolynomial3D, ...]:
    """Generate a deterministic real basis of homogeneous harmonic polynomials."""

    maximum = int(maximum_degree)
    if maximum < 0:
        raise ValueError("harmonic degree must be nonnegative")
    output = []
    for degree in range(1, maximum + 1):
        columns = _degree_monomials(degree)
        rows = _degree_monomials(degree - 2) if degree >= 2 else tuple()
        row_lookup = {exponent: index for index, exponent in enumerate(rows)}
        laplacian = [[0.0 for _ in columns] for _ in rows]
        for column, exponent in enumerate(columns):
            for axis in range(3):
                order = exponent[axis]
                if order < 2:
                    continue
                reduced = list(exponent)
                reduced[axis] -= 2
                laplacian[row_lookup[tuple(reduced)]][column] += order * (order - 1)
        basis = _nullspace(laplacian, len(columns))
        expected = 2 * degree + 1
        if len(basis) != expected:
            raise RuntimeError("solid-harmonic nullspace has the wrong dimension")
        for index, coefficients in enumerate(basis):
            terms = tuple(
                (exponent, coefficient)
                for exponent, coefficient in zip(columns, coefficients, strict=True)
                if _abs(coefficient) > 1.0e-15
            )
            output.append(HarmonicPolynomial3D(degree, index, terms))
    return tuple(output)


class MeshDifferentialGeometry:
    """Sparse cotangent geometry and curvature jets for a triangle mesh."""

    def __init__(
        self,
        points: Iterable[Iterable[float]],
        faces: Iterable[Iterable[int]],
        weights: Iterable[float],
        normals: Iterable[Iterable[float]] | None = None,
    ) -> None:
        self.points = tuple(_point(point) for point in points)
        self.faces = tuple(tuple(int(index) for index in face) for face in faces)
        self.weights = tuple(float(value) for value in weights)
        self.n = len(self.points)
        if len(self.weights) != self.n or any(value <= 0.0 for value in self.weights):
            raise ValueError("mesh repayment requires one positive weight per vertex")
        adjacency = [dict() for _ in self.points]
        neighbors = [set() for _ in self.points]
        normal_sums = [[0.0, 0.0, 0.0] for _ in self.points]
        angle_sums = [0.0 for _ in self.points]
        edge_counts: dict[tuple[int, int], int] = {}
        signed_volume = 0.0
        for face in self.faces:
            if len(face) != 3 or len(set(face)) != 3:
                raise ValueError("mesh repayment requires triangle faces")
            if any(index < 0 or index >= self.n for index in face):
                raise ValueError("mesh repayment face index is out of range")
            first, second, third = (self.points[index] for index in face)
            left = _sub(second, first)
            right = _sub(third, first)
            cross = _cross(left, right)
            twice_area = _norm(cross)
            if twice_area <= 1.0e-30:
                raise ValueError("mesh repayment does not support degenerate faces")
            unit_normal = _scale(1.0 / twice_area, cross)
            area = 0.5 * twice_area
            signed_volume += _dot(first, _cross(second, third)) / 6.0
            for vertex in face:
                for axis in range(3):
                    normal_sums[vertex][axis] += area * unit_normal[axis]
            for offset in range(3):
                vertex = face[offset]
                left_index = face[(offset + 1) % 3]
                right_index = face[(offset + 2) % 3]
                first_edge = _sub(self.points[left_index], self.points[vertex])
                second_edge = _sub(self.points[right_index], self.points[vertex])
                edge_cross = _norm(_cross(first_edge, second_edge))
                edge_dot = _dot(first_edge, second_edge)
                angle_sums[vertex] += _atan2(edge_cross, edge_dot)
                cotangent = edge_dot / edge_cross
                coefficient = 0.5 * cotangent
                adjacency[left_index][right_index] = (
                    adjacency[left_index].get(right_index, 0.0) + coefficient
                )
                adjacency[right_index][left_index] = (
                    adjacency[right_index].get(left_index, 0.0) + coefficient
                )
                neighbors[vertex].add(left_index)
                neighbors[vertex].add(right_index)
                edge = (min(vertex, left_index), max(vertex, left_index))
                edge_counts[edge] = edge_counts.get(edge, 0) + 1
        orientation = -1.0 if signed_volume < 0.0 else 1.0
        if normals is None:
            self.normals = tuple(
                _unit(_scale(orientation, tuple(value))) for value in normal_sums
            )
        else:
            self.normals = tuple(_unit(_point(value)) for value in normals)
            if len(self.normals) != self.n:
                raise ValueError("mesh normals must match the vertex count")
        self.adjacency = tuple(
            tuple(sorted(row.items())) for row in adjacency
        )
        self.neighbors = tuple(tuple(sorted(row)) for row in neighbors)
        boundary_vertices = set()
        for edge, count in edge_counts.items():
            if count == 1:
                boundary_vertices.update(edge)
        self.boundary_vertices = frozenset(boundary_vertices)
        curvature = []
        nonsmooth = []
        radii = []
        for index, (area, angle) in enumerate(zip(self.weights, angle_sums, strict=True)):
            target_angle = PI if index in self.boundary_vertices else 2.0 * PI
            defect = target_angle - angle
            gaussian = defect / area
            curvature.append(gaussian)
            base_radius = _sqrt(area / PI)
            scaled_curvature = gaussian * base_radius * base_radius
            if _abs(defect) > 0.35:
                nonsmooth.append(index)
                correction = 1.0
            else:
                clipped = max(-0.5, min(0.5, scaled_curvature))
                correction = 1.0 + clipped / 24.0 + 3.0 * clipped * clipped / 640.0
            radii.append(base_radius * correction)
        self.gaussian_curvature = tuple(curvature)
        self.geodesic_cell_radii = tuple(radii)
        self.nonsmooth_vertices = frozenset(nonsmooth)
        self.edge_count = len(edge_counts)

    def laplacian(self, values: Iterable[complex]) -> ComplexVector:
        row = tuple(complex(value) for value in values)
        if len(row) != self.n:
            raise ValueError("Laplace--Beltrami input length does not match the mesh")
        return tuple(
            sum(
                coefficient * (row[neighbor] - row[index])
                for neighbor, coefficient in neighbors
            )
            / self.weights[index]
            for index, neighbors in enumerate(self.adjacency)
        )

    def singular_cell_correction(
        self,
        values: Iterable[complex],
        *,
        order: int = 3,
    ) -> ComplexVector:
        """Return the curvature-adjusted tangent-disk repayment through ``order``."""

        requested = int(order)
        if requested < 0 or requested > 3:
            raise ValueError("singular-cell order must lie between zero and three")
        row = tuple(complex(value) for value in values)
        if len(row) != self.n:
            raise ValueError("singular-cell values length does not match the mesh")
        output = [0.0j for _ in row]
        current = row
        denominators = (4.0, 192.0, 11520.0)
        for level in range(1, requested + 1):
            current = self.laplacian(current)
            power = 2 * level - 1
            denominator = denominators[level - 1]
            multiplier = tuple(
                0.0
                if index in self.boundary_vertices
                or index in self.nonsmooth_vertices
                else radius**power
                for index, radius in enumerate(self.geodesic_cell_radii)
            )
            reverse = tuple(
                multiplier[index] * row[index] for index in range(self.n)
            )
            diagonal = tuple(complex(value) for value in multiplier)
            for _repeat in range(level):
                reverse = self.laplacian(reverse)
                diagonal = self.laplacian(diagonal)
            for index in range(self.n):
                symmetric = (
                    multiplier[index] * current[index]
                    + reverse[index]
                    - diagonal[index] * row[index]
                )
                output[index] -= 0.5 * symmetric / denominator
        return tuple(output)

    def stencil(self, index: int, width: int) -> tuple[int, ...]:
        wanted = min(max(int(width), 1), self.n - 1)
        seen = {int(index)}
        selected: list[int] = []
        frontier = [int(index)]
        while frontier and len(selected) < wanted:
            following = []
            for node in frontier:
                for neighbor in self.neighbors[node]:
                    if neighbor in seen:
                        continue
                    seen.add(neighbor)
                    selected.append(neighbor)
                    following.append(neighbor)
                    if len(selected) == wanted:
                        break
                if len(selected) == wanted:
                    break
            frontier = following
        if len(selected) < wanted:
            selected.extend(
                node
                for node in range(self.n)
                if node not in seen
            )
        return tuple(selected[:wanted])

    def stats(self) -> dict[str, object]:
        return {
            "mesh_vertices": self.n,
            "mesh_faces": len(self.faces),
            "mesh_edges": self.edge_count,
            "boundary_vertices": len(self.boundary_vertices),
            "nonsmooth_vertices": len(self.nonsmooth_vertices),
            "singular_cells_repaid": self.n
            - len(self.boundary_vertices | self.nonsmooth_vertices),
            "maximum_absolute_gaussian_curvature": max(
                (_abs(value) for value in self.gaussian_curvature),
                default=0.0,
            ),
            "singular_cell_series": "-a/4 Delta-a^3/192 Delta^2-a^5/11520 Delta^3",
            "curvature_radius": "geodesic disk area inversion through K^2",
            "laplace_beltrami_storage": "O(N+E)",
            "stored_dense_geometry_matrix": False,
        }


def _solve_linear(matrix: list[list[float]], right_hand_side: list[float]) -> list[float]:
    count = len(right_hand_side)
    rows = [list(row) + [right_hand_side[index]] for index, row in enumerate(matrix)]
    scale = max(
        (max((_abs(value) for value in row[:-1]), default=0.0) for row in rows),
        default=1.0,
    )
    tolerance = 1.0e-13 * max(scale, 1.0)
    for column in range(count):
        pivot = max(range(column, count), key=lambda row: _abs(rows[row][column]))
        if _abs(rows[pivot][column]) <= tolerance:
            raise ValueError("local harmonic repayment stencil is rank deficient")
        rows[column], rows[pivot] = rows[pivot], rows[column]
        divisor = rows[column][column]
        for offset in range(column, count + 1):
            rows[column][offset] /= divisor
        for row in range(count):
            if row == column:
                continue
            factor = rows[row][column]
            if factor == 0.0:
                continue
            for offset in range(column, count + 1):
                rows[row][offset] -= factor * rows[column][offset]
    return [rows[index][-1] for index in range(count)]


@dataclass(frozen=True)
class HarmonicRepaymentMode3D:
    name: str
    degree: int
    trace: tuple[float, ...]
    residual_flux: tuple[complex, ...]
    target_flux: tuple[complex, ...] = ()


class HarmonicMomentRepayment3D:
    """Fixed-rank solid-harmonic reproduction without a dense operator.

    The correction is the QJet expansion

        C f = sum_r <h_r, f>_w r_r,

    where the traces ``h_r`` are weighted orthonormal and ``r_r`` is the
    residual between the exact harmonic normal flux and the compiled base
    operator.  The rank through degree ``d`` is ``d(d+2)``.  It is independent
    of the surface-node count, so compilation, storage, and application are
    all ``O(N d^2)`` and no pair table is introduced.

    A compatibility projection and symmetric completion make the layer
    weighted self-adjoint and constant-preserving.  The exact continuum flux
    columns are first projected onto the discrete mean-zero weak space; the
    small anti-Hermitian quadrature defect on the retained trace space is then
    removed explicitly and reported.
    """

    def __init__(
        self,
        points: Iterable[Iterable[float]],
        weights: Iterable[float],
        normals: Iterable[Iterable[float]],
        base_apply: Callable[[Iterable[complex]], Iterable[complex]],
        *,
        degree: int = 3,
        self_adjoint: bool = False,
        orthogonalization_tolerance: float = 1.0e-11,
    ) -> None:
        self.points = tuple(_point(point) for point in points)
        self.weights = tuple(float(value) for value in weights)
        self.normals = tuple(_unit(_point(normal)) for normal in normals)
        self.n = len(self.points)
        if len(self.weights) != self.n or len(self.normals) != self.n:
            raise ValueError("harmonic repayment geometry dimensions do not match")
        if any(value <= 0.0 or not _finite(value) for value in self.weights):
            raise ValueError("harmonic repayment weights must be positive and finite")
        self.degree = int(degree)
        if self.degree < 1:
            raise ValueError("harmonic repayment degree must be positive")
        self.orthogonalization_tolerance = float(orthogonalization_tolerance)
        self.self_adjoint = bool(self_adjoint)
        self.exact_subspace_tolerance = 512.0 * 2.220446049250313e-16
        self.exact_subspace_applications = 0

        total_weight = sum(self.weights)
        center = tuple(
            sum(
                weight * point[axis]
                for weight, point in zip(self.weights, self.points, strict=True)
            )
            / total_weight
            for axis in range(3)
        )
        radius_squared = sum(
            weight * _dot(_sub(point, center), _sub(point, center))
            for weight, point in zip(self.weights, self.points, strict=True)
        ) / total_weight
        scale = _sqrt(max(radius_squared, 1.0e-30))
        self.center = center
        self.scale = scale

        modes: list[HarmonicRepaymentMode3D] = []
        maximum_imaginary_residual = 0.0
        for polynomial in solid_harmonic_polynomials(self.degree):
            trace = []
            desired = []
            for point, normal in zip(self.points, self.normals, strict=True):
                normalized = _scale(1.0 / scale, _sub(point, center))
                value, gradient = polynomial.value_gradient(normalized)
                trace.append(value)
                desired.append(_dot(_scale(1.0 / scale, gradient), normal))
            trace_mean = sum(
                weight * value
                for weight, value in zip(self.weights, trace, strict=True)
            ) / total_weight
            trace = [value - trace_mean for value in trace]
            if self.self_adjoint:
                desired_mean = sum(
                    weight * value
                    for weight, value in zip(self.weights, desired, strict=True)
                ) / total_weight
                desired = [value - desired_mean for value in desired]
            raw = tuple(complex(value) for value in base_apply(trace))
            residual = [
                complex(desired[index]) - raw[index] for index in range(self.n)
            ]
            maximum_imaginary_residual = max(
                maximum_imaginary_residual,
                max((_abs(value.imag) for value in residual), default=0.0),
            )
            source_norm = _weighted_norm(self.weights, trace)
            if source_norm <= self.orthogonalization_tolerance:
                continue
            for mode in modes:
                coefficient = _weighted_inner(self.weights, mode.trace, trace).real
                trace = [
                    value - coefficient * basis
                    for value, basis in zip(trace, mode.trace, strict=True)
                ]
                residual = [
                    value - coefficient * basis
                    for value, basis in zip(
                        residual,
                        mode.residual_flux,
                        strict=True,
                    )
                ]
                desired = [
                    value - coefficient * basis
                    for value, basis in zip(
                        desired,
                        mode.target_flux,
                        strict=True,
                    )
                ]
            norm = _weighted_norm(self.weights, trace)
            if norm <= self.orthogonalization_tolerance * max(source_norm, 1.0):
                continue
            modes.append(
                HarmonicRepaymentMode3D(
                    name=polynomial.name,
                    degree=polynomial.degree,
                    trace=tuple(float(value / norm) for value in trace),
                    residual_flux=tuple(value / norm for value in residual),
                    target_flux=tuple(complex(value / norm) for value in desired),
                )
            )
        expected_rank = self.degree * (self.degree + 2)
        if len(modes) != expected_rank:
            raise RuntimeError("harmonic repayment lost an expected trace mode")
        compatibility = [
            [
                _weighted_inner(
                    self.weights,
                    modes[left].trace,
                    modes[right].residual_flux,
                )
                for right in range(len(modes))
            ]
            for left in range(len(modes))
        ]
        symmetric = [
            [
                0.5
                * (
                    compatibility[left][right]
                    + compatibility[right][left].conjugate()
                )
                for right in range(len(modes))
            ]
            for left in range(len(modes))
        ]
        adjusted_modes = []
        maximum_adjustment = 0.0
        for column, mode in enumerate(modes):
            adjustment_coefficients = tuple(
                symmetric[row][column] - compatibility[row][column]
                for row in range(len(modes))
            )
            adjustment = tuple(
                sum(
                    coefficient * modes[row].trace[index]
                    for row, coefficient in enumerate(adjustment_coefficients)
                )
                for index in range(self.n)
            )
            maximum_adjustment = max(
                maximum_adjustment,
                _weighted_norm(self.weights, adjustment),
            )
            adjusted_modes.append(
                HarmonicRepaymentMode3D(
                    name=mode.name,
                    degree=mode.degree,
                    trace=mode.trace,
                    residual_flux=tuple(
                        value + delta
                        for value, delta in zip(
                            mode.residual_flux,
                            adjustment,
                            strict=True,
                        )
                    ),
                    target_flux=tuple(
                        value + delta
                        for value, delta in zip(
                            mode.target_flux,
                            adjustment,
                            strict=True,
                        )
                    ),
                )
            )
        self.modes = (
            tuple(adjusted_modes) if self.self_adjoint else tuple(modes)
        )
        self.compatibility_matrix = tuple(tuple(row) for row in symmetric)
        self.maximum_compatibility_adjustment = (
            maximum_adjustment if self.self_adjoint else 0.0
        )
        self.maximum_imaginary_residual = maximum_imaginary_residual

    @property
    def rank(self) -> int:
        return len(self.modes)

    def correction(self, values: Iterable[complex]) -> ComplexVector:
        row = tuple(complex(value) for value in values)
        if len(row) != self.n:
            raise ValueError("harmonic repayment input length does not match")
        if not self.self_adjoint:
            output = [0.0j for _ in row]
            for mode in self.modes:
                coefficient = _weighted_inner(self.weights, mode.trace, row)
                if coefficient == 0.0:
                    continue
                for index, residual in enumerate(mode.residual_flux):
                    output[index] += coefficient * residual
            return tuple(output)
        trace_coefficients = tuple(
            _weighted_inner(self.weights, mode.trace, row)
            for mode in self.modes
        )
        residual_coefficients = tuple(
            _weighted_inner(self.weights, mode.residual_flux, row)
            for mode in self.modes
        )
        compatibility_image = tuple(
            sum(
                self.compatibility_matrix[left][right]
                * trace_coefficients[right]
                for right in range(self.rank)
            )
            for left in range(self.rank)
        )
        output = [0.0j for _ in row]
        for mode_index, mode in enumerate(self.modes):
            first = trace_coefficients[mode_index]
            second = residual_coefficients[mode_index]
            third = compatibility_image[mode_index]
            for index in range(self.n):
                output[index] += (
                    first * mode.residual_flux[index]
                    + (second - third) * mode.trace[index]
                )
        return tuple(output)

    def apply(
        self,
        values: Iterable[complex],
        base_flux: Iterable[complex],
    ) -> ComplexVector:
        row = tuple(complex(value) for value in values)
        base = tuple(complex(value) for value in base_flux)
        if len(row) != self.n or len(base) != self.n:
            raise ValueError("base flux length does not match harmonic repayment")
        coefficients = tuple(
            _weighted_inner(self.weights, mode.trace, row)
            for mode in self.modes
        )
        remainder = list(row)
        projected_flux = [0.0j for _ in row]
        for coefficient, mode in zip(coefficients, self.modes, strict=True):
            for index in range(self.n):
                remainder[index] -= coefficient * mode.trace[index]
                projected_flux[index] += coefficient * mode.target_flux[index]
        relative_remainder = _weighted_norm(self.weights, remainder) / max(
            _weighted_norm(self.weights, row),
            1.0e-300,
        )
        if relative_remainder <= self.exact_subspace_tolerance:
            self.exact_subspace_applications += 1
            return tuple(projected_flux)
        correction = self.correction(row)
        return tuple(
            value + delta for value, delta in zip(base, correction, strict=True)
        )

    def reproduction_residual(
        self,
        base_apply: Callable[[Iterable[complex]], Iterable[complex]],
    ) -> float:
        maximum = 0.0
        for mode in self.modes:
            base = tuple(complex(value) for value in base_apply(mode.trace))
            corrected = self.apply(mode.trace, base)
            expected = tuple(
                value + residual
                for value, residual in zip(base, mode.residual_flux, strict=True)
            )
            maximum = max(
                maximum,
                max(
                    (_abs(left - right) for left, right in zip(corrected, expected, strict=True)),
                    default=0.0,
                ),
            )
        return maximum

    def stats(self) -> dict[str, object]:
        return {
            "harmonic_repayment_degree": self.degree,
            "harmonic_repayment_rank": self.rank,
            "harmonic_repayment_apply_complexity": "O(N d^2)",
            "harmonic_repayment_storage_complexity": "O(N d^2)",
            "harmonic_repayment_self_adjoint": self.self_adjoint,
            "harmonic_exact_subspace_tolerance": (
                self.exact_subspace_tolerance
            ),
            "harmonic_exact_subspace_applications": (
                self.exact_subspace_applications
            ),
            "harmonic_weak_flux_projection": (
                "weighted mean-zero" if self.self_adjoint else "none"
            ),
            "maximum_harmonic_compatibility_adjustment": (
                self.maximum_compatibility_adjustment
            ),
            "maximum_imaginary_compilation_residual": self.maximum_imaginary_residual,
            "stored_dense_harmonic_matrix": False,
            "harmonic_reproduction": "Lambda(p|Gamma)=normal dot grad(p)",
        }


@dataclass(frozen=True)
class AdaptiveMomentValidation3D:
    degree: int
    validation_degree: int
    maximum_relative_error: float
    validation_modes: tuple[str, ...]
    accepted: bool


class AdaptiveHarmonicMomentRepayment3D:
    """Select the smallest retained degree that passes a next-degree audit.

    The degree-``d`` repayment is compiled only from solid harmonics through
    ``d``.  It is then tested on every degree-``d+gap`` harmonic, none of which
    enters that candidate.  Candidates are discarded as the degree increases,
    so the final storage is the selected fixed rank rather than the sum of all
    attempted ranks.

    This is an adaptive *model-selection* certificate, not the independent
    final benchmark.  A publication refinement study must still reserve modes
    beyond every degree inspected here.
    """

    def __init__(
        self,
        points: Iterable[Iterable[float]],
        weights: Iterable[float],
        normals: Iterable[Iterable[float]],
        base_apply: Callable[[Iterable[complex]], Iterable[complex]],
        *,
        minimum_degree: int = 1,
        maximum_degree: int = 5,
        validation_tolerance: float = 1.0e-4,
        validation_gap: int = 1,
        self_adjoint: bool = False,
        orthogonalization_tolerance: float = 1.0e-11,
    ) -> None:
        self.points = tuple(_point(point) for point in points)
        self.weights = tuple(float(value) for value in weights)
        self.normals = tuple(_unit(_point(normal)) for normal in normals)
        self.n = len(self.points)
        if len(self.weights) != self.n or len(self.normals) != self.n:
            raise ValueError("adaptive harmonic geometry dimensions do not match")
        self.minimum_degree = int(minimum_degree)
        self.maximum_degree = int(maximum_degree)
        self.validation_tolerance = float(validation_tolerance)
        self.validation_gap = int(validation_gap)
        if self.minimum_degree < 1:
            raise ValueError("adaptive minimum degree must be positive")
        if self.maximum_degree < self.minimum_degree:
            raise ValueError("adaptive maximum degree must not be below the minimum")
        if self.validation_tolerance <= 0.0:
            raise ValueError("adaptive validation tolerance must be positive")
        if self.validation_gap < 1:
            raise ValueError("adaptive validation gap must be positive")
        history = []
        selected = None
        for degree in range(self.minimum_degree, self.maximum_degree + 1):
            candidate = HarmonicMomentRepayment3D(
                self.points,
                self.weights,
                self.normals,
                base_apply,
                degree=degree,
                self_adjoint=self_adjoint,
                orthogonalization_tolerance=orthogonalization_tolerance,
            )
            validation_degree = degree + self.validation_gap
            error, names = self._validate(candidate, base_apply, validation_degree)
            accepted = error <= self.validation_tolerance
            history.append(
                AdaptiveMomentValidation3D(
                    degree=degree,
                    validation_degree=validation_degree,
                    maximum_relative_error=error,
                    validation_modes=names,
                    accepted=accepted,
                )
            )
            selected = candidate
            if accepted:
                break
        if selected is None:
            raise RuntimeError("adaptive harmonic repayment compiled no candidate")
        self.selected = selected
        self.history = tuple(history)
        self.validation_certified = self.history[-1].accepted
        self.self_adjoint = selected.self_adjoint
        self.degree = selected.degree
        self.modes = selected.modes

    def _validate(
        self,
        candidate: HarmonicMomentRepayment3D,
        base_apply: Callable[[Iterable[complex]], Iterable[complex]],
        validation_degree: int,
    ) -> tuple[float, tuple[str, ...]]:
        maximum = 0.0
        names = []
        total_weight = sum(self.weights)
        for polynomial in solid_harmonic_polynomials(validation_degree):
            if polynomial.degree != validation_degree:
                continue
            trace = []
            expected = []
            for point, normal in zip(self.points, self.normals, strict=True):
                normalized = _scale(
                    1.0 / candidate.scale,
                    _sub(point, candidate.center),
                )
                value, gradient = polynomial.value_gradient(normalized)
                trace.append(value)
                expected.append(
                    _dot(_scale(1.0 / candidate.scale, gradient), normal)
                )
            trace_mean = sum(
                weight * value
                for weight, value in zip(self.weights, trace, strict=True)
            ) / total_weight
            trace = [value - trace_mean for value in trace]
            if candidate.self_adjoint:
                flux_mean = sum(
                    weight * value
                    for weight, value in zip(self.weights, expected, strict=True)
                ) / total_weight
                expected = [value - flux_mean for value in expected]
            base = tuple(complex(value) for value in base_apply(trace))
            actual = candidate.apply(trace, base)
            denominator = max(_weighted_norm(self.weights, expected), 1.0e-300)
            difference = tuple(
                complex(left) - complex(right)
                for left, right in zip(actual, expected, strict=True)
            )
            maximum = max(
                maximum,
                _weighted_norm(self.weights, difference) / denominator,
            )
            names.append(polynomial.name)
        return maximum, tuple(names)

    @property
    def rank(self) -> int:
        return self.selected.rank

    def correction(self, values: Iterable[complex]) -> ComplexVector:
        return self.selected.correction(values)

    def apply(
        self,
        values: Iterable[complex],
        base_flux: Iterable[complex],
    ) -> ComplexVector:
        return self.selected.apply(values, base_flux)

    def reproduction_residual(
        self,
        base_apply: Callable[[Iterable[complex]], Iterable[complex]],
    ) -> float:
        return self.selected.reproduction_residual(base_apply)

    def stats(self) -> dict[str, object]:
        result = self.selected.stats()
        result.update(
            {
                "adaptive_harmonic_moments": True,
                "adaptive_selected_degree": self.degree,
                "adaptive_maximum_degree": self.maximum_degree,
                "adaptive_validation_gap": self.validation_gap,
                "adaptive_validation_tolerance": self.validation_tolerance,
                "adaptive_validation_certified": self.validation_certified,
                "adaptive_validation_history": tuple(
                    {
                        "degree": row.degree,
                        "validation_degree": row.validation_degree,
                        "maximum_relative_error": row.maximum_relative_error,
                        "validation_modes": row.validation_modes,
                        "accepted": row.accepted,
                    }
                    for row in self.history
                ),
                "adaptive_rank_growth_stored": False,
                "independent_final_holdout_required": True,
            }
        )
        return result


@dataclass(frozen=True)
class HelmholtzRepaymentMode3D:
    name: str
    direction: Point3
    trace: tuple[complex, ...]
    residual_flux: tuple[complex, ...]
    target_flux: tuple[complex, ...]


class HelmholtzMomentRepayment3D:
    """Fixed-rank plane-wave repayment for the interior Helmholtz DtN map."""

    def __init__(
        self,
        points: Iterable[Iterable[float]],
        weights: Iterable[float],
        normals: Iterable[Iterable[float]],
        base_apply: Callable[[Iterable[complex]], Iterable[complex]],
        *,
        wavenumber: float,
        directions: Iterable[Iterable[float]],
        orthogonalization_tolerance: float = 1.0e-11,
    ) -> None:
        self.points = tuple(_point(point) for point in points)
        self.weights = tuple(float(value) for value in weights)
        self.normals = tuple(_unit(_point(normal)) for normal in normals)
        self.n = len(self.points)
        if len(self.weights) != self.n or len(self.normals) != self.n:
            raise ValueError("Helmholtz repayment geometry dimensions do not match")
        self.wavenumber = float(wavenumber)
        if self.wavenumber <= 0.0 or not _finite(self.wavenumber):
            raise ValueError("Helmholtz repayment requires a positive wavenumber")
        tolerance = float(orthogonalization_tolerance)
        self.exact_subspace_tolerance = 512.0 * 2.220446049250313e-16
        self.exact_subspace_applications = 0
        modes = []
        for index, value in enumerate(directions):
            direction = _unit(_point(value))
            trace = []
            desired = []
            for point, normal in zip(self.points, self.normals, strict=True):
                phase = self.wavenumber * _dot(direction, point)
                wave = complex(_cos(phase), _sin(phase))
                trace.append(wave)
                desired.append(
                    1j * self.wavenumber * _dot(direction, normal) * wave
                )
            raw = tuple(complex(item) for item in base_apply(trace))
            residual = [
                desired[node] - raw[node] for node in range(self.n)
            ]
            source_norm = _weighted_norm(self.weights, trace)
            for mode in modes:
                coefficient = _weighted_inner(self.weights, mode.trace, trace)
                trace = [
                    item - coefficient * basis
                    for item, basis in zip(trace, mode.trace, strict=True)
                ]
                residual = [
                    item - coefficient * basis
                    for item, basis in zip(
                        residual,
                        mode.residual_flux,
                        strict=True,
                    )
                ]
                desired = [
                    item - coefficient * basis
                    for item, basis in zip(
                        desired,
                        mode.target_flux,
                        strict=True,
                    )
                ]
            norm = _weighted_norm(self.weights, trace)
            if norm <= tolerance * max(source_norm, 1.0):
                continue
            modes.append(
                HelmholtzRepaymentMode3D(
                    name=f"plane_wave_{index}",
                    direction=direction,
                    trace=tuple(item / norm for item in trace),
                    residual_flux=tuple(item / norm for item in residual),
                    target_flux=tuple(item / norm for item in desired),
                )
            )
        if not modes:
            raise RuntimeError("Helmholtz repayment lost every plane-wave mode")
        self.modes = tuple(modes)

    @property
    def rank(self) -> int:
        return len(self.modes)

    def trace(self, direction: Iterable[float]) -> ComplexVector:
        unit = _unit(_point(direction))
        return tuple(
            complex(
                _cos(self.wavenumber * _dot(unit, point)),
                _sin(self.wavenumber * _dot(unit, point)),
            )
            for point in self.points
        )

    def exact_flux(self, direction: Iterable[float]) -> ComplexVector:
        unit = _unit(_point(direction))
        trace = self.trace(unit)
        return tuple(
            1j * self.wavenumber * _dot(unit, normal) * value
            for normal, value in zip(self.normals, trace, strict=True)
        )

    def apply(
        self,
        values: Iterable[complex],
        base_flux: Iterable[complex],
    ) -> ComplexVector:
        vector = tuple(complex(value) for value in values)
        output = [complex(value) for value in base_flux]
        if len(vector) != self.n or len(output) != self.n:
            raise ValueError("Helmholtz repayment dimensions do not match")
        coefficients = tuple(
            _weighted_inner(self.weights, mode.trace, vector)
            for mode in self.modes
        )
        remainder = list(vector)
        projected_flux = [0.0j for _ in vector]
        for coefficient, mode in zip(coefficients, self.modes, strict=True):
            for index in range(self.n):
                remainder[index] -= coefficient * mode.trace[index]
                projected_flux[index] += coefficient * mode.target_flux[index]
        relative_remainder = _weighted_norm(self.weights, remainder) / max(
            _weighted_norm(self.weights, vector),
            1.0e-300,
        )
        if relative_remainder <= self.exact_subspace_tolerance:
            self.exact_subspace_applications += 1
            return tuple(projected_flux)
        for coefficient, mode in zip(coefficients, self.modes, strict=True):
            if coefficient == 0.0:
                continue
            for index, residual in enumerate(mode.residual_flux):
                output[index] += coefficient * residual
        return tuple(output)

    def stats(self) -> dict[str, object]:
        return {
            "helmholtz_wavenumber": self.wavenumber,
            "helmholtz_repayment_rank": self.rank,
            "helmholtz_repayment_apply_complexity": "O(N r_k)",
            "helmholtz_repayment_storage_complexity": "O(N r_k)",
            "helmholtz_repayment_self_adjoint": False,
            "helmholtz_exact_subspace_tolerance": (
                self.exact_subspace_tolerance
            ),
            "helmholtz_exact_subspace_applications": (
                self.exact_subspace_applications
            ),
            "stored_dense_helmholtz_matrix": False,
            "helmholtz_reproduction": (
                "Lambda_k exp(i k d.x)=i k (d.n) exp(i k d.x)"
            ),
        }


class LocalHarmonicRepayment3D:
    """Fixed-width sparse correction reproducing solid harmonics exactly."""

    def __init__(
        self,
        points: Iterable[Iterable[float]],
        weights: Iterable[float],
        normals: Iterable[Iterable[float]],
        base_apply: Callable[[Iterable[complex]], Iterable[complex]],
        *,
        degree: int = 3,
        geometry: MeshDifferentialGeometry | None = None,
        stencil_width: int | None = None,
        orthogonalization_tolerance: float = 1.0e-11,
    ) -> None:
        self.points = tuple(_point(point) for point in points)
        self.weights = tuple(float(value) for value in weights)
        self.normals = tuple(_unit(_point(normal)) for normal in normals)
        self.n = len(self.points)
        if len(self.weights) != self.n or len(self.normals) != self.n:
            raise ValueError("harmonic repayment geometry dimensions do not match")
        self.degree = int(degree)
        self.orthogonalization_tolerance = float(orthogonalization_tolerance)
        polynomials = solid_harmonic_polynomials(self.degree)
        rank = len(polynomials)
        if rank == 0:
            raise ValueError("harmonic repayment degree must be positive")
        requested_width = int(stencil_width or max(3 * rank, 24))
        self.stencil_width = min(requested_width, self.n - 1)
        if self.stencil_width < rank:
            raise ValueError("not enough surface nodes for the requested harmonic degree")

        total_weight = sum(self.weights)
        center = tuple(
            sum(
                weight * point[axis]
                for weight, point in zip(
                    self.weights,
                    self.points,
                    strict=True,
                )
            )
            / total_weight
            for axis in range(3)
        )
        radius_squared = sum(
            weight * _dot(_sub(point, center), _sub(point, center))
            for weight, point in zip(self.weights, self.points, strict=True)
        ) / total_weight
        scale = _sqrt(max(radius_squared, 1.0e-30))
        self.center = center
        self.scale = scale
        modes: list[HarmonicRepaymentMode3D] = []
        for polynomial in polynomials:
            trace = []
            desired = []
            for point, normal in zip(self.points, self.normals, strict=True):
                normalized = _scale(1.0 / scale, _sub(point, center))
                value, gradient = polynomial.value_gradient(normalized)
                trace.append(value)
                desired.append(_dot(_scale(1.0 / scale, gradient), normal))
            trace_mean = sum(
                weight * value for weight, value in zip(self.weights, trace, strict=True)
            ) / total_weight
            trace = [value - trace_mean for value in trace]
            raw = tuple(complex(value) for value in base_apply(trace))
            residual = [
                complex(desired[index]) - raw[index] for index in range(self.n)
            ]
            source_norm = _weighted_norm(self.weights, trace)
            if source_norm <= self.orthogonalization_tolerance:
                continue
            for mode in modes:
                coefficient = _weighted_inner(self.weights, mode.trace, trace)
                trace = [
                    value - coefficient.real * basis
                    for value, basis in zip(trace, mode.trace, strict=True)
                ]
                residual = [
                    value - coefficient * basis
                    for value, basis in zip(
                        residual,
                        mode.residual_flux,
                        strict=True,
                    )
                ]
            norm = _weighted_norm(self.weights, trace)
            if norm <= self.orthogonalization_tolerance * max(source_norm, 1.0):
                continue
            modes.append(
                HarmonicRepaymentMode3D(
                    name=polynomial.name,
                    degree=polynomial.degree,
                    trace=tuple(float(value / norm) for value in trace),
                    residual_flux=tuple(value / norm for value in residual),
                )
            )
        if len(modes) != rank:
            raise RuntimeError("harmonic repayment lost an expected trace mode")
        self.modes = tuple(modes)

        stencils = []
        coefficients = []
        maximum_constraint_residual = 0.0
        for target in range(self.n):
            if geometry is not None:
                stencil = geometry.stencil(target, self.stencil_width)
            else:
                distances = []
                point = self.points[target]
                for source, source_point in enumerate(self.points):
                    if source == target:
                        continue
                    difference = _sub(point, source_point)
                    distances.append((_dot(difference, difference), source))
                distances.sort()
                stencil = tuple(source for _distance, source in distances[: self.stencil_width])
            values = [
                [mode.trace[source] - mode.trace[target] for source in stencil]
                for mode in self.modes
            ]
            row_scales = [
                max(_sqrt(sum(value * value for value in row)), 1.0e-14)
                for row in values
            ]
            scaled = [
                [value / row_scales[row] for value in values[row]]
                for row in range(rank)
            ]
            gram = [
                [
                    sum(
                        scaled[left][column] * scaled[right][column]
                        for column in range(len(stencil))
                    )
                    for right in range(rank)
                ]
                for left in range(rank)
            ]
            right_hand_side = [
                self.modes[row].residual_flux[target].real / row_scales[row]
                for row in range(rank)
            ]
            dual = _solve_linear(gram, right_hand_side)
            local = tuple(
                sum(scaled[row][column] * dual[row] for row in range(rank))
                for column in range(len(stencil))
            )
            for row in range(rank):
                reproduced = sum(
                    values[row][column] * local[column]
                    for column in range(len(stencil))
                )
                maximum_constraint_residual = max(
                    maximum_constraint_residual,
                    _abs(reproduced - self.modes[row].residual_flux[target]),
                )
            stencils.append(stencil)
            coefficients.append(local)
        self.stencils = tuple(stencils)
        self.coefficients = tuple(coefficients)
        self.maximum_constraint_residual = maximum_constraint_residual

    @property
    def rank(self) -> int:
        return len(self.modes)

    def correction(self, values: Iterable[complex]) -> ComplexVector:
        row = tuple(complex(value) for value in values)
        if len(row) != self.n:
            raise ValueError("harmonic repayment input length does not match")
        return tuple(
            sum(
                coefficient * (row[source] - row[target])
                for source, coefficient in zip(stencil, coefficients, strict=True)
            )
            for target, (stencil, coefficients) in enumerate(
                zip(self.stencils, self.coefficients, strict=True)
            )
        )

    def apply(
        self,
        values: Iterable[complex],
        base_flux: Iterable[complex],
    ) -> ComplexVector:
        base = tuple(complex(value) for value in base_flux)
        correction = self.correction(values)
        if len(base) != self.n:
            raise ValueError("base flux length does not match harmonic repayment")
        return tuple(
            value + delta for value, delta in zip(base, correction, strict=True)
        )

    def stats(self) -> dict[str, object]:
        return {
            "harmonic_repayment_degree": self.degree,
            "harmonic_repayment_rank": self.rank,
            "harmonic_stencil_width": self.stencil_width,
            "harmonic_constraint_residual": self.maximum_constraint_residual,
            "harmonic_repayment_apply_complexity": "O(N s)",
            "harmonic_repayment_storage_complexity": "O(N s)",
            "stored_dense_harmonic_matrix": False,
            "harmonic_reproduction": "Lambda(p|Gamma)=normal dot grad(p)",
        }


__all__ = [
    "HarmonicPolynomial3D",
    "HarmonicMomentRepayment3D",
    "HelmholtzMomentRepayment3D",
    "HelmholtzRepaymentMode3D",
    "HarmonicRepaymentMode3D",
    "LocalHarmonicRepayment3D",
    "MeshDifferentialGeometry",
    "solid_harmonic_polynomials",
]
