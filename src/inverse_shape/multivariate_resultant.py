"""Sparse multivariate resultant generator for peeled Euclidean jets.

For q_j(x)=|x-X_j|^2, define

    D = product_j q_j,
    N_a = sum_j a_j product_{k != j} q_k.

At X_i in R^3, the finite part of N_a/D is

    sum_{j != i} a_j/q_j(X_i)
      = (Delta N_a - a_i Delta^2 D/20) / Delta D.

The module builds D and any number of N_a channels together in a normalized
sparse product tree.  It never stores a pair matrix.  Generic three-variable
support grows cubically with degree, so a strict support budget is enforced;
overflow or failed numerical audit is repaid by an exact pair stream.
"""

from inverse_shape.quadrature import (
    BorrowComputeRepayLedger,
    _abs,
    _clean_scalar,
    _finite,
    _sqrt,
)


class ResultantSupportOverflow(RuntimeError):
    pass


class ResultantNumericalFailure(RuntimeError):
    pass


def _point(value):
    row = tuple(float(component) for component in value)
    if len(row) != 3:
        raise ValueError("each resultant point must have three coordinates")
    if any(not _finite(component) for component in row):
        raise ValueError("resultant coordinates must be finite")
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


def _add_term(polynomial, monomial, value):
    updated = polynomial.get(monomial, 0.0 + 0.0j) + value
    if updated == 0.0:
        polynomial.pop(monomial, None)
    else:
        polynomial[monomial] = updated


def _kahan_add(polynomial, compensation, monomial, value):
    current = polynomial.get(monomial, 0.0 + 0.0j)
    corrected = value - compensation.get(monomial, 0.0 + 0.0j)
    updated = current + corrected
    compensation[monomial] = (updated - current) - corrected
    polynomial[monomial] = updated


class _SparseResultantBuilder:
    def __init__(self, points, residue_channels, support_budget):
        self.points = points
        self.residue_channels = residue_channels
        self.support_budget = int(support_budget)
        self.scalar_multiplications = 0
        self.maximum_polynomial_support = 0
        self.maximum_total_support = 0

    def _check(self, polynomials):
        maximum = max((len(poly) for poly in polynomials), default=0)
        total = sum(len(poly) for poly in polynomials)
        self.maximum_polynomial_support = max(
            self.maximum_polynomial_support,
            maximum,
        )
        self.maximum_total_support = max(self.maximum_total_support, total)
        if maximum > self.support_budget:
            raise ResultantSupportOverflow(
                "multivariate resultant support exceeded the configured budget"
            )

    def _multiply(self, left, right):
        output = {}
        compensation = {}
        for left_monomial, left_value in left.items():
            for right_monomial, right_value in right.items():
                monomial = (
                    left_monomial[0] + right_monomial[0],
                    left_monomial[1] + right_monomial[1],
                    left_monomial[2] + right_monomial[2],
                )
                _kahan_add(
                    output,
                    compensation,
                    monomial,
                    left_value * right_value,
                )
                self.scalar_multiplications += 1
                if len(output) > self.support_budget:
                    raise ResultantSupportOverflow(
                        "multivariate resultant support exceeded the configured budget"
                    )
        return {
            monomial: value for monomial, value in output.items() if value != 0.0
        }

    def _sum_products(self, left_a, right_a, left_b, right_b):
        output = self._multiply(left_a, right_a)
        second = self._multiply(left_b, right_b)
        compensation = {}
        for monomial, value in second.items():
            _kahan_add(output, compensation, monomial, value)
        if len(output) > self.support_budget:
            raise ResultantSupportOverflow(
                "multivariate numerator support exceeded the configured budget"
            )
        return output

    @staticmethod
    def _normalize(denominator, numerators):
        magnitude = max(
            (_abs(value) for value in denominator.values()),
            default=0.0,
        )
        for numerator in numerators:
            magnitude = max(
                magnitude,
                max((_abs(value) for value in numerator.values()), default=0.0),
            )
        if magnitude <= 0.0 or not _finite(magnitude):
            raise ResultantNumericalFailure("resultant bundle has zero scale")
        inverse = 1.0 / magnitude
        normalized_denominator = {
            monomial: value * inverse
            for monomial, value in denominator.items()
        }
        normalized_numerators = tuple(
            {
                monomial: value * inverse
                for monomial, value in numerator.items()
            }
            for numerator in numerators
        )
        return normalized_denominator, normalized_numerators

    def _leaf(self, index):
        x, y, z = self.points[index]
        denominator = {
            (2, 0, 0): 1.0 + 0.0j,
            (0, 2, 0): 1.0 + 0.0j,
            (0, 0, 2): 1.0 + 0.0j,
        }
        for monomial, value in (
            ((1, 0, 0), -2.0 * x),
            ((0, 1, 0), -2.0 * y),
            ((0, 0, 1), -2.0 * z),
            ((0, 0, 0), x * x + y * y + z * z),
        ):
            if value != 0.0:
                denominator[monomial] = complex(value)
        numerators = tuple(
            (
                {(0, 0, 0): residues[index]}
                if residues[index] != 0.0
                else {}
            )
            for residues in self.residue_channels
        )
        bundle = self._normalize(denominator, numerators)
        self._check((bundle[0],) + bundle[1])
        return bundle

    def _combine(self, left, right):
        left_denominator, left_numerators = left
        right_denominator, right_numerators = right
        denominator = self._multiply(left_denominator, right_denominator)
        numerators = tuple(
            self._sum_products(
                left_numerator,
                right_denominator,
                left_denominator,
                right_numerator,
            )
            for left_numerator, right_numerator in zip(
                left_numerators,
                right_numerators,
                strict=True,
            )
        )
        bundle = self._normalize(denominator, numerators)
        self._check((bundle[0],) + bundle[1])
        return bundle

    def _build_range(self, first, last):
        if last - first == 1:
            return self._leaf(first)
        middle = (first + last) // 2
        return self._combine(
            self._build_range(first, middle),
            self._build_range(middle, last),
        )

    def build(self):
        return self._build_range(0, len(self.points))


def _powers(value, degree):
    output = [1.0 + 0.0j]
    for _index in range(degree):
        output.append(output[-1] * value)
    return output


def _falling(value, order):
    result = 1
    for offset in range(order):
        result *= value - offset
    return result


def _derivative_term(coefficient, exponents, orders, power_tables):
    for exponent, order in zip(exponents, orders, strict=True):
        if exponent < order:
            return 0.0 + 0.0j
    multiplier = coefficient
    for axis in range(3):
        multiplier *= _falling(exponents[axis], orders[axis])
        multiplier *= power_tables[axis][exponents[axis] - orders[axis]]
    return multiplier


def _laplacian_and_bilaplacian(polynomial, point, need_bilaplacian):
    maximum = max(
        (max(monomial) for monomial in polynomial),
        default=0,
    )
    power_tables = tuple(_powers(complex(value), maximum) for value in point)
    laplacian = 0.0 + 0.0j
    bilaplacian = 0.0 + 0.0j
    laplacian_compensation = 0.0 + 0.0j
    bilaplacian_compensation = 0.0 + 0.0j

    def add_laplacian(value):
        nonlocal laplacian, laplacian_compensation
        corrected = value - laplacian_compensation
        updated = laplacian + corrected
        laplacian_compensation = (updated - laplacian) - corrected
        laplacian = updated

    def add_bilaplacian(value):
        nonlocal bilaplacian, bilaplacian_compensation
        corrected = value - bilaplacian_compensation
        updated = bilaplacian + corrected
        bilaplacian_compensation = (updated - bilaplacian) - corrected
        bilaplacian = updated

    for exponents, coefficient in polynomial.items():
        for axis in range(3):
            orders = [0, 0, 0]
            orders[axis] = 2
            add_laplacian(
                _derivative_term(
                    coefficient,
                    exponents,
                    tuple(orders),
                    power_tables,
                )
            )
        if not need_bilaplacian:
            continue
        for axis in range(3):
            orders = [0, 0, 0]
            orders[axis] = 4
            add_bilaplacian(
                _derivative_term(
                    coefficient,
                    exponents,
                    tuple(orders),
                    power_tables,
                )
            )
        for left in range(3):
            for right in range(left + 1, 3):
                orders = [0, 0, 0]
                orders[left] = 2
                orders[right] = 2
                add_bilaplacian(
                    2.0
                    * _derivative_term(
                        coefficient,
                        exponents,
                        tuple(orders),
                        power_tables,
                    )
                )
    return laplacian, bilaplacian


class MultivariateResultantEvaluation:
    def __init__(self, values, inverse_square_sums, ledger, stats):
        self.values = values
        self.inverse_square_sums = inverse_square_sums
        self.ledger = ledger
        self.stats = stats


class MultivariateResultantPeeledJetQJet:
    """Sparse D/N resultant generator with a matrix-free exact fallback."""

    def __init__(
        self,
        points,
        weights,
        support_budget=50000,
        audit_mode="full",
        audit_nodes=8,
        audit_tolerance=5.0e-13,
        fallback=True,
    ):
        self.points = tuple(_point(value) for value in points)
        self.n = len(self.points)
        if self.n < 2:
            raise ValueError("the resultant generator requires at least two nodes")
        if len(set(self.points)) != self.n:
            raise ValueError("resultant nodes must be distinct")
        self.weights = tuple(complex(value) for value in weights)
        if len(self.weights) != self.n:
            raise ValueError("weights must contain one value per node")
        if any(
            not _finite(value.real) or not _finite(value.imag)
            for value in self.weights
        ):
            raise ValueError("weights must be finite")
        self.support_budget = int(support_budget)
        self.audit_mode = str(audit_mode)
        self.audit_nodes = int(audit_nodes)
        self.audit_tolerance = float(audit_tolerance)
        self.fallback = bool(fallback)
        if self.support_budget < 7:
            raise ValueError("support_budget must be at least seven")
        if self.audit_mode not in ("full", "sampled", "none"):
            raise ValueError("audit_mode must be 'full', 'sampled', or 'none'")
        if self.audit_nodes < 1:
            raise ValueError("audit_nodes must be positive")
        if self.audit_tolerance <= 0.0 or not _finite(self.audit_tolerance):
            raise ValueError("audit_tolerance must be positive and finite")
        self.center = tuple(
            sum(point[axis] for point in self.points) / self.n
            for axis in range(3)
        )
        self.scale = max(
            _vnorm(_vsub(point, self.center)) for point in self.points
        )
        if self.scale <= 1.0e-14 or not _finite(self.scale):
            raise ValueError("resultant geometry has zero spatial scale")
        inverse_scale = 1.0 / self.scale
        self.normalized_points = tuple(
            tuple(
                (point[axis] - self.center[axis]) * inverse_scale
                for axis in range(3)
            )
            for point in self.points
        )
        self.last_stats = {}
        self.last_inverse_square_sums = None

    def _vectors(self, residue_channels):
        rows = tuple(
            tuple(complex(value) for value in row) for row in residue_channels
        )
        if not rows or any(len(row) != self.n for row in rows):
            raise ValueError("each residue channel must contain one value per node")
        if any(
            not _finite(value.real) or not _finite(value.imag)
            for row in rows
            for value in row
        ):
            raise ValueError("residue channels must be finite")
        return rows

    def _resultant_sums(self, residue_channels):
        builder = _SparseResultantBuilder(
            self.normalized_points,
            residue_channels,
            self.support_budget,
        )
        denominator, numerators = builder.build()
        inverse_scale_squared = 1.0 / (self.scale * self.scale)
        output = [[] for _ in residue_channels]
        for index, point in enumerate(self.normalized_points):
            denominator_laplacian, denominator_bilaplacian = (
                _laplacian_and_bilaplacian(
                    denominator,
                    point,
                    need_bilaplacian=True,
                )
            )
            if _abs(denominator_laplacian) <= 1.0e-280:
                raise ResultantNumericalFailure(
                    "peeled denominator Laplacian is numerically singular"
                )
            for channel, numerator in enumerate(numerators):
                numerator_laplacian, _unused = _laplacian_and_bilaplacian(
                    numerator,
                    point,
                    need_bilaplacian=False,
                )
                value = (
                    numerator_laplacian
                    - residue_channels[channel][index]
                    * denominator_bilaplacian
                    / 20.0
                ) / denominator_laplacian
                value *= inverse_scale_squared
                if not _finite(value.real) or not _finite(value.imag):
                    raise ResultantNumericalFailure(
                        "resultant finite part is not finite"
                    )
                output[channel].append(_clean_scalar(value))
        stats = {
            "method": "sparse_multivariate_peeled_resultant",
            "denominator_support": len(denominator),
            "numerator_supports": tuple(len(value) for value in numerators),
            "maximum_polynomial_support": builder.maximum_polynomial_support,
            "maximum_total_support": builder.maximum_total_support,
            "scalar_polynomial_multiplications": builder.scalar_multiplications,
            "support_budget": self.support_budget,
            "resultant_degree": 2 * self.n,
            "stored_dense_matrix": False,
        }
        return tuple(tuple(row) for row in output), stats

    def _audit_indices(self):
        if self.audit_mode == "none":
            return tuple()
        if self.audit_mode == "full":
            return tuple(range(self.n))
        return _even_indices(self.n, self.audit_nodes)

    def _direct_sums(self, residue_channels, indices=None):
        targets = tuple(range(self.n)) if indices is None else tuple(indices)
        output = [[0.0 + 0.0j for _ in targets] for _ in residue_channels]
        for target_local, left in enumerate(targets):
            for right in range(self.n):
                if left == right:
                    continue
                difference = _vsub(self.points[left], self.points[right])
                distance_squared = _vdot(difference, difference)
                if distance_squared <= 1.0e-28:
                    raise ValueError("distinct resultant nodes numerically collide")
                for channel, residues in enumerate(residue_channels):
                    output[channel][target_local] += (
                        residues[right] / distance_squared
                    )
        return tuple(tuple(row) for row in output)

    def _audit(self, candidate, residue_channels):
        indices = self._audit_indices()
        if not indices:
            return 0.0, None
        reference = self._direct_sums(residue_channels, indices)
        maximum = 0.0
        for channel in range(len(residue_channels)):
            numerator = sum(
                _abs(complex(candidate[channel][index]) - reference[channel][local])
                ** 2
                for local, index in enumerate(indices)
            )
            denominator = sum(_abs(value) ** 2 for value in reference[channel])
            relative = _sqrt(numerator / max(denominator, 1.0e-300))
            maximum = max(maximum, relative)
        return maximum, reference

    def generate_inverse_square_sums(self, residue_channels):
        rows = self._vectors(residue_channels)
        try:
            candidate, stats = self._resultant_sums(rows)
            audit_error, _audit_reference = self._audit(candidate, rows)
            stats["audit_mode"] = self.audit_mode
            stats["audit_relative_error"] = audit_error
            stats["audit_passed"] = audit_error <= self.audit_tolerance
            if audit_error > self.audit_tolerance:
                raise ResultantNumericalFailure(
                    "resultant failed its independent finite-part audit"
                )
            self.last_stats = stats
            self.last_inverse_square_sums = candidate
            return candidate
        except (ResultantSupportOverflow, ResultantNumericalFailure) as error:
            if not self.fallback:
                raise
            direct = self._direct_sums(rows)
            self.last_stats = {
                "method": "exact_streamed_resultant_repayment",
                "fallback_reason": str(error),
                "support_budget": self.support_budget,
                "audit_mode": self.audit_mode,
                "stored_dense_matrix": False,
            }
            self.last_inverse_square_sums = direct
            return direct

    def apply(self, values):
        row = tuple(complex(value) for value in values)
        if len(row) != self.n:
            raise ValueError("values must contain one entry per resultant node")
        weighted_field = tuple(
            self.weights[index] * row[index] for index in range(self.n)
        )
        weight_sums, field_sums = self.generate_inverse_square_sums(
            (self.weights, weighted_field)
        )
        return tuple(
            _clean_scalar(
                row[index] * complex(weight_sums[index])
                - complex(field_sums[index])
            )
            for index in range(self.n)
        )

    def stats(self):
        result = {
            "nodes": self.n,
            "support_budget": self.support_budget,
            "audit_mode": self.audit_mode,
            "audit_tolerance": self.audit_tolerance,
            "fallback_enabled": self.fallback,
            "persistent_geometry_entries": 4 * self.n,
            "stored_dense_distance_matrix": False,
            "stored_dense_operator_matrix": False,
            "generic_support_bound": "Theta(N^3) monomials in three variables",
            "resultant_apply_formula": (
                "(Delta N_a-a_i Delta^2 D/20)/Delta D"
            ),
        }
        result.update(self.last_stats)
        return result

    def evaluate(self, values):
        result = self.apply(values)
        stats = self.stats()
        fallback = stats.get("method") == "exact_streamed_resultant_repayment"
        ledger = BorrowComputeRepayLedger(
            borrowed=(
                "normalized three-dimensional squared-distance factors",
                "sparse multivariate D/N product tree",
                "four-jet denominator and two-jet numerator traces",
            ),
            computed=(
                "peeled multivariate resultant finite parts",
                "weighted inverse-square graph action",
            ),
            repaid=(
                "independent finite-part audit",
                "strict sparse-support budget",
                (
                    "exact streamed fallback"
                    if fallback
                    else "all temporary sparse polynomials"
                ),
            ),
            residuals=(
                (
                    "audit_relative_error",
                    float(stats.get("audit_relative_error", 0.0)),
                ),
            ),
            residual_norm=float(stats.get("audit_relative_error", 0.0)),
            status="borrowed_repaid",
            notes=(
                "The resultant identity is exact. Generic support is cubic; "
                "the support budget prevents a hidden dense polynomial from "
                "replacing the forbidden dense pair matrix."
            ),
        )
        return MultivariateResultantEvaluation(
            result,
            self.last_inverse_square_sums,
            ledger,
            stats,
        )


__all__ = [
    "MultivariateResultantEvaluation",
    "MultivariateResultantPeeledJetQJet",
    "ResultantNumericalFailure",
    "ResultantSupportOverflow",
]
