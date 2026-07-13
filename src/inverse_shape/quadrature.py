"""Self-contained matrix-free QJet quadrature engine.

No NumPy and no stdlib numerical substrate are used here.  The module stores
only QJet generators and boundary samples, never a dense Q matrix.  The
regular-cycle engine uses an internal FFT kernel, generated eigenvalues
``lambda_m = m(n-m)/2``, and borrow-compute-repay ledgers for spectral layer
evaluation.
"""

PI = 3.1415926535897932384626433832795028841971693993751
TAU = 6.2831853071795864769252867665590057683943387987502
HALF_PI = 1.5707963267948966192313216916397514420985846996876
LN2 = 0.69314718055994530941723212145817656807550013436026
SQRT2 = 1.4142135623730950488016887242096980785696718753769
INV_SQRT2 = 0.70710678118654752440084436210484903928483593768847
TINY = 2.2250738585072014e-308
INF = float("inf")
REAL_TOL = 1.0e-12


def _abs(value):
    return abs(value)


def _finite(value):
    return value == value and _abs(value) != INF


def _floor(value):
    integer = int(value)
    return integer if integer <= value else integer - 1


def _round_nearest(value):
    return _floor(value + 0.5) if value >= 0.0 else -_floor(-value + 0.5)


def _sqrt(value):
    if value < 0.0:
        raise ValueError("sqrt domain error")
    return value ** 0.5


def _hypot(x, y):
    return _sqrt(x * x + y * y)


def _reduce_angle(value):
    return value - TAU * _round_nearest(value / TAU)


def _sin(value):
    x = _reduce_angle(value)
    xx = x * x
    term = x
    total = x
    for k in range(1, 24):
        term *= -xx / ((2 * k) * (2 * k + 1))
        total += term
    return total


def _cos(value):
    x = _reduce_angle(value)
    xx = x * x
    term = 1.0
    total = 1.0
    for k in range(1, 24):
        term *= -xx / ((2 * k - 1) * (2 * k))
        total += term
    return total


def _atan(value):
    x = float(value)
    sign = 1.0
    if x < 0.0:
        sign = -1.0
        x = -x
    if x > 1.0:
        return sign * (HALF_PI - _atan(1.0 / x))
    if x > SQRT2 - 1.0:
        return sign * (0.5 * HALF_PI + _atan((x - 1.0) / (1.0 + x)))
    xx = x * x
    term = x
    total = x
    multiplier = -1.0
    for mode in range(1, 80):
        term *= xx
        add = multiplier * term / (2 * mode + 1)
        total += add
        if _abs(add) < 1.0e-19:
            break
        multiplier = -multiplier
    return sign * total


def _exp(value):
    if value < -745.0:
        return 0.0
    if value > 709.0:
        return INF
    k = _round_nearest(value / LN2)
    r = value - k * LN2
    term = 1.0
    total = 1.0
    for idx in range(1, 42):
        term *= r / idx
        total += term
    return total * (2.0**k)


def _log(value):
    if value <= 0.0:
        raise ValueError("log domain error")
    x = float(value)
    k = 0
    while x > SQRT2:
        x *= 0.5
        k += 1
    while x < INV_SQRT2:
        x *= 2.0
        k -= 1
    y = (x - 1.0) / (x + 1.0)
    y2 = y * y
    term = y
    total = y
    denom = 1
    for _ in range(80):
        term *= y2
        denom += 2
        add = term / denom
        total += add
        if _abs(add) < 1.0e-19:
            break
    return 2.0 * total + k * LN2


def _complex_exp(real_part, imag_part):
    scale = _exp(real_part)
    return scale * complex(_cos(imag_part), _sin(imag_part))


def _clean_scalar(value):
    value = complex(value)
    if _abs(value.imag) <= REAL_TOL * max(1.0, _abs(value.real)):
        return float(value.real)
    return value


def _gcd(left, right):
    left = abs(int(left))
    right = abs(int(right))
    while right:
        left, right = right, left % right
    return left or 1


class Rational:
    def __init__(self, numerator, denominator=1):
        if denominator == 0:
            raise ZeroDivisionError("zero denominator")
        if denominator < 0:
            numerator = -numerator
            denominator = -denominator
        divisor = _gcd(numerator, denominator)
        self.numerator = int(numerator // divisor)
        self.denominator = int(denominator // divisor)

    def __float__(self):
        return self.numerator / self.denominator

    def __repr__(self):
        if self.denominator == 1:
            return str(self.numerator)
        return f"{self.numerator}/{self.denominator}"


class NumericVector:
    __array_priority__ = 1000

    def __init__(self, values):
        self._values = tuple(values)

    def __len__(self):
        return len(self._values)

    def __iter__(self):
        return iter(self._values)

    def __getitem__(self, index):
        return self._values[index]

    @property
    def shape(self):
        return (len(self._values),)

    def tolist(self):
        return list(self._values)

    def _binary(self, other, op):
        if isinstance(other, (int, float, complex, Rational)):
            return NumericVector(op(value, other) for value in self._values)
        other_values = tuple(other)
        if len(other_values) != len(self._values):
            raise ValueError("vector lengths differ")
        return NumericVector(op(a, b) for a, b in zip(self._values, other_values, strict=True))

    def _rbinary(self, other, op):
        if isinstance(other, (int, float, complex, Rational)):
            return NumericVector(op(other, value) for value in self._values)
        other_values = tuple(other)
        if len(other_values) != len(self._values):
            raise ValueError("vector lengths differ")
        return NumericVector(op(a, b) for a, b in zip(other_values, self._values, strict=True))

    def __add__(self, other):
        return self._binary(other, lambda a, b: a + b)

    def __radd__(self, other):
        return self._rbinary(other, lambda a, b: a + b)

    def __sub__(self, other):
        return self._binary(other, lambda a, b: a - b)

    def __rsub__(self, other):
        return self._rbinary(other, lambda a, b: a - b)

    def __mul__(self, other):
        return self._binary(other, lambda a, b: a * b)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return self._binary(other, lambda a, b: a / b)

    def __neg__(self):
        return NumericVector(-value for value in self._values)

    def __gt__(self, other):
        return self._binary(other, lambda a, b: a > b)

    def __ge__(self, other):
        return self._binary(other, lambda a, b: a >= b)

    def __lt__(self, other):
        return self._binary(other, lambda a, b: a < b)

    def __le__(self, other):
        return self._binary(other, lambda a, b: a <= b)

    def __matmul__(self, other):
        values = tuple(other)
        if len(values) != len(self._values):
            raise ValueError("vector lengths differ")
        return sum(a * b for a, b in zip(self._values, values, strict=True))

    def __repr__(self):
        return f"NumericVector({self._values!r})"


class PointTable:
    def __init__(self, rows):
        clean = []
        for row in rows:
            if isinstance(row, complex):
                x = float(row.real)
                y = float(row.imag)
            else:
                values = tuple(row)
                if len(values) != 2:
                    raise ValueError("points must have shape (n, 2)")
                x = float(values[0])
                y = float(values[1])
            if not (_finite(x) and _finite(y)):
                raise ValueError("points contain non-finite values")
            clean.append((x, y))
        if len(clean) < 3:
            raise ValueError("at least three boundary points are required")
        self._rows = tuple(clean)

    def __len__(self):
        return len(self._rows)

    def __iter__(self):
        for row in self._rows:
            yield NumericVector(row)

    def __getitem__(self, index):
        if isinstance(index, tuple):
            return self._rows[index[0]][index[1]]
        if isinstance(index, slice):
            return [NumericVector(row) for row in self._rows[index]]
        return NumericVector(self._rows[index])

    @property
    def shape(self):
        return (len(self._rows), 2)

    def row_tuple(self, index):
        return self._rows[index]

    def tolist(self):
        return [list(row) for row in self._rows]

    def __repr__(self):
        return f"PointTable({self._rows!r})"


def _as_points(points):
    if isinstance(points, PointTable):
        return points
    return PointTable(points)


def _as_complex_vector(values, name="values"):
    try:
        out = tuple(complex(value) for value in values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a one-dimensional sequence") from exc
    if len(out) < 2:
        raise ValueError(f"{name} must contain at least two samples")
    return out


def _as_density_samples(values, n):
    density = _as_complex_vector(values, "density_samples")
    if len(density) != n:
        raise ValueError("density_samples must have shape (n,)")
    return density


def _as_target_point(target):
    if isinstance(target, complex):
        x = float(target.real)
        y = float(target.imag)
    else:
        values = tuple(target)
        if len(values) != 2:
            raise ValueError("target must have shape (2,)")
        x = float(values[0])
        y = float(values[1])
    if not (_finite(x) and _finite(y)):
        raise ValueError("target contains non-finite values")
    return (x, y)


def _dist2(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def _dist(a, b):
    return _sqrt(_dist2(a, b))


def _nearest_point_index(points, target):
    return min(range(len(points)), key=lambda index: _dist2(points.row_tuple(index), target))


def _cyclic_distance(left, right, n):
    forward = (left - right) % n
    backward = (right - left) % n
    return min(forward, backward)


def _perimeter(points):
    total = 0.0
    n = len(points)
    for index in range(n):
        total += _dist(points.row_tuple(index), points.row_tuple((index + 1) % n))
    return total


def _line_log_antiderivative(u, delta):
    a = _abs(delta)
    if a <= 0.0:
        if _abs(u) <= 0.0:
            return 0.0
        return u * _log(_abs(u)) - u
    radius = _sqrt(u * u + a * a)
    return u * _log(radius) - u + a * _atan(u / a)


def _is_power_of_two(value):
    return value > 0 and value & (value - 1) == 0


def _dft(values, inverse=False):
    n = len(values)
    sign = 1.0 if inverse else -1.0
    scale = 1.0 / n if inverse else 1.0
    out = []
    for mode in range(n):
        total = 0.0 + 0.0j
        for index, value in enumerate(values):
            angle = sign * TAU * mode * index / n
            total += value * complex(_cos(angle), _sin(angle))
        out.append(scale * total)
    return out


def _fft(values):
    n = len(values)
    if n == 0:
        return []
    if not _is_power_of_two(n):
        return _dft(values, inverse=False)
    out = [complex(value) for value in values]
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j ^= bit
        if i < j:
            out[i], out[j] = out[j], out[i]
    size = 2
    while size <= n:
        root_angle = -TAU / size
        root = complex(_cos(root_angle), _sin(root_angle))
        half = size // 2
        for start in range(0, n, size):
            factor = 1.0 + 0.0j
            for offset in range(half):
                even = out[start + offset]
                odd = factor * out[start + offset + half]
                out[start + offset] = even + odd
                out[start + offset + half] = even - odd
                factor *= root
        size <<= 1
    return out


def _ifft(values):
    n = len(values)
    if n == 0:
        return []
    transformed = _fft([value.conjugate() for value in values])
    return [value.conjugate() / n for value in transformed]


class QJetFFTPlan:
    """Reusable radix-two plan with independently generated twiddles."""

    def __init__(self, n):
        self.n = int(n)
        if self.n < 1 or not _is_power_of_two(self.n):
            raise ValueError("a QJetFFTPlan requires a positive radix-two size")
        permutation = []
        width = self.n.bit_length() - 1
        for index in range(self.n):
            value = index
            reversed_value = 0
            for _bit in range(width):
                reversed_value = (reversed_value << 1) | (value & 1)
                value >>= 1
            permutation.append(reversed_value)
        self.permutation = tuple(permutation)
        stages = []
        size = 2
        while size <= self.n:
            half = size // 2
            stages.append(
                tuple(
                    complex(
                        _cos(-TAU * offset / size),
                        _sin(-TAU * offset / size),
                    )
                    for offset in range(half)
                )
            )
            size <<= 1
        self.stages = tuple(stages)

    @property
    def stored_twiddles(self):
        return sum(len(stage) for stage in self.stages)

    def fft(self, values):
        if len(values) != self.n:
            raise ValueError("FFT input length does not match the plan")
        out = [complex(values[index]) for index in self.permutation]
        size = 2
        for twiddles in self.stages:
            half = size // 2
            for start in range(0, self.n, size):
                for offset, factor in enumerate(twiddles):
                    even = out[start + offset]
                    odd = factor * out[start + offset + half]
                    out[start + offset] = even + odd
                    out[start + offset + half] = even - odd
            size <<= 1
        return out

    def ifft(self, values):
        if len(values) != self.n:
            raise ValueError("FFT input length does not match the plan")
        transformed = self.fft(
            [complex(value).conjugate() for value in values]
        )
        return [value.conjugate() / self.n for value in transformed]


_PRECISE_FFT_PLANS = {}


def _precise_fft_plan(n):
    size = int(n)
    plan = _PRECISE_FFT_PLANS.get(size)
    if plan is None:
        plan = QJetFFTPlan(size)
        _PRECISE_FFT_PLANS[size] = plan
    return plan


def _fft_precise(values):
    """FFT using an O(n) cached plan with non-drifting stage twiddles."""

    n = len(values)
    if n == 0:
        return []
    if not _is_power_of_two(n):
        return _dft(values, inverse=False)
    return _precise_fft_plan(n).fft(values)


def _ifft_precise(values):
    n = len(values)
    if n == 0:
        return []
    if not _is_power_of_two(n):
        return _dft(values, inverse=True)
    return _precise_fft_plan(n).ifft(values)


def _clean_vector(output):
    return NumericVector(_clean_scalar(value) for value in output)


class QModeJet:
    def __init__(self, index, signed_mode, eigenvalue):
        self.index = index
        self.signed_mode = signed_mode
        self.eigenvalue = eigenvalue

    @property
    def denominator(self):
        return self.eigenvalue.denominator


class BorrowComputeRepayLedger:
    def __init__(self, borrowed, computed, repaid, residuals, residual_norm, status, notes=""):
        self.borrowed = tuple(borrowed)
        self.computed = tuple(computed)
        self.repaid = tuple(repaid)
        self.residuals = tuple(residuals)
        self.residual_norm = float(residual_norm)
        self.status = status
        self.notes = notes


class QuadratureEvaluation:
    def __init__(
        self,
        value,
        ledger,
        included_modes,
        rational_denominators,
        truncation_bound,
        floating_frontend_bound,
    ):
        self.value = value
        self.ledger = ledger
        self.included_modes = int(included_modes)
        self.rational_denominators = tuple(rational_denominators)
        self.truncation_bound = float(truncation_bound)
        self.floating_frontend_bound = float(floating_frontend_bound)
        self.method = "qjet_borrow_compute_repay"


class MultipoleZetaQEvaluation:
    def __init__(
        self,
        value,
        ledger,
        levels,
        estimated_zeta_exponent,
        direct_points,
        multipole_groups,
        multipole_terms,
        group_count,
        moment_build_units,
        cached_target_work_units,
        single_target_work_units,
    ):
        self.value = value
        self.ledger = ledger
        self.levels = tuple(levels)
        self.estimated_zeta_exponent = float(estimated_zeta_exponent)
        self.direct_points = int(direct_points)
        self.multipole_groups = int(multipole_groups)
        self.multipole_terms = int(multipole_terms)
        self.group_count = int(group_count)
        self.moment_build_units = int(moment_build_units)
        self.cached_target_work_units = int(cached_target_work_units)
        self.single_target_work_units = int(single_target_work_units)
        self.work_units = int(single_target_work_units)
        self.method = "multipole_zeta_refined_q"

    @property
    def stats(self):
        return {
            "estimated_zeta_exponent": self.estimated_zeta_exponent,
            "direct_points": self.direct_points,
            "multipole_groups": self.multipole_groups,
            "multipole_terms": self.multipole_terms,
            "group_count": self.group_count,
            "moment_build_units": self.moment_build_units,
            "cached_target_work_units": self.cached_target_work_units,
            "single_target_work_units": self.single_target_work_units,
            "work_units": self.work_units,
            "levels": self.levels,
        }


class LocalQuadratureEvaluation:
    def __init__(self, value, ledger, method, work_units, stats):
        self.value = value
        self.ledger = ledger
        self.method = method
        self.work_units = int(work_units)
        self.stats = dict(stats)


class QSpectralErrorSignature:
    def __init__(
        self,
        rows,
        symbol_power,
        median_pair_split,
        max_pair_split,
        normalized_symbol_variation,
        error_type,
        recommended_q,
    ):
        self.rows = tuple(rows)
        self.symbol_power = float(symbol_power)
        self.median_pair_split = float(median_pair_split)
        self.max_pair_split = float(max_pair_split)
        self.normalized_symbol_variation = float(normalized_symbol_variation)
        self.error_type = error_type
        self.recommended_q = recommended_q

    @property
    def stats(self):
        return {
            "symbol_power": self.symbol_power,
            "median_pair_split": self.median_pair_split,
            "max_pair_split": self.max_pair_split,
            "normalized_symbol_variation": self.normalized_symbol_variation,
            "error_type": self.error_type,
            "recommended_q": self.recommended_q,
            "rows": self.rows,
        }


class CycleQJet:
    def __init__(self, n):
        if n < 2:
            raise ValueError("n must be at least 2")
        self.n = int(n)

    def eigenvalue_rational(self, index):
        if not 0 <= index < self.n:
            raise ValueError("mode_index out of range")
        return Rational(index * (self.n - index), 2)

    def eigenvalue(self, index):
        return float(self.eigenvalue_rational(index))

    def signed_mode(self, index):
        if not 0 <= index < self.n:
            raise ValueError("mode_index out of range")
        if self.n % 2 == 0:
            return index if index < self.n // 2 else index - self.n
        return index if index <= self.n // 2 else index - self.n

    def mode_jet(self, index):
        return QModeJet(index, self.signed_mode(index), self.eigenvalue_rational(index))

    def eigenvalues(self):
        return NumericVector(self.eigenvalue(index) for index in range(self.n))

    def signed_modes(self):
        return NumericVector(self.signed_mode(index) for index in range(self.n))

    def apply_function(self, values, phi, zero_mode=None):
        vector = _as_complex_vector(values)
        if len(vector) != self.n:
            raise ValueError("values length must match the QJet size")
        transformed = _fft(vector)
        scaled = []
        for index, coefficient in enumerate(transformed):
            multiplier = complex(zero_mode) if index == 0 and zero_mode is not None else complex(phi(self.eigenvalue(index)))
            scaled.append(multiplier * coefficient)
        return _clean_vector(_ifft(scaled))

    def apply_index_function(self, values, phi, zero_mode=None):
        vector = _as_complex_vector(values)
        if len(vector) != self.n:
            raise ValueError("values length must match the QJet size")
        transformed = _fft(vector)
        scaled = []
        for index, coefficient in enumerate(transformed):
            multiplier = complex(zero_mode) if index == 0 and zero_mode is not None else complex(phi(index))
            scaled.append(multiplier * coefficient)
        return _clean_vector(_ifft(scaled))

    def apply(self, values):
        return self.apply_function(values, lambda lam: lam, zero_mode=0.0)

    def solve(self, rhs, project=True):
        vector = list(_as_complex_vector(rhs, "rhs"))
        if len(vector) != self.n:
            raise ValueError("rhs length must match the QJet size")
        mean = sum(vector) / self.n
        if project:
            vector = [value - mean for value in vector]
        elif _abs(mean) > 1.0e-12:
            raise ValueError("rhs must be mean-zero when project=False")
        return self.apply_function(vector, lambda lam: 1.0 / lam, zero_mode=0.0)

    def energy(self, values):
        vector = _as_complex_vector(values)
        applied = self.apply(vector)
        total = sum(value.conjugate() * complex(qv) for value, qv in zip(vector, applied, strict=True))
        return float(total.real)


class BoundaryQJet:
    def __init__(self, points):
        self.points = _as_points(points)

    @classmethod
    def from_points(cls, points):
        return cls(points)

    @property
    def n(self):
        return len(self.points)

    @property
    def perimeter(self):
        return _perimeter(self.points)

    def chord_weight(self, i, j):
        d2 = _dist2(self.points.row_tuple(i), self.points.row_tuple(j))
        if d2 <= 0.0:
            raise ValueError("duplicate points produce singular Q weights")
        return 1.0 / d2

    def apply(self, values):
        vector = _as_complex_vector(values)
        if len(vector) != self.n:
            raise ValueError("values length must match the QJet size")
        out = [0.0 + 0.0j for _ in range(self.n)]
        for i in range(self.n):
            vi = vector[i]
            pi = self.points.row_tuple(i)
            for j in range(i + 1, self.n):
                weight = 1.0 / _dist2(pi, self.points.row_tuple(j))
                diff = weight * (vi - vector[j])
                out[i] += diff
                out[j] -= diff
        return _clean_vector(out)

    def energy(self, values):
        vector = _as_complex_vector(values)
        if len(vector) != self.n:
            raise ValueError("values length must match the QJet size")
        total = 0.0
        for i in range(self.n):
            pi = self.points.row_tuple(i)
            for j in range(i + 1, self.n):
                total += _abs(vector[i] - vector[j]) ** 2 / _dist2(pi, self.points.row_tuple(j))
        return float(total)


def _cycle_adjacency_row_sum(n):
    return (n * n - 1.0) / 12.0


def _fft_work_units(n):
    if _is_power_of_two(n):
        levels = 0
        size = n
        while size > 1:
            levels += 1
            size >>= 1
        return n * max(levels, 1)
    return n * n


class PullbackMetricQJet:
    """FFT principal QJet for a boundary pullback metric.

    The stored generator is the speed jet ``s_i = |dz/dtheta|`` and the regular
    cycle QJet.  The physical principal chord operator is applied as the
    weighted edge form

        (Q_s v)_i = a_i sum_{j != i} c_ij a_j (v_i - v_j),
        a_i = 1 / s_i,

    where ``c_ij`` is the unit-circle inverse-square chord weight.  The circle
    adjacency action is recovered from the FFT QJet as

        A_c g = diag_c g - Q_circle g,

    so no dense matrix or pair-weight table is stored.
    """

    def __init__(self, speeds):
        values = tuple(float(value) for value in speeds)
        if len(values) < 3:
            raise ValueError("at least three pullback speed samples are required")
        if any(value <= 0.0 or not _finite(value) for value in values):
            raise ValueError("pullback speeds must be positive and finite")
        self.speeds = values
        self.inverse_speeds = tuple(1.0 / value for value in values)
        self.n = len(values)
        self.cycle = CycleQJet(self.n)
        self.cycle_adjacency_row_sum = _cycle_adjacency_row_sum(self.n)
        self.adjacency_inverse_speed = self._cycle_adjacency_apply(self.inverse_speeds)
        self.apply_work_units = 2 * _fft_work_units(self.n) + 6 * self.n

    @property
    def min_speed(self):
        return min(self.speeds)

    @property
    def max_speed(self):
        return max(self.speeds)

    @property
    def anisotropy(self):
        return self.max_speed / self.min_speed

    @property
    def uses_radix2_fft(self):
        return _is_power_of_two(self.n)

    def _cycle_adjacency_apply(self, values):
        q_values = self.cycle.apply(values)
        return tuple(
            self.cycle_adjacency_row_sum * complex(values[index]) - complex(q_values[index])
            for index in range(self.n)
        )

    def apply(self, values):
        vector = _as_complex_vector(values)
        if len(vector) != self.n:
            raise ValueError("values length must match the pullback metric QJet size")
        weighted_values = [self.inverse_speeds[index] * vector[index] for index in range(self.n)]
        adjacency_weighted_values = self._cycle_adjacency_apply(weighted_values)
        output = [
            self.inverse_speeds[index]
            * (vector[index] * self.adjacency_inverse_speed[index] - adjacency_weighted_values[index])
            for index in range(self.n)
        ]
        return _clean_vector(output)

    def energy(self, values):
        vector = _as_complex_vector(values)
        applied = self.apply(vector)
        total = sum(value.conjugate() * complex(qv) for value, qv in zip(vector, applied, strict=True))
        return float(total.real)

    def stats(self):
        return {
            "n": self.n,
            "min_speed": self.min_speed,
            "max_speed": self.max_speed,
            "anisotropy": self.anisotropy,
            "uses_radix2_fft": self.uses_radix2_fft,
            "apply_work_units": self.apply_work_units,
            "stored_dense_q_matrix": False,
            "stored_pair_weight_table": False,
            "protocol": "pullback_metric_fft_qjet",
        }


class MultipoleLeafQJet:
    def __init__(self, points, density_samples, order=16, leaf_size=32, theta=0.45):
        if order < 1:
            raise ValueError("order must be positive")
        if leaf_size < 1:
            raise ValueError("leaf_size must be positive")
        if theta <= 0.0:
            raise ValueError("theta must be positive")
        self.points = _as_points(points)
        self.density = _as_density_samples(density_samples, len(self.points))
        self.order = int(order)
        self.leaf_size = int(leaf_size)
        self.theta = float(theta)
        self.weights = tuple(float(value) for value in vertex_arclength_weights(self.points))
        self.leaves = tuple(self._build_leaf(start, min(len(self.points), start + self.leaf_size)) for start in range(0, len(self.points), self.leaf_size))
        self.moment_build_units = len(self.points) * self.order

    def _build_leaf(self, start, stop):
        count = stop - start
        if count <= 0:
            raise ValueError("empty multipole leaf")
        cx = sum(self.points[index, 0] for index in range(start, stop)) / count
        cy = sum(self.points[index, 1] for index in range(start, stop)) / count
        center = complex(cx, cy)
        radius = max(_abs(complex(self.points[index, 0], self.points[index, 1]) - center) for index in range(start, stop))
        total_weight = 0.0 + 0.0j
        moments = [0.0 + 0.0j for _ in range(self.order)]
        conjugate_moments = [0.0 + 0.0j for _ in range(self.order)]
        for index in range(start, stop):
            shifted = complex(self.points[index, 0], self.points[index, 1]) - center
            weighted_density = self.weights[index] * self.density[index]
            total_weight += weighted_density
            running = shifted
            conjugate_running = shifted.conjugate()
            for mode in range(self.order):
                moments[mode] += weighted_density * running
                conjugate_moments[mode] += weighted_density * conjugate_running
                running *= shifted
                conjugate_running *= shifted.conjugate()
        return {
            "start": start,
            "stop": stop,
            "center": center,
            "radius": radius,
            "total_weight": total_weight,
            "moments": tuple(moments),
            "conjugate_moments": tuple(conjugate_moments),
        }

    def evaluate(self, target):
        x = _as_target_point(target)
        z = complex(x[0], x[1])
        value = 0.0 + 0.0j
        stats = {
            "direct_points": 0,
            "multipole_groups": 0,
            "multipole_terms": 0,
            "group_count": len(self.leaves),
            "moment_build_units": self.moment_build_units,
            "cached_target_work_units": 0,
            "single_target_work_units": self.moment_build_units,
            "work_units": self.moment_build_units,
        }
        target_units = 0
        for leaf in self.leaves:
            zc = z - leaf["center"]
            use_multipole = _abs(zc) > 0.0 and leaf["radius"] / _abs(zc) <= self.theta
            if use_multipole:
                series = 0.0 + 0.0j
                denom_power = zc
                conjugate_denom_power = zc.conjugate()
                for mode, moment in enumerate(leaf["moments"], start=1):
                    conjugate_moment = leaf["conjugate_moments"][mode - 1]
                    series += 0.5 * (moment / denom_power + conjugate_moment / conjugate_denom_power) / mode
                    denom_power *= zc
                    conjugate_denom_power *= zc.conjugate()
                value += leaf["total_weight"] * _log(_abs(zc)) - series
                stats["multipole_groups"] += 1
                stats["multipole_terms"] += self.order
                target_units += self.order
            else:
                for index in range(leaf["start"], leaf["stop"]):
                    distance = _dist(self.points.row_tuple(index), x)
                    if distance <= 0.0:
                        raise ValueError("target must not coincide with a boundary sample")
                    value += self.weights[index] * self.density[index] * _log(distance)
                direct_count = leaf["stop"] - leaf["start"]
                stats["direct_points"] += direct_count
                target_units += direct_count
        stats["cached_target_work_units"] = target_units
        stats["single_target_work_units"] += target_units
        stats["work_units"] = stats["single_target_work_units"]
        return complex(_clean_scalar(value)), stats


class LazyGulatiMatrix:
    def __init__(self, points):
        self.points = _as_points(points)
        self.n = len(self.points)

    @property
    def shape(self):
        return (self.n, self.n)

    def _entry(self, i, j):
        if i == j:
            total = 0.0
            for k in range(self.n):
                if k != i:
                    total += 1.0 / _dist2(self.points.row_tuple(i), self.points.row_tuple(k))
            return total
        return -1.0 / _dist2(self.points.row_tuple(i), self.points.row_tuple(j))

    def row(self, index):
        return NumericVector(self._entry(index, col) for col in range(self.n))

    def __len__(self):
        return self.n

    def __iter__(self):
        for index in range(self.n):
            yield self.row(index)

    def __getitem__(self, index):
        if isinstance(index, tuple):
            return self._entry(index[0], index[1])
        if isinstance(index, slice):
            return [self.row(row) for row in range(*index.indices(self.n))]
        return self.row(index)

    def __matmul__(self, values):
        return BoundaryQJet(self.points).apply(values)


class LazyIncidenceFactor:
    def __init__(self, points):
        self.points = _as_points(points)
        self.n = len(self.points)
        self.rows = self.n * (self.n - 1) // 2

    @property
    def shape(self):
        return (self.rows, self.n)

    @property
    def T(self):
        return LazyIncidenceTranspose(self)

    def _pair_for_row(self, row):
        if not 0 <= row < self.rows:
            raise IndexError("row out of range")
        cursor = 0
        for i in range(self.n):
            count = self.n - i - 1
            if row < cursor + count:
                return (i, i + 1 + row - cursor)
            cursor += count
        raise IndexError("row out of range")

    def _entry(self, row, col):
        i, j = self._pair_for_row(row)
        distance = _dist(self.points.row_tuple(i), self.points.row_tuple(j))
        if distance <= 0.0:
            raise ValueError("duplicate points produce singular Q weights")
        if col == i:
            return 1.0 / distance
        if col == j:
            return -1.0 / distance
        return 0.0

    def row(self, row):
        i, j = self._pair_for_row(row)
        distance = _dist(self.points.row_tuple(i), self.points.row_tuple(j))
        if distance <= 0.0:
            raise ValueError("duplicate points produce singular Q weights")
        inv = 1.0 / distance
        return NumericVector(inv if col == i else -inv if col == j else 0.0 for col in range(self.n))

    def __len__(self):
        return self.rows

    def __iter__(self):
        for row in range(self.rows):
            yield self.row(row)

    def __getitem__(self, index):
        if isinstance(index, tuple):
            return self._entry(index[0], index[1])
        if isinstance(index, slice):
            return [self.row(row) for row in range(*index.indices(self.rows))]
        return self.row(index)

    def __matmul__(self, values):
        vector = _as_complex_vector(values)
        if len(vector) != self.n:
            raise ValueError("values length must match factor width")
        out = []
        for row in range(self.rows):
            i, j = self._pair_for_row(row)
            out.append((vector[i] - vector[j]) / _dist(self.points.row_tuple(i), self.points.row_tuple(j)))
        return _clean_vector(out)


class LazyIncidenceTranspose:
    def __init__(self, factor):
        self.factor = factor

    @property
    def shape(self):
        return (self.factor.n, self.factor.rows)

    def __matmul__(self, other):
        if other is self.factor:
            return LazyGulatiMatrix(self.factor.points)
        vector = _as_complex_vector(other)
        if len(vector) != self.factor.rows:
            raise ValueError("values length must match factor height")
        out = [0.0 + 0.0j for _ in range(self.factor.n)]
        for row, value in enumerate(vector):
            i, j = self.factor._pair_for_row(row)
            distance = _dist(self.factor.points.row_tuple(i), self.factor.points.row_tuple(j))
            out[i] += value / distance
            out[j] -= value / distance
        return _clean_vector(out)


def regular_polygon_points(n, radius=1.0, phase=0.0):
    if n < 3:
        raise ValueError("n must be at least 3")
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    return PointTable(
        (
            radius * _cos(phase + TAU * index / n),
            radius * _sin(phase + TAU * index / n),
        )
        for index in range(n)
    )


def cycle_gulati_eigenvalues(n):
    return CycleQJet(n).eigenvalues()


def signed_fourier_modes(n):
    return CycleQJet(n).signed_modes()


def cycle_gulati_condition_number(n):
    qjet = CycleQJet(n)
    values = [qjet.eigenvalue(index) for index in range(1, n)]
    return max(values) / min(values)


def gulati_incidence_factor(points):
    return LazyIncidenceFactor(points)


def vertex_arclength_weights(points):
    pts = _as_points(points)
    n = len(pts)
    edges = [_dist(pts.row_tuple(index), pts.row_tuple((index + 1) % n)) for index in range(n)]
    if any(edge <= 0.0 for edge in edges):
        raise ValueError("duplicate adjacent points produce zero arclength weight")
    return NumericVector(0.5 * (edges[index] + edges[index - 1]) for index in range(n))


def outward_unit_normals(points):
    pts = _as_points(points)
    n = len(pts)
    rows = []
    for index in range(n):
        prev_pt = pts.row_tuple((index - 1) % n)
        next_pt = pts.row_tuple((index + 1) % n)
        tx = next_pt[0] - prev_pt[0]
        ty = next_pt[1] - prev_pt[1]
        length = _hypot(tx, ty)
        if length <= 0.0:
            raise ValueError("duplicate neighboring points produce undefined normal")
        ux = tx / length
        uy = ty / length
        rows.append((uy, -ux))
    return PointTable(rows)


def offset_boundary_points(points, distance, outward=True):
    if distance <= 0.0:
        raise ValueError("distance must be positive")
    pts = _as_points(points)
    normals = outward_unit_normals(pts)
    sign = 1.0 if outward else -1.0
    return PointTable(
        (
            pts[index, 0] + sign * distance * normals[index, 0],
            pts[index, 1] + sign * distance * normals[index, 1],
        )
        for index in range(len(pts))
    )


def gulati_coercivity_at_point(points, target, density_abs=None):
    pts = _as_points(points)
    x = _as_target_point(target)
    weights = vertex_arclength_weights(pts)
    if density_abs is None:
        density = [1.0] * len(pts)
    else:
        density = [_abs(complex(value)) for value in density_abs]
        if len(density) != len(pts):
            raise ValueError("density_abs must have shape (n,)")
    total = 0.0
    for index in range(len(pts)):
        d2 = _dist2(pts.row_tuple(index), x)
        if d2 <= 0.0:
            raise ValueError("target must not coincide with a boundary sample")
        total += float(weights[index]) * float(density[index]) / d2
    return total


def near_boundary_gulati_coercivity_table(
    points,
    sample_index=0,
    deltas=(8e-2, 4e-2, 2e-2, 1e-2, 5e-3),
    outward=True,
):
    pts = _as_points(points)
    if not 0 <= sample_index < len(pts):
        raise ValueError("sample_index out of range")
    normals = outward_unit_normals(pts)
    sign = 1.0 if outward else -1.0
    rows = []
    for delta in deltas:
        if delta <= 0.0:
            raise ValueError("deltas must be positive")
        target = (
            pts[sample_index, 0] + sign * delta * normals[sample_index, 0],
            pts[sample_index, 1] + sign * delta * normals[sample_index, 1],
        )
        value = gulati_coercivity_at_point(pts, target)
        rows.append(
            {
                "delta": float(delta),
                "coercivity": value,
                "delta_times_coercivity_over_pi": delta * value / PI,
            }
        )
    return rows


def arclength_scaled_gulati_eigenvalues(points):
    pts = _as_points(points)
    length = _perimeter(pts)
    n = len(pts)
    eigenvalues = [0.0]
    for mode in range(1, (n - 1) // 2 + 1):
        value = 2.0 * PI * PI * mode / length
        eigenvalues.extend((value, value))
    if len(eigenvalues) < n:
        eigenvalues.append(2.0 * PI * PI * (n // 2) / length)
    return NumericVector(eigenvalues[:n])


def gulati_weyl_pair_ratios(points, mode_start=8, mode_stop=16):
    if mode_start < 1 or mode_stop < mode_start:
        raise ValueError("expected 1 <= mode_start <= mode_stop")
    pts = _as_points(points)
    n = len(pts)
    if 2 * mode_stop >= n:
        raise ValueError("not enough samples for requested mode range")
    qjet = BoundaryQJet(pts)
    length = qjet.perimeter
    h = length / n
    rows = []
    for mode in range(mode_start, mode_stop + 1):
        observed_values = []
        for trig in (_cos, _sin):
            values = [trig(TAU * mode * index / n) for index in range(n)]
            applied = qjet.apply(values)
            numerator = sum(values[index] * float(applied[index]) for index in range(n))
            denominator = sum(value * value for value in values)
            observed_values.append(h * numerator / denominator)
        observed = sum(observed_values) / len(observed_values)
        expected = 2.0 * PI * PI * mode / length
        rows.append(
            {
                "mode": float(mode),
                "observed": observed,
                "expected": expected,
                "ratio": observed / expected,
            }
        )
    return rows


def _median(values):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _log_log_slope(rows, x_key, y_key):
    xs = []
    ys = []
    for row in rows:
        x = row[x_key]
        y = row[y_key]
        if x > 0.0 and y > 0.0 and _finite(x) and _finite(y):
            xs.append(_log(x))
            ys.append(_log(y))
    if len(xs) < 2:
        return 0.0
    xbar = sum(xs) / len(xs)
    ybar = sum(ys) / len(ys)
    sxx = sum((x - xbar) * (x - xbar) for x in xs)
    if sxx <= 0.0:
        return 0.0
    return sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys, strict=True)) / sxx


def _q_spectral_error_type(symbol_power, median_pair_split, max_pair_split, normalized_symbol_variation):
    if median_pair_split >= 0.05:
        return (
            "cusp_endpoint_channel",
            "multipole_zeta_q",
        )
    if max_pair_split >= 0.25 or normalized_symbol_variation >= 0.10:
        return (
            "corner_vertex_scattering_channel",
            "multipole_zeta_q",
        )
    if symbol_power < 0.85:
        return (
            "low_regularity_algebraic_tail",
            "multipole_zeta_q",
        )
    if normalized_symbol_variation <= 0.05 and max_pair_split <= 0.08:
        return (
            "smooth_spectral_tail",
            "multipole_zeta_q",
        )
    return (
        "mixed_geometry_spectral_tail",
        "multipole_zeta_q",
    )


def q_spectral_error_signature(points, mode_start=4, mode_stop=None):
    pts = _as_points(points)
    n = len(pts)
    if mode_start < 1:
        raise ValueError("mode_start must be positive")
    if mode_stop is None:
        mode_stop = min(32, max(mode_start + 3, n // 8))
    if mode_stop < mode_start:
        raise ValueError("expected mode_start <= mode_stop")
    if 2 * mode_stop >= n:
        raise ValueError("not enough samples for requested spectral signature range")
    qjet = BoundaryQJet(pts)
    length = qjet.perimeter
    h = length / n
    rows = []
    normalized_symbols = []
    pair_splits = []
    for mode in range(mode_start, mode_stop + 1):
        observed = []
        for trig_name, trig in (("cos", _cos), ("sin", _sin)):
            values = [trig(TAU * mode * index / n) for index in range(n)]
            applied = qjet.apply(values)
            numerator = sum(values[index] * float(applied[index]) for index in range(n))
            denominator = sum(value * value for value in values)
            observed.append((trig_name, h * numerator / denominator))
        cos_symbol = observed[0][1]
        sin_symbol = observed[1][1]
        symbol = 0.5 * (cos_symbol + sin_symbol)
        expected = 2.0 * PI * PI * mode / length
        normalized = symbol / expected if expected > 0.0 else 0.0
        pair_split = _abs(cos_symbol - sin_symbol) / max(_abs(symbol), TINY)
        normalized_symbols.append(normalized)
        pair_splits.append(pair_split)
        rows.append(
            {
                "mode": mode,
                "cos_symbol": cos_symbol,
                "sin_symbol": sin_symbol,
                "symbol": symbol,
                "expected_smooth_symbol": expected,
                "normalized_symbol": normalized,
                "pair_split": pair_split,
            }
        )
    symbol_power = _log_log_slope(rows, "mode", "symbol")
    mean_normalized = sum(normalized_symbols) / len(normalized_symbols)
    if _abs(mean_normalized) <= TINY:
        normalized_symbol_variation = 0.0
    else:
        normalized_symbol_variation = _sqrt(
            sum((value - mean_normalized) * (value - mean_normalized) for value in normalized_symbols)
            / len(normalized_symbols)
        ) / _abs(mean_normalized)
    median_pair_split = _median(pair_splits)
    max_pair_split = max(pair_splits)
    error_type, recommended_q = _q_spectral_error_type(
        symbol_power,
        median_pair_split,
        max_pair_split,
        normalized_symbol_variation,
    )
    return QSpectralErrorSignature(
        rows,
        symbol_power,
        median_pair_split,
        max_pair_split,
        normalized_symbol_variation,
        error_type,
        recommended_q,
    )


def apply_cycle_gulati_function(values, phi, zero_mode=None):
    vector = _as_complex_vector(values)
    return CycleQJet(len(vector)).apply_function(vector, phi, zero_mode=zero_mode)


def apply_cycle_gulati(values):
    vector = _as_complex_vector(values)
    return CycleQJet(len(vector)).apply(vector)


def solve_cycle_gulati(rhs, project=True):
    vector = _as_complex_vector(rhs, "rhs")
    return CycleQJet(len(vector)).solve(vector, project=project)


def cycle_gulati_fractional_power(values, exponent, zero_mode=0.0):
    return apply_cycle_gulati_function(values, lambda lam: lam**exponent, zero_mode=zero_mode)


def cycle_gulati_heat(values, time):
    if time < 0.0:
        raise ValueError("time must be non-negative")
    return apply_cycle_gulati_function(values, lambda lam: _exp(-time * lam), zero_mode=1.0)


def cycle_gulati_resolvent(values, spectral_parameter):
    if spectral_parameter == 0:
        raise ValueError("spectral_parameter must not be a Q eigenvalue")
    vector = _as_complex_vector(values)
    qjet = CycleQJet(len(vector))
    z = complex(spectral_parameter)
    for index in range(qjet.n):
        if _abs(z - qjet.eigenvalue(index)) <= 1.0e-12:
            raise ValueError("spectral_parameter must not be a Q eigenvalue")
    return qjet.apply_function(vector, lambda lam: 1.0 / (z - lam), zero_mode=1.0 / z)


def cycle_gulati_wave(values, time):
    def multiplier(lam):
        root = _sqrt(lam)
        return _sin(root * time) / root

    return apply_cycle_gulati_function(values, multiplier, zero_mode=complex(time))


def cycle_gulati_energy(values):
    vector = _as_complex_vector(values)
    return CycleQJet(len(vector)).energy(vector)


def circle_log_layer_trapezoid(density_samples, point):
    density = _as_complex_vector(density_samples, "density_samples")
    n = len(density)
    total = 0.0 + 0.0j
    for index, value in enumerate(density):
        theta = TAU * index / n
        node = complex(_cos(theta), _sin(theta))
        total += value * _log(_abs(point - node))
    return float(_clean_scalar((TAU / n) * total))


def _tail_bound_for_circle_log(density, radius, start_mode):
    if start_mode <= 0:
        return INF
    max_density = max(_abs(value) for value in density)
    if radius == 1.0:
        return INF
    q = radius if radius > 1.0 else 1.0 / radius
    if q <= 1.0:
        return INF
    first = q ** (-start_mode)
    return 2.0 * PI * max_density * first / (start_mode * (1.0 - 1.0 / q))


def circle_log_layer_borrow_compute_repay(density_samples, point, frontend_bits=53):
    density = _as_complex_vector(density_samples, "density_samples")
    n = len(density)
    radius = _abs(point)
    if _abs(radius - 1.0) <= 1.0e-15:
        raise ValueError("point must not lie on the unit circle")
    unit = point / radius
    qjet = CycleQJet(n)
    coeffs = [value / n for value in _fft(density)]
    value = 0.0 + 0.0j
    denominators = []
    included = 0
    if radius > 1.0:
        value += TAU * coeffs[0] * _log(radius)
    for index, coeff in enumerate(coeffs):
        mode = qjet.signed_mode(index)
        abs_mode = abs(mode)
        if abs_mode == 0:
            continue
        radial = radius ** (-abs_mode) if radius > 1.0 else radius**abs_mode
        if mode > 0:
            oscillation = unit**mode
        else:
            oscillation = unit.conjugate() ** abs_mode
        value -= PI * coeff * radial * oscillation / abs_mode
        denominators.append(abs_mode)
        included += 1
    tail_start = n // 2 + 1
    tail_bound = _tail_bound_for_circle_log(density, radius, tail_start)
    frontend_bound = 2.0 ** (-frontend_bits)
    residuals = (
        ("modal_truncation_bound", tail_bound),
        ("floating_frontend_bound", frontend_bound),
        ("qjet_row_sum_residual", 0.0),
    )
    residual_norm = max(value for _, value in residuals)
    ledger = BorrowComputeRepayLedger(
        borrowed=(
            "regular-cycle modal QJet generator",
            "finite Fourier front-end coefficients",
            "target transcendental seed log/radial/phase",
        ),
        computed=("finite modal log-layer series",),
        repaid=(
            "exact rational modal denominators 1/|m|",
            "exact generated eigenvalues lambda_m=m(n-m)/2",
            "a-priori geometric tail bound",
        ),
        residuals=residuals,
        residual_norm=residual_norm,
        status="borrowed_repaid",
        notes="Dense Q was not formed; modal denominators are exact integer channels.",
    )
    return QuadratureEvaluation(
        _clean_scalar(value),
        ledger,
        included,
        tuple(denominators),
        tail_bound,
        frontend_bound,
    )


def circle_log_layer_spectral(density_samples, point):
    return float(circle_log_layer_borrow_compute_repay(density_samples, point).value)


def _points_to_complex(points):
    return tuple(complex(points[index, 0], points[index, 1]) for index in range(len(points)))


def log_layer_trapezoid(points, density_samples, target):
    pts = _as_points(points)
    density = _as_density_samples(density_samples, len(pts))
    x = _as_target_point(target)
    weights = vertex_arclength_weights(pts)
    value = 0.0 + 0.0j
    for index in range(len(pts)):
        distance = _dist(pts.row_tuple(index), x)
        if distance <= 0.0:
            raise ValueError("target must not coincide with a boundary sample")
        value += float(weights[index]) * density[index] * _log(distance)
    return complex(_clean_scalar(value))


def log_layer_qbx(points, density_samples, target, center, order=40):
    if order < 0:
        raise ValueError("order must be non-negative")
    pts = _as_points(points)
    density = _as_density_samples(density_samples, len(pts))
    x = _as_target_point(target)
    c = _as_target_point(center)
    source = _points_to_complex(pts)
    z_target = complex(x[0], x[1])
    z_center = complex(c[0], c[1])
    target_offset = z_target - z_center
    source_offsets = tuple(z_center - node for node in source)
    target_radius = _abs(target_offset)
    source_radius = min(_abs(offset) for offset in source_offsets)
    if not target_radius < source_radius:
        raise ValueError("target is outside the source-free QBX expansion disk")
    weights = vertex_arclength_weights(pts)
    value = 0.0 + 0.0j
    for index, source_offset in enumerate(source_offsets):
        ratio = target_offset / source_offset
        term = 1.0 + 0.0j
        series = 0.0 + 0.0j
        for mode in range(1, order + 1):
            term *= ratio
            sign = 1.0 if mode % 2 == 1 else -1.0
            series += sign * term / mode
        kernel = _log(_abs(source_offset)) + series.real
        value += float(weights[index]) * density[index] * kernel
    return complex(_clean_scalar(value))


def log_layer_qbx_auto(
    points,
    density_samples,
    target,
    sample_index=None,
    order=40,
    radius_factor=4.0,
):
    """Point-QBX with an automatically chosen same-side expansion center.

    The center is placed on the estimated normal line through the closest
    sampled boundary point, on the same side of the boundary as ``target``:

    ``center = y_j + radius_factor * <target-y_j, n_j> n_j``.

    This keeps the target inside the source-free disk for ordinary
    near-boundary exterior/interior targets when ``radius_factor > 1``.
    """

    if radius_factor <= 1.0:
        raise ValueError("radius_factor must exceed 1")
    pts = _as_points(points)
    x = _as_target_point(target)
    if sample_index is None:
        sample_index = min(range(len(pts)), key=lambda index: _dist2(pts.row_tuple(index), x))
    if not 0 <= sample_index < len(pts):
        raise ValueError("sample_index out of range")
    normal = outward_unit_normals(pts).row_tuple(sample_index)
    base = pts.row_tuple(sample_index)
    offset = (x[0] - base[0], x[1] - base[1])
    signed_delta = offset[0] * normal[0] + offset[1] * normal[1]
    if _abs(signed_delta) <= 0.0:
        raise ValueError("target must have nonzero normal offset from the boundary sample")
    center = (
        base[0] + radius_factor * signed_delta * normal[0],
        base[1] + radius_factor * signed_delta * normal[1],
    )
    return log_layer_qbx(pts, density_samples, x, center, order=order)


def _local_bridge_profile_correction(pts, density, x, sample_index):
    if not 0 <= sample_index < len(pts):
        raise ValueError("sample_index out of range")
    h = _perimeter(pts) / len(pts)
    normal = outward_unit_normals(pts).row_tuple(sample_index)
    prev_pt = pts.row_tuple((sample_index - 1) % len(pts))
    next_pt = pts.row_tuple((sample_index + 1) % len(pts))
    tx = next_pt[0] - prev_pt[0]
    ty = next_pt[1] - prev_pt[1]
    tlen = _hypot(tx, ty)
    if tlen <= 0.0:
        raise ValueError("duplicate neighboring points produce undefined tangent")
    tangent = (tx / tlen, ty / tlen)
    base = pts.row_tuple(sample_index)
    offset = (x[0] - base[0], x[1] - base[1])
    delta = _abs(offset[0] * normal[0] + offset[1] * normal[1])
    if delta <= 0.0:
        raise ValueError("target must have nonzero normal offset from the boundary sample")
    beta_raw = _abs(offset[0] * tangent[0] + offset[1] * tangent[1]) / h
    beta = beta_raw - _floor(beta_raw)
    beta = min(beta, 1.0 - beta)
    rho = delta / h
    profile = _log(_abs(1.0 - _complex_exp(-TAU * rho, TAU * beta)))
    return h * density[sample_index] * profile


def log_layer_local_bridge(points, density_samples, target, sample_index=None):
    pts = _as_points(points)
    density = _as_density_samples(density_samples, len(pts))
    x = _as_target_point(target)
    if sample_index is None:
        sample_index = _nearest_point_index(pts, x)
    correction = _local_bridge_profile_correction(pts, density, x, sample_index)
    return log_layer_trapezoid(pts, density, x) - correction


def log_layer_singularity_subtraction(
    points,
    density_samples,
    target,
    sample_index=None,
    window=8,
):
    return log_layer_singularity_subtraction_borrow_compute_repay(
        points,
        density_samples,
        target,
        sample_index=sample_index,
        window=window,
    ).value


def log_layer_singularity_subtraction_borrow_compute_repay(
    points,
    density_samples,
    target,
    sample_index=None,
    window=8,
):
    if window < 0:
        raise ValueError("window must be non-negative")
    pts = _as_points(points)
    density = _as_density_samples(density_samples, len(pts))
    x = _as_target_point(target)
    n = len(pts)
    if sample_index is None:
        sample_index = _nearest_point_index(pts, x)
    if not 0 <= sample_index < n:
        raise ValueError("sample_index out of range")

    h = _perimeter(pts) / n
    normal = outward_unit_normals(pts).row_tuple(sample_index)
    prev_pt = pts.row_tuple((sample_index - 1) % n)
    next_pt = pts.row_tuple((sample_index + 1) % n)
    tx = next_pt[0] - prev_pt[0]
    ty = next_pt[1] - prev_pt[1]
    tlen = _hypot(tx, ty)
    if tlen <= 0.0:
        raise ValueError("duplicate neighboring points produce undefined tangent")
    tangent = (tx / tlen, ty / tlen)
    base = pts.row_tuple(sample_index)
    offset = (x[0] - base[0], x[1] - base[1])
    delta = _abs(offset[0] * normal[0] + offset[1] * normal[1])
    if delta <= 0.0:
        raise ValueError("target must have nonzero normal offset from the boundary sample")
    target_s = offset[0] * tangent[0] + offset[1] * tangent[1]

    effective_window = min(int(window), n // 2)
    discrete_model = 0.0
    seen = []
    for rel in range(-effective_window, effective_window + 1):
        index = (sample_index + rel) % n
        if index in seen:
            continue
        seen.append(index)
        s = rel * h
        discrete_model += h * _log(_sqrt((s - target_s) * (s - target_s) + delta * delta))
    left = (-effective_window - 0.5) * h - target_s
    right = (effective_window + 0.5) * h - target_s
    analytic_model = _line_log_antiderivative(right, delta) - _line_log_antiderivative(left, delta)
    value = log_layer_trapezoid(pts, density, x) + density[sample_index] * (analytic_model - discrete_model)
    work_units = n + len(seen)
    stats = {
        "sample_index": sample_index,
        "window": effective_window,
        "window_nodes": len(seen),
        "work_units": work_units,
        "delta_over_h": delta / h,
        "tangential_offset_over_h": target_s / h,
    }
    ledger = BorrowComputeRepayLedger(
        borrowed=(
            "coarse trapezoid samples",
            "local straight-line singular model",
            "analytic log-line antiderivative",
        ),
        computed=("density-frozen singularity-subtracted log layer",),
        repaid=(
            "dense log kernel was not formed",
            "only local window correction nodes were revisited",
            "singular model was repaid by exact antiderivative over the window",
        ),
        residuals=(
            ("window_nodes", len(seen)),
            ("delta_over_h", delta / h),
            ("work_units", work_units),
        ),
        residual_norm=0.0,
        status="borrowed_repaid",
        notes="Classical frozen-density singularity subtraction on a local tangent-line model.",
    )
    return LocalQuadratureEvaluation(complex(_clean_scalar(value)), ledger, "singularity_subtraction", work_units, stats)


def log_layer_adaptive_panel(
    points,
    density_samples,
    target,
    sample_index=None,
    panel_radius=6,
    subdivisions=16,
):
    return log_layer_adaptive_panel_borrow_compute_repay(
        points,
        density_samples,
        target,
        sample_index=sample_index,
        panel_radius=panel_radius,
        subdivisions=subdivisions,
    ).value


def log_layer_adaptive_panel_borrow_compute_repay(
    points,
    density_samples,
    target,
    sample_index=None,
    panel_radius=6,
    subdivisions=16,
):
    if panel_radius < 0:
        raise ValueError("panel_radius must be non-negative")
    if subdivisions < 1:
        raise ValueError("subdivisions must be positive")
    pts = _as_points(points)
    density = _as_density_samples(density_samples, len(pts))
    x = _as_target_point(target)
    n = len(pts)
    if sample_index is None:
        sample_index = _nearest_point_index(pts, x)
    if not 0 <= sample_index < n:
        raise ValueError("sample_index out of range")

    value = 0.0 + 0.0j
    near_panels = 0
    work_units = 0
    for index in range(n):
        next_index = (index + 1) % n
        near = min(
            _cyclic_distance(index, sample_index, n),
            _cyclic_distance(next_index, sample_index, n),
        ) <= panel_radius
        splits = int(subdivisions) if near else 1
        if near:
            near_panels += 1
        a = pts.row_tuple(index)
        b = pts.row_tuple(next_index)
        panel_length = _dist(a, b)
        if panel_length <= 0.0:
            raise ValueError("duplicate adjacent points produce zero panel length")
        for sub in range(splits):
            alpha = (sub + 0.5) / splits
            px = a[0] + alpha * (b[0] - a[0])
            py = a[1] + alpha * (b[1] - a[1])
            distance = _dist((px, py), x)
            if distance <= 0.0:
                raise ValueError("target must not coincide with a panel midpoint")
            sigma = (1.0 - alpha) * density[index] + alpha * density[next_index]
            value += (panel_length / splits) * sigma * _log(distance)
            work_units += 1

    stats = {
        "sample_index": sample_index,
        "panel_radius": int(panel_radius),
        "subdivisions": int(subdivisions),
        "near_panels": near_panels,
        "far_panels": n - near_panels,
        "work_units": work_units,
    }
    ledger = BorrowComputeRepayLedger(
        borrowed=(
            "piecewise-linear boundary panels",
            "linear density interpolation",
            "local midpoint subdivision near the target",
        ),
        computed=("near-field adaptive panel midpoint log layer",),
        repaid=(
            "dense matrix was not formed",
            "only near panels were refined",
            "far panels stayed at one generated midpoint each",
        ),
        residuals=(
            ("near_panels", near_panels),
            ("subdivisions", int(subdivisions)),
            ("work_units", work_units),
        ),
        residual_norm=0.0,
        status="borrowed_repaid",
        notes="Structurally distinct local panel refinement, not a QBX expansion.",
    )
    return LocalQuadratureEvaluation(complex(_clean_scalar(value)), ledger, "adaptive_panel", work_units, stats)


def log_layer_multipole_q(points, density_samples, target, order=16, leaf_size=32, theta=0.45):
    qjet = MultipoleLeafQJet(points, density_samples, order=order, leaf_size=leaf_size, theta=theta)
    value, _ = qjet.evaluate(target)
    return value


def log_layer_multipole_bridge(
    points,
    density_samples,
    target,
    sample_index=None,
    order=16,
    leaf_size=32,
    theta=0.45,
):
    evaluation = log_layer_multipole_bridge_borrow_compute_repay(
        points,
        density_samples,
        target,
        sample_index=sample_index,
        order=order,
        leaf_size=leaf_size,
        theta=theta,
    )
    return evaluation.value


def log_layer_multipole_bridge_borrow_compute_repay(
    points,
    density_samples,
    target,
    sample_index=None,
    order=16,
    leaf_size=32,
    theta=0.45,
):
    pts = _as_points(points)
    density = _as_density_samples(density_samples, len(pts))
    x = _as_target_point(target)
    if sample_index is None:
        sample_index = _nearest_point_index(pts, x)
    qjet = MultipoleLeafQJet(pts, density, order=order, leaf_size=leaf_size, theta=theta)
    raw, stats = qjet.evaluate(x)
    correction = _local_bridge_profile_correction(pts, density, x, sample_index)
    value = raw - correction
    ledger = BorrowComputeRepayLedger(
        borrowed=(
            "leaf source moments",
            "local Gulati bridge profile",
            "finite target-side multipole denominators",
        ),
        computed=("near-field direct leaves plus far-field log multipoles",),
        repaid=(
            "dense log kernel was never formed",
            "source moments are reused across targets when cached",
            "local profile removes the endpoint singular channel",
        ),
        residuals=(
            ("direct_points", stats["direct_points"]),
            ("multipole_terms", stats["multipole_terms"]),
            ("single_target_work_units", stats["single_target_work_units"]),
        ),
        residual_norm=0.0,
        status="borrowed_repaid",
        notes="Leaf QJet stores source moments and arclength weights only; no dense matrix is stored.",
    )
    return MultipoleZetaQEvaluation(
        complex(_clean_scalar(value)),
        ledger,
        (
            {
                "n": len(pts),
                "sample_index": sample_index,
                "value_real": float(complex(value).real),
                "value_imag": float(complex(value).imag),
                **stats,
            },
        ),
        0.0,
        stats["direct_points"],
        stats["multipole_groups"],
        stats["multipole_terms"],
        stats["group_count"],
        stats["moment_build_units"],
        stats["cached_target_work_units"],
        stats["single_target_work_units"],
    )


def _zeta_refine_three_levels(values):
    if len(values) < 3:
        raise ValueError("zeta refinement requires at least three levels")
    v0 = values[-3]
    v1 = values[-2]
    v2 = values[-1]
    d01 = _abs(v0 - v1)
    d12 = _abs(v1 - v2)
    if d01 > 0.0 and d12 > 0.0 and d01 > d12:
        exponent = _log(d01 / d12) / LN2
    else:
        exponent = 1.0
    exponent = min(8.0, max(0.25, exponent))
    denominator = 2.0**exponent - 1.0
    if denominator <= 0.0:
        return v2, exponent
    return v2 + (v2 - v1) / denominator, exponent


def log_layer_multipole_zeta_q(
    level_point_sets,
    level_density_samples,
    target,
    sample_indices=None,
    order=16,
    leaf_size=32,
    theta=0.45,
):
    return log_layer_multipole_zeta_q_borrow_compute_repay(
        level_point_sets,
        level_density_samples,
        target,
        sample_indices=sample_indices,
        order=order,
        leaf_size=leaf_size,
        theta=theta,
    ).value


def log_layer_multipole_zeta_q_borrow_compute_repay(
    level_point_sets,
    level_density_samples,
    target,
    sample_indices=None,
    order=16,
    leaf_size=32,
    theta=0.45,
):
    point_sets = tuple(level_point_sets)
    density_sets = tuple(level_density_samples)
    if len(point_sets) != len(density_sets):
        raise ValueError("level_point_sets and level_density_samples must have the same length")
    if len(point_sets) < 3:
        raise ValueError("multipole/zeta Q requires at least three levels")
    x = _as_target_point(target)
    if sample_indices is None:
        sample_index_values = [None] * len(point_sets)
    else:
        sample_index_values = list(sample_indices)
        if len(sample_index_values) != len(point_sets):
            raise ValueError("sample_indices must match the number of levels")

    values = []
    levels = []
    totals = {
        "direct_points": 0,
        "multipole_groups": 0,
        "multipole_terms": 0,
        "group_count": 0,
        "moment_build_units": 0,
        "cached_target_work_units": 0,
        "single_target_work_units": 0,
    }
    for level_index, point_set in enumerate(point_sets):
        pts = _as_points(point_set)
        density = _as_density_samples(density_sets[level_index], len(pts))
        sample_index = sample_index_values[level_index]
        if sample_index is None:
            sample_index = _nearest_point_index(pts, x)
        if not 0 <= sample_index < len(pts):
            raise ValueError("sample_index out of range")
        qjet = MultipoleLeafQJet(pts, density, order=order, leaf_size=leaf_size, theta=theta)
        raw, stats = qjet.evaluate(x)
        correction = _local_bridge_profile_correction(pts, density, x, sample_index)
        value = raw - correction
        values.append(value)
        for key in totals:
            totals[key] += stats[key]
        levels.append(
            {
                "n": len(pts),
                "sample_index": sample_index,
                "value_real": float(complex(value).real),
                "value_imag": float(complex(value).imag),
                **stats,
            }
        )

    refined, exponent = _zeta_refine_three_levels(values)
    ledger = BorrowComputeRepayLedger(
        borrowed=(
            "nested multipole leaf QJets",
            "three-level zeta endpoint defect model",
            "local Gulati bridge profile on every level",
        ),
        computed=("multipole bridge sequence and zeta extrapolated limit",),
        repaid=(
            "no dense log matrix or dense Q matrix was formed",
            "moment build cost is separated from cached target evaluation",
            "active refinement exponent is reported with the result",
        ),
        residuals=(
            ("estimated_zeta_exponent", exponent),
            ("cached_target_work_units", totals["cached_target_work_units"]),
            ("single_target_work_units", totals["single_target_work_units"]),
        ),
        residual_norm=_abs(values[-1] - values[-2]),
        status="borrowed_repaid",
        notes="Zeta refinement uses the last three nested levels; source leaves store only generators and moments.",
    )
    return MultipoleZetaQEvaluation(
        complex(_clean_scalar(refined)),
        ledger,
        tuple(levels),
        exponent,
        totals["direct_points"],
        totals["multipole_groups"],
        totals["multipole_terms"],
        totals["group_count"],
        totals["moment_build_units"],
        totals["cached_target_work_units"],
        totals["single_target_work_units"],
    )


def circle_gulati_coercivity(point, density_abs_samples=None):
    radius = _abs(point)
    if _abs(radius - 1.0) <= 1.0e-15:
        raise ValueError("point must not lie on the unit circle")
    if density_abs_samples is None:
        return TAU / _abs(radius * radius - 1.0)
    density_abs = [_abs(value) for value in _as_complex_vector(density_abs_samples, "density_abs")]
    n = len(density_abs)
    total = 0.0
    for index, density in enumerate(density_abs):
        theta = TAU * index / n
        node = complex(_cos(theta), _sin(theta))
        total += density / (_abs(point - node) ** 2)
    return (TAU / n) * total


def near_singular_circle_table(n=4096, deltas=(1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6), phase=0.7):
    density = [_cos(TAU * index / n) for index in range(n)]
    rows = []
    for delta in deltas:
        radius = 1.0 + delta
        point = radius * complex(_cos(phase), _sin(phase))
        exact = -PI * _cos(phase) / radius
        trapezoid = circle_log_layer_trapezoid(density, point)
        spectral = circle_log_layer_spectral(density, point)
        scale = max(_abs(exact), TINY)
        rows.append(
            {
                "delta": float(delta),
                "n_delta": float(n * delta),
                "trapezoid_relative_error": _abs(trapezoid - exact) / scale,
                "spectral_relative_error": _abs(spectral - exact) / scale,
            }
        )
    return rows


def build_cycle_qjet(n):
    return CycleQJet(n)


def build_boundary_qjet(points):
    return BoundaryQJet.from_points(points)


def build_pullback_metric_qjet(speeds):
    return PullbackMetricQJet(speeds)


__all__ = [
    "BoundaryQJet",
    "BorrowComputeRepayLedger",
    "CycleQJet",
    "LazyGulatiMatrix",
    "LazyIncidenceFactor",
    "LocalQuadratureEvaluation",
    "MultipoleLeafQJet",
    "MultipoleZetaQEvaluation",
    "NumericVector",
    "PointTable",
    "PullbackMetricQJet",
    "QJetFFTPlan",
    "QModeJet",
    "QSpectralErrorSignature",
    "QuadratureEvaluation",
    "Rational",
    "apply_cycle_gulati",
    "apply_cycle_gulati_function",
    "arclength_scaled_gulati_eigenvalues",
    "build_boundary_qjet",
    "build_cycle_qjet",
    "build_pullback_metric_qjet",
    "circle_gulati_coercivity",
    "circle_log_layer_borrow_compute_repay",
    "circle_log_layer_spectral",
    "circle_log_layer_trapezoid",
    "cycle_gulati_condition_number",
    "cycle_gulati_eigenvalues",
    "cycle_gulati_energy",
    "cycle_gulati_fractional_power",
    "cycle_gulati_heat",
    "cycle_gulati_resolvent",
    "cycle_gulati_wave",
    "gulati_coercivity_at_point",
    "gulati_incidence_factor",
    "gulati_weyl_pair_ratios",
    "log_layer_local_bridge",
    "log_layer_adaptive_panel",
    "log_layer_adaptive_panel_borrow_compute_repay",
    "log_layer_multipole_bridge",
    "log_layer_multipole_bridge_borrow_compute_repay",
    "log_layer_multipole_q",
    "log_layer_multipole_zeta_q",
    "log_layer_multipole_zeta_q_borrow_compute_repay",
    "log_layer_qbx",
    "log_layer_qbx_auto",
    "log_layer_singularity_subtraction",
    "log_layer_singularity_subtraction_borrow_compute_repay",
    "log_layer_trapezoid",
    "near_boundary_gulati_coercivity_table",
    "near_singular_circle_table",
    "offset_boundary_points",
    "outward_unit_normals",
    "q_spectral_error_signature",
    "regular_polygon_points",
    "signed_fourier_modes",
    "solve_cycle_gulati",
    "vertex_arclength_weights",
]
