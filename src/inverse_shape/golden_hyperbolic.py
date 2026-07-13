"""Golden Cayley normalization for analytic scale-phase charts.

Write an axisymmetric meridian point as ``w = z + i r`` in the hyperbolic
upper half-plane. An oriented PSL(2,R) frame sends a selected point and
tangent to ``(i, positive vertical)``. The golden coordinate is then

    tau = sqrt(5) * (g(w) - i) / (g(w) + i).

It maps the whole upper half-plane biholomorphically to ``|tau| < sqrt(5)``.
The points at signed hyperbolic distance ``2 log(phi)`` on the normalized
geodesic map exactly to ``tau = +/-1``. No logarithm branch is needed for the
primary coordinate.
"""

from inverse_shape.joukowski_endpoint import GOLDEN_MU, PHI
from inverse_shape.quadrature import (
    HALF_PI,
    PI,
    _abs,
    _atan,
    _complex_exp,
    _cos,
    _exp,
    _finite,
    _hypot,
    _log,
    _sin,
    _sqrt,
)


SQRT5 = _sqrt(5.0)
GOLDEN_TETRATION_MULTIPLIER = 1.0 / (2.0 * PHI)
GOLDEN_RADIAL_CONTRACTION = 1.0 / (PHI * PHI)
GOLDEN_FOURIER_RATIO = GOLDEN_RADIAL_CONTRACTION**2
GOLDEN_TRANSLATION_LENGTH = 2.0 * GOLDEN_MU
GOLDEN_ANALYTIC_RADIUS = SQRT5
GOLDEN_BERNSTEIN_RADIUS = SQRT5 + 2.0


def integer_trace_half_length(trace):
    value = int(trace)
    if value < 3:
        raise ValueError("a hyperbolic integer trace must be at least three")
    return _log(0.5 * (value + _sqrt(value * value - 4.0)))


def integer_trace_analytic_radius(trace):
    value = int(trace)
    if value < 3:
        raise ValueError("a hyperbolic integer trace must be at least three")
    return _sqrt((value + 2.0) / (value - 2.0))


def integer_trace_bernstein_radius(trace):
    radius = integer_trace_analytic_radius(trace)
    return radius + _sqrt(radius * radius - 1.0)


def golden_coordinate_geometric_factor(order):
    degree = int(order)
    if degree < 0:
        raise ValueError("expansion order must be nonnegative")
    return GOLDEN_BERNSTEIN_RADIUS ** (-degree)


def _atan2(y_value, x_value):
    y = float(y_value)
    x = float(x_value)
    if x > 0.0:
        return _atan(y / x)
    if x < 0.0:
        return _atan(y / x) + (PI if y >= 0.0 else -PI)
    if y > 0.0:
        return HALF_PI
    if y < 0.0:
        return -HALF_PI
    raise ValueError("the argument of zero is undefined")


def _complex_log(value):
    number = complex(value)
    radius = _hypot(number.real, number.imag)
    if radius <= 0.0:
        raise ValueError("complex log domain error")
    return complex(_log(radius), _atan2(number.imag, number.real))


def _upper_point(value, name="point"):
    point = complex(value)
    if (
        not _finite(point.real)
        or not _finite(point.imag)
        or point.imag <= 0.0
    ):
        raise ValueError(f"{name} must lie in the upper half-plane")
    return point


def _complex_jet4(value, name):
    result = tuple(complex(component) for component in value)
    if len(result) != 4 or any(
        not _finite(component.real) or not _finite(component.imag)
        for component in result
    ):
        raise ValueError(f"{name} must contain four finite complex values")
    return result


def _mobius_jet(coefficients, source_jet):
    a_value, b_value, c_value, d_value = coefficients
    source = _complex_jet4(source_jet, "source jet")
    denominator = c_value * source[0] + d_value
    if _abs(denominator) <= 1.0e-30:
        raise ValueError("Mobius jet meets its pole")
    determinant = a_value * d_value - b_value * c_value
    first_map = determinant / denominator**2
    second_map = -2.0 * c_value * determinant / denominator**3
    third_map = 6.0 * c_value**2 * determinant / denominator**4
    return (
        (a_value * source[0] + b_value) / denominator,
        first_map * source[1],
        second_map * source[1] ** 2 + first_map * source[2],
        third_map * source[1] ** 3
        + 3.0 * second_map * source[1] * source[2]
        + first_map * source[3],
    )


class HolomorphicThreeJet:
    """Value-through-third-derivative data for a complex chart."""

    def __init__(self, values):
        self.values = _complex_jet4(values, "holomorphic jet")

    @property
    def value(self):
        return self.values[0]

    def stats(self):
        return {
            "jet_order": 3,
            "stored_complex_scalars": 4,
            "stored_dense_matrix": False,
        }


def meridian_upper_point(radius, height):
    radius_value = float(radius)
    height_value = float(height)
    if (
        radius_value <= 0.0
        or not _finite(radius_value)
        or not _finite(height_value)
    ):
        raise ValueError("a meridian point requires positive finite radius")
    return complex(height_value, radius_value)


def hyperbolic_cosh_upper(left, right):
    first = _upper_point(left, "left point")
    second = _upper_point(right, "right point")
    difference = first - second
    return 1.0 + (
        difference.real * difference.real
        + difference.imag * difference.imag
    ) / (2.0 * first.imag * second.imag)


def hyperbolic_cosh_golden(left, right):
    first = complex(left)
    second = complex(right)
    first_floor = 5.0 - _abs(first) ** 2
    second_floor = 5.0 - _abs(second) ** 2
    if first_floor <= 0.0 or second_floor <= 0.0:
        raise ValueError("golden coordinates must satisfy |tau| < sqrt(5)")
    difference = first - second
    return 1.0 + 10.0 * _abs(difference) ** 2 / (
        first_floor * second_floor
    )


def golden_pseudohyperbolic_distance(left, right):
    first = complex(left)
    second = complex(right)
    if _abs(first) >= SQRT5 or _abs(second) >= SQRT5:
        raise ValueError("golden coordinates must lie in the analytic disk")
    denominator = 5.0 - first.conjugate() * second
    value = _abs(SQRT5 * (first - second) / denominator)
    if value >= 1.0 + 2.0e-14:
        raise RuntimeError("pseudohyperbolic distance escaped the unit disk")
    return min(value, 1.0)


def golden_hyperbolic_decay(left, right):
    distance = golden_pseudohyperbolic_distance(left, right)
    return (1.0 - distance) / (1.0 + distance)


def golden_tau_from_xi(value):
    xi = complex(value)
    exponential = _complex_exp(xi.real, xi.imag)
    return SQRT5 * (exponential - 1.0) / (exponential + 1.0)


def golden_xi_from_tau(value):
    tau = complex(value)
    if _abs(tau) >= SQRT5:
        raise ValueError("golden coordinate must satisfy |tau| < sqrt(5)")
    ratio = (SQRT5 + tau) / (SQRT5 - tau)
    return _complex_log(ratio)


def scale_phase_cosh_from_xi(left, right):
    first = complex(left)
    second = complex(right)
    delta_rho = first.real - second.real
    cosine_hyperbolic = 0.5 * (
        _exp(delta_rho) + _exp(-delta_rho)
    )
    numerator = cosine_hyperbolic - _sin(first.imag) * _sin(second.imag)
    denominator = _cos(first.imag) * _cos(second.imag)
    if denominator <= 0.0:
        raise ValueError("scale-phase points left the branch-free strip")
    return numerator / denominator


class GoldenHyperbolicFrame:
    """Oriented PSL(2,R) frame with a golden Cayley readout."""

    def __init__(self, center, tangent):
        self.center = _upper_point(center, "frame center")
        self.tangent = complex(tangent)
        tangent_norm = _abs(self.tangent)
        if tangent_norm <= 0.0 or not _finite(tangent_norm):
            raise ValueError("frame tangent must be finite and nonzero")
        self.tangent_unit = self.tangent / tangent_norm
        rotation = 1j / self.tangent_unit
        rotation /= _abs(rotation)
        cosine = max(-1.0, min(1.0, rotation.real))
        if cosine > -1.0 + 1.0e-15:
            self.rotation_cosine_half = _sqrt(0.5 * (1.0 + cosine))
            self.rotation_sine_half = (
                rotation.imag / (2.0 * self.rotation_cosine_half)
            )
        else:
            self.rotation_cosine_half = 0.0
            self.rotation_sine_half = 1.0
        self.tangent_norm = tangent_norm
        x_value = self.center.real
        y_value = self.center.imag
        cosine = self.rotation_cosine_half
        sine = self.rotation_sine_half
        self.mobius_coefficients = (
            complex(cosine),
            complex(-cosine * x_value + sine * y_value),
            complex(-sine),
            complex(sine * x_value + cosine * y_value),
        )
        a_value, b_value, c_value, d_value = self.mobius_coefficients
        self.golden_coefficients = (
            SQRT5 * (a_value - 1j * c_value),
            SQRT5 * (b_value - 1j * d_value),
            a_value + 1j * c_value,
            b_value + 1j * d_value,
        )
        golden_a, golden_b, golden_c, golden_d = self.golden_coefficients
        self.inverse_golden_coefficients = (
            golden_d,
            -golden_b,
            -golden_c,
            golden_a,
        )

    @classmethod
    def from_meridian(cls, radius, height, radius_derivative, height_derivative):
        return cls(
            meridian_upper_point(radius, height),
            complex(float(height_derivative), float(radius_derivative)),
        )

    def mobius(self, value):
        point = _upper_point(value)
        normalized = (point - self.center.real) / self.center.imag
        cosine = self.rotation_cosine_half
        sine = self.rotation_sine_half
        denominator = -sine * normalized + cosine
        result = (cosine * normalized + sine) / denominator
        return _upper_point(result, "Mobius image")

    def inverse_mobius(self, value):
        point = _upper_point(value)
        cosine = self.rotation_cosine_half
        sine = self.rotation_sine_half
        normalized = (cosine * point - sine) / (sine * point + cosine)
        result = self.center.real + self.center.imag * normalized
        return _upper_point(result, "inverse Mobius image")

    def tau(self, value):
        normalized = self.mobius(value)
        return SQRT5 * (normalized - 1j) / (normalized + 1j)

    def transform_jet(self, source_jet):
        source_values = (
            source_jet.values
            if isinstance(source_jet, HolomorphicThreeJet)
            else source_jet
        )
        source = _complex_jet4(source_values, "upper-half-plane jet")
        _upper_point(source[0], "jet base point")
        transformed = _mobius_jet(self.golden_coefficients, source)
        if _abs(transformed[0]) >= SQRT5 * (1.0 + 2.0e-14):
            raise RuntimeError("golden jet base escaped the analytic disk")
        return HolomorphicThreeJet(transformed)

    def inverse_transform_jet(self, golden_jet):
        source = (
            golden_jet.values
            if isinstance(golden_jet, HolomorphicThreeJet)
            else golden_jet
        )
        source = _complex_jet4(source, "golden jet")
        if _abs(source[0]) >= SQRT5:
            raise ValueError("golden jet base must satisfy |tau| < sqrt(5)")
        transformed = _mobius_jet(
            self.inverse_golden_coefficients,
            source,
        )
        _upper_point(transformed[0], "inverse jet base point")
        return HolomorphicThreeJet(transformed)

    def inverse_tau(self, value):
        tau = complex(value)
        if _abs(tau) >= SQRT5:
            raise ValueError("golden coordinate must lie in |tau| < sqrt(5)")
        normalized = 1j * (SQRT5 + tau) / (SQRT5 - tau)
        return self.inverse_mobius(normalized)

    def xi(self, value):
        return golden_xi_from_tau(self.tau(value))

    def certificate(self):
        center_tau = self.tau(self.center)
        center_xi = self.xi(self.center)
        denominator = complex(
            self.rotation_cosine_half,
            -self.rotation_sine_half,
        )
        mapped_tangent = self.tangent / (
            self.center.imag * denominator * denominator
        )
        return {
            "center_tau_residual": _abs(center_tau),
            "center_xi_residual": _abs(center_xi),
            "mapped_tangent_real_residual": _abs(mapped_tangent.real),
            "mapped_tangent_imaginary": mapped_tangent.imag,
            "golden_mu": GOLDEN_MU,
            "golden_trace_residual": _abs(2.0 * _cosh(GOLDEN_MU) - 3.0),
            "golden_endpoint_tau_residual": max(
                _abs(golden_tau_from_xi(GOLDEN_MU) - 1.0),
                _abs(golden_tau_from_xi(-GOLDEN_MU) + 1.0),
            ),
            "analytic_disk_radius": GOLDEN_ANALYTIC_RADIUS,
            "bernstein_radius": GOLDEN_BERNSTEIN_RADIUS,
            "order_24_geometric_factor": golden_coordinate_geometric_factor(24),
            "holomorphic_jet_order": 3,
            "stored_frame_complex_scalars": 4,
            "stored_dense_matrix": False,
        }


def _cosh(value):
    argument = float(value)
    return 0.5 * (_exp(argument) + _exp(-argument))


def _jet4(value, name):
    result = tuple(float(component) for component in value)
    if len(result) != 4 or any(not _finite(component) for component in result):
        raise ValueError(f"{name} must contain four finite values")
    return result


def hyperbolic_speed_two_jet(radius_jet, height_jet):
    radius = _jet4(radius_jet, "radius jet")
    height = _jet4(height_jet, "height jet")
    if radius[0] <= 0.0:
        raise ValueError("hyperbolic arclength requires positive radius")
    metric = radius[1] ** 2 + height[1] ** 2
    if metric <= 0.0:
        raise ValueError("meridian jet must have nonzero tangent")
    metric_first = 2.0 * (
        radius[1] * radius[2] + height[1] * height[2]
    )
    metric_second = 2.0 * (
        radius[2] ** 2
        + radius[1] * radius[3]
        + height[2] ** 2
        + height[1] * height[3]
    )
    euclidean_speed = _sqrt(metric)
    euclidean_first = metric_first / (2.0 * euclidean_speed)
    euclidean_second = (
        metric_second / (2.0 * euclidean_speed)
        - metric_first * metric_first / (4.0 * euclidean_speed**3)
    )
    value = euclidean_speed / radius[0]
    first = (
        euclidean_first / radius[0]
        - euclidean_speed * radius[1] / radius[0] ** 2
    )
    second = (
        euclidean_second / radius[0]
        - 2.0 * euclidean_first * radius[1] / radius[0] ** 2
        - euclidean_speed * radius[2] / radius[0] ** 2
        + 2.0 * euclidean_speed * radius[1] ** 2 / radius[0] ** 3
    )
    return value, first, second


def _reparameterize_jet(function_jet, coordinate_jet):
    function = _jet4(function_jet, "function jet")
    speed, speed_first, speed_second = coordinate_jet
    if speed <= 0.0:
        raise ValueError("new coordinate must be strictly increasing")
    first = function[1] / speed
    second = function[2] / speed**2 - function[1] * speed_first / speed**3
    third = (
        function[3] / speed**3
        - 3.0 * function[2] * speed_first / speed**4
        - function[1] * speed_second / speed**4
        + 3.0 * function[1] * speed_first**2 / speed**5
    )
    return function[0], first, second, third


def golden_arclength_coordinate(signed_distance):
    distance = float(signed_distance)
    sign = -1.0 if distance < 0.0 else 1.0
    decay = _exp(-_abs(distance))
    return sign * SQRT5 * (1.0 - decay) / (1.0 + decay)


class GoldenNormalizedMeridianJet:
    def __init__(
        self,
        coordinate,
        radius_jet,
        height_jet,
        hyperbolic_speed_jet,
    ):
        self.coordinate = float(coordinate)
        self.radius_jet = tuple(radius_jet)
        self.height_jet = tuple(height_jet)
        self.hyperbolic_speed_jet = tuple(hyperbolic_speed_jet)

    def stats(self):
        return {
            "coordinate": self.coordinate,
            "stored_scalars": 1 + 4 + 4 + 3,
            "jet_order": 3,
            "stored_dense_matrix": False,
        }


def golden_reparameterize_meridian_jet(
    radius_jet,
    height_jet,
    signed_hyperbolic_distance,
):
    radius = _jet4(radius_jet, "radius jet")
    height = _jet4(height_jet, "height jet")
    speed_jet = hyperbolic_speed_two_jet(radius, height)
    radius_s = _reparameterize_jet(radius, speed_jet)
    height_s = _reparameterize_jet(height, speed_jet)
    coordinate = golden_arclength_coordinate(signed_hyperbolic_distance)
    coordinate_speed = 0.5 * SQRT5 * (1.0 - coordinate * coordinate / 5.0)
    coordinate_first = -(coordinate / SQRT5) * coordinate_speed
    coordinate_second = -(
        coordinate_speed * coordinate_speed
        + coordinate * coordinate_first
    ) / SQRT5
    coordinate_jet = (
        coordinate_speed,
        coordinate_first,
        coordinate_second,
    )
    radius_tau = _reparameterize_jet(radius_s, coordinate_jet)
    height_tau = _reparameterize_jet(height_s, coordinate_jet)
    return GoldenNormalizedMeridianJet(
        coordinate,
        radius_tau,
        height_tau,
        speed_jet,
    )


__all__ = [
    "GOLDEN_ANALYTIC_RADIUS",
    "GOLDEN_BERNSTEIN_RADIUS",
    "GOLDEN_FOURIER_RATIO",
    "GOLDEN_MU",
    "GOLDEN_RADIAL_CONTRACTION",
    "GOLDEN_TETRATION_MULTIPLIER",
    "GOLDEN_TRANSLATION_LENGTH",
    "GoldenHyperbolicFrame",
    "HolomorphicThreeJet",
    "GoldenNormalizedMeridianJet",
    "golden_arclength_coordinate",
    "golden_coordinate_geometric_factor",
    "golden_hyperbolic_decay",
    "golden_pseudohyperbolic_distance",
    "golden_reparameterize_meridian_jet",
    "golden_tau_from_xi",
    "golden_xi_from_tau",
    "hyperbolic_cosh_golden",
    "hyperbolic_cosh_upper",
    "hyperbolic_speed_two_jet",
    "integer_trace_analytic_radius",
    "integer_trace_bernstein_radius",
    "integer_trace_half_length",
    "meridian_upper_point",
    "scale_phase_cosh_from_xi",
]
