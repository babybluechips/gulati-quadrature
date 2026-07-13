"""Fixed-width weighted discriminant three-jet contraction.

This module implements the exact contraction after generating jets are known.
It deliberately does not hide the cost of constructing those jets.

For simple roots of ``P`` and residues ``a_i``, define the numerator

    A_a(z) = sum_j a_j P(z)/(z-z_j).

Given ``P'(z_i), P''(z_i), P'''(z_i)`` and
``A_a'(z_i), A_a''(z_i)``, the weighted inverse-square sum is a fixed-width
local contraction.  Applying it to residues ``w`` and ``w*f`` gives the full
holomorphic graph action.  Persistent work is O(n), application is O(n), no
rank is selected, and no pair matrix is formed.

For arbitrary nodal densities, generating the numerator jets remains a
separate product/remainder problem.  A sparse rational QJet density can supply
them directly; this module does not claim that every sampled density has such
a fixed-width generator.  The common-circle case is complete for arbitrary
sampled densities: ``RootOfUnityDiscriminantQJet`` uses the foundational QJet
FFT, with mixed-radix and Bluestein reductions for arbitrary lengths, to
generate those jets in O(n log n) time and O(n) workspace.
"""

from inverse_shape.quadrature import (
    PI,
    TAU,
    BorrowComputeRepayLedger,
    _abs,
    _clean_scalar,
    _cos,
    _fft_precise,
    _finite,
    _ifft_precise,
    _is_power_of_two,
    _sin,
)


def _next_power_of_two(value):
    size = 1
    while size < value:
        size <<= 1
    return size


_SMALL_DFT_TABLES = {}
_MIXED_RADIX_TWIDDLES = {}
_BLUESTEIN_PLANS = {}


def _small_dft_table(count):
    table = _SMALL_DFT_TABLES.get(count)
    if table is None:
        table = tuple(
            tuple(
                complex(
                    _cos(-TAU * mode * index / count),
                    _sin(-TAU * mode * index / count),
                )
                for index in range(count)
            )
            for mode in range(count)
        )
        _SMALL_DFT_TABLES[count] = table
    return table


def _direct_small_fft(row):
    count = len(row)
    return tuple(
        sum(
            value * twiddle
            for value, twiddle in zip(
                row,
                _small_dft_table(count)[mode],
                strict=True,
            )
        )
        for mode in range(count)
    )


def _smallest_factor(value):
    if value % 2 == 0:
        return 2
    factor = 3
    while factor * factor <= value:
        if value % factor == 0:
            return factor
        factor += 2
    return value


def _fft_strategy(value):
    if _is_power_of_two(value):
        return "radix_two"
    if value <= 16:
        return "direct_small_factor"
    factor = _smallest_factor(value)
    if factor == value or factor > 16:
        return "bluestein"
    child = _fft_strategy(value // factor)
    if child in ("radix_two", "direct_small_factor"):
        return "mixed_radix"
    if "bluestein" in child:
        return "mixed_radix+bluestein"
    return "mixed_radix"


def _bluestein_plan(count):
    plan = _BLUESTEIN_PLANS.get(count)
    if plan is not None:
        return plan
    work_size = _next_power_of_two(2 * count - 1)
    right = [0.0 + 0.0j for _ in range(work_size)]
    output_chirp = []
    for index in range(count):
        angle = PI * index * index / count
        forward = complex(_cos(angle), -_sin(angle))
        reverse = forward.conjugate()
        right[index] = reverse
        if index:
            right[work_size - index] = reverse
        output_chirp.append(forward)
    plan = (
        work_size,
        tuple(output_chirp),
        tuple(_fft_precise(right)),
    )
    _BLUESTEIN_PLANS[count] = plan
    return plan


def _bluestein_fft(row):
    count = len(row)
    work_size, output_chirp, right_hat = _bluestein_plan(count)
    left = [0.0 + 0.0j for _ in range(work_size)]
    for index, value in enumerate(row):
        left[index] = value * output_chirp[index]
    left_hat = _fft_precise(left)
    convolution = _ifft_precise(
        tuple(
            left_hat[index] * right_hat[index]
            for index in range(work_size)
        )
    )
    return tuple(
        convolution[index] * output_chirp[index]
        for index in range(count)
    )


def _mixed_radix_twiddles(count, factor):
    key = (count, factor)
    twiddles = _MIXED_RADIX_TWIDDLES.get(key)
    if twiddles is None:
        twiddles = tuple(
            tuple(
                complex(
                    _cos(-TAU * offset * mode / count),
                    _sin(-TAU * offset * mode / count),
                )
                for offset in range(factor)
            )
            for mode in range(count)
        )
        _MIXED_RADIX_TWIDDLES[key] = twiddles
    return twiddles


def _fft_any(values):
    """Foundational O(n log n) FFT for every input length."""

    row = tuple(complex(value) for value in values)
    count = len(row)
    if count == 0:
        return tuple()
    if _is_power_of_two(count):
        return tuple(_fft_precise(row))
    if count <= 16:
        return _direct_small_fft(row)
    factor = _smallest_factor(count)
    if factor == count or factor > 16:
        return _bluestein_fft(row)
    sub_length = count // factor
    transformed = tuple(
        _fft_any(row[offset::factor]) for offset in range(factor)
    )
    twiddles = _mixed_radix_twiddles(count, factor)
    output = [0.0 + 0.0j for _ in range(count)]
    for base_mode in range(sub_length):
        for branch in range(factor):
            output_mode = base_mode + sub_length * branch
            output[output_mode] = sum(
                transformed[offset][base_mode]
                * twiddles[output_mode][offset]
                for offset in range(factor)
            )
    return tuple(output)


def _ifft_any(values):
    row = tuple(complex(value) for value in values)
    count = len(row)
    if count == 0:
        return tuple()
    transformed = _fft_any(tuple(value.conjugate() for value in row))
    return tuple(value.conjugate() / count for value in transformed)


def _vector(values, length, name):
    result = tuple(complex(value) for value in values)
    if len(result) != length:
        raise ValueError(f"{name} must contain {length} entries")
    if any(not _finite(value.real) or not _finite(value.imag) for value in result):
        raise ValueError(f"{name} must contain finite entries")
    return result


def weighted_discriminant_s2_from_jets(
    residues,
    p1,
    p2,
    p3,
    numerator_first,
    numerator_second,
    scale=1.0,
):
    """Contract one generated numerator jet into weighted inverse squares."""

    residue_values = tuple(complex(value) for value in residues)
    count = len(residue_values)
    if count == 0:
        return tuple()
    p1_values = _vector(p1, count, "p1")
    p2_values = _vector(p2, count, "p2")
    p3_values = _vector(p3, count, "p3")
    first_values = _vector(numerator_first, count, "numerator_first")
    second_values = _vector(numerator_second, count, "numerator_second")
    scale_value = complex(scale)
    if _abs(scale_value) <= 1.0e-300:
        raise ValueError("scale must be nonzero")
    inverse_scale_squared = 1.0 / (scale_value * scale_value)
    output = []
    for index, residue in enumerate(residue_values):
        derivative = p1_values[index]
        if _abs(derivative) <= 1.0e-300:
            raise ValueError("the discriminant jet is not on the simple-root stratum")
        half_second = 0.5 * p2_values[index]
        peeled_value = first_values[index] - residue * half_second
        peeled_slope = (
            0.5 * second_values[index]
            - residue * p3_values[index] / 6.0
        )
        output.append(
            _clean_scalar(
                inverse_scale_squared
                * (
                    peeled_value * half_second
                    - peeled_slope * derivative
                )
                / (derivative * derivative)
            )
        )
    return tuple(output)


class GeneratedDiscriminantThreeJetQJet:
    """O(n) graph action from P and generated weighted-numerator jets."""

    def __init__(self, p1, p2, p3, scale=1.0):
        self.p1 = tuple(complex(value) for value in p1)
        self.n = len(self.p1)
        if self.n == 0:
            raise ValueError("at least one discriminant jet is required")
        self.p2 = _vector(p2, self.n, "p2")
        self.p3 = _vector(p3, self.n, "p3")
        self.scale = complex(scale)
        if _abs(self.scale) <= 1.0e-300:
            raise ValueError("scale must be nonzero")
        if any(_abs(value) <= 1.0e-300 for value in self.p1):
            raise ValueError("P' vanishes on the supplied simple-root jet")
        self.last_apply_stats = {}

    @property
    def persistent_complex_entries(self):
        return 3 * self.n

    def apply_fields(
        self,
        fields,
        source_weights,
        numerator_jets,
    ):
        rows = tuple(_vector(row, self.n, "field") for row in fields)
        if not rows:
            raise ValueError("at least one field is required")
        weights = _vector(source_weights, self.n, "source_weights")
        jets = tuple(numerator_jets)
        if len(jets) != len(rows) + 1:
            raise ValueError(
                "numerator_jets must contain the weight jet followed by "
                "one weighted-field jet per field"
            )
        parsed_jets = tuple(
            (
                _vector(jet[0], self.n, "numerator_first"),
                _vector(jet[1], self.n, "numerator_second"),
            )
            for jet in jets
        )
        weight_s2 = weighted_discriminant_s2_from_jets(
            weights,
            self.p1,
            self.p2,
            self.p3,
            parsed_jets[0][0],
            parsed_jets[0][1],
            self.scale,
        )
        output = []
        for field_index, row in enumerate(rows):
            weighted_field = tuple(
                weights[index] * row[index] for index in range(self.n)
            )
            field_s2 = weighted_discriminant_s2_from_jets(
                weighted_field,
                self.p1,
                self.p2,
                self.p3,
                parsed_jets[field_index + 1][0],
                parsed_jets[field_index + 1][1],
                self.scale,
            )
            output.append(
                tuple(
                    _clean_scalar(
                        row[index] * complex(weight_s2[index])
                        - complex(field_s2[index])
                    )
                    for index in range(self.n)
                )
            )
        self.last_apply_stats = {
            "method": "fixed_width_generated_discriminant_three_jet",
            "nodes": self.n,
            "scalar_jet_contractions": (len(rows) + 1) * self.n,
            "adaptive_rank": 0,
            "stored_dense_matrix": False,
        }
        return tuple(output)

    def apply(self, values, source_weights, weight_jet, weighted_field_jet):
        return self.apply_fields(
            (values,),
            source_weights,
            (weight_jet, weighted_field_jet),
        )[0]

    def stats(self):
        result = {
            "nodes": self.n,
            "persistent_complex_entries": self.persistent_complex_entries,
            "persistent_storage": "O(n) generated P three-jets",
            "apply_complexity": "O(n) after numerator-jet generation",
            "adaptive_rank": 0,
            "stored_dense_distance_matrix": False,
            "stored_dense_operator_matrix": False,
            "kernel": "holomorphic signed inverse square",
        }
        result.update(self.last_apply_stats)
        return result

    def evaluate(
        self,
        values,
        source_weights,
        weight_jet,
        weighted_field_jet,
    ):
        result = self.apply(
            values,
            source_weights,
            weight_jet,
            weighted_field_jet,
        )
        ledger = BorrowComputeRepayLedger(
            borrowed=(
                "P value/three-jets on the simple-root stratum",
                "generated numerator jets A_w and A_wf",
            ),
            computed=("fixed-width weighted inverse-square graph action",),
            repaid=(
                "all numerator jets remain caller-owned",
                "no pair matrix and no numerical rank",
            ),
            residuals=tuple(),
            residual_norm=0.0,
            status="borrowed_repaid",
            notes=(
                "O(n) applies after jet generation. The generator complexity "
                "depends on the density representation."
            ),
        )
        return result, ledger, self.stats()


def root_of_unity_discriminant_p_jets(n):
    """Return exact analytic P jets for ``P(z)=z^n-1`` on its roots."""

    count = int(n)
    if count < 2:
        raise ValueError("root-of-unity jets require at least two nodes")
    roots = tuple(
        complex(_cos(TAU * index / count), _sin(TAU * index / count))
        for index in range(count)
    )
    p1 = tuple(count / root for root in roots)
    p2 = tuple(count * (count - 1) / (root * root) for root in roots)
    p3 = tuple(
        count * (count - 1) * (count - 2) / (root * root * root)
        for root in roots
    )
    return roots, (p1, p2, p3)


def _root_of_unity_numerator_jets(residues, roots, p1):
    values = tuple(complex(value) for value in residues)
    count = len(values)
    if len(roots) != count or len(p1) != count:
        raise ValueError("root and P-prime jets must match the residue length")
    interpolation_values = tuple(
        values[index] * p1[index] for index in range(count)
    )
    coefficients = tuple(
        value / count for value in _fft_any(interpolation_values)
    )
    first_coefficients = tuple(
        mode * coefficients[mode] for mode in range(count)
    )
    second_coefficients = tuple(
        mode * (mode - 1) * coefficients[mode] for mode in range(count)
    )
    first_base = _ifft_any(first_coefficients)
    second_base = _ifft_any(second_coefficients)
    first = tuple(
        count * first_base[index] / roots[index] for index in range(count)
    )
    second = tuple(
        count
        * second_base[index]
        / (roots[index] * roots[index])
        for index in range(count)
    )
    return first, second


def root_of_unity_numerator_jets(residues):
    """Generate ``A_a'`` and ``A_a''`` by one spectral interpolation."""

    values = tuple(complex(value) for value in residues)
    roots, p_jet = root_of_unity_discriminant_p_jets(len(values))
    return _root_of_unity_numerator_jets(values, roots, p_jet[0])


class RootOfUnityDiscriminantQJet:
    """Unconditional rank-free Q on a common circle via generated three-jets."""

    def __init__(self, n):
        self.roots, p_jet = root_of_unity_discriminant_p_jets(n)
        self.n = len(self.roots)
        self.generated = GeneratedDiscriminantThreeJetQJet(*p_jet)
        self.last_apply_stats = {}

    def apply_signed(self, values, source_weights):
        row = _vector(values, self.n, "values")
        weights = _vector(source_weights, self.n, "source_weights")
        weighted_field = tuple(
            weights[index] * row[index] for index in range(self.n)
        )
        weight_jet = _root_of_unity_numerator_jets(
            weights,
            self.roots,
            self.generated.p1,
        )
        field_jet = _root_of_unity_numerator_jets(
            weighted_field,
            self.roots,
            self.generated.p1,
        )
        result = self.generated.apply(row, weights, weight_jet, field_jet)
        self.last_apply_stats = {
            "method": "root_of_unity_generated_discriminant_three_jet",
            "numerator_qjet_transforms": 6,
            "adaptive_rank": 0,
            "stored_dense_matrix": False,
        }
        return result

    def apply_circle_euclidean(self, values, source_weights, radius=1.0):
        radius_value = float(radius)
        if radius_value <= 0.0 or not _finite(radius_value):
            raise ValueError("circle radius must be positive and finite")
        weights = _vector(source_weights, self.n, "source_weights")
        signed_weights = tuple(
            weights[index] * self.roots[index] for index in range(self.n)
        )
        signed = self.apply_signed(values, signed_weights)
        inverse_radius_squared = 1.0 / (radius_value * radius_value)
        self.last_apply_stats = {
            **self.last_apply_stats,
            "method": "root_of_unity_euclidean_metric_closure",
            "circle_radius": radius_value,
        }
        return tuple(
            _clean_scalar(
                -self.roots[index]
                * complex(signed[index])
                * inverse_radius_squared
            )
            for index in range(self.n)
        )

    def stats(self):
        result = {
            "nodes": self.n,
            "persistent_complex_entries": 4 * self.n,
            "persistent_storage": "O(n) roots plus analytic P three-jets",
            "shared_fft_plan_storage": "O(n)",
            "numerator_jet_generation": (
                "O(n log n) QJet mixed-radix/Bluestein FFT for every n"
            ),
            "fft_strategy": _fft_strategy(self.n),
            "jet_contraction": "O(n)",
            "total_apply_complexity": "O(n log n)",
            "adaptive_rank": 0,
            "stored_dense_distance_matrix": False,
            "stored_dense_operator_matrix": False,
        }
        result.update(self.last_apply_stats)
        return result


__all__ = [
    "GeneratedDiscriminantThreeJetQJet",
    "RootOfUnityDiscriminantQJet",
    "root_of_unity_discriminant_p_jets",
    "root_of_unity_numerator_jets",
    "weighted_discriminant_s2_from_jets",
]
