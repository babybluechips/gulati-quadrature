import ast

import inverse_shape.discriminant_three_jet as discriminant_module
from inverse_shape.discriminant_three_jet import (
    GeneratedDiscriminantThreeJetQJet,
    RootOfUnityDiscriminantQJet,
    weighted_discriminant_s2_from_jets,
)
from inverse_shape.quadrature import TAU, _cos, _sin


def _poly_multiply(left, right):
    output = [0.0 + 0.0j for _ in range(len(left) + len(right) - 1)]
    for left_index, left_value in enumerate(left):
        for right_index, right_value in enumerate(right):
            output[left_index + right_index] += left_value * right_value
    return tuple(output)


def _poly_add(left, right):
    length = max(len(left), len(right))
    return tuple(
        (left[index] if index < len(left) else 0.0)
        + (right[index] if index < len(right) else 0.0)
        for index in range(length)
    )


def _poly_derivative(coefficients, order=1):
    result = tuple(coefficients)
    for _ in range(order):
        result = tuple(
            degree * result[degree] for degree in range(1, len(result))
        )
    return result


def _poly_evaluate(coefficients, value):
    output = 0.0 + 0.0j
    for coefficient in reversed(coefficients):
        output = output * value + coefficient
    return output


def _polynomial_from_roots(roots):
    output = (1.0 + 0.0j,)
    for root in roots:
        output = _poly_multiply(output, (-root, 1.0 + 0.0j))
    return output


def _numerator_from_residues(roots, residues):
    output = (0.0 + 0.0j,)
    for target, residue in enumerate(residues):
        term = (complex(residue),)
        for index, root in enumerate(roots):
            if index != target:
                term = _poly_multiply(term, (-root, 1.0 + 0.0j))
        output = _poly_add(output, term)
    return output


def _jets(roots, residues):
    polynomial = _polynomial_from_roots(roots)
    numerator = _numerator_from_residues(roots, residues)
    p1 = tuple(
        _poly_evaluate(_poly_derivative(polynomial, 1), root)
        for root in roots
    )
    p2 = tuple(
        _poly_evaluate(_poly_derivative(polynomial, 2), root)
        for root in roots
    )
    p3 = tuple(
        _poly_evaluate(_poly_derivative(polynomial, 3), root)
        for root in roots
    )
    a1 = tuple(
        _poly_evaluate(_poly_derivative(numerator, 1), root)
        for root in roots
    )
    a2 = tuple(
        _poly_evaluate(_poly_derivative(numerator, 2), root)
        for root in roots
    )
    return (p1, p2, p3), (a1, a2)


def _direct_s2(roots, residues):
    return tuple(
        sum(
            residues[right] / (roots[left] - roots[right]) ** 2
            for right in range(len(roots))
            if right != left
        )
        for left in range(len(roots))
    )


def _relative_error(reference, candidate):
    numerator = sum(
        abs(complex(left) - complex(right)) ** 2
        for left, right in zip(reference, candidate, strict=True)
    )
    denominator = sum(abs(complex(value)) ** 2 for value in reference)
    return (numerator / max(denominator, 1.0e-300)) ** 0.5


ROOTS = (
    -1.7 + 0.2j,
    -0.8 - 0.9j,
    -0.1 + 1.1j,
    0.6 - 0.4j,
    1.3 + 0.8j,
    2.1 - 0.2j,
)


def test_discriminant_kernel_has_no_external_numerical_dependency() -> None:
    with open(discriminant_module.__file__, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert imported == ["inverse_shape.quadrature"]


def test_weighted_three_jet_identity_matches_direct_inverse_squares() -> None:
    residues = tuple(
        0.7 + 0.11 * index + 0.05j * (index - 2)
        for index in range(len(ROOTS))
    )
    p_jet, numerator_jet = _jets(ROOTS, residues)
    candidate = weighted_discriminant_s2_from_jets(
        residues,
        p_jet[0],
        p_jet[1],
        p_jet[2],
        numerator_jet[0],
        numerator_jet[1],
    )
    assert _relative_error(_direct_s2(ROOTS, residues), candidate) < 2.0e-13


def test_generated_weight_and_field_jets_give_full_graph_action() -> None:
    center = sum(ROOTS) / len(ROOTS)
    scale = max(abs(root - center) for root in ROOTS)
    normalized = tuple((root - center) / scale for root in ROOTS)
    weights = tuple(0.8 + 0.07 * index for index in range(len(ROOTS)))
    values = tuple(
        root.real - 0.3 * root.imag + 0.1j * index
        for index, root in enumerate(ROOTS)
    )
    weighted_field = tuple(
        weights[index] * values[index] for index in range(len(ROOTS))
    )
    p_jet, weight_jet = _jets(normalized, weights)
    _unused, field_jet = _jets(normalized, weighted_field)
    qjet = GeneratedDiscriminantThreeJetQJet(*p_jet, scale=scale)
    candidate = qjet.apply(values, weights, weight_jet, field_jet)
    weight_s2 = _direct_s2(ROOTS, weights)
    field_s2 = _direct_s2(ROOTS, weighted_field)
    reference = tuple(
        values[index] * weight_s2[index] - field_s2[index]
        for index in range(len(ROOTS))
    )
    assert _relative_error(reference, candidate) < 3.0e-13
    stats = qjet.stats()
    assert stats["apply_complexity"] == "O(n) after numerator-jet generation"
    assert stats["adaptive_rank"] == 0
    assert stats["stored_dense_operator_matrix"] is False


def test_constant_graph_channel_cancels_exactly_from_identical_jets() -> None:
    weights = tuple(1.0 + 0.05 * index for index in range(len(ROOTS)))
    p_jet, weight_jet = _jets(ROOTS, weights)
    qjet = GeneratedDiscriminantThreeJetQJet(*p_jet)
    result = qjet.apply(
        (1.0,) * len(ROOTS),
        weights,
        weight_jet,
        weight_jet,
    )
    assert max(abs(complex(value)) for value in result) == 0.0


def test_common_circle_closes_euclidean_chord_kernel_holomorphically() -> None:
    roots = (
        1.8 + 0.0j,
        0.9 + 1.5588457268119895j,
        -0.9 + 1.5588457268119895j,
        -1.8 + 0.0j,
        -0.9 - 1.5588457268119895j,
        0.9 - 1.5588457268119895j,
    )
    radius_squared = 1.8**2
    weights = tuple(0.7 + 0.08 * index for index in range(len(roots)))
    values = tuple(
        root.real + 0.2 * root.imag + 0.05j * index
        for index, root in enumerate(roots)
    )
    signed_weights = tuple(
        weights[index] * roots[index] / radius_squared
        for index in range(len(roots))
    )
    signed_field = tuple(
        signed_weights[index] * values[index]
        for index in range(len(roots))
    )
    p_jet, weight_jet = _jets(roots, signed_weights)
    _unused, field_jet = _jets(roots, signed_field)
    qjet = GeneratedDiscriminantThreeJetQJet(*p_jet)
    signed = qjet.apply(values, signed_weights, weight_jet, field_jet)
    candidate = tuple(
        -roots[index] * complex(signed[index]) for index in range(len(roots))
    )
    reference = tuple(
        sum(
            weights[right]
            * (values[left] - values[right])
            / abs(roots[left] - roots[right]) ** 2
            for right in range(len(roots))
            if right != left
        )
        for left in range(len(roots))
    )
    assert _relative_error(reference, candidate) < 4.0e-13


def test_root_of_unity_generator_is_rank_free_and_matches_direct_circle_q() -> None:
    count = 256
    radius = 1.7
    qjet = RootOfUnityDiscriminantQJet(count)
    values = tuple(
        _cos(3.0 * TAU * index / count)
        + 0.2 * _sin(7.0 * TAU * index / count)
        for index in range(count)
    )
    weights = tuple(
        1.0 + 0.1 * _cos(2.0 * TAU * index / count)
        for index in range(count)
    )
    candidate = qjet.apply_circle_euclidean(values, weights, radius)
    roots = tuple(radius * root for root in qjet.roots)
    reference = tuple(
        sum(
            weights[right]
            * (values[left] - values[right])
            / abs(roots[left] - roots[right]) ** 2
            for right in range(count)
            if right != left
        )
        for left in range(count)
    )
    assert _relative_error(reference, candidate) < 4.0e-13
    constant = qjet.apply_circle_euclidean((1.0,) * count, weights, radius)
    assert max(abs(complex(value)) for value in constant) == 0.0
    stats = qjet.stats()
    assert stats["total_apply_complexity"] == "O(n log n)"
    assert stats["adaptive_rank"] == 0
    assert stats["stored_dense_operator_matrix"] is False


def test_non_radix_two_root_of_unity_generator_stays_subquadratic() -> None:
    for count, expected_strategy in ((150, "mixed_radix"), (151, "bluestein")):
        qjet = RootOfUnityDiscriminantQJet(count)
        values = tuple(
            _cos(3.0 * TAU * index / count)
            + 0.2 * _sin(7.0 * TAU * index / count)
            for index in range(count)
        )
        weights = tuple(
            1.0 + 0.1 * _cos(2.0 * TAU * index / count)
            for index in range(count)
        )
        candidate = qjet.apply_circle_euclidean(
            values,
            weights,
            radius=1.7,
        )
        roots = tuple(1.7 * root for root in qjet.roots)
        reference = tuple(
            sum(
                weights[right]
                * (values[left] - values[right])
                / abs(roots[left] - roots[right]) ** 2
                for right in range(count)
                if right != left
            )
            for left in range(count)
        )
        assert _relative_error(reference, candidate) < 2.0e-13
        stats = qjet.stats()
        assert "for every n" in stats["numerator_jet_generation"]
        assert stats["total_apply_complexity"] == "O(n log n)"
        assert stats["fft_strategy"] == expected_strategy
