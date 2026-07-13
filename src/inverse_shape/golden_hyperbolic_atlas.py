"""Sparse three-jet atlas in the golden hyperbolic coordinate."""

from inverse_shape.axisymmetric_scale_phase import (
    AxisymmetricScalePhaseQJet,
    MeridianThreeJetSpline,
)
from inverse_shape.golden_hyperbolic import (
    GOLDEN_MU,
    GOLDEN_TRANSLATION_LENGTH,
    golden_reparameterize_meridian_jet,
)
from inverse_shape.quadrature import _finite, _sqrt


_GAUSS_NODES = (
    -0.9602898564975363,
    -0.7966664774136267,
    -0.5255324099163290,
    -0.1834346424956498,
    0.1834346424956498,
    0.5255324099163290,
    0.7966664774136267,
    0.9602898564975363,
)

_GAUSS_WEIGHTS = (
    0.1012285362903763,
    0.2223810344533745,
    0.3137066458778873,
    0.3626837833783620,
    0.3626837833783620,
    0.3137066458778873,
    0.2223810344533745,
    0.1012285362903763,
)


def _hyperbolic_speed(radius_jet, height_jet):
    radius = radius_jet[0]
    if radius <= 0.0:
        raise ValueError("golden atlas requires positive radius")
    return _sqrt(radius_jet[1] ** 2 + height_jet[1] ** 2) / radius


def _integrate_hyperbolic_segment(spline, left, right):
    center = 0.5 * (left + right)
    half_width = 0.5 * (right - left)
    total = 0.0
    for node, weight in zip(_GAUSS_NODES, _GAUSS_WEIGHTS, strict=True):
        coordinate = center + half_width * node
        radius_jet, height_jet = spline.evaluate_jet(coordinate)
        total += weight * _hyperbolic_speed(radius_jet, height_jet)
    return half_width * total


class GoldenMeridianPatch:
    """One trace-three patch represented only by normalized three-jets."""

    def __init__(
        self,
        first,
        last,
        center_distance,
        hyperbolic_distances,
        radius_jets,
        height_jets,
    ):
        self.first = int(first)
        self.last = int(last)
        self.center_distance = float(center_distance)
        self.hyperbolic_distances = tuple(
            float(value) for value in hyperbolic_distances
        )
        normalized = tuple(
            golden_reparameterize_meridian_jet(
                radius_jet,
                height_jet,
                distance - self.center_distance,
            )
            for distance, radius_jet, height_jet in zip(
                self.hyperbolic_distances,
                radius_jets,
                height_jets,
                strict=True,
            )
        )
        self.coordinates = tuple(value.coordinate for value in normalized)
        if any(
            self.coordinates[index + 1] <= self.coordinates[index]
            for index in range(len(self.coordinates) - 1)
        ):
            raise RuntimeError("golden patch coordinate is not increasing")
        if max(abs(value) for value in self.coordinates) > 1.0 + 2.0e-12:
            raise RuntimeError("golden patch escaped the canonical interval")
        self.radius_jets = tuple(value.radius_jet for value in normalized)
        self.height_jets = tuple(value.height_jet for value in normalized)
        self.spline = MeridianThreeJetSpline(
            self.coordinates,
            self.radius_jets,
            self.height_jets,
        )

    @property
    def n_source_nodes(self):
        return self.last - self.first

    @property
    def hyperbolic_span(self):
        return self.hyperbolic_distances[-1] - self.hyperbolic_distances[0]

    def uniform_coordinates(self, count):
        size = int(count)
        if size < 2:
            raise ValueError("a golden patch requires at least two nodes")
        start = self.coordinates[0]
        stop = self.coordinates[-1]
        return tuple(
            start + (stop - start) * index / (size - 1)
            for index in range(size)
        )

    def generated_meridional_weights(self, coordinates):
        values = tuple(float(value) for value in coordinates)
        if len(values) < 2:
            raise ValueError("at least two generated coordinates are required")
        weights = []
        for index, coordinate in enumerate(values):
            radius_jet, height_jet = self.spline.evaluate_jet(coordinate)
            density = radius_jet[0] * _sqrt(
                radius_jet[1] ** 2 + height_jet[1] ** 2
            )
            if index == 0:
                cell = 0.5 * (values[1] - values[0])
            elif index == len(values) - 1:
                cell = 0.5 * (values[-1] - values[-2])
            else:
                cell = 0.5 * (values[index + 1] - values[index - 1])
            weights.append(density * cell)
        return tuple(weights)

    def qjet(self, n_scale, n_theta, **options):
        coordinates = self.uniform_coordinates(n_scale)
        qjet = AxisymmetricScalePhaseQJet(
            coordinates,
            self.spline,
            n_theta,
            self.generated_meridional_weights(coordinates),
            **options,
        )
        qjet.golden_patch = self
        return qjet

    def stats(self):
        return {
            "first": self.first,
            "last": self.last,
            "source_nodes": self.n_source_nodes,
            "hyperbolic_span": self.hyperbolic_span,
            "golden_half_length": GOLDEN_MU,
            "coordinate_minimum": self.coordinates[0],
            "coordinate_maximum": self.coordinates[-1],
            "stored_geometry_scalars": 8 * self.n_source_nodes,
            "stored_dense_matrix": False,
            "quadratic_fallback": False,
        }


class GoldenHyperbolicJetAtlas:
    """Canonical trace-three partition of an arbitrary regular meridian."""

    def __init__(self, coordinates, radius_jets, height_jets):
        self.coordinates = tuple(float(value) for value in coordinates)
        self.radius_jets = tuple(tuple(float(item) for item in jet) for jet in radius_jets)
        self.height_jets = tuple(tuple(float(item) for item in jet) for jet in height_jets)
        count = len(self.coordinates)
        if count < 2:
            raise ValueError("a golden atlas requires at least two nodes")
        if len(self.radius_jets) != count or len(self.height_jets) != count:
            raise ValueError("one radius and height jet is required per node")
        if any(
            not _finite(value) for value in self.coordinates
        ) or any(
            self.coordinates[index + 1] <= self.coordinates[index]
            for index in range(count - 1)
        ):
            raise ValueError("atlas coordinates must be finite and increasing")
        self.raw_spline = MeridianThreeJetSpline(
            self.coordinates,
            self.radius_jets,
            self.height_jets,
        )
        distances = [0.0]
        for left, right in zip(
            self.coordinates[:-1],
            self.coordinates[1:],
            strict=True,
        ):
            increment = _integrate_hyperbolic_segment(
                self.raw_spline,
                left,
                right,
            )
            if increment <= 0.0 or not _finite(increment):
                raise ValueError("meridian has invalid hyperbolic length")
            if increment > GOLDEN_TRANSLATION_LENGTH * (1.0 + 2.0e-12):
                raise ValueError(
                    "one source interval exceeds the golden chart length; "
                    "subdivide its generating three-jet"
                )
            distances.append(distances[-1] + increment)
        self.hyperbolic_distances = tuple(distances)
        self.patches = self._compile_patches()

    def _compile_patches(self):
        patches = []
        start = 0
        count = len(self.coordinates)
        while start < count - 1:
            stop = start + 1
            while (
                stop + 1 < count
                and self.hyperbolic_distances[stop + 1]
                - self.hyperbolic_distances[start]
                <= GOLDEN_TRANSLATION_LENGTH * (1.0 + 2.0e-14)
            ):
                stop += 1
            first_distance = self.hyperbolic_distances[start]
            last_distance = self.hyperbolic_distances[stop]
            center = 0.5 * (first_distance + last_distance)
            patches.append(
                GoldenMeridianPatch(
                    start,
                    stop + 1,
                    center,
                    self.hyperbolic_distances[start : stop + 1],
                    self.radius_jets[start : stop + 1],
                    self.height_jets[start : stop + 1],
                )
            )
            start = stop
        return tuple(patches)

    def stats(self):
        return {
            "source_nodes": len(self.coordinates),
            "patch_count": len(self.patches),
            "total_hyperbolic_length": (
                self.hyperbolic_distances[-1]
                - self.hyperbolic_distances[0]
            ),
            "maximum_patch_span": max(
                patch.hyperbolic_span for patch in self.patches
            ),
            "golden_translation_length": GOLDEN_TRANSLATION_LENGTH,
            "stored_geometry_scalars": 8 * len(self.coordinates),
            "stored_dense_matrix": False,
            "quadratic_fallback": False,
            "construction_complexity": "O(n_scale)",
        }


__all__ = [
    "GoldenHyperbolicJetAtlas",
    "GoldenMeridianPatch",
]
