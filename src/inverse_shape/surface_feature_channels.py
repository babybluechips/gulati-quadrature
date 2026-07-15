"""Mellin--Kondratiev edge and vertex channels on curved surface panels.

Sharp edges and vertices do not admit the smooth tangent-disk expansion used
inside a panel atlas.  Their local flux densities instead have power-log
forms determined by dihedral and spherical-link pencils.  This module
compiles those forms into four-coefficient channels.

For a channel basis ``b_j`` the production quadrature stores only

    M_j = integral_high_order b_j - sum_coarse w_i b_j(x_i),  j=0,...,3.

At application time the local amplitude is projected onto the four basis
functions and ``sum a_j M_j`` is repaid.  The high-order moments are evaluated
on the same cubic geometry, so the construction remains valid for nonuniform
panel nodes and does not pretend that a Duffy grid is an equispaced midpoint
grid.  The associated Mellin pencil and optional Hurwitz midpoint checksum are
reported separately.

All channel supports are local.  Compilation and application are linear in
the number of retained support nodes, with four coefficients per channel and
no dense global matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from inverse_shape.curved_panels import (
    CurvedPanelSurface,
    _gauss_legendre_unit,
)
from inverse_shape.polyhedral_kondratiev import (
    EdgeMellinPencil,
    SparseSphericalDirichletPencil,
    VertexMellinPencil,
    mellin_midpoint_defect,
)


Point3 = tuple[float, float, float]


def _sub(left: Point3, right: Point3) -> Point3:
    return (
        left[0] - right[0],
        left[1] - right[1],
        left[2] - right[2],
    )


def _add(left: Point3, right: Point3) -> Point3:
    return (
        left[0] + right[0],
        left[1] + right[1],
        left[2] + right[2],
    )


def _scale(factor: float, value: Point3) -> Point3:
    return (factor * value[0], factor * value[1], factor * value[2])


def _dot(left: Point3, right: Point3) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _cross(left: Point3, right: Point3) -> Point3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _norm(value: Point3) -> float:
    return math.sqrt(max(_dot(value, value), 0.0))


def _unit(value: Point3) -> Point3:
    length = _norm(value)
    if length <= 1.0e-30:
        raise ValueError("cannot normalize a zero feature direction")
    return _scale(1.0 / length, value)


def _distance_to_segment(point: Point3, first: Point3, second: Point3) -> tuple[float, float]:
    direction = _sub(second, first)
    denominator = _dot(direction, direction)
    if denominator <= 1.0e-30:
        return _norm(_sub(point, first)), 0.0
    phase = max(0.0, min(1.0, _dot(_sub(point, first), direction) / denominator))
    closest = _add(first, _scale(phase, direction))
    return _norm(_sub(point, closest)), phase


def _solve_complex(matrix: list[list[float]], right: list[complex]) -> list[complex]:
    size = len(right)
    rows = [
        [complex(value) for value in matrix[index]] + [complex(right[index])]
        for index in range(size)
    ]
    scale = max((max(abs(value) for value in row[:-1]) for row in rows), default=1.0)
    tolerance = 2.0e-13 * max(scale, 1.0)
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(rows[row][column]))
        if abs(rows[pivot][column]) <= tolerance:
            raise ValueError("Mellin channel amplitude fit is rank deficient")
        rows[column], rows[pivot] = rows[pivot], rows[column]
        divisor = rows[column][column]
        for offset in range(column, size + 1):
            rows[column][offset] /= divisor
        for row in range(size):
            if row == column:
                continue
            factor = rows[row][column]
            if factor == 0.0:
                continue
            for offset in range(column, size + 1):
                rows[row][offset] -= factor * rows[column][offset]
    return [rows[index][-1] for index in range(size)]


def _link_cycle(surface: CurvedPanelSurface, vertex: int) -> tuple[int, ...]:
    pairs = []
    for face in surface.topology.faces:
        if vertex not in face:
            continue
        offset = face.index(vertex)
        pairs.append((face[(offset + 1) % 3], face[(offset + 2) % 3]))
    adjacency: dict[int, set[int]] = {}
    for left, right in pairs:
        adjacency.setdefault(left, set()).add(right)
        adjacency.setdefault(right, set()).add(left)
    if not adjacency or any(len(row) != 2 for row in adjacency.values()):
        raise ValueError("vertex link is not a single manifold cycle")
    start = min(adjacency)
    cycle = [start]
    previous = None
    current = start
    while True:
        choices = sorted(value for value in adjacency[current] if value != previous)
        following = choices[0]
        if following == start:
            break
        if following in cycle:
            raise ValueError("vertex link closes before visiting every edge direction")
        cycle.append(following)
        previous, current = current, following
    if len(cycle) != len(adjacency):
        raise ValueError("vertex link contains multiple cycles")
    return tuple(cycle)


def _angle(first: Point3, second: Point3) -> float:
    return math.acos(max(-1.0, min(1.0, _dot(first, second))))


def _simplify_spherical_cycle(points: tuple[Point3, ...]) -> tuple[Point3, ...]:
    """Remove subdivision rays lying on one minor great-circle arc."""

    output = list(points)
    changed = True
    while changed and len(output) > 3:
        changed = False
        for index in range(len(output)):
            previous = output[(index - 1) % len(output)]
            current = output[index]
            following = output[(index + 1) % len(output)]
            split_arc = _angle(previous, current) + _angle(current, following)
            direct_arc = _angle(previous, following)
            if abs(split_arc - direct_arc) <= 2.0e-12:
                output.pop(index)
                changed = True
                break
    return tuple(output)


def _spherical_link_center(
    boundary: tuple[Point3, ...],
    orientation_hint: Point3,
) -> Point3:
    center = (0.0, 0.0, 0.0)
    for index, first in enumerate(boundary):
        second = boundary[(index + 1) % len(boundary)]
        normal = _cross(first, second)
        length = _norm(normal)
        if length <= 1.0e-14:
            continue
        center = _add(
            center,
            _scale(_angle(first, second) / length, normal),
        )
    if _dot(center, orientation_hint) < 0.0:
        center = _scale(-1.0, center)
    return _unit(center)


def _refine_spherical_link(
    points: list[Point3],
    triangles: list[tuple[int, int, int]],
    boundary_edges: set[tuple[int, int]],
    refinements: int,
) -> tuple[list[Point3], list[tuple[int, int, int]], set[int]]:
    boundary_nodes = {index for edge in boundary_edges for index in edge}
    for _level in range(refinements):
        cache: dict[tuple[int, int], int] = {}
        next_boundary_edges: set[tuple[int, int]] = set()

        def midpoint(left: int, right: int) -> int:
            edge = (min(left, right), max(left, right))
            if edge not in cache:
                cache[edge] = len(points)
                points.append(_unit(_add(points[left], points[right])))
                if edge in boundary_edges:
                    boundary_nodes.add(cache[edge])
                    next_boundary_edges.add((min(left, cache[edge]), max(left, cache[edge])))
                    next_boundary_edges.add((min(right, cache[edge]), max(right, cache[edge])))
            return cache[edge]

        refined = []
        for first, second, third in triangles:
            first_second = midpoint(first, second)
            second_third = midpoint(second, third)
            third_first = midpoint(third, first)
            refined.extend(
                (
                    (first, first_second, third_first),
                    (first_second, second, second_third),
                    (third_first, second_third, third),
                    (first_second, second_third, third_first),
                )
            )
        triangles = refined
        boundary_edges = next_boundary_edges
    return points, triangles, boundary_nodes


def vertex_mellin_pencil(
    surface: CurvedPanelSurface,
    vertex: int,
    *,
    refinements: int = 1,
) -> VertexMellinPencil:
    """Solve a sparse spherical-link pencil at one repaired mesh vertex."""

    cycle = _link_cycle(surface, vertex)
    origin = surface.topology.vertices[vertex]
    boundary = _simplify_spherical_cycle(
        tuple(
            _unit(_sub(surface.topology.vertices[index], origin))
            for index in cycle
        )
    )
    inward = _scale(-1.0, surface.topology.vertex_normals[vertex])
    center = _spherical_link_center(boundary, inward)
    points = [center, *boundary]
    triangles = [
        (0, 1 + index, 1 + ((index + 1) % len(boundary)))
        for index in range(len(boundary))
    ]
    boundary_edges = {
        (
            min(1 + index, 1 + ((index + 1) % len(boundary))),
            max(1 + index, 1 + ((index + 1) % len(boundary))),
        )
        for index in range(len(boundary))
    }
    points, triangles, boundary_nodes = _refine_spherical_link(
        points,
        triangles,
        boundary_edges,
        int(refinements),
    )
    link = SparseSphericalDirichletPencil(points, triangles, boundary_nodes)
    return VertexMellinPencil.from_spherical_link(
        link,
        tolerance=2.0e-9,
        maximum_inverse_iterations=80,
        maximum_cg_iterations=400,
    )


@dataclass(frozen=True)
class FeatureChannelConfig:
    jet_order: int = 3
    reference_quadrature_order: int = 10
    vertex_link_refinements: int = 3
    minimum_vertex_feature_degree: int = 3

    def __post_init__(self) -> None:
        if self.jet_order < 0 or self.jet_order > 3:
            raise ValueError("feature jet_order must lie between zero and three")
        if self.reference_quadrature_order < 4 or self.reference_quadrature_order > 16:
            raise ValueError("reference_quadrature_order must lie between four and sixteen")
        if self.vertex_link_refinements < 0 or self.vertex_link_refinements > 3:
            raise ValueError("vertex_link_refinements must lie between zero and three")
        if self.minimum_vertex_feature_degree < 2:
            raise ValueError("minimum_vertex_feature_degree must be at least two")


@dataclass(frozen=True)
class PanelMellinChannel:
    label: str
    kind: str
    support: tuple[int, ...]
    basis_values: tuple[tuple[float, ...], ...]
    moment_defects: tuple[float, ...]
    exact_moments: tuple[float, ...]
    coarse_moments: tuple[float, ...]
    exponent: float
    pencil_certificate: dict[str, object]
    characteristic_radius: float

    @property
    def rank(self) -> int:
        return len(self.basis_values)

    def coefficients(
        self,
        weights: tuple[float, ...],
        values: tuple[complex, ...],
    ) -> tuple[complex, ...]:
        matrix = [
            [
                sum(
                    weights[index]
                    * self.basis_values[left][offset]
                    * self.basis_values[right][offset]
                    for offset, index in enumerate(self.support)
                )
                for right in range(self.rank)
            ]
            for left in range(self.rank)
        ]
        right = [
            sum(
                weights[index] * basis[offset] * values[index]
                for offset, index in enumerate(self.support)
            )
            for basis in self.basis_values
        ]
        return tuple(_solve_complex(matrix, right))

    def correction(
        self,
        weights: tuple[float, ...],
        values: tuple[complex, ...],
    ) -> complex:
        return sum(
            coefficient * defect
            for coefficient, defect in zip(
                self.coefficients(weights, values),
                self.moment_defects,
                strict=True,
            )
        )

    def hurwitz_midpoint_checksum(self, step: float) -> tuple[float, ...]:
        return tuple(
            mellin_midpoint_defect(self.exponent + degree, step)[0]
            for degree in range(self.rank)
        )


@dataclass(frozen=True)
class FeatureRepaymentEvaluation:
    value: complex
    correction: complex
    channel_corrections: tuple[tuple[str, complex], ...]
    stats: dict[str, object]


class MellinKondratievPanelRepayment3D:
    """Fixed-rank edge/vertex repayment for scalar panel quadrature."""

    def __init__(
        self,
        surface: CurvedPanelSurface,
        *,
        config: FeatureChannelConfig | None = None,
    ) -> None:
        self.surface = surface
        self.config = config or FeatureChannelConfig()
        channels = []
        for edge in surface.topology.sharp_edges:
            channels.append(self._edge_channel(edge))
        incident_sharp: dict[int, list[object]] = {}
        for edge in surface.topology.sharp_edges:
            for vertex in edge.vertices:
                incident_sharp.setdefault(vertex, []).append(edge)
        self.vertex_pencil_failures: list[tuple[int, str]] = []
        for vertex, edges in sorted(incident_sharp.items()):
            if len(edges) < self.config.minimum_vertex_feature_degree:
                continue
            try:
                pencil = vertex_mellin_pencil(
                    surface,
                    vertex,
                    refinements=self.config.vertex_link_refinements,
                )
            except (RuntimeError, ValueError, ZeroDivisionError) as error:
                # A link solve can fail on a severely warped local fan.  The
                # conservative edge-envelope exponent keeps the channel
                # explicit and records that the vertex pencil was not certified.
                exponent = min(edge.kondratiev_exponent for edge in edges)
                pencil = VertexMellinPencil.from_exponent(
                    exponent,
                    label="uncertified_edge_envelope",
                )
                self.vertex_pencil_failures.append((vertex, str(error)))
            channels.append(self._vertex_channel(vertex, pencil))
        self.channels = tuple(channels)

    def _high_order_moments(
        self,
        panel_indices: tuple[int, ...],
        evaluator,
    ) -> tuple[float, ...]:
        moments = [0.0 for _ in range(self.config.jet_order + 1)]
        rule = _gauss_legendre_unit(self.config.reference_quadrature_order)
        for panel_index in panel_indices:
            panel = self.surface.panels[panel_index]
            for first, first_weight in rule:
                for second, second_weight in rule:
                    u_value = first
                    v_value = (1.0 - first) * second
                    jet = panel.evaluate(u_value, v_value)
                    weight = (
                        first_weight
                        * second_weight
                        * (1.0 - first)
                        * jet.jacobian
                    )
                    basis = evaluator(jet.position)
                    for degree, value in enumerate(basis):
                        moments[degree] += weight * value
        return tuple(moments)

    def _edge_channel(self, edge) -> PanelMellinChannel:
        pencil = EdgeMellinPencil(edge.opening_angle)
        panels = tuple(edge.incident_faces)
        support = tuple(
            index
            for panel in panels
            for index in range(*self.surface.panel_node_ranges[panel])
        )
        first = self.surface.topology.vertices[edge.vertices[0]]
        second = self.surface.topology.vertices[edge.vertices[1]]
        distances = tuple(
            _distance_to_segment(self.surface.points[index], first, second)
            for index in support
        )
        radius = max(value[0] for value in distances)

        def evaluate(point: Point3) -> tuple[float, ...]:
            distance, phase = _distance_to_segment(point, first, second)
            normalized = max(distance / radius, 1.0e-15)
            window = 16.0 * phase * phase * (1.0 - phase) * (1.0 - phase)
            return tuple(
                window * normalized ** (pencil.exponent - 1.0 + degree)
                for degree in range(self.config.jet_order + 1)
            )

        basis = tuple(
            zip(
                *(evaluate(self.surface.points[index]) for index in support),
                strict=True,
            )
        )
        coarse = tuple(
            sum(
                self.surface.weights[index] * basis[degree][offset]
                for offset, index in enumerate(support)
            )
            for degree in range(len(basis))
        )
        exact = self._high_order_moments(panels, evaluate)
        return PanelMellinChannel(
            label=f"edge_{edge.index}",
            kind="edge",
            support=support,
            basis_values=tuple(tuple(row) for row in basis),
            moment_defects=tuple(
                left - right for left, right in zip(exact, coarse, strict=True)
            ),
            exact_moments=exact,
            coarse_moments=coarse,
            exponent=pencil.boundary_quadrature_exponent,
            pencil_certificate=pencil.certificate(),
            characteristic_radius=radius,
        )

    def _vertex_channel(
        self,
        vertex: int,
        pencil: VertexMellinPencil,
    ) -> PanelMellinChannel:
        panels = tuple(
            index
            for index, face in enumerate(self.surface.topology.faces)
            if vertex in face
        )
        support = tuple(
            index
            for panel in panels
            for index in range(*self.surface.panel_node_ranges[panel])
        )
        origin = self.surface.topology.vertices[vertex]
        radius = max(_norm(_sub(self.surface.points[index], origin)) for index in support)

        def evaluate(point: Point3) -> tuple[float, ...]:
            normalized = max(_norm(_sub(point, origin)) / radius, 1.0e-15)
            window = (1.0 - min(normalized, 1.0)) ** 2
            return tuple(
                window * normalized ** (pencil.exponent - 1.0 + degree)
                for degree in range(self.config.jet_order + 1)
            )

        basis = tuple(
            zip(
                *(evaluate(self.surface.points[index]) for index in support),
                strict=True,
            )
        )
        coarse = tuple(
            sum(
                self.surface.weights[index] * basis[degree][offset]
                for offset, index in enumerate(support)
            )
            for degree in range(len(basis))
        )
        exact = self._high_order_moments(panels, evaluate)
        return PanelMellinChannel(
            label=f"vertex_{vertex}",
            kind="vertex",
            support=support,
            basis_values=tuple(tuple(row) for row in basis),
            moment_defects=tuple(
                left - right for left, right in zip(exact, coarse, strict=True)
            ),
            exact_moments=exact,
            coarse_moments=coarse,
            exponent=pencil.boundary_quadrature_exponent,
            pencil_certificate=pencil.certificate(),
            characteristic_radius=radius,
        )

    def repay_integral(
        self,
        values: Iterable[complex],
        *,
        borrowed_value: complex | None = None,
        channel_labels: Iterable[str] | None = None,
    ) -> FeatureRepaymentEvaluation:
        row = tuple(complex(value) for value in values)
        if len(row) != len(self.surface.points):
            raise ValueError("feature repayment requires one value per panel node")
        selected = None if channel_labels is None else set(channel_labels)
        corrections = []
        for channel in self.channels:
            if selected is not None and channel.label not in selected:
                continue
            corrections.append(
                (channel.label, channel.correction(self.surface.weights, row))
            )
        correction = sum((value for _label, value in corrections), 0.0j)
        raw = (
            sum(weight * value for weight, value in zip(self.surface.weights, row, strict=True))
            if borrowed_value is None
            else complex(borrowed_value)
        )
        return FeatureRepaymentEvaluation(
            value=raw + correction,
            correction=correction,
            channel_corrections=tuple(corrections),
            stats=self.stats(),
        )

    def stats(self) -> dict[str, object]:
        return {
            "mellin_kondratiev_channels": len(self.channels),
            "edge_mellin_channels": sum(
                channel.kind == "edge" for channel in self.channels
            ),
            "vertex_mellin_channels": sum(
                channel.kind == "vertex" for channel in self.channels
            ),
            "vertex_link_pencil_failures": len(self.vertex_pencil_failures),
            "feature_channel_rank": self.config.jet_order + 1,
            "stored_feature_support_entries": sum(
                len(channel.support) for channel in self.channels
            ),
            "feature_reference_quadrature_order": self.config.reference_quadrature_order,
            "feature_apply_big_o": "O(total retained feature support)",
            "feature_storage_big_o": "O(total retained feature support)",
            "stored_dense_feature_matrix": False,
            "stored_pair_table": False,
        }


__all__ = [
    "FeatureChannelConfig",
    "FeatureRepaymentEvaluation",
    "MellinKondratievPanelRepayment3D",
    "PanelMellinChannel",
    "vertex_mellin_pencil",
]
