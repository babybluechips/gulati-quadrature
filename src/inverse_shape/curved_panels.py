"""Cubic curved triangle panels and sparse smooth-cell repayment.

The repaired triangle mesh supplies topology and feature locations.  This
module raises each triangle to a cubic point-normal (PN) patch, integrates it
with a tensor-product Duffy rule, and compiles a local surface differential
operator on the resulting quadrature nodes.  The construction retains only
panel control points, one tensor differentiation jet, and fixed-ring moments.

For every node whose panel is smooth, the odd principal-value defect is repaid
from the first tangent moment.  The stable even singular-cell term is

    -a/4 Delta_Gamma

where ``a`` is the equal-area geodesic-cell radius.  The Laplace--Beltrami
operator is a measure-symmetric weak Duffy spectral jet.  No global dense
differentiation or boundary-operator matrix is formed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

from inverse_shape.surface_manifold import RepairedTriangleMesh


Point3 = tuple[float, float, float]
ControlKey = tuple[int, int, int]


def _add(left: Point3, right: Point3) -> Point3:
    return (
        left[0] + right[0],
        left[1] + right[1],
        left[2] + right[2],
    )


def _sub(left: Point3, right: Point3) -> Point3:
    return (
        left[0] - right[0],
        left[1] - right[1],
        left[2] - right[2],
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


def _unit(value: Point3, fallback: Point3 = (1.0, 0.0, 0.0)) -> Point3:
    length = _norm(value)
    if length <= 1.0e-30:
        value = fallback
        length = _norm(value)
    if length <= 1.0e-30:
        return (1.0, 0.0, 0.0)
    return _scale(1.0 / length, value)


def _sum_points(values: Iterable[Point3]) -> Point3:
    output = (0.0, 0.0, 0.0)
    for value in values:
        output = _add(output, value)
    return output


def _multinomial(degree: int, key: ControlKey) -> float:
    value = math.factorial(degree)
    for order in key:
        value //= math.factorial(order)
    return float(value)


def _bernstein(degree: int, key: ControlKey, barycentric: Point3) -> float:
    return (
        _multinomial(degree, key)
        * barycentric[0] ** key[0]
        * barycentric[1] ** key[1]
        * barycentric[2] ** key[2]
    )


def _degree_keys(degree: int) -> tuple[ControlKey, ...]:
    return tuple(
        (first, second, degree - first - second)
        for first in range(degree, -1, -1)
        for second in range(degree - first, -1, -1)
    )


def _gauss_legendre_unit(order: int) -> tuple[tuple[float, float], ...]:
    """Return the ``order`` Gauss--Legendre nodes and weights on ``[0,1]``."""

    n = int(order)
    if n < 2:
        raise ValueError("curved-panel quadrature order must be at least two")
    roots = [0.0 for _ in range(n)]
    weights = [0.0 for _ in range(n)]
    half = (n + 1) // 2
    for index in range(half):
        root = math.cos(math.pi * (index + 0.75) / (n + 0.5))
        derivative = 0.0
        for _iteration in range(64):
            previous = 1.0
            current = root
            if n == 0:
                polynomial = previous
            elif n == 1:
                polynomial = current
            else:
                for degree in range(2, n + 1):
                    following = (
                        (2 * degree - 1) * root * current
                        - (degree - 1) * previous
                    ) / degree
                    previous, current = current, following
                polynomial = current
            derivative = n * (root * polynomial - previous) / (root * root - 1.0)
            update = polynomial / derivative
            root -= update
            if abs(update) <= 4.0e-16:
                break
        weight = 2.0 / ((1.0 - root * root) * derivative * derivative)
        roots[index] = -root
        roots[n - 1 - index] = root
        weights[index] = weight
        weights[n - 1 - index] = weight
    return tuple(
        (0.5 * (root + 1.0), 0.5 * weight)
        for root, weight in zip(roots, weights, strict=True)
    )


def _pn_edge_control(first: Point3, second: Point3, normal: Point3) -> Point3:
    projection = _dot(_sub(second, first), normal)
    return _scale(
        1.0 / 3.0,
        _sub(_add(_scale(2.0, first), second), _scale(projection, normal)),
    )


def _linear_edge_control(first: Point3, second: Point3) -> Point3:
    return _scale(1.0 / 3.0, _add(_scale(2.0, first), second))


@dataclass(frozen=True)
class CurvedPanelJet:
    position: Point3
    derivative_u: Point3
    derivative_v: Point3
    derivative_uu: Point3
    derivative_uv: Point3
    derivative_vv: Point3
    normal: Point3
    jacobian: float
    gaussian_curvature: float


@dataclass(frozen=True)
class CubicPNTriangle:
    """One cubic point-normal triangle with exact first and second jets."""

    face_index: int
    vertex_indices: tuple[int, int, int]
    controls: tuple[tuple[ControlKey, Point3], ...]
    smooth: bool
    sharp_edge_count: int

    @property
    def control_map(self) -> dict[ControlKey, Point3]:
        return dict(self.controls)

    def evaluate(self, u_value: float, v_value: float) -> CurvedPanelJet:
        u = float(u_value)
        v = float(v_value)
        if u < -1.0e-14 or v < -1.0e-14 or u + v > 1.0 + 1.0e-14:
            raise ValueError("triangle coordinates must lie in the reference simplex")
        barycentric = (1.0 - u - v, u, v)
        control = self.control_map
        position = _sum_points(
            _scale(_bernstein(3, key, barycentric), value)
            for key, value in self.controls
        )
        derivative_u = (0.0, 0.0, 0.0)
        derivative_v = (0.0, 0.0, 0.0)
        for i, j, k in _degree_keys(2):
            basis = 3.0 * _bernstein(2, (i, j, k), barycentric)
            derivative_u = _add(
                derivative_u,
                _scale(
                    basis,
                    _sub(control[(i, j + 1, k)], control[(i + 1, j, k)]),
                ),
            )
            derivative_v = _add(
                derivative_v,
                _scale(
                    basis,
                    _sub(control[(i, j, k + 1)], control[(i + 1, j, k)]),
                ),
            )
        derivative_uu = (0.0, 0.0, 0.0)
        derivative_uv = (0.0, 0.0, 0.0)
        derivative_vv = (0.0, 0.0, 0.0)
        for i, j, k in _degree_keys(1):
            basis = 6.0 * _bernstein(1, (i, j, k), barycentric)
            derivative_uu = _add(
                derivative_uu,
                _scale(
                    basis,
                    _add(
                        _sub(control[(i, j + 2, k)], _scale(2.0, control[(i + 1, j + 1, k)])),
                        control[(i + 2, j, k)],
                    ),
                ),
            )
            derivative_uv = _add(
                derivative_uv,
                _scale(
                    basis,
                    _add(
                        _sub(
                            control[(i, j + 1, k + 1)],
                            control[(i + 1, j + 1, k)],
                        ),
                        _sub(
                            control[(i + 2, j, k)],
                            control[(i + 1, j, k + 1)],
                        ),
                    ),
                ),
            )
            derivative_vv = _add(
                derivative_vv,
                _scale(
                    basis,
                    _add(
                        _sub(control[(i, j, k + 2)], _scale(2.0, control[(i + 1, j, k + 1)])),
                        control[(i + 2, j, k)],
                    ),
                ),
            )
        cross = _cross(derivative_u, derivative_v)
        jacobian = _norm(cross)
        if jacobian <= 1.0e-30:
            raise ValueError("a cubic panel has a singular geometric Jacobian")
        normal = _scale(1.0 / jacobian, cross)
        first_form_uu = _dot(derivative_u, derivative_u)
        first_form_uv = _dot(derivative_u, derivative_v)
        first_form_vv = _dot(derivative_v, derivative_v)
        denominator = first_form_uu * first_form_vv - first_form_uv**2
        second_uu = _dot(normal, derivative_uu)
        second_uv = _dot(normal, derivative_uv)
        second_vv = _dot(normal, derivative_vv)
        gaussian = (
            (second_uu * second_vv - second_uv**2) / denominator
            if denominator > 1.0e-30
            else 0.0
        )
        return CurvedPanelJet(
            position=position,
            derivative_u=derivative_u,
            derivative_v=derivative_v,
            derivative_uu=derivative_uu,
            derivative_uv=derivative_uv,
            derivative_vv=derivative_vv,
            normal=normal,
            jacobian=jacobian,
            gaussian_curvature=gaussian,
        )


@dataclass(frozen=True)
class RadialQuadricTriangle:
    """Exact radial chart of an origin-centered ellipsoid.

    If ``A=diag(a,b,c)`` and the three vertices lie on ``A S^2``, the map is

        X(u,v) = A normalize(A^{-1} X_linear(u,v)).

    Adjacent panels agree along their full common edge.  The first and second
    jets below are analytic, so this chart is useful as an independent
    geometry oracle in refinement studies.
    """

    face_index: int
    vertex_indices: tuple[int, int, int]
    vertices: tuple[Point3, Point3, Point3]
    axes: Point3
    smooth: bool
    sharp_edge_count: int

    def _physical(self, value: Point3) -> Point3:
        return tuple(self.axes[axis] * value[axis] for axis in range(3))

    def _normalized(self, value: Point3) -> Point3:
        return tuple(value[axis] / self.axes[axis] for axis in range(3))

    @staticmethod
    def _normalization_first(value: Point3, direction: Point3) -> Point3:
        radius = _norm(value)
        inner = _dot(value, direction)
        return _sub(
            _scale(1.0 / radius, direction),
            _scale(inner / radius**3, value),
        )

    @staticmethod
    def _normalization_second(
        value: Point3,
        first: Point3,
        second: Point3,
    ) -> Point3:
        radius = _norm(value)
        first_inner = _dot(value, first)
        second_inner = _dot(value, second)
        return _add(
            _scale(
                -1.0 / radius**3,
                _add(
                    _add(
                        _scale(second_inner, first),
                        _scale(first_inner, second),
                    ),
                    _scale(_dot(first, second), value),
                ),
            ),
            _scale(
                3.0 * first_inner * second_inner / radius**5,
                value,
            ),
        )

    def evaluate(self, u_value: float, v_value: float) -> CurvedPanelJet:
        u = float(u_value)
        v = float(v_value)
        barycentric = (1.0 - u - v, u, v)
        linear = _sum_points(
            _scale(coefficient, point)
            for coefficient, point in zip(barycentric, self.vertices, strict=True)
        )
        derivative_u_linear = _sub(self.vertices[1], self.vertices[0])
        derivative_v_linear = _sub(self.vertices[2], self.vertices[0])
        normalized = self._normalized(linear)
        normalized_u = self._normalized(derivative_u_linear)
        normalized_v = self._normalized(derivative_v_linear)
        unit = _unit(normalized)
        first_u = self._normalization_first(normalized, normalized_u)
        first_v = self._normalization_first(normalized, normalized_v)
        second_uu = self._normalization_second(
            normalized,
            normalized_u,
            normalized_u,
        )
        second_uv = self._normalization_second(
            normalized,
            normalized_u,
            normalized_v,
        )
        second_vv = self._normalization_second(
            normalized,
            normalized_v,
            normalized_v,
        )
        position = self._physical(unit)
        derivative_u = self._physical(first_u)
        derivative_v = self._physical(first_v)
        derivative_uu = self._physical(second_uu)
        derivative_uv = self._physical(second_uv)
        derivative_vv = self._physical(second_vv)
        cross = _cross(derivative_u, derivative_v)
        jacobian = _norm(cross)
        if jacobian <= 1.0e-30:
            raise ValueError("radial quadric chart has a singular Jacobian")
        normal = _scale(1.0 / jacobian, cross)
        first_form_uu = _dot(derivative_u, derivative_u)
        first_form_uv = _dot(derivative_u, derivative_v)
        first_form_vv = _dot(derivative_v, derivative_v)
        denominator = first_form_uu * first_form_vv - first_form_uv**2
        second_form_uu = _dot(normal, derivative_uu)
        second_form_uv = _dot(normal, derivative_uv)
        second_form_vv = _dot(normal, derivative_vv)
        gaussian = (
            (second_form_uu * second_form_vv - second_form_uv**2) / denominator
            if denominator > 1.0e-30
            else 0.0
        )
        return CurvedPanelJet(
            position=position,
            derivative_u=derivative_u,
            derivative_v=derivative_v,
            derivative_uu=derivative_uu,
            derivative_uv=derivative_uv,
            derivative_vv=derivative_vv,
            normal=normal,
            jacobian=jacobian,
            gaussian_curvature=gaussian,
        )


@dataclass(frozen=True)
class CurvedPanelConfig:
    """Panel geometry and quadrature settings."""

    quadrature_order: int = 4
    curvature_blend: float = 1.0

    def __post_init__(self) -> None:
        if self.quadrature_order < 2 or self.quadrature_order > 12:
            raise ValueError("quadrature_order must lie between two and twelve")
        if not 0.0 <= self.curvature_blend <= 1.0:
            raise ValueError("curvature_blend must lie between zero and one")


@dataclass(frozen=True)
class CurvedPanelSurface:
    """Quadrature-node surface generated by a repaired manifold."""

    topology: RepairedTriangleMesh
    panels: tuple[CubicPNTriangle | RadialQuadricTriangle, ...]
    points: tuple[Point3, ...]
    weights: tuple[float, ...]
    normals: tuple[Point3, ...]
    tangent_u: tuple[Point3, ...]
    gaussian_curvature: tuple[float, ...]
    panel_ids: tuple[int, ...]
    reference_coordinates: tuple[tuple[float, float], ...]
    panel_node_ranges: tuple[tuple[int, int], ...]
    panel_neighbors: tuple[tuple[int, ...], ...]
    smooth_nodes: tuple[bool, ...]
    maximum_panel_seam_gap: float
    quadrature_order: int

    @property
    def stats(self) -> dict[str, object]:
        curved_area = sum(self.weights)
        flat_area = sum(self.topology.face_areas)
        return {
            "curved_panel_count": len(self.panels),
            "curved_quadrature_nodes": len(self.points),
            "quadrature_order": self.quadrature_order,
            "nodes_per_panel": self.quadrature_order**2,
            "smooth_panels": sum(panel.smooth for panel in self.panels),
            "feature_panels": sum(not panel.smooth for panel in self.panels),
            "smooth_quadrature_nodes": sum(self.smooth_nodes),
            "curved_area": curved_area,
            "flat_mesh_area": flat_area,
            "curved_to_flat_area_ratio": curved_area / flat_area,
            "minimum_panel_jacobian_weight": min(self.weights),
            "maximum_absolute_panel_curvature": max(
                abs(value) for value in self.gaussian_curvature
            ),
            "panel_geometry_order": (
                3 if isinstance(self.panels[0], CubicPNTriangle) else "exact"
            ),
            "panel_geometry_kind": type(self.panels[0]).__name__,
            "maximum_panel_seam_gap": self.maximum_panel_seam_gap,
            "curved_atlas_watertight": self.maximum_panel_seam_gap <= 2.0e-12,
            "panel_storage_big_o": "O(number of panels + quadrature nodes)",
            "stored_dense_matrix": False,
            "stored_pair_table": False,
        }


def _build_panel(
    topology: RepairedTriangleMesh,
    face_index: int,
    curvature_blend: float,
    sharp_lookup: set[tuple[int, int]],
) -> CubicPNTriangle:
    face = topology.faces[face_index]
    points = tuple(topology.vertices[index] for index in face)
    normals = topology.corner_normals[face_index]
    if curvature_blend < 1.0:
        flat = topology.face_normals[face_index]
        normals = tuple(
            _unit(
                _add(
                    _scale(curvature_blend, normal),
                    _scale(1.0 - curvature_blend, flat),
                ),
                flat,
            )
            for normal in normals
        )
    def edge_control(
        first_index: int,
        second_index: int,
        normal: Point3,
    ) -> Point3:
        edge = (
            min(face[first_index], face[second_index]),
            max(face[first_index], face[second_index]),
        )
        if edge in sharp_lookup:
            return _linear_edge_control(
                points[first_index],
                points[second_index],
            )
        return _pn_edge_control(
            points[first_index],
            points[second_index],
            normal,
        )

    controls: dict[ControlKey, Point3] = {
        (3, 0, 0): points[0],
        (0, 3, 0): points[1],
        (0, 0, 3): points[2],
        (2, 1, 0): edge_control(0, 1, normals[0]),
        (1, 2, 0): edge_control(1, 0, normals[1]),
        (2, 0, 1): edge_control(0, 2, normals[0]),
        (1, 0, 2): edge_control(2, 0, normals[2]),
        (0, 2, 1): edge_control(1, 2, normals[1]),
        (0, 1, 2): edge_control(2, 1, normals[2]),
    }
    edge_average = _scale(
        1.0 / 6.0,
        _sum_points(
            controls[key]
            for key in (
                (2, 1, 0),
                (1, 2, 0),
                (2, 0, 1),
                (1, 0, 2),
                (0, 2, 1),
                (0, 1, 2),
            )
        ),
    )
    vertex_average = _scale(1.0 / 3.0, _sum_points(points))
    controls[(1, 1, 1)] = _add(
        edge_average,
        _scale(0.5, _sub(edge_average, vertex_average)),
    )
    panel_edges = tuple(
        (min(face[offset], face[(offset + 1) % 3]), max(face[offset], face[(offset + 1) % 3]))
        for offset in range(3)
    )
    sharp_count = sum(edge in sharp_lookup for edge in panel_edges)
    return CubicPNTriangle(
        face_index=face_index,
        vertex_indices=face,
        controls=tuple(sorted(controls.items())),
        smooth=sharp_count == 0,
        sharp_edge_count=sharp_count,
    )


def _panel_edge_point(
    panel: CubicPNTriangle | RadialQuadricTriangle,
    edge: tuple[int, int],
    phase: float,
) -> Point3:
    barycentric = [0.0, 0.0, 0.0]
    barycentric[panel.vertex_indices.index(edge[0])] = 1.0 - phase
    barycentric[panel.vertex_indices.index(edge[1])] = phase
    return panel.evaluate(barycentric[1], barycentric[2]).position


def _maximum_panel_seam_gap(
    topology: RepairedTriangleMesh,
    panels: tuple[CubicPNTriangle | RadialQuadricTriangle, ...],
) -> float:
    maximum = 0.0
    for edge in topology.feature_edges:
        left = panels[edge.incident_faces[0]]
        right = panels[edge.incident_faces[1]]
        for phase in (0.25, 0.5, 0.75):
            maximum = max(
                maximum,
                _norm(
                    _sub(
                        _panel_edge_point(left, edge.vertices, phase),
                        _panel_edge_point(right, edge.vertices, phase),
                    )
                ),
            )
    return maximum


def build_curved_panel_surface(
    topology: RepairedTriangleMesh,
    *,
    config: CurvedPanelConfig | None = None,
) -> CurvedPanelSurface:
    """Lift a repaired triangle manifold to cubic panels and Duffy nodes."""

    settings = config or CurvedPanelConfig()
    if not topology.certificate.production_ready:
        raise ValueError("curved panels require a certified closed manifold")
    sharp_lookup = {edge.vertices for edge in topology.sharp_edges}
    panels = tuple(
        _build_panel(topology, index, settings.curvature_blend, sharp_lookup)
        for index in range(len(topology.faces))
    )
    edge_rows: dict[tuple[int, int], list[int]] = {}
    for face_index, face in enumerate(topology.faces):
        for offset in range(3):
            edge = (
                min(face[offset], face[(offset + 1) % 3]),
                max(face[offset], face[(offset + 1) % 3]),
            )
            edge_rows.setdefault(edge, []).append(face_index)
    neighbors = [set() for _ in panels]
    for edge, rows in edge_rows.items():
        if len(rows) == 2 and edge not in sharp_lookup:
            neighbors[rows[0]].add(rows[1])
            neighbors[rows[1]].add(rows[0])

    rule = _gauss_legendre_unit(settings.quadrature_order)
    points = []
    weights = []
    normals = []
    tangents = []
    curvatures = []
    panel_ids = []
    coordinates = []
    ranges = []
    smooth_nodes = []
    for panel_index, panel in enumerate(panels):
        start = len(points)
        for first, first_weight in rule:
            for second, second_weight in rule:
                u_value = first
                v_value = (1.0 - first) * second
                jet = panel.evaluate(u_value, v_value)
                reference_jacobian = 1.0 - first
                weight = (
                    first_weight
                    * second_weight
                    * reference_jacobian
                    * jet.jacobian
                )
                if weight <= 0.0 or not math.isfinite(weight):
                    raise ValueError("curved panel produced a nonpositive weight")
                points.append(jet.position)
                weights.append(weight)
                normals.append(jet.normal)
                tangents.append(_unit(jet.derivative_u))
                curvatures.append(jet.gaussian_curvature)
                panel_ids.append(panel_index)
                coordinates.append((u_value, v_value))
                smooth_nodes.append(panel.smooth)
        ranges.append((start, len(points)))
    return CurvedPanelSurface(
        topology=topology,
        panels=panels,
        points=tuple(points),
        weights=tuple(weights),
        normals=tuple(normals),
        tangent_u=tuple(tangents),
        gaussian_curvature=tuple(curvatures),
        panel_ids=tuple(panel_ids),
        reference_coordinates=tuple(coordinates),
        panel_node_ranges=tuple(ranges),
        panel_neighbors=tuple(tuple(sorted(row)) for row in neighbors),
        smooth_nodes=tuple(smooth_nodes),
        maximum_panel_seam_gap=_maximum_panel_seam_gap(topology, panels),
        quadrature_order=settings.quadrature_order,
    )


def build_radial_quadric_panel_surface(
    topology: RepairedTriangleMesh,
    axes: Iterable[float],
    *,
    config: CurvedPanelConfig | None = None,
) -> CurvedPanelSurface:
    """Build exact radial sphere/ellipsoid panels for independent audits."""

    settings = config or CurvedPanelConfig()
    axis_values = tuple(float(value) for value in axes)
    if len(axis_values) != 3 or any(value <= 0.0 for value in axis_values):
        raise ValueError("radial quadric axes must contain three positive values")
    sharp_lookup = {edge.vertices for edge in topology.sharp_edges}
    panels = []
    for face_index, face in enumerate(topology.faces):
        panel_edges = tuple(
            (
                min(face[offset], face[(offset + 1) % 3]),
                max(face[offset], face[(offset + 1) % 3]),
            )
            for offset in range(3)
        )
        sharp_count = sum(edge in sharp_lookup for edge in panel_edges)
        panels.append(
            RadialQuadricTriangle(
                face_index=face_index,
                vertex_indices=face,
                vertices=tuple(topology.vertices[index] for index in face),
                axes=axis_values,
                smooth=sharp_count == 0,
                sharp_edge_count=sharp_count,
            )
        )
    edge_rows: dict[tuple[int, int], list[int]] = {}
    for face_index, face in enumerate(topology.faces):
        for offset in range(3):
            edge = (
                min(face[offset], face[(offset + 1) % 3]),
                max(face[offset], face[(offset + 1) % 3]),
            )
            edge_rows.setdefault(edge, []).append(face_index)
    neighbors = [set() for _ in panels]
    for edge, rows in edge_rows.items():
        if len(rows) == 2 and edge not in sharp_lookup:
            neighbors[rows[0]].add(rows[1])
            neighbors[rows[1]].add(rows[0])
    rule = _gauss_legendre_unit(settings.quadrature_order)
    points = []
    weights = []
    normals = []
    tangents = []
    curvatures = []
    panel_ids = []
    coordinates = []
    ranges = []
    smooth_nodes = []
    for panel_index, panel in enumerate(panels):
        start = len(points)
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
                points.append(jet.position)
                weights.append(weight)
                normals.append(jet.normal)
                tangents.append(_unit(jet.derivative_u))
                curvatures.append(jet.gaussian_curvature)
                panel_ids.append(panel_index)
                coordinates.append((u_value, v_value))
                smooth_nodes.append(panel.smooth)
        ranges.append((start, len(points)))
    return CurvedPanelSurface(
        topology=topology,
        panels=tuple(panels),
        points=tuple(points),
        weights=tuple(weights),
        normals=tuple(normals),
        tangent_u=tuple(tangents),
        gaussian_curvature=tuple(curvatures),
        panel_ids=tuple(panel_ids),
        reference_coordinates=tuple(coordinates),
        panel_node_ranges=tuple(ranges),
        panel_neighbors=tuple(tuple(sorted(row)) for row in neighbors),
        smooth_nodes=tuple(smooth_nodes),
        maximum_panel_seam_gap=_maximum_panel_seam_gap(topology, tuple(panels)),
        quadrature_order=settings.quadrature_order,
    )


@dataclass(frozen=True)
class PanelSingularRepaymentConfig:
    """Sparse odd-moment and stable even-cell repayment settings."""

    series_order: int = 1
    maximum_panel_rings: int = 2

    def __post_init__(self) -> None:
        if self.series_order not in (0, 1):
            raise ValueError("curved-panel series_order must be zero or one")
        if self.maximum_panel_rings < 1:
            raise ValueError("maximum_panel_rings must be positive")


def _barycentric_differentiation(nodes: tuple[float, ...]) -> tuple[tuple[float, ...], ...]:
    weights = []
    for column, point in enumerate(nodes):
        product = 1.0
        for other, value in enumerate(nodes):
            if other != column:
                product *= point - value
        weights.append(1.0 / product)
    rows = []
    for left, point in enumerate(nodes):
        row = []
        for right, value in enumerate(nodes):
            if left == right:
                row.append(0.0)
            else:
                row.append(
                    weights[right]
                    / weights[left]
                    / (point - value)
                )
        row[left] = -sum(row)
        rows.append(tuple(row))
    return tuple(rows)


class PanelSingularRepayment3D:
    """Weak-spectral singular-cell repayment on every smooth curved panel."""

    def __init__(
        self,
        surface: CurvedPanelSurface,
        *,
        config: PanelSingularRepaymentConfig | None = None,
    ) -> None:
        self.surface = surface
        self.config = config or PanelSingularRepaymentConfig()
        self.n = len(surface.points)
        self.q = surface.quadrature_order
        rule = _gauss_legendre_unit(self.q)
        self.differentiation = _barycentric_differentiation(
            tuple(node for node, _weight in rule)
        )
        weak_aa = [0.0 for _ in range(self.n)]
        weak_ab = [0.0 for _ in range(self.n)]
        weak_bb = [0.0 for _ in range(self.n)]
        gradient_a = [(0.0, 0.0, 0.0) for _ in range(self.n)]
        gradient_b = [(0.0, 0.0, 0.0) for _ in range(self.n)]
        for panel_index, panel in enumerate(surface.panels):
            first_node = surface.panel_node_ranges[panel_index][0]
            for first_index, (first, first_weight) in enumerate(rule):
                for second_index, (second, second_weight) in enumerate(rule):
                    node = first_node + first_index * self.q + second_index
                    u_value = first
                    v_value = (1.0 - first) * second
                    jet = panel.evaluate(u_value, v_value)
                    derivative_a = _sub(
                        jet.derivative_u,
                        _scale(second, jet.derivative_v),
                    )
                    derivative_b = _scale(1.0 - first, jet.derivative_v)
                    metric_aa = _dot(derivative_a, derivative_a)
                    metric_ab = _dot(derivative_a, derivative_b)
                    metric_bb = _dot(derivative_b, derivative_b)
                    determinant = metric_aa * metric_bb - metric_ab * metric_ab
                    if determinant <= 1.0e-30:
                        raise ValueError("Duffy panel metric is singular")
                    jacobian = math.sqrt(determinant)
                    reference_weight = first_weight * second_weight
                    factor = reference_weight * jacobian / determinant
                    weak_aa[node] = factor * metric_bb
                    weak_ab[node] = -factor * metric_ab
                    weak_bb[node] = factor * metric_aa
                    inverse_aa = metric_bb / determinant
                    inverse_ab = -metric_ab / determinant
                    inverse_bb = metric_aa / determinant
                    gradient_a[node] = _add(
                        _scale(inverse_aa, derivative_a),
                        _scale(inverse_ab, derivative_b),
                    )
                    gradient_b[node] = _add(
                        _scale(inverse_ab, derivative_a),
                        _scale(inverse_bb, derivative_b),
                    )
                    expected_mass = reference_weight * jacobian
                    if abs(expected_mass - surface.weights[node]) > (
                        2.0e-12 * max(expected_mass, surface.weights[node])
                    ):
                        raise RuntimeError("panel metric and quadrature mass disagree")
        self.weak_aa = tuple(weak_aa)
        self.weak_ab = tuple(weak_ab)
        self.weak_bb = tuple(weak_bb)
        self.gradient_a = tuple(gradient_a)
        self.gradient_b = tuple(gradient_b)
        radii = [0.0 for _ in range(self.n)]
        for panel, (first, last) in enumerate(surface.panel_node_ranges):
            panel_area = sum(surface.weights[first:last])
            curvature = sum(
                surface.weights[index] * surface.gaussian_curvature[index]
                for index in range(first, last)
            ) / panel_area
            base = math.sqrt(panel_area / (self.q * self.q * math.pi))
            scaled = max(-0.5, min(0.5, curvature * base * base))
            radius = base * (
                1.0 + scaled / 24.0 + 3.0 * scaled * scaled / 640.0
            )
            for index in range(first, last):
                radii[index] = radius if surface.panels[panel].smooth else 0.0
        self.geodesic_cell_radii = tuple(radii)
        odd_moments = []
        for target, panel in enumerate(surface.panel_ids):
            if not surface.smooth_nodes[target]:
                odd_moments.append((0.0, 0.0, 0.0))
                continue
            panels = {panel}
            frontier = {panel}
            for _ring in range(self.config.maximum_panel_rings):
                following = {
                    neighbor
                    for current in frontier
                    for neighbor in surface.panel_neighbors[current]
                    if neighbor not in panels
                }
                panels.update(following)
                frontier = following
            origin = surface.points[target]
            normal = surface.normals[target]
            moment = [0.0, 0.0, 0.0]
            for source_panel in panels:
                first, last = surface.panel_node_ranges[source_panel]
                for source in range(first, last):
                    if source == target:
                        continue
                    displacement = _sub(surface.points[source], origin)
                    distance = _norm(displacement)
                    if distance <= 1.0e-30:
                        continue
                    tangent = _sub(
                        displacement,
                        _scale(_dot(displacement, normal), normal),
                    )
                    factor = surface.weights[source] / distance**3
                    for axis in range(3):
                        moment[axis] += factor * tangent[axis]
            odd_moments.append(tuple(moment))
        self.odd_tangent_moments = tuple(odd_moments)

    def surface_gradient(
        self,
        values: Iterable[complex],
    ) -> tuple[tuple[complex, complex, complex], ...]:
        """Evaluate the strong tangent gradient from the panel QJet."""

        row = tuple(complex(value) for value in values)
        if len(row) != self.n:
            raise ValueError("panel surface-gradient input length does not match")
        output = [(0.0j, 0.0j, 0.0j) for _ in row]
        derivative = self.differentiation
        for first, _last in self.surface.panel_node_ranges:
            for left in range(self.q):
                for right in range(self.q):
                    derivative_a = sum(
                        derivative[left][column]
                        * row[first + column * self.q + right]
                        for column in range(self.q)
                    )
                    derivative_b = sum(
                        derivative[right][column]
                        * row[first + left * self.q + column]
                        for column in range(self.q)
                    )
                    node = first + left * self.q + right
                    output[node] = tuple(
                        derivative_a * self.gradient_a[node][axis]
                        + derivative_b * self.gradient_b[node][axis]
                        for axis in range(3)
                    )
        return tuple(output)

    def laplacian(self, values: Iterable[complex]) -> tuple[complex, ...]:
        row = tuple(complex(value) for value in values)
        if len(row) != self.n:
            raise ValueError("panel Laplace--Beltrami input length does not match")
        output = [0.0j for _ in row]
        derivative = self.differentiation
        for first, last in self.surface.panel_node_ranges:
            values_a = [[0.0j for _ in range(self.q)] for _ in range(self.q)]
            values_b = [[0.0j for _ in range(self.q)] for _ in range(self.q)]
            for left in range(self.q):
                for right in range(self.q):
                    values_a[left][right] = sum(
                        derivative[left][column]
                        * row[first + column * self.q + right]
                        for column in range(self.q)
                    )
                    values_b[left][right] = sum(
                        derivative[right][column]
                        * row[first + left * self.q + column]
                        for column in range(self.q)
                    )
            flux_a = [[0.0j for _ in range(self.q)] for _ in range(self.q)]
            flux_b = [[0.0j for _ in range(self.q)] for _ in range(self.q)]
            for left in range(self.q):
                for right in range(self.q):
                    node = first + left * self.q + right
                    flux_a[left][right] = (
                        self.weak_aa[node] * values_a[left][right]
                        + self.weak_ab[node] * values_b[left][right]
                    )
                    flux_b[left][right] = (
                        self.weak_ab[node] * values_a[left][right]
                        + self.weak_bb[node] * values_b[left][right]
                    )
            for left in range(self.q):
                for right in range(self.q):
                    node = first + left * self.q + right
                    stiffness = sum(
                        derivative[column][left] * flux_a[column][right]
                        for column in range(self.q)
                    ) + sum(
                        derivative[column][right] * flux_b[left][column]
                        for column in range(self.q)
                    )
                    output[node] = -stiffness / self.surface.weights[node]
        return tuple(output)

    def correction(
        self,
        values: Iterable[complex],
        *,
        order: int | None = None,
    ) -> tuple[complex, ...]:
        requested = self.config.series_order if order is None else int(order)
        if requested not in (0, 1):
            raise ValueError("curved-panel singular-cell order must be zero or one")
        row = tuple(complex(value) for value in values)
        if len(row) != self.n:
            raise ValueError("panel singular-cell input length does not match")
        output = [0.0j for _ in row]
        gradients = self.surface_gradient(row)
        for index, (gradient, moment) in enumerate(
            zip(gradients, self.odd_tangent_moments, strict=True)
        ):
            output[index] += sum(
                gradient[axis] * moment[axis] for axis in range(3)
            ) / (2.0 * math.pi)
        current = row
        denominators = (4.0,)
        for level in range(1, requested + 1):
            current = self.laplacian(current)
            power = 2 * level - 1
            for index in range(self.n):
                output[index] -= (
                    self.geodesic_cell_radii[index] ** power
                    * current[index]
                    / denominators[level - 1]
                )
        return tuple(output)

    def stats(self) -> dict[str, object]:
        return {
            "panel_singular_cells_repaid": sum(self.surface.smooth_nodes),
            "panel_feature_cells_deferred": self.n - sum(self.surface.smooth_nodes),
            "panel_singular_cell_order": self.config.series_order,
            "minimum_local_polynomial_degree": self.q - 1,
            "maximum_local_polynomial_degree": self.q - 1,
            "maximum_local_stencil_width": self.q * self.q,
            "stored_local_stencil_coefficients": (
                self.q * self.q + 3 * self.n
            ),
            "local_differential_form": "measure-symmetric weak Duffy spectral jet",
            "odd_tangent_moment_rings": self.config.maximum_panel_rings,
            "stored_odd_tangent_moment_coefficients": 3 * self.n,
            "odd_pv_channel": "annihilated discrete first tangent moment",
            "local_singular_apply_big_o": "O(N q) at panel order q",
            "local_singular_storage_big_o": "O(N+q^2)",
            "local_singular_series": "odd tangent moment - a/4 Delta",
            "stored_dense_panel_matrix": False,
            "stored_pair_table": False,
        }


__all__ = [
    "CubicPNTriangle",
    "CurvedPanelConfig",
    "CurvedPanelJet",
    "CurvedPanelSurface",
    "PanelSingularRepayment3D",
    "PanelSingularRepaymentConfig",
    "RadialQuadricTriangle",
    "build_curved_panel_surface",
    "build_radial_quadric_panel_surface",
]
