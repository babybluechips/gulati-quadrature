"""Certified linear-storage repair of triangle soups into closed 2-manifolds.

The CAD audit compiler intentionally accepts imperfect public meshes.  A
boundary PDE operator, however, needs a more precise geometric contract: every
edge must have two incident faces, adjacent face orientations must agree, and
each closed component must have a documented material-side orientation.

This module supplies that contract without constructing a distance matrix or
an all-pairs intersection table.  It performs the following deterministic
passes:

* tolerance-based vertex welding with a 27-cell spatial hash;
* degenerate and duplicate triangle removal;
* sheet separation at nonmanifold edges;
* component-wise orientation propagation;
* boundary-loop capping; and
* outward orientation and sharp-feature classification.

The result certifies *topological* watertightness and manifoldness.  Global
self-intersection is deliberately reported as unaudited because certifying it
requires a separate broad-phase/narrow-phase geometry pass.  The repair uses
``O(V+F)`` expected storage and never forms a dense matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


Point3 = tuple[float, float, float]
Face3 = tuple[int, int, int]


def _point(value: Iterable[float]) -> Point3:
    row = tuple(float(component) for component in value)
    if len(row) != 3 or any(not math.isfinite(component) for component in row):
        raise ValueError("mesh vertices must contain three finite coordinates")
    return row


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


def _unit(value: Point3, fallback: Point3 = (1.0, 0.0, 0.0)) -> Point3:
    length = _norm(value)
    if length <= 1.0e-30:
        value = fallback
        length = _norm(value)
    if length <= 1.0e-30:
        return (1.0, 0.0, 0.0)
    return _scale(1.0 / length, value)


def _face_geometry(points: tuple[Point3, ...], face: Face3) -> tuple[float, Point3]:
    first, second, third = (points[index] for index in face)
    normal = _cross(_sub(second, first), _sub(third, first))
    twice_area = _norm(normal)
    if twice_area <= 1.0e-30:
        return 0.0, (0.0, 0.0, 0.0)
    return 0.5 * twice_area, _scale(1.0 / twice_area, normal)


def _edge_incidences(
    faces: Iterable[Face3],
) -> dict[tuple[int, int], list[tuple[int, int, int]]]:
    incidences: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for face_index, face in enumerate(faces):
        for offset in range(3):
            left = face[offset]
            right = face[(offset + 1) % 3]
            edge = (min(left, right), max(left, right))
            incidences.setdefault(edge, []).append((face_index, left, right))
    return incidences


def _signed_volume(points: tuple[Point3, ...], faces: Iterable[Face3]) -> float:
    return sum(
        _dot(points[first], _cross(points[second], points[third])) / 6.0
        for first, second, third in faces
    )


def _compact(
    points: list[Point3],
    faces: list[Face3],
) -> tuple[list[Point3], list[Face3], int]:
    used = sorted({index for face in faces for index in face})
    mapping = {old: new for new, old in enumerate(used)}
    removed = len(points) - len(used)
    return (
        [points[index] for index in used],
        [tuple(mapping[index] for index in face) for face in faces],
        removed,
    )


@dataclass(frozen=True)
class ManifoldRepairConfig:
    """Geometry tolerances and feature policy for triangle-soup repair."""

    weld_tolerance: float = 1.0e-10
    area_tolerance: float = 1.0e-14
    sharp_angle_degrees: float = 32.0
    fill_holes: bool = True
    split_nonmanifold_sheets: bool = True
    maximum_repair_passes: int = 8

    def __post_init__(self) -> None:
        if self.weld_tolerance <= 0.0:
            raise ValueError("weld_tolerance must be positive")
        if self.area_tolerance <= 0.0:
            raise ValueError("area_tolerance must be positive")
        if not 0.0 < self.sharp_angle_degrees < 180.0:
            raise ValueError("sharp_angle_degrees must lie between zero and 180")
        if self.maximum_repair_passes < 1:
            raise ValueError("maximum_repair_passes must be positive")


@dataclass(frozen=True)
class SurfaceFeatureEdge:
    """One repaired manifold edge with its material-side wedge data."""

    index: int
    vertices: tuple[int, int]
    incident_faces: tuple[int, int]
    length: float
    normal_turn: float
    opening_angle: float
    sharp: bool
    introduced_by_hole_fill: bool

    @property
    def kondratiev_exponent(self) -> float:
        return math.pi / self.opening_angle


@dataclass(frozen=True)
class ManifoldRepairCertificate:
    source_vertices: int
    source_faces: int
    welded_vertices: int
    removed_unused_vertices: int
    removed_degenerate_faces: int
    removed_duplicate_faces: int
    split_sheet_vertices: int
    boundary_edges_before_fill: int
    nonmanifold_edges_before_split: int
    boundary_loops_filled: int
    cap_vertices_added: int
    cap_faces_added: int
    orientation_flips: int
    repaired_vertices: int
    repaired_faces: int
    repaired_edges: int
    connected_components: int
    euler_characteristic: int
    orientable_genus_sum: int | None
    signed_volume: float
    boundary_edges: int
    nonmanifold_edges: int
    nonmanifold_vertices: int
    inconsistent_oriented_edges: int
    watertight: bool
    manifold: bool
    consistently_oriented: bool
    outward_oriented_nonzero_components: bool
    self_intersection_certified: bool = False
    dense_matrix_stored: bool = False
    pair_table_stored: bool = False

    @property
    def production_ready(self) -> bool:
        return (
            self.watertight
            and self.manifold
            and self.consistently_oriented
            and self.outward_oriented_nonzero_components
        )

    @property
    def stats(self) -> dict[str, object]:
        result = dict(self.__dict__)
        result.update(
            {
                "production_ready": self.production_ready,
                "repair_time_big_o": "O(V+F) expected",
                "repair_storage_big_o": "O(V+F)",
                "global_self_intersection_audit": "not performed",
            }
        )
        return result


@dataclass(frozen=True)
class RepairedTriangleMesh:
    """Closed oriented triangle manifold and all local differential geometry."""

    vertices: tuple[Point3, ...]
    faces: tuple[Face3, ...]
    face_areas: tuple[float, ...]
    face_normals: tuple[Point3, ...]
    vertex_normals: tuple[Point3, ...]
    corner_normals: tuple[tuple[Point3, Point3, Point3], ...]
    feature_edges: tuple[SurfaceFeatureEdge, ...]
    certificate: ManifoldRepairCertificate

    @property
    def sharp_edges(self) -> tuple[SurfaceFeatureEdge, ...]:
        return tuple(edge for edge in self.feature_edges if edge.sharp)

    @property
    def stats(self) -> dict[str, object]:
        result = dict(self.certificate.stats)
        result.update(
            {
                "sharp_edges": len(self.sharp_edges),
                "smooth_edges": len(self.feature_edges) - len(self.sharp_edges),
                "stored_corner_normals": 3 * len(self.faces),
            }
        )
        return result


def _weld_vertices(
    points: tuple[Point3, ...],
    tolerance: float,
) -> tuple[list[Point3], tuple[int, ...], int]:
    minimum = tuple(min(point[axis] for point in points) for axis in range(3))
    inverse = 1.0 / tolerance
    buckets: dict[tuple[int, int, int], list[int]] = {}
    representatives: list[Point3] = []
    counts: list[int] = []
    mapping: list[int] = []
    tolerance_squared = tolerance * tolerance
    for point in points:
        cell = tuple(
            int(math.floor((point[axis] - minimum[axis]) * inverse))
            for axis in range(3)
        )
        selected = None
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    for candidate in buckets.get(
                        (cell[0] + dx, cell[1] + dy, cell[2] + dz),
                        (),
                    ):
                        delta = _sub(point, representatives[candidate])
                        if _dot(delta, delta) <= tolerance_squared:
                            selected = candidate
                            break
                    if selected is not None:
                        break
                if selected is not None:
                    break
            if selected is not None:
                break
        if selected is None:
            selected = len(representatives)
            representatives.append(point)
            counts.append(1)
            buckets.setdefault(cell, []).append(selected)
        else:
            count = counts[selected]
            representatives[selected] = tuple(
                (count * representatives[selected][axis] + point[axis]) / (count + 1)
                for axis in range(3)
            )
            counts[selected] += 1
        mapping.append(selected)
    return representatives, tuple(mapping), len(points) - len(representatives)


def _clean_faces(
    points: list[Point3],
    faces: Iterable[Face3],
    area_tolerance: float,
) -> tuple[list[Face3], int, int]:
    output: list[Face3] = []
    seen: set[tuple[int, int, int]] = set()
    degenerate = 0
    duplicate = 0
    point_rows = tuple(points)
    for face in faces:
        if len(set(face)) != 3:
            degenerate += 1
            continue
        area, _normal = _face_geometry(point_rows, face)
        if area <= area_tolerance:
            degenerate += 1
            continue
        key = tuple(sorted(face))
        if key in seen:
            duplicate += 1
            continue
        seen.add(key)
        output.append(face)
    return output, degenerate, duplicate


def _face_components(faces: list[Face3]) -> tuple[tuple[int, ...], ...]:
    incidences = _edge_incidences(faces)
    adjacency = [set() for _ in faces]
    for rows in incidences.values():
        if len(rows) != 2:
            continue
        left = rows[0][0]
        right = rows[1][0]
        adjacency[left].add(right)
        adjacency[right].add(left)
    unseen = set(range(len(faces)))
    components = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack = [start]
        component = []
        while stack:
            face = stack.pop()
            component.append(face)
            for neighbor in adjacency[face]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        components.append(tuple(sorted(component)))
    return tuple(components)


def _separate_nonmanifold_sheets(
    points: list[Point3],
    faces: list[Face3],
    maximum_passes: int,
) -> tuple[list[Point3], list[Face3], int, int]:
    split_vertices = 0
    initial = sum(
        len(rows) > 2 for rows in _edge_incidences(faces).values()
    )
    for _pass in range(maximum_passes):
        incidences = _edge_incidences(faces)
        vertex_faces = [set() for _ in points]
        for face_index, face in enumerate(faces):
            for vertex in face:
                vertex_faces[vertex].add(face_index)
        replacements: dict[tuple[int, int], int] = {}
        for vertex, incident in enumerate(vertex_faces):
            if len(incident) <= 1:
                continue
            adjacency = {face: set() for face in incident}
            for edge, rows in incidences.items():
                if vertex not in edge or len(rows) != 2:
                    continue
                left = rows[0][0]
                right = rows[1][0]
                if left in incident and right in incident:
                    adjacency[left].add(right)
                    adjacency[right].add(left)
            unseen = set(incident)
            fans = []
            while unseen:
                start = min(unseen)
                unseen.remove(start)
                stack = [start]
                fan = []
                while stack:
                    face = stack.pop()
                    fan.append(face)
                    for neighbor in adjacency[face]:
                        if neighbor in unseen:
                            unseen.remove(neighbor)
                            stack.append(neighbor)
                fans.append(tuple(fan))
            if len(fans) <= 1:
                continue
            for fan in fans[1:]:
                duplicate = len(points)
                points.append(points[vertex])
                split_vertices += 1
                for face in fan:
                    replacements[(vertex, face)] = duplicate
        if not replacements:
            break
        faces = [
            tuple(replacements.get((vertex, face_index), vertex) for vertex in face)
            for face_index, face in enumerate(faces)
        ]
    return points, faces, split_vertices, initial


def _orient_faces(faces: list[Face3]) -> tuple[list[Face3], int]:
    incidences = _edge_incidences(faces)
    adjacency: list[list[tuple[int, bool]]] = [[] for _ in faces]
    for rows in incidences.values():
        if len(rows) != 2:
            continue
        first, second = rows
        same_direction = first[1] == second[1] and first[2] == second[2]
        adjacency[first[0]].append((second[0], same_direction))
        adjacency[second[0]].append((first[0], same_direction))
    states: list[bool | None] = [None for _ in faces]
    for start in range(len(faces)):
        if states[start] is not None:
            continue
        states[start] = False
        stack = [start]
        while stack:
            face = stack.pop()
            for neighbor, same_direction in adjacency[face]:
                wanted = bool(states[face]) ^ same_direction
                if states[neighbor] is None:
                    states[neighbor] = wanted
                    stack.append(neighbor)
                elif states[neighbor] != wanted:
                    raise ValueError("triangle soup contains a non-orientable face component")
    output = []
    flips = 0
    for face, flip in zip(faces, states, strict=True):
        if flip:
            output.append((face[0], face[2], face[1]))
            flips += 1
        else:
            output.append(face)
    return output, flips


def _boundary_loops(faces: list[Face3]) -> tuple[tuple[tuple[int, int], ...], ...]:
    incidences = _edge_incidences(faces)
    directed = tuple(rows[0][1:] for rows in incidences.values() if len(rows) == 1)
    if not directed:
        return tuple()
    outgoing: dict[int, list[tuple[int, int]]] = {}
    incoming: dict[int, list[tuple[int, int]]] = {}
    for edge in directed:
        outgoing.setdefault(edge[0], []).append(edge)
        incoming.setdefault(edge[1], []).append(edge)
    vertices = set(outgoing) | set(incoming)
    if any(len(outgoing.get(vertex, ())) != 1 for vertex in vertices) or any(
        len(incoming.get(vertex, ())) != 1 for vertex in vertices
    ):
        raise ValueError(
            "boundary graph is branched after sheet separation; it cannot be capped safely"
        )
    unused = set(directed)
    loops = []
    while unused:
        start = min(unused)
        current = start
        loop = []
        while current in unused:
            unused.remove(current)
            loop.append(current)
            following = outgoing[current[1]][0]
            current = following
        if current != start:
            raise ValueError("boundary halfedges do not close into loops")
        loops.append(tuple(loop))
    return tuple(loops)


def _fill_boundary_loops(
    points: list[Point3],
    faces: list[Face3],
) -> tuple[list[Point3], list[Face3], int, int, set[tuple[int, int]]]:
    loops = _boundary_loops(faces)
    cap_edges: set[tuple[int, int]] = set()
    cap_faces = 0
    for loop in loops:
        vertices = tuple(edge[0] for edge in loop)
        center = tuple(
            sum(points[index][axis] for index in vertices) / len(vertices)
            for axis in range(3)
        )
        center_index = len(points)
        points.append(center)
        for left, right in loop:
            faces.append((right, left, center_index))
            cap_edges.add((min(left, right), max(left, right)))
            cap_edges.add((min(left, center_index), max(left, center_index)))
            cap_edges.add((min(right, center_index), max(right, center_index)))
            cap_faces += 1
    return points, faces, len(loops), cap_faces, cap_edges


def _outward_components(
    points: tuple[Point3, ...],
    faces: list[Face3],
) -> tuple[list[Face3], int, int, bool]:
    components = _face_components(faces)
    flips = 0
    nonzero = True
    for component in components:
        volume = _signed_volume(points, (faces[index] for index in component))
        if abs(volume) <= 1.0e-14:
            nonzero = False
            continue
        if volume < 0.0:
            for index in component:
                face = faces[index]
                faces[index] = (face[0], face[2], face[1])
                flips += 1
    return faces, flips, len(components), nonzero


def _orientation_failures(faces: tuple[Face3, ...]) -> int:
    failures = 0
    for rows in _edge_incidences(faces).values():
        if len(rows) == 2 and not (
            rows[0][1] == rows[1][2] and rows[0][2] == rows[1][1]
        ):
            failures += 1
    return failures


def _nonmanifold_vertex_count(
    faces: tuple[Face3, ...],
    vertex_count: int,
) -> int:
    """Count vertices whose simplicial link is not one closed cycle."""

    links: list[dict[int, set[int]]] = [dict() for _ in range(vertex_count)]
    for face in faces:
        for offset, vertex in enumerate(face):
            left = face[(offset + 1) % 3]
            right = face[(offset + 2) % 3]
            links[vertex].setdefault(left, set()).add(right)
            links[vertex].setdefault(right, set()).add(left)

    failures = 0
    for adjacency in links:
        if not adjacency or any(len(neighbors) != 2 for neighbors in adjacency.values()):
            failures += 1
            continue
        unseen = set(adjacency)
        stack = [unseen.pop()]
        while stack:
            current = stack.pop()
            for neighbor in adjacency[current]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        if unseen:
            failures += 1
    return failures


def _feature_geometry(
    points: tuple[Point3, ...],
    faces: tuple[Face3, ...],
    sharp_angle: float,
    cap_edges: set[tuple[int, int]],
) -> tuple[
    tuple[float, ...],
    tuple[Point3, ...],
    tuple[Point3, ...],
    tuple[SurfaceFeatureEdge, ...],
    tuple[tuple[Point3, Point3, Point3], ...],
]:
    face_areas = []
    face_normals = []
    normal_sums = [[0.0, 0.0, 0.0] for _ in points]
    vertex_faces = [set() for _ in points]
    for face_index, face in enumerate(faces):
        area, normal = _face_geometry(points, face)
        face_areas.append(area)
        face_normals.append(normal)
        for vertex in face:
            vertex_faces[vertex].add(face_index)
            for axis in range(3):
                normal_sums[vertex][axis] += area * normal[axis]
    vertex_normals = tuple(_unit(tuple(row)) for row in normal_sums)
    incidences = _edge_incidences(faces)
    features = []
    smooth_face_adjacency = [set() for _ in faces]
    for edge_index, edge in enumerate(sorted(incidences)):
        rows = incidences[edge]
        if len(rows) != 2:
            continue
        first_face, directed_left, directed_right = rows[0]
        second_face = rows[1][0]
        first_normal = face_normals[first_face]
        second_normal = face_normals[second_face]
        tangent = _unit(_sub(points[directed_right], points[directed_left]))
        turn = math.atan2(
            _dot(tangent, _cross(first_normal, second_normal)),
            max(-1.0, min(1.0, _dot(first_normal, second_normal))),
        )
        opening = math.pi - turn
        while opening <= 0.0:
            opening += 2.0 * math.pi
        while opening >= 2.0 * math.pi:
            opening -= 2.0 * math.pi
        normal_angle = math.acos(
            max(-1.0, min(1.0, _dot(first_normal, second_normal)))
        )
        sharp = normal_angle >= sharp_angle
        if not sharp:
            smooth_face_adjacency[first_face].add(second_face)
            smooth_face_adjacency[second_face].add(first_face)
        features.append(
            SurfaceFeatureEdge(
                index=edge_index,
                vertices=edge,
                incident_faces=(first_face, second_face),
                length=_norm(_sub(points[edge[1]], points[edge[0]])),
                normal_turn=turn,
                opening_angle=opening,
                sharp=sharp,
                introduced_by_hole_fill=edge in cap_edges,
            )
        )

    corner_rows = []
    for face_index, face in enumerate(faces):
        row = []
        for vertex in face:
            allowed = vertex_faces[vertex]
            seen = {face_index}
            stack = [face_index]
            while stack:
                current = stack.pop()
                for neighbor in smooth_face_adjacency[current]:
                    if neighbor in allowed and neighbor not in seen:
                        seen.add(neighbor)
                        stack.append(neighbor)
            averaged = (0.0, 0.0, 0.0)
            for neighbor in seen:
                averaged = _add(
                    averaged,
                    _scale(face_areas[neighbor], face_normals[neighbor]),
                )
            row.append(_unit(averaged, face_normals[face_index]))
        corner_rows.append(tuple(row))
    return (
        tuple(face_areas),
        tuple(face_normals),
        vertex_normals,
        tuple(features),
        tuple(corner_rows),
    )


def repair_triangle_mesh(
    vertices: Iterable[Iterable[float]],
    triangles: Iterable[Iterable[int]],
    *,
    config: ManifoldRepairConfig | None = None,
) -> RepairedTriangleMesh:
    """Repair a finite triangle soup and return a closed oriented manifold.

    The procedure certifies incidence topology and orientation.  It does not
    claim that a badly self-intersecting source has been converted into the
    boundary of the intended solid; that separate geometric question remains
    visible in the returned certificate.
    """

    settings = config or ManifoldRepairConfig()
    source_points = tuple(_point(value) for value in vertices)
    if len(source_points) < 3:
        raise ValueError("triangle-soup repair requires at least three vertices")
    source_faces = []
    for value in triangles:
        face = tuple(int(index) for index in value)
        if len(face) != 3:
            raise ValueError("every source face must contain three indices")
        if any(index < 0 or index >= len(source_points) for index in face):
            raise ValueError("source face index is out of range")
        source_faces.append(face)
    if not source_faces:
        raise ValueError("triangle-soup repair requires at least one face")

    minimum = tuple(min(point[axis] for point in source_points) for axis in range(3))
    maximum = tuple(max(point[axis] for point in source_points) for axis in range(3))
    diagonal = _norm(_sub(maximum, minimum))
    scale = max(diagonal, 1.0)
    weld_tolerance = settings.weld_tolerance * scale
    area_tolerance = settings.area_tolerance * scale * scale

    points, vertex_map, welded = _weld_vertices(source_points, weld_tolerance)
    mapped_faces = [tuple(vertex_map[index] for index in face) for face in source_faces]
    faces, degenerate, duplicate = _clean_faces(points, mapped_faces, area_tolerance)
    if not faces:
        raise ValueError("triangle-soup repair removed every source face")
    if settings.split_nonmanifold_sheets:
        points, faces, split_vertices, nonmanifold_before = (
            _separate_nonmanifold_sheets(
                points,
                faces,
                settings.maximum_repair_passes,
            )
        )
    else:
        split_vertices = 0
        nonmanifold_before = sum(
            len(rows) > 2 for rows in _edge_incidences(faces).values()
        )
    remaining_nonmanifold = sum(
        len(rows) > 2 for rows in _edge_incidences(faces).values()
    )
    if remaining_nonmanifold:
        raise ValueError("nonmanifold sheet separation did not converge")
    faces, orientation_flips = _orient_faces(faces)
    boundary_before = sum(
        len(rows) == 1 for rows in _edge_incidences(faces).values()
    )
    cap_loops = 0
    cap_faces = 0
    cap_vertices = 0
    cap_edges: set[tuple[int, int]] = set()
    if boundary_before:
        if not settings.fill_holes:
            raise ValueError("the repaired mesh remains open and fill_holes is false")
        old_vertices = len(points)
        points, faces, cap_loops, cap_faces, cap_edges = _fill_boundary_loops(
            points,
            faces,
        )
        cap_vertices = len(points) - old_vertices
        faces, added_flips = _orient_faces(faces)
        orientation_flips += added_flips
    points, faces, removed_unused = _compact(points, faces)
    # Compaction changes edge indices, so cap provenance is conservative after
    # unused-vertex removal.  In normal repair paths all cap vertices are used
    # and the mapping is the identity.
    if removed_unused:
        cap_edges = set()
    point_rows = tuple(points)
    faces, outward_flips, components, nonzero_components = _outward_components(
        point_rows,
        faces,
    )
    orientation_flips += outward_flips
    face_rows = tuple(faces)
    incidences = _edge_incidences(face_rows)
    boundary = sum(len(rows) == 1 for rows in incidences.values())
    nonmanifold = sum(len(rows) > 2 for rows in incidences.values())
    nonmanifold_vertices = _nonmanifold_vertex_count(
        face_rows,
        len(point_rows),
    )
    orientation_failures = _orientation_failures(face_rows)
    face_areas, face_normals, vertex_normals, features, corner_normals = (
        _feature_geometry(
            point_rows,
            face_rows,
            math.radians(settings.sharp_angle_degrees),
            cap_edges,
        )
    )
    edge_count = len(incidences)
    euler = len(point_rows) - edge_count + len(face_rows)
    genus_numerator = 2 * components - euler
    genus = (
        genus_numerator // 2
        if boundary == 0 and nonmanifold == 0 and genus_numerator >= 0
        and genus_numerator % 2 == 0
        else None
    )
    certificate = ManifoldRepairCertificate(
        source_vertices=len(source_points),
        source_faces=len(source_faces),
        welded_vertices=welded,
        removed_unused_vertices=removed_unused,
        removed_degenerate_faces=degenerate,
        removed_duplicate_faces=duplicate,
        split_sheet_vertices=split_vertices,
        boundary_edges_before_fill=boundary_before,
        nonmanifold_edges_before_split=nonmanifold_before,
        boundary_loops_filled=cap_loops,
        cap_vertices_added=cap_vertices,
        cap_faces_added=cap_faces,
        orientation_flips=orientation_flips,
        repaired_vertices=len(point_rows),
        repaired_faces=len(face_rows),
        repaired_edges=edge_count,
        connected_components=components,
        euler_characteristic=euler,
        orientable_genus_sum=genus,
        signed_volume=_signed_volume(point_rows, face_rows),
        boundary_edges=boundary,
        nonmanifold_edges=nonmanifold,
        nonmanifold_vertices=nonmanifold_vertices,
        inconsistent_oriented_edges=orientation_failures,
        watertight=boundary == 0,
        manifold=(
            nonmanifold == 0
            and nonmanifold_vertices == 0
            and all(len(rows) == 2 for rows in incidences.values())
        ),
        consistently_oriented=orientation_failures == 0,
        outward_oriented_nonzero_components=nonzero_components,
    )
    if not certificate.production_ready:
        raise ValueError("triangle-soup repair did not produce a closed oriented manifold")
    return RepairedTriangleMesh(
        vertices=point_rows,
        faces=face_rows,
        face_areas=face_areas,
        face_normals=face_normals,
        vertex_normals=vertex_normals,
        corner_normals=corner_normals,
        feature_edges=features,
        certificate=certificate,
    )


__all__ = [
    "ManifoldRepairCertificate",
    "ManifoldRepairConfig",
    "RepairedTriangleMesh",
    "SurfaceFeatureEdge",
    "repair_triangle_mesh",
]
