"""Linear-storage compilation of exact CAD meshes into PDE audit surfaces.

The lossless ``QCAD3J`` archive may contain hundreds of thousands of vertices.
Repeated pure-Python PDE solves on every source vertex are not an informative
interactive audit, so this module compiles the *entire* source mesh into a
smaller topology-bearing surface.  Every source triangle contributes to the
vertex mass and normal jets before clustering.

Vertices are grouped by a Cartesian cell and one of six normal orientations.
That second key prevents the two sides of a thin component from being merged.
Faces are then lowered through the vertex map and duplicate or collapsed faces
are removed.  The algorithm uses ``O(V+F)`` memory, no pair table, and no dense
matrix.  The result is a nondimensional CAD audit mesh; it is not claimed to be
a lossless replacement for the source archive.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from inverse_shape.quadrature import _abs, _sqrt
from inverse_shape.reversible_cad_qjet import ExactMesh, decode_mesh


Point3 = tuple[float, float, float]
Face3 = tuple[int, int, int]


def _sub(left: Point3, right: Point3) -> Point3:
    return (
        left[0] - right[0],
        left[1] - right[1],
        left[2] - right[2],
    )


def _cross(left: Point3, right: Point3) -> Point3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _dot(left: Point3, right: Point3) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _norm(value: Point3) -> float:
    return _sqrt(max(_dot(value, value), 0.0))


def _unit(value: Point3, fallback: Point3) -> Point3:
    length = _norm(value)
    if length <= 1.0e-30:
        value = fallback
        length = _norm(value)
    if length <= 1.0e-30:
        return (1.0, 0.0, 0.0)
    return tuple(component / length for component in value)


def _orientation_bucket(normal: Point3) -> int:
    axis = max(range(3), key=lambda index: _abs(normal[index]))
    return 2 * axis + (1 if normal[axis] < 0.0 else 0)


def _cell_key(
    point: Point3,
    bounds_min: Point3,
    bounds_max: Point3,
    resolution: int,
    orientation: int,
    group: int,
) -> tuple[int, int, int, int, int]:
    indices = []
    for axis in range(3):
        extent = bounds_max[axis] - bounds_min[axis]
        if extent <= 1.0e-30:
            indices.append(0)
            continue
        coordinate = int(
            resolution * (point[axis] - bounds_min[axis]) / extent
        )
        indices.append(max(0, min(resolution - 1, coordinate)))
    return (group, indices[0], indices[1], indices[2], orientation)


@dataclass(frozen=True)
class CompiledCadSurface:
    """A nondimensional topology-bearing CAD surface and its audit metadata."""

    vertices: tuple[Point3, ...]
    faces: tuple[Face3, ...]
    weights: tuple[float, ...]
    normals: tuple[Point3, ...]
    source_name: str
    source_vertices: int
    source_faces: int
    processed_source_faces: int
    collapsed_source_faces: int
    grid_resolution: int
    coordinate_center: Point3
    coordinate_scale: float
    source_area: float
    compiled_area: float
    boundary_edges: int
    nonmanifold_edges: int

    @property
    def stats(self) -> dict[str, object]:
        return {
            "source_name": self.source_name,
            "source_vertices": self.source_vertices,
            "source_faces": self.source_faces,
            "processed_source_faces": self.processed_source_faces,
            "collapsed_source_faces": self.collapsed_source_faces,
            "compiled_vertices": len(self.vertices),
            "compiled_faces": len(self.faces),
            "grid_resolution": self.grid_resolution,
            "coordinate_center": self.coordinate_center,
            "coordinate_scale": self.coordinate_scale,
            "source_area_nondimensional": self.source_area,
            "compiled_area_nondimensional": self.compiled_area,
            "compiled_measure_nondimensional": sum(self.weights),
            "compiled_to_source_area_ratio": (
                self.compiled_area / self.source_area
                if self.source_area > 0.0
                else 0.0
            ),
            "compiled_measure_to_source_area_ratio": (
                sum(self.weights) / self.source_area
                if self.source_area > 0.0
                else 0.0
            ),
            "boundary_edges": self.boundary_edges,
            "nonmanifold_edges": self.nonmanifold_edges,
            "all_source_faces_scanned": (
                self.processed_source_faces == self.source_faces
            ),
            "compiler_time_big_o": "O(V+F) per fixed grid trial",
            "compiler_storage_big_o": "O(V+F)",
            "stored_dense_matrix": False,
            "stored_pair_table": False,
            "lossless_geometry_channel": False,
            "purpose": "nondimensional PDE audit surface",
        }


@dataclass(frozen=True)
class CompiledCadPanelSurface:
    """Area-exact panel atlas compiled from every nondegenerate CAD triangle."""

    points: tuple[Point3, ...]
    weights: tuple[float, ...]
    normals: tuple[Point3, ...]
    source_name: str
    source_vertices: int
    source_faces: int
    processed_source_faces: int
    degenerate_source_faces: int
    grid_resolution: int
    coordinate_center: Point3
    coordinate_scale: float
    source_area: float

    @property
    def stats(self) -> dict[str, object]:
        return {
            "source_name": self.source_name,
            "source_vertices": self.source_vertices,
            "source_faces": self.source_faces,
            "processed_source_faces": self.processed_source_faces,
            "degenerate_source_faces": self.degenerate_source_faces,
            "compiled_panels": len(self.points),
            "grid_resolution": self.grid_resolution,
            "coordinate_center": self.coordinate_center,
            "coordinate_scale": self.coordinate_scale,
            "source_area_nondimensional": self.source_area,
            "compiled_area_nondimensional": sum(self.weights),
            "compiled_to_source_area_ratio": sum(self.weights) / self.source_area,
            "all_source_faces_scanned": (
                self.processed_source_faces == self.source_faces
            ),
            "compiler_time_big_o": "O(V+F) per fixed grid trial",
            "compiler_storage_big_o": "O(V+F)+O(P)",
            "operator_storage_after_compile": "O(P)",
            "stored_dense_matrix": False,
            "stored_pair_table": False,
            "lossless_geometry_channel": False,
            "topology_retained": False,
            "purpose": "area-exact nondimensional boundary-panel PDE audit",
        }


def _resolution_for_target(
    points: tuple[Point3, ...],
    orientations: tuple[int, ...],
    groups: tuple[int, ...],
    bounds_min: Point3,
    bounds_max: Point3,
    target_vertices: int,
    maximum_resolution: int,
) -> int:
    def count(resolution: int) -> int:
        return len(
            {
                _cell_key(
                    point,
                    bounds_min,
                    bounds_max,
                    resolution,
                    orientation,
                    group,
                )
                for point, orientation, group in zip(
                    points,
                    orientations,
                    groups,
                    strict=True,
                )
            }
        )

    low = 1
    high = max(1, int(maximum_resolution))
    candidates = {low, high}
    while low <= high:
        middle = (low + high) // 2
        candidates.add(middle)
        occupied = count(middle)
        if occupied < target_vertices:
            low = middle + 1
        elif occupied > target_vertices:
            high = middle - 1
        else:
            return middle
    for center in tuple(candidates):
        for offset in (-2, -1, 0, 1, 2):
            if 1 <= center + offset <= maximum_resolution:
                candidates.add(center + offset)
    return min(
        candidates,
        key=lambda resolution: (
            _abs(count(resolution) - target_vertices),
            resolution,
        ),
    )


def compile_cad_surface(
    mesh: ExactMesh,
    *,
    target_vertices: int = 192,
    maximum_resolution: int = 64,
) -> CompiledCadSurface:
    """Compile every triangle of ``mesh`` into a bounded CAD PDE audit mesh."""

    target = int(target_vertices)
    if target < 24:
        raise ValueError("target_vertices must be at least 24")
    if not mesh.vertices or not mesh.faces:
        raise ValueError("CAD surface compilation requires a nonempty mesh")

    vertex_count = len(mesh.vertices)
    normal_sums = [[0.0, 0.0, 0.0] for _ in mesh.vertices]
    vertex_masses = [0.0 for _ in mesh.vertices]
    processed_faces = 0
    source_area_physical = 0.0
    for face in mesh.faces:
        first, second, third = (mesh.vertices[index] for index in face)
        cross = _cross(_sub(second, first), _sub(third, first))
        area = 0.5 * _norm(cross)
        processed_faces += 1
        if area <= 1.0e-30:
            continue
        source_area_physical += area
        for index in face:
            vertex_masses[index] += area / 3.0
            for axis in range(3):
                normal_sums[index][axis] += cross[axis]
    if source_area_physical <= 0.0:
        raise ValueError("CAD mesh has no positive-area triangles")

    total_mass = sum(vertex_masses)
    center = tuple(
        sum(
            mass * point[axis]
            for mass, point in zip(vertex_masses, mesh.vertices, strict=True)
        )
        / total_mass
        for axis in range(3)
    )
    radius_squared = sum(
        mass * _dot(_sub(point, center), _sub(point, center))
        for mass, point in zip(vertex_masses, mesh.vertices, strict=True)
    ) / total_mass
    scale = _sqrt(max(radius_squared, 1.0e-30))
    points = tuple(
        tuple((point[axis] - center[axis]) / scale for axis in range(3))
        for point in mesh.vertices
    )
    vertex_normals = tuple(
        _unit(tuple(values), point)
        for values, point in zip(normal_sums, points, strict=True)
    )
    orientations = tuple(_orientation_bucket(normal) for normal in vertex_normals)
    vertex_groups = [-1 for _ in mesh.vertices]
    for group, part in enumerate(mesh.parts):
        stop = part.face_start + part.face_count
        for face in mesh.faces[part.face_start:stop]:
            for index in face:
                if vertex_groups[index] < 0:
                    vertex_groups[index] = group
    groups = tuple(max(group, 0) for group in vertex_groups)
    bounds_min = tuple(min(point[axis] for point in points) for axis in range(3))
    bounds_max = tuple(max(point[axis] for point in points) for axis in range(3))
    resolution = _resolution_for_target(
        points,
        orientations,
        groups,
        bounds_min,
        bounds_max,
        target,
        int(maximum_resolution),
    )

    keys = tuple(
        _cell_key(
            point,
            bounds_min,
            bounds_max,
            resolution,
            orientation,
            group,
        )
        for point, orientation, group in zip(
            points,
            orientations,
            groups,
            strict=True,
        )
    )
    ordered_keys = tuple(sorted(set(keys)))
    key_to_index = {key: index for index, key in enumerate(ordered_keys)}
    source_to_cluster = tuple(key_to_index[key] for key in keys)
    cluster_mass = [0.0 for _ in ordered_keys]
    cluster_position = [[0.0, 0.0, 0.0] for _ in ordered_keys]
    cluster_normal = [[0.0, 0.0, 0.0] for _ in ordered_keys]
    for source, cluster in enumerate(source_to_cluster):
        mass = vertex_masses[source]
        if mass <= 0.0:
            mass = 1.0e-30
        cluster_mass[cluster] += mass
        for axis in range(3):
            cluster_position[cluster][axis] += mass * points[source][axis]
            cluster_normal[cluster][axis] += mass * vertex_normals[source][axis]
    coarse_points = tuple(
        tuple(value / cluster_mass[index] for value in cluster_position[index])
        for index in range(len(ordered_keys))
    )
    coarse_normals = tuple(
        _unit(tuple(cluster_normal[index]), coarse_points[index])
        for index in range(len(ordered_keys))
    )

    unique_faces = set()
    coarse_faces = []
    collapsed = 0
    for face in mesh.faces:
        mapped = tuple(source_to_cluster[index] for index in face)
        if len(set(mapped)) != 3:
            collapsed += 1
            continue
        first, second, third = (coarse_points[index] for index in mapped)
        if _norm(_cross(_sub(second, first), _sub(third, first))) <= 1.0e-18:
            collapsed += 1
            continue
        canonical = tuple(sorted(mapped))
        if canonical in unique_faces:
            collapsed += 1
            continue
        unique_faces.add(canonical)
        coarse_faces.append(mapped)

    used = sorted({index for face in coarse_faces for index in face})
    if len(used) < 6 or len(coarse_faces) < 8:
        raise ValueError("CAD clustering collapsed too much surface topology")
    remap = {old: new for new, old in enumerate(used)}
    vertices = tuple(coarse_points[index] for index in used)
    normals = tuple(coarse_normals[index] for index in used)
    faces = tuple(
        tuple(remap[index] for index in face) for face in coarse_faces
    )

    compiled_area = 0.0
    edge_counts: dict[tuple[int, int], int] = {}
    for face in faces:
        first, second, third = (vertices[index] for index in face)
        compiled_area += 0.5 * _norm(
            _cross(_sub(second, first), _sub(third, first))
        )
        for offset in range(3):
            left = face[offset]
            right = face[(offset + 1) % 3]
            edge = (min(left, right), max(left, right))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    source_area = source_area_physical / (scale * scale)
    raw_weights = tuple(cluster_mass[index] / (scale * scale) for index in used)
    retained_measure = sum(raw_weights)
    measure_scale = source_area / retained_measure
    weights = tuple(value * measure_scale for value in raw_weights)
    return CompiledCadSurface(
        vertices=vertices,
        faces=faces,
        weights=weights,
        normals=normals,
        source_name=mesh.name,
        source_vertices=vertex_count,
        source_faces=len(mesh.faces),
        processed_source_faces=processed_faces,
        collapsed_source_faces=collapsed,
        grid_resolution=resolution,
        coordinate_center=center,
        coordinate_scale=scale,
        source_area=source_area,
        compiled_area=compiled_area,
        boundary_edges=sum(count == 1 for count in edge_counts.values()),
        nonmanifold_edges=sum(count > 2 for count in edge_counts.values()),
    )


def _faces_with_groups(mesh: ExactMesh):
    part_index = 0
    part_stop = mesh.parts[0].face_count
    for face_index, face in enumerate(mesh.faces):
        while face_index >= part_stop and part_index + 1 < len(mesh.parts):
            part_index += 1
            part = mesh.parts[part_index]
            part_stop = part.face_start + part.face_count
        yield part_index, face


def compile_cad_panels(
    mesh: ExactMesh,
    *,
    target_panels: int = 128,
    maximum_resolution: int = 64,
) -> CompiledCadPanelSurface:
    """Compile every CAD triangle into an area-exact oriented panel atlas."""

    target = int(target_panels)
    if target < 16:
        raise ValueError("target_panels must be at least 16")
    if not mesh.vertices or not mesh.faces:
        raise ValueError("CAD panel compilation requires a nonempty mesh")

    physical_area = 0.0
    first_moment = [0.0, 0.0, 0.0]
    degenerate = 0
    for _group, face in _faces_with_groups(mesh):
        first, second, third = (mesh.vertices[index] for index in face)
        cross = _cross(_sub(second, first), _sub(third, first))
        area = 0.5 * _norm(cross)
        if area <= 1.0e-30:
            degenerate += 1
            continue
        centroid = tuple(
            (first[axis] + second[axis] + third[axis]) / 3.0
            for axis in range(3)
        )
        physical_area += area
        for axis in range(3):
            first_moment[axis] += area * centroid[axis]
    if physical_area <= 0.0:
        raise ValueError("CAD mesh has no positive-area triangles")
    center = tuple(value / physical_area for value in first_moment)
    radius_squared = 0.0
    for _group, face in _faces_with_groups(mesh):
        first, second, third = (mesh.vertices[index] for index in face)
        cross = _cross(_sub(second, first), _sub(third, first))
        area = 0.5 * _norm(cross)
        if area <= 1.0e-30:
            continue
        centroid = tuple(
            (first[axis] + second[axis] + third[axis]) / 3.0
            for axis in range(3)
        )
        radius_squared += area * _dot(
            _sub(centroid, center),
            _sub(centroid, center),
        )
    scale = _sqrt(max(radius_squared / physical_area, 1.0e-30))
    bounds_min = tuple(
        min((point[axis] - center[axis]) / scale for point in mesh.vertices)
        for axis in range(3)
    )
    bounds_max = tuple(
        max((point[axis] - center[axis]) / scale for point in mesh.vertices)
        for axis in range(3)
    )

    def face_key(_group: int, face: Face3, resolution: int):
        first, second, third = (mesh.vertices[index] for index in face)
        cross = _cross(_sub(second, first), _sub(third, first))
        length = _norm(cross)
        if length <= 1.0e-30:
            return None
        centroid = tuple(
            (
                (first[axis] + second[axis] + third[axis]) / 3.0
                - center[axis]
            )
            / scale
            for axis in range(3)
        )
        normal = tuple(value / length for value in cross)
        return _cell_key(
            centroid,
            bounds_min,
            bounds_max,
            resolution,
            _orientation_bucket(normal),
            0,
        )

    def occupied(resolution: int) -> int:
        keys = set()
        for group, face in _faces_with_groups(mesh):
            key = face_key(group, face, resolution)
            if key is not None:
                keys.add(key)
        return len(keys)

    candidates = []
    for resolution in range(1, int(maximum_resolution) + 1):
        count = occupied(resolution)
        candidates.append((resolution, count))
        if count >= target:
            break
    resolution, _count = min(
        candidates,
        key=lambda row: (_abs(row[1] - target), row[0]),
    )

    accumulators: dict[
        tuple[int, int, int, int, int], list[float]
    ] = {}
    for group, face in _faces_with_groups(mesh):
        key = face_key(group, face, resolution)
        if key is None:
            continue
        first, second, third = (mesh.vertices[index] for index in face)
        cross = _cross(_sub(second, first), _sub(third, first))
        area_physical = 0.5 * _norm(cross)
        area = area_physical / (scale * scale)
        centroid = tuple(
            (
                (first[axis] + second[axis] + third[axis]) / 3.0
                - center[axis]
            )
            / scale
            for axis in range(3)
        )
        row = accumulators.setdefault(key, [0.0] * 7)
        row[0] += area
        for axis in range(3):
            row[1 + axis] += area * centroid[axis]
            row[4 + axis] += 0.5 * cross[axis] / (scale * scale)
    points = []
    weights = []
    normals = []
    for key in sorted(accumulators):
        row = accumulators[key]
        weight = row[0]
        point = tuple(row[1 + axis] / weight for axis in range(3))
        points.append(point)
        weights.append(weight)
        normals.append(_unit(tuple(row[4:7]), point))
    source_area = physical_area / (scale * scale)
    return CompiledCadPanelSurface(
        points=tuple(points),
        weights=tuple(weights),
        normals=tuple(normals),
        source_name=mesh.name,
        source_vertices=len(mesh.vertices),
        source_faces=len(mesh.faces),
        processed_source_faces=len(mesh.faces),
        degenerate_source_faces=degenerate,
        grid_resolution=resolution,
        coordinate_center=center,
        coordinate_scale=scale,
        source_area=source_area,
    )


def load_compiled_cad_surface(
    path: str | Path,
    *,
    target_vertices: int = 192,
    maximum_resolution: int = 64,
) -> CompiledCadSurface:
    """Decode a QCAD3J archive and compile its complete mesh for PDE audits."""

    archive_path = Path(path)
    mesh = decode_mesh(archive_path.read_bytes())
    return compile_cad_surface(
        mesh,
        target_vertices=target_vertices,
        maximum_resolution=maximum_resolution,
    )


def load_compiled_cad_panels(
    path: str | Path,
    *,
    target_panels: int = 128,
    maximum_resolution: int = 64,
) -> CompiledCadPanelSurface:
    """Decode a QCAD3J archive and compile all triangles into panel QJets."""

    archive_path = Path(path)
    mesh = decode_mesh(archive_path.read_bytes())
    return compile_cad_panels(
        mesh,
        target_panels=target_panels,
        maximum_resolution=maximum_resolution,
    )


__all__ = [
    "CompiledCadSurface",
    "CompiledCadPanelSurface",
    "compile_cad_panels",
    "compile_cad_surface",
    "load_compiled_cad_panels",
    "load_compiled_cad_surface",
]
