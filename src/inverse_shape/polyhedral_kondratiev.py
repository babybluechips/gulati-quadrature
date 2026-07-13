"""Sparse Mellin--Kondratiev channels for polyhedral edges and vertices.

The smooth arbitrary-surface backend only approximates a discrete pair graph.
This module supplies the missing continuum corner bookkeeping.  A Dirichlet
edge of opening ``omega`` has exponents ``lambda_m=m*pi/omega``.  A conical
vertex with spherical-link eigenvalue ``mu`` has exponent determined by
``lambda*(lambda+1)=mu``.  Boundary fluxes therefore have local forms

    edge:   r**(lambda-1) sum_j a_j r**j (log r)**q dr,
    vertex: r**(lambda-1) sum_j a_j r**j (log r)**q r dr.

The four coefficients ``a_0,...,a_3`` are the persistent three-jet.  Their
midpoint-grid defects are evaluated directly through differentiated Hurwitz
zeta values.  No corner refinement, pair table, or dense matrix is stored.
"""

from inverse_shape.quadrature import (
    BorrowComputeRepayLedger,
    HALF_PI,
    PI,
    TAU,
    _abs,
    _atan,
    _clean_scalar,
    _exp,
    _finite,
    _log,
    _sqrt,
)


_BERNOULLI_EVEN_OVER_FACTORIAL = (
    1.0 / 12.0,
    -1.0 / 720.0,
    1.0 / 30240.0,
    -1.0 / 1209600.0,
    1.0 / 47900160.0,
    -691.0 / 1307674368000.0,
    1.0 / 74724249600.0,
    -3617.0 / 10670622842880000.0,
)


def _point(value):
    row = tuple(float(component) for component in value)
    if len(row) != 3:
        raise ValueError("each mesh point must have three coordinates")
    if any(not _finite(component) for component in row):
        raise ValueError("mesh coordinates must be finite")
    return row


def _vsub(left, right):
    return (
        left[0] - right[0],
        left[1] - right[1],
        left[2] - right[2],
    )


def _vscale(scale, value):
    return (scale * value[0], scale * value[1], scale * value[2])


def _vdot(left, right):
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _vcross(left, right):
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _vnorm(value):
    return _sqrt(max(_vdot(value, value), 0.0))


def _unit(value):
    norm = _vnorm(value)
    if norm <= 1.0e-30:
        raise ValueError("zero-length geometric direction")
    return _vscale(1.0 / norm, value)


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
    return 0.0


def _face_geometry(points, face):
    left = _vsub(points[face[1]], points[face[0]])
    right = _vsub(points[face[2]], points[face[0]])
    cross = _vcross(left, right)
    area_twice = _vnorm(cross)
    if area_twice <= 1.0e-30:
        raise ValueError("degenerate mesh triangles are not supported")
    return 0.5 * area_twice, _vscale(1.0 / area_twice, cross)


class PolyhedralEdgeRecord:
    """One oriented-manifold edge and its local dihedral pencil data."""

    def __init__(
        self,
        index,
        vertices,
        incident_faces,
        length,
        opening_angle,
        material_side_certified,
    ):
        self.index = int(index)
        self.vertices = tuple(vertices)
        self.incident_faces = tuple(incident_faces)
        self.length = float(length)
        self.opening_angle = (
            None if opening_angle is None else float(opening_angle)
        )
        self.material_side_certified = bool(material_side_certified)

    @property
    def is_boundary(self):
        return len(self.incident_faces) == 1

    @property
    def is_manifold(self):
        return len(self.incident_faces) in (1, 2)

    @property
    def is_reentrant(self):
        return self.opening_angle is not None and self.opening_angle > PI

    @property
    def is_geometric(self):
        return (
            self.opening_angle is not None
            and _abs(self.opening_angle - PI) > 1.0e-12
        )

    def dirichlet_exponent(self, mode=1):
        if self.opening_angle is None:
            raise ValueError("a boundary edge has no interior dihedral angle")
        order = int(mode)
        if order < 1:
            raise ValueError("Dirichlet edge modes start at one")
        return order * PI / self.opening_angle

    def certificate(self):
        return {
            "index": self.index,
            "vertices": self.vertices,
            "incident_faces": self.incident_faces,
            "length": self.length,
            "opening_angle": self.opening_angle,
            "is_boundary": self.is_boundary,
            "is_geometric": self.is_geometric,
            "is_reentrant": self.is_reentrant,
            "material_side_certified": self.material_side_certified,
            "first_dirichlet_exponent": (
                None if self.opening_angle is None else self.dirichlet_exponent()
            ),
        }


class PolyhedralVertexRecord:
    """Connectivity retained at one polyhedral vertex."""

    def __init__(self, index, incident_edges, incident_faces):
        self.index = int(index)
        self.incident_edges = tuple(sorted(incident_edges))
        self.incident_faces = tuple(sorted(incident_faces))

    def certificate(self):
        return {
            "index": self.index,
            "incident_edges": self.incident_edges,
            "incident_faces": self.incident_faces,
            "edge_degree": len(self.incident_edges),
            "face_degree": len(self.incident_faces),
        }


class PolyhedralMeshTopology:
    """Validated triangle topology with signed dihedral opening angles."""

    def __init__(self, vertices, triangles, normalize_orientation=True):
        self.points = tuple(_point(value) for value in vertices)
        if len(self.points) < 4:
            raise ValueError("a polyhedral mesh requires at least four vertices")
        faces = []
        for value in triangles:
            face = tuple(int(index) for index in value)
            if len(face) != 3 or len(set(face)) != 3:
                raise ValueError("each mesh face must contain three distinct vertices")
            if any(index < 0 or index >= len(self.points) for index in face):
                raise ValueError("triangle vertex index is out of range")
            _face_geometry(self.points, face)
            faces.append(face)
        if not faces:
            raise ValueError("a polyhedral mesh requires at least one triangle")

        edge_degree = self._edge_degrees(faces)
        self.closed = all(value == 2 for value in edge_degree.values())
        if any(value > 2 for value in edge_degree.values()):
            raise ValueError("nonmanifold mesh edges are not supported")
        components = self._face_components(faces)
        reversed_components = 0
        component_volumes = []
        if self.closed:
            for component in components:
                volume = self._signed_volume(tuple(faces[index] for index in component))
                if _abs(volume) <= 1.0e-30:
                    raise ValueError("a closed polyhedral component has zero signed volume")
                if normalize_orientation and volume < 0.0:
                    for index in component:
                        face = faces[index]
                        faces[index] = (face[0], face[2], face[1])
                    volume = -volume
                    reversed_components += 1
                component_volumes.append(volume)
        signed_volume = sum(component_volumes)
        self.orientation_reversed = reversed_components > 0
        self.orientation_reversed_components = reversed_components
        self.connected_components = len(components)
        self.material_side_dihedral_certified = self.closed and all(
            volume > 0.0 for volume in component_volumes
        )
        self.faces = tuple(faces)
        self.signed_volume = signed_volume
        self.face_areas = []
        self.face_normals = []
        for face in self.faces:
            area, normal = _face_geometry(self.points, face)
            self.face_areas.append(area)
            self.face_normals.append(normal)
        self.face_areas = tuple(self.face_areas)
        self.face_normals = tuple(self.face_normals)

        incidences = {}
        vertex_faces = [set() for _ in self.points]
        for face_index, face in enumerate(self.faces):
            for vertex in face:
                vertex_faces[vertex].add(face_index)
            for offset in range(3):
                left = face[offset]
                right = face[(offset + 1) % 3]
                key = (min(left, right), max(left, right))
                incidences.setdefault(key, []).append((face_index, left, right))

        edges = []
        vertex_edges = [set() for _ in self.points]
        orientation_failures = 0
        for key in sorted(incidences):
            rows = incidences[key]
            if len(rows) == 2 and not (
                rows[0][1] == rows[1][2] and rows[0][2] == rows[1][1]
            ):
                orientation_failures += 1
            left_point = self.points[key[0]]
            right_point = self.points[key[1]]
            length = _vnorm(_vsub(right_point, left_point))
            opening = None
            if len(rows) == 2:
                first_face, directed_left, directed_right = rows[0]
                second_face = rows[1][0]
                tangent = _unit(
                    _vsub(
                        self.points[directed_right],
                        self.points[directed_left],
                    )
                )
                first_normal = self.face_normals[first_face]
                second_normal = self.face_normals[second_face]
                turn = _atan2(
                    _vdot(tangent, _vcross(first_normal, second_normal)),
                    _vdot(first_normal, second_normal),
                )
                # With outward-oriented faces, the directed normal turn is the
                # exterior fold.  The material-side interior wedge is its
                # supplement; this gives pi/2 on a convex cube and 3*pi/2 on
                # a Fichera re-entrant edge.
                opening = PI - turn
                if opening <= 1.0e-12:
                    opening += TAU
                if opening >= TAU - 1.0e-12:
                    opening -= TAU
            edge = PolyhedralEdgeRecord(
                len(edges),
                key,
                tuple(row[0] for row in rows),
                length,
                opening,
                self.material_side_dihedral_certified,
            )
            edges.append(edge)
            vertex_edges[key[0]].add(edge.index)
            vertex_edges[key[1]].add(edge.index)
        if orientation_failures:
            raise ValueError(
                "adjacent triangle orientations are inconsistent across an edge"
            )
        self.edges = tuple(edges)
        self.vertices = tuple(
            PolyhedralVertexRecord(index, vertex_edges[index], vertex_faces[index])
            for index in range(len(self.points))
        )
        self._edge_lookup = {edge.vertices: edge for edge in self.edges}

    @staticmethod
    def _edge_degrees(faces):
        result = {}
        for face in faces:
            for offset in range(3):
                left = face[offset]
                right = face[(offset + 1) % 3]
                key = (min(left, right), max(left, right))
                result[key] = result.get(key, 0) + 1
        return result

    @staticmethod
    def _face_components(faces):
        edge_faces = {}
        for face_index, face in enumerate(faces):
            for offset in range(3):
                left = face[offset]
                right = face[(offset + 1) % 3]
                key = (min(left, right), max(left, right))
                edge_faces.setdefault(key, []).append(face_index)
        adjacency = [set() for _ in faces]
        for incident in edge_faces.values():
            if len(incident) == 2:
                left, right = incident
                adjacency[left].add(right)
                adjacency[right].add(left)
        unseen = set(range(len(faces)))
        components = []
        while unseen:
            seed = min(unseen)
            unseen.remove(seed)
            stack = [seed]
            component = []
            while stack:
                current = stack.pop()
                component.append(current)
                for neighbor in adjacency[current]:
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        stack.append(neighbor)
            components.append(tuple(sorted(component)))
        return tuple(components)

    def _signed_volume(self, faces):
        return sum(
            _vdot(
                self.points[face[0]],
                _vcross(self.points[face[1]], self.points[face[2]]),
            )
            for face in faces
        ) / 6.0

    def edge(self, value):
        if isinstance(value, int):
            return self.edges[value]
        key = tuple(sorted(int(index) for index in value))
        try:
            return self._edge_lookup[key]
        except KeyError as error:
            raise KeyError(f"mesh has no edge {key}") from error

    def stats(self):
        interior = tuple(edge for edge in self.edges if not edge.is_boundary)
        geometric = tuple(edge for edge in interior if edge.is_geometric)
        reentrant = tuple(edge for edge in interior if edge.is_reentrant)
        return {
            "mesh_vertices": len(self.points),
            "mesh_faces": len(self.faces),
            "mesh_edges": len(self.edges),
            "closed": self.closed,
            "connected_components": self.connected_components,
            "signed_volume": self.signed_volume,
            "orientation_reversed": self.orientation_reversed,
            "orientation_reversed_components": self.orientation_reversed_components,
            "material_side_dihedral_certified": (
                self.material_side_dihedral_certified
            ),
            "boundary_edges": sum(edge.is_boundary for edge in self.edges),
            "interior_edges": len(interior),
            "geometric_edges": len(geometric),
            "coplanar_triangulation_seams": len(interior) - len(geometric),
            "reentrant_edges": len(reentrant),
            "minimum_opening_angle": min(
                (edge.opening_angle for edge in interior),
                default=None,
            ),
            "maximum_opening_angle": max(
                (edge.opening_angle for edge in interior),
                default=None,
            ),
            "stored_dense_matrix": False,
            "storage_complexity": "O(V+E+F)",
        }


class EdgeMellinPencil:
    """Dirichlet or nonconstant Neumann pencil on a dihedral cross-section."""

    def __init__(self, opening_angle, mode=1, boundary_condition="dirichlet"):
        self.opening_angle = float(opening_angle)
        self.mode = int(mode)
        self.boundary_condition = str(boundary_condition).lower()
        if self.opening_angle <= 0.0 or self.opening_angle >= TAU:
            raise ValueError("edge opening angle must lie between zero and 2*pi")
        if self.mode < 1:
            raise ValueError("the first nonconstant edge mode is mode one")
        if self.boundary_condition not in ("dirichlet", "neumann"):
            raise ValueError("edge boundary condition must be dirichlet or neumann")
        self.exponent = self.mode * PI / self.opening_angle

    @classmethod
    def from_edge(cls, edge, mode=1, boundary_condition="dirichlet"):
        if not isinstance(edge, PolyhedralEdgeRecord):
            raise TypeError("edge must be a PolyhedralEdgeRecord")
        if edge.opening_angle is None:
            raise ValueError("boundary edges do not define a dihedral pencil")
        if not edge.material_side_certified:
            raise ValueError(
                "automatic material-side edge pencils require a closed mesh; "
                "supply EdgeMellinPencil(opening_angle) explicitly for an open mesh"
            )
        if not edge.is_geometric:
            raise ValueError("a coplanar triangulation seam has no edge pencil")
        return cls(edge.opening_angle, mode, boundary_condition)

    @property
    def boundary_quadrature_exponent(self):
        return self.exponent

    def certificate(self):
        return {
            "kind": "edge",
            "opening_angle": self.opening_angle,
            "mode": self.mode,
            "boundary_condition": self.boundary_condition,
            "kondratiev_exponent": self.exponent,
            "flux_radial_power": self.exponent - 1.0,
            "surface_measure_radial_power": 0,
            "boundary_quadrature_exponent": self.boundary_quadrature_exponent,
        }


class VertexMellinPencil:
    """Scalar Laplace pencil on a spherical vertex link."""

    def __init__(self, angular_eigenvalue, label="vertex"):
        self.angular_eigenvalue = float(angular_eigenvalue)
        self.label = str(label)
        if self.angular_eigenvalue <= 0.0 or not _finite(self.angular_eigenvalue):
            raise ValueError("a Dirichlet spherical-link eigenvalue must be positive")
        self.exponent = 0.5 * (
            _sqrt(1.0 + 4.0 * self.angular_eigenvalue) - 1.0
        )

    @classmethod
    def from_exponent(cls, exponent, label="vertex"):
        value = float(exponent)
        if value <= 0.0 or not _finite(value):
            raise ValueError("a vertex Kondratiev exponent must be positive")
        return cls(value * (value + 1.0), label)

    @classmethod
    def from_spherical_link(cls, link, **solve_options):
        if not isinstance(link, SparseSphericalDirichletPencil):
            raise TypeError("link must be a SparseSphericalDirichletPencil")
        eigenpair = link.solve_lowest(**solve_options)
        result = cls(eigenpair.eigenvalue, "spherical_link")
        result.spherical_link_eigenpair = eigenpair
        return result

    @property
    def boundary_quadrature_exponent(self):
        return self.exponent + 1.0

    @property
    def pencil_residual(self):
        return _abs(
            self.exponent * (self.exponent + 1.0)
            - self.angular_eigenvalue
        )

    def certificate(self):
        result = {
            "kind": "vertex",
            "label": self.label,
            "angular_eigenvalue": self.angular_eigenvalue,
            "kondratiev_exponent": self.exponent,
            "pencil_residual": self.pencil_residual,
            "flux_radial_power": self.exponent - 1.0,
            "surface_measure_radial_power": 1,
            "boundary_quadrature_exponent": self.boundary_quadrature_exponent,
        }
        eigenpair = getattr(self, "spherical_link_eigenpair", None)
        if eigenpair is not None:
            result["spherical_link"] = eigenpair.certificate()
        return result


class SphericalPencilEigenpair:
    def __init__(
        self,
        eigenvalue,
        values,
        residual,
        inverse_iterations,
        cg_iterations,
    ):
        self.eigenvalue = float(eigenvalue)
        self.values = tuple(values)
        self.residual = float(residual)
        self.inverse_iterations = int(inverse_iterations)
        self.cg_iterations = int(cg_iterations)

    @property
    def kondratiev_exponent(self):
        return 0.5 * (_sqrt(1.0 + 4.0 * self.eigenvalue) - 1.0)

    def certificate(self):
        return {
            "angular_eigenvalue": self.eigenvalue,
            "kondratiev_exponent": self.kondratiev_exponent,
            "relative_residual": self.residual,
            "inverse_iterations": self.inverse_iterations,
            "cg_iterations": self.cg_iterations,
            "stored_dense_matrix": False,
        }


class SparseSphericalDirichletPencil:
    """P1 spherical-link pencil assembled and solved in sparse row form."""

    def __init__(self, points, triangles, boundary_nodes):
        self.points = tuple(_unit(_point(value)) for value in points)
        self.triangles = tuple(tuple(int(index) for index in face) for face in triangles)
        self.boundary_nodes = frozenset(int(index) for index in boundary_nodes)
        if not self.triangles:
            raise ValueError("a spherical link requires triangles")
        if any(
            len(face) != 3
            or len(set(face)) != 3
            or any(index < 0 or index >= len(self.points) for index in face)
            for face in self.triangles
        ):
            raise ValueError("invalid spherical-link triangle")
        if any(index < 0 or index >= len(self.points) for index in self.boundary_nodes):
            raise ValueError("spherical-link boundary index is out of range")
        self.interior_nodes = tuple(
            index for index in range(len(self.points)) if index not in self.boundary_nodes
        )
        if not self.interior_nodes:
            raise ValueError("a spherical link needs interior degrees of freedom")
        self._local = {node: index for index, node in enumerate(self.interior_nodes)}
        stiffness = [dict() for _ in self.interior_nodes]
        mass = [dict() for _ in self.interior_nodes]
        for face in self.triangles:
            self._assemble_face(face, stiffness, mass)
        self.stiffness_rows = tuple(
            tuple(sorted(row.items())) for row in stiffness
        )
        self.mass_rows = tuple(tuple(sorted(row.items())) for row in mass)
        self.stiffness_nnz = sum(len(row) for row in self.stiffness_rows)
        self.mass_nnz = sum(len(row) for row in self.mass_rows)

    @staticmethod
    def _accumulate(rows, left, right, value):
        rows[left][right] = rows[left].get(right, 0.0) + value

    def _assemble_face(self, face, stiffness, mass):
        p0, p1, p2 = (self.points[index] for index in face)
        v01 = _vsub(p1, p0)
        v02 = _vsub(p2, p0)
        area_twice = _vnorm(_vcross(v01, v02))
        if area_twice <= 1.0e-30:
            raise ValueError("degenerate spherical-link triangle")
        area = 0.5 * area_twice
        cot0 = _vdot(v01, v02) / area_twice
        cot1 = _vdot(_vsub(p0, p1), _vsub(p2, p1)) / area_twice
        cot2 = _vdot(_vsub(p0, p2), _vsub(p1, p2)) / area_twice
        local_k = (
            (0.5 * (cot1 + cot2), -0.5 * cot2, -0.5 * cot1),
            (-0.5 * cot2, 0.5 * (cot0 + cot2), -0.5 * cot0),
            (-0.5 * cot1, -0.5 * cot0, 0.5 * (cot0 + cot1)),
        )
        local_m = (
            (area / 6.0, area / 12.0, area / 12.0),
            (area / 12.0, area / 6.0, area / 12.0),
            (area / 12.0, area / 12.0, area / 6.0),
        )
        for left_offset, left_global in enumerate(face):
            left = self._local.get(left_global)
            if left is None:
                continue
            for right_offset, right_global in enumerate(face):
                right = self._local.get(right_global)
                if right is None:
                    continue
                self._accumulate(
                    stiffness,
                    left,
                    right,
                    local_k[left_offset][right_offset],
                )
                self._accumulate(
                    mass,
                    left,
                    right,
                    local_m[left_offset][right_offset],
                )

    @staticmethod
    def _matvec(rows, values):
        return tuple(
            sum(coefficient * values[column] for column, coefficient in row)
            for row in rows
        )

    @staticmethod
    def _dot(left, right):
        return sum(a * b for a, b in zip(left, right, strict=True))

    def _mass_norm(self, values):
        applied = self._matvec(self.mass_rows, values)
        return _sqrt(max(self._dot(values, applied), 0.0))

    def _cg(self, rhs, tolerance, maximum_iterations):
        count = len(rhs)
        solution = [0.0 for _ in range(count)]
        residual = list(rhs)
        direction = list(residual)
        rr = self._dot(residual, residual)
        initial = _sqrt(max(rr, 0.0))
        if initial == 0.0:
            return tuple(solution), 0
        target = float(tolerance) * initial
        for iteration in range(1, int(maximum_iterations) + 1):
            applied = self._matvec(self.stiffness_rows, direction)
            denominator = self._dot(direction, applied)
            if denominator <= 0.0:
                raise RuntimeError("spherical stiffness lost positive definiteness")
            alpha = rr / denominator
            for index in range(count):
                solution[index] += alpha * direction[index]
                residual[index] -= alpha * applied[index]
            next_rr = self._dot(residual, residual)
            if _sqrt(max(next_rr, 0.0)) <= target:
                return tuple(solution), iteration
            beta = next_rr / rr
            for index in range(count):
                direction[index] = residual[index] + beta * direction[index]
            rr = next_rr
        return tuple(solution), int(maximum_iterations)

    def solve_lowest(
        self,
        tolerance=2.0e-10,
        maximum_inverse_iterations=120,
        maximum_cg_iterations=None,
    ):
        tolerance = float(tolerance)
        if tolerance <= 0.0:
            raise ValueError("pencil tolerance must be positive")
        count = len(self.interior_nodes)
        cg_limit = (
            max(80, 12 * count)
            if maximum_cg_iterations is None
            else int(maximum_cg_iterations)
        )
        values = tuple(
            1.0 + 0.125 * ((37 * index + 11) % 17) / 17.0
            for index in range(count)
        )
        scale = self._mass_norm(values)
        values = tuple(value / scale for value in values)
        eigenvalue = 0.0
        total_cg = 0
        relative_residual = float("inf")
        for iteration in range(1, int(maximum_inverse_iterations) + 1):
            rhs = self._matvec(self.mass_rows, values)
            candidate, cg_iterations = self._cg(
                rhs,
                min(1.0e-12, 0.02 * tolerance),
                cg_limit,
            )
            total_cg += cg_iterations
            norm = self._mass_norm(candidate)
            if norm <= 1.0e-30:
                raise RuntimeError("inverse iteration produced a zero mode")
            values = tuple(value / norm for value in candidate)
            stiffness_values = self._matvec(self.stiffness_rows, values)
            mass_values = self._matvec(self.mass_rows, values)
            next_eigenvalue = self._dot(values, stiffness_values) / self._dot(
                values,
                mass_values,
            )
            residual = tuple(
                stiffness_values[index]
                - next_eigenvalue * mass_values[index]
                for index in range(count)
            )
            denominator = max(
                _sqrt(self._dot(stiffness_values, stiffness_values)),
                1.0e-300,
            )
            relative_residual = _sqrt(self._dot(residual, residual)) / denominator
            change = _abs(next_eigenvalue - eigenvalue) / max(
                1.0,
                _abs(next_eigenvalue),
            )
            eigenvalue = next_eigenvalue
            if change <= tolerance and relative_residual <= 8.0 * tolerance:
                return SphericalPencilEigenpair(
                    eigenvalue,
                    values,
                    relative_residual,
                    iteration,
                    total_cg,
                )
        return SphericalPencilEigenpair(
            eigenvalue,
            values,
            relative_residual,
            int(maximum_inverse_iterations),
            total_cg,
        )

    def stats(self):
        return {
            "spherical_nodes": len(self.points),
            "spherical_triangles": len(self.triangles),
            "boundary_nodes": len(self.boundary_nodes),
            "interior_dofs": len(self.interior_nodes),
            "stiffness_nnz": self.stiffness_nnz,
            "mass_nnz": self.mass_nnz,
            "stored_dense_matrix": False,
            "assembly_complexity": "O(number of spherical-link triangles)",
            "solve_complexity": "O(iterations * sparse nonzeros)",
        }


class _TaylorJet:
    """Small Taylor algebra in the Mellin exponent, through order three."""

    ORDER = 3

    def __init__(self, coefficients):
        values = tuple(float(value) for value in coefficients)
        if len(values) > self.ORDER + 1:
            raise ValueError("Mellin exponent jets stop at third order")
        self.c = values + (0.0,) * (self.ORDER + 1 - len(values))

    @classmethod
    def constant(cls, value):
        return cls((value,))

    @classmethod
    def variable(cls, value):
        return cls((value, 1.0))

    @staticmethod
    def _coerce(value):
        return value if isinstance(value, _TaylorJet) else _TaylorJet.constant(value)

    def __add__(self, other):
        right = self._coerce(other)
        return _TaylorJet(tuple(self.c[k] + right.c[k] for k in range(4)))

    __radd__ = __add__

    def __neg__(self):
        return _TaylorJet(tuple(-value for value in self.c))

    def __sub__(self, other):
        return self + (-self._coerce(other))

    def __rsub__(self, other):
        return self._coerce(other) - self

    def __mul__(self, other):
        right = self._coerce(other)
        return _TaylorJet(
            tuple(
                sum(self.c[j] * right.c[k - j] for j in range(k + 1))
                for k in range(4)
            )
        )

    __rmul__ = __mul__

    def reciprocal(self):
        if _abs(self.c[0]) <= 1.0e-300:
            raise ZeroDivisionError("Taylor-jet reciprocal has a zero constant")
        output = [1.0 / self.c[0]]
        for order in range(1, 4):
            output.append(
                -sum(self.c[j] * output[order - j] for j in range(1, order + 1))
                / self.c[0]
            )
        return _TaylorJet(output)

    def __truediv__(self, other):
        return self * self._coerce(other).reciprocal()

    def exp(self):
        output = [_exp(self.c[0])]
        for order in range(1, 4):
            output.append(
                sum(
                    index * self.c[index] * output[order - index]
                    for index in range(1, order + 1)
                )
                / order
            )
        return _TaylorJet(output)

    def derivative(self, order):
        index = int(order)
        if index < 0 or index > 3:
            raise ValueError("power-log order must lie between zero and three")
        factorial = (1, 1, 2, 6)[index]
        return factorial * self.c[index]


def _positive_power_jet(base, exponent):
    value = float(base)
    if value <= 0.0 or not _finite(value):
        raise ValueError("Mellin power base must be positive and finite")
    return (exponent * _log(value)).exp()


def _hurwitz_zeta_jet(s, phase, cutoff=40, corrections=7):
    exponent = _TaylorJet._coerce(s)
    beta = float(phase)
    terms = int(cutoff)
    orders = int(corrections)
    if _abs(exponent.c[0] - 1.0) <= 1.0e-14:
        raise ValueError("Hurwitz zeta has a pole at s=1")
    if beta <= 0.0 or not _finite(beta):
        raise ValueError("Hurwitz phase must be positive and finite")
    if terms < 4:
        raise ValueError("Hurwitz cutoff must be at least four")
    if orders < 0 or orders >= len(_BERNOULLI_EVEN_OVER_FACTORIAL):
        raise ValueError("Hurwitz corrections exceed the Bernoulli ladder")
    total = _TaylorJet.constant(0.0)
    for index in range(terms):
        total += _positive_power_jet(beta + index, -exponent)
    endpoint = beta + terms
    total += _positive_power_jet(endpoint, 1.0 - exponent) / (exponent - 1.0)
    total += 0.5 * _positive_power_jet(endpoint, -exponent)

    def correction(order):
        rising = _TaylorJet.constant(1.0)
        for offset in range(2 * order - 1):
            rising *= exponent + offset
        return (
            _BERNOULLI_EVEN_OVER_FACTORIAL[order - 1]
            * rising
            * _positive_power_jet(endpoint, -exponent - 2 * order + 1)
        )

    for order in range(1, orders + 1):
        total += correction(order)
    return total, correction(orders + 1)


def mellin_midpoint_defect(
    exponent,
    step,
    phase=0.5,
    log_order=0,
    cutoff=40,
    corrections=7,
):
    """Return ``d_nu^q[h^nu zeta(1-nu, phase)]`` and an estimate.

    For a localized singular term ``r**(nu-1) (log r)**q``, the punctured
    midpoint sum minus the integral has this leading value.  A repayment
    therefore subtracts the returned defect.
    """

    nu_value = float(exponent)
    h = float(step)
    order = int(log_order)
    if nu_value <= 0.0 or not _finite(nu_value):
        raise ValueError("Mellin quadrature exponent must be positive")
    if h <= 0.0 or not _finite(h):
        raise ValueError("Mellin step must be positive")
    if order < 0 or order > 3:
        raise ValueError("power-log order must lie between zero and three")
    nu = _TaylorJet.variable(nu_value)
    zeta, next_term = _hurwitz_zeta_jet(
        1.0 - nu,
        phase,
        cutoff,
        corrections,
    )
    scale = _positive_power_jet(h, nu)
    defect = (scale * zeta).derivative(order)
    estimate = _abs((scale * next_term).derivative(order))
    return defect, estimate


class MellinThreeJetChannel:
    """One edge or vertex power-log family generated by four coefficients."""

    def __init__(
        self,
        pencil,
        coefficients,
        log_order=0,
        phase=0.5,
        label=None,
    ):
        if not isinstance(pencil, (EdgeMellinPencil, VertexMellinPencil)):
            raise TypeError("pencil must be an edge or vertex Mellin pencil")
        values = tuple(complex(value) for value in coefficients)
        if not values or len(values) > 4:
            raise ValueError("a sparse Mellin three-jet needs one to four coefficients")
        if any(not _finite(value.real) or not _finite(value.imag) for value in values):
            raise ValueError("Mellin three-jet coefficients must be finite")
        self.pencil = pencil
        self.coefficients = values + (0.0j,) * (4 - len(values))
        self.log_order = int(log_order)
        self.phase = float(phase)
        self.label = str(label or pencil.certificate()["kind"])
        if self.log_order < 0 or self.log_order > 3:
            raise ValueError("power-log order must lie between zero and three")
        if self.phase <= 0.0 or not _finite(self.phase):
            raise ValueError("Mellin grid phase must be positive")

    @property
    def base_quadrature_exponent(self):
        return self.pencil.boundary_quadrature_exponent

    def evaluate(self, step, cutoff=40, corrections=7):
        total = 0.0 + 0.0j
        estimate = 0.0
        rungs = []
        for degree, coefficient in enumerate(self.coefficients):
            if coefficient == 0.0:
                continue
            exponent = self.base_quadrature_exponent + degree
            defect, error = mellin_midpoint_defect(
                exponent,
                step,
                self.phase,
                self.log_order,
                cutoff,
                corrections,
            )
            correction = -coefficient * defect
            total += correction
            estimate += _abs(coefficient) * error
            rungs.append(
                {
                    "degree": degree,
                    "quadrature_exponent": exponent,
                    "coefficient": _clean_scalar(coefficient),
                    "midpoint_defect": defect,
                    "correction": _clean_scalar(correction),
                    "hurwitz_evaluator_next_term_estimate": (
                        _abs(coefficient) * error
                    ),
                }
            )
        return {
            "label": self.label,
            "kind": self.pencil.certificate()["kind"],
            "correction": _clean_scalar(total),
            "hurwitz_evaluator_next_term_estimate": estimate,
            "amplitude_jet_remainder_bound": None,
            "step": float(step),
            "phase": self.phase,
            "log_order": self.log_order,
            "rungs": tuple(rungs),
            "pencil": self.pencil.certificate(),
            "stored_dense_matrix": False,
        }

    def stats(self):
        return {
            "label": self.label,
            "kind": self.pencil.certificate()["kind"],
            "stored_coefficients": 4,
            "nonzero_coefficients": sum(value != 0.0 for value in self.coefficients),
            "log_order": self.log_order,
            "grid_refinement_iterations": 0,
            "adaptive_rank": 0,
            "stored_dense_matrix": False,
            "apply_complexity": "O(1) per corner three-jet",
            "full_corner_error_certificate": False,
        }


class CornerRepaymentEvaluation:
    def __init__(self, value, correction, ledger, channels, stats):
        self.value = value
        self.correction = correction
        self.ledger = ledger
        self.channels = channels
        self.stats = stats


class MellinKondratievRepayment:
    """A fixed list of sparse edge/vertex channels with no mesh refinement."""

    def __init__(self, channels):
        self.channels = tuple(channels)
        if not self.channels:
            raise ValueError("at least one edge or vertex channel is required")
        if any(not isinstance(channel, MellinThreeJetChannel) for channel in self.channels):
            raise TypeError("all corner channels must be MellinThreeJetChannel instances")

    def evaluate(self, step, cutoff=40, corrections=7):
        if isinstance(step, dict):
            steps = tuple(float(step[channel.label]) for channel in self.channels)
        else:
            steps = (float(step),) * len(self.channels)
        certificates = tuple(
            channel.evaluate(value, cutoff, corrections)
            for channel, value in zip(self.channels, steps, strict=True)
        )
        correction = _clean_scalar(
            sum(complex(row["correction"]) for row in certificates)
        )
        estimate = sum(
            row["hurwitz_evaluator_next_term_estimate"]
            for row in certificates
        )
        return {
            "correction": correction,
            "hurwitz_evaluator_next_term_estimate": estimate,
            "amplitude_jet_remainder_bound": None,
            "full_corner_error_certificate": False,
            "channels": certificates,
            "channel_count": len(certificates),
            "edge_channels": sum(row["kind"] == "edge" for row in certificates),
            "vertex_channels": sum(row["kind"] == "vertex" for row in certificates),
            "grid_refinement_iterations": 0,
            "adaptive_rank": 0,
            "stored_dense_matrix": False,
            "complexity": "O(number of retained edge/vertex three-jets)",
        }

    def repay(self, borrowed_value, step, cutoff=40, corrections=7):
        result = self.evaluate(step, cutoff, corrections)
        value = _clean_scalar(
            complex(borrowed_value) + complex(result["correction"])
        )
        residual = result["hurwitz_evaluator_next_term_estimate"]
        ledger = BorrowComputeRepayLedger(
            borrowed=(
                "smooth-chart midpoint layer-potential quadrature",
                "polyhedral edge and vertex pencil exponents",
                "four-coefficient local amplitude jets",
            ),
            computed=(
                "differentiated Hurwitz defects for every retained power-log rung",
            ),
            repaid=(
                "dihedral edge flux channels",
                "conical vertex flux channels",
                "surface-measure exponent shift at vertices",
            ),
            residuals=(("hurwitz_evaluator_first_omitted_term", residual),),
            residual_norm=residual,
            status="borrowed_repaid",
            notes=(
                "The corner cost is fixed by the retained local pencils and jets; "
                "no physical-grid refinement or dense corner operator is formed. "
                "The reported residual covers Hurwitz evaluation only; a full "
                "continuum bound also needs the supplied amplitude-jet remainder "
                "and the smooth partition-of-unity quadrature bound."
            ),
        )
        stats = {
            "channel_count": len(self.channels),
            "stored_real_or_complex_coefficients": 4 * len(self.channels),
            "grid_refinement_iterations": 0,
            "adaptive_rank": 0,
            "stored_dense_matrix": False,
            "full_corner_error_certificate": False,
            "apply_complexity": "O(number of retained corner channels)",
            "storage_complexity": "O(number of retained corner channels)",
        }
        return CornerRepaymentEvaluation(
            value,
            result["correction"],
            ledger,
            result["channels"],
            stats,
        )


class CertifiedPolyhedralSurfaceQJet:
    """Topology-preserving graph backend plus explicit continuum corner ledger."""

    def __init__(self, vertices, triangles, corner_channels=(), **surface_options):
        from inverse_shape.arbitrary_surface import (
            CertifiedArbitrarySurfaceQJet,
            triangle_lumped_vertex_weights,
        )

        self.topology = PolyhedralMeshTopology(vertices, triangles)
        weights = triangle_lumped_vertex_weights(
            self.topology.points,
            self.topology.faces,
        )
        self.smooth_backend = CertifiedArbitrarySurfaceQJet(
            self.topology.points,
            weights,
            **surface_options,
        )
        self.corner_channels = tuple(corner_channels)
        if any(
            not isinstance(channel, MellinThreeJetChannel)
            for channel in self.corner_channels
        ):
            raise TypeError("corner_channels must contain MellinThreeJetChannel values")
        self.corner_repayment = (
            MellinKondratievRepayment(self.corner_channels)
            if self.corner_channels
            else None
        )

    def edge_pencil(self, edge, mode=1, boundary_condition="dirichlet"):
        return EdgeMellinPencil.from_edge(
            self.topology.edge(edge),
            mode,
            boundary_condition,
        )

    @staticmethod
    def vertex_pencil_from_exponent(exponent, label="vertex"):
        return VertexMellinPencil.from_exponent(exponent, label)

    def apply(self, values):
        return self.smooth_backend.apply(values)

    def repay_corner_integral(self, borrowed_value, step, **options):
        if self.corner_repayment is None:
            raise ValueError("no edge or vertex corner channels were compiled")
        return self.corner_repayment.repay(borrowed_value, step, **options)

    def stats(self):
        result = self.smooth_backend.stats()
        result.update(self.topology.stats())
        result.update(
            {
                "topology_preserved": True,
                "topology_assumption": "consistently oriented manifold triangle mesh",
                "orientation_required": True,
                "material_side_dihedral_certified": (
                    self.topology.material_side_dihedral_certified
                ),
                "corner_channel_count": len(self.corner_channels),
                "edge_corner_channels": sum(
                    isinstance(channel.pencil, EdgeMellinPencil)
                    for channel in self.corner_channels
                ),
                "vertex_corner_channels": sum(
                    isinstance(channel.pencil, VertexMellinPencil)
                    for channel in self.corner_channels
                ),
                "corner_channel_apply_complexity": "O(number of retained channels)",
                "corner_channel_storage_complexity": "O(number of retained channels)",
                "stored_dense_corner_matrix": False,
            }
        )
        return result


__all__ = [
    "CertifiedPolyhedralSurfaceQJet",
    "CornerRepaymentEvaluation",
    "EdgeMellinPencil",
    "MellinKondratievRepayment",
    "MellinThreeJetChannel",
    "PolyhedralEdgeRecord",
    "PolyhedralMeshTopology",
    "PolyhedralVertexRecord",
    "SparseSphericalDirichletPencil",
    "SphericalPencilEigenpair",
    "VertexMellinPencil",
    "mellin_midpoint_defect",
]
