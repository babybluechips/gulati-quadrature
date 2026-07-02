"""Volumetric P1 FEM comparators for boundary DtN benchmarks.

The Q/DtN engine is matrix-free; this module is only a baseline comparator.
It assembles a sparse P1 stiffness matrix on a radial fan mesh, forms the
boundary Schur-complement DtN operator, and applies boundary operator
functions through the generalized Steklov eigenproblem.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from time import perf_counter
from typing import Callable, Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import splu

from inverse_shape.geometry import as_points, centroid, polygon_area

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


def _timed(fn):
    start = perf_counter()
    value = fn()
    return value, 1000.0 * (perf_counter() - start)


def _point_in_polygon(point: tuple[float, float], polygon: FloatArray) -> bool:
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i in range(len(polygon)):
        xi, yi = float(polygon[i, 0]), float(polygon[i, 1])
        xj, yj = float(polygon[j, 0]), float(polygon[j, 1])
        if (yi > y) != (yj > y):
            x_intersect = (xj - xi) * (y - yi) / (yj - yi + 1.0e-300) + xi
            if x < x_intersect:
                inside = not inside
        j = i
    return inside


def star_fan_center(boundary: ArrayLike) -> FloatArray:
    """Choose an interior center for radial fan meshing."""

    points = as_points(boundary)
    candidates = [
        np.array([0.0, 0.0], dtype=np.float64),
        centroid(points),
        points.mean(axis=0),
    ]
    for candidate in candidates:
        if _point_in_polygon((float(candidate[0]), float(candidate[1])), points):
            return candidate
    raise ValueError("could not find an interior radial-fan center")


def boundary_lumped_mass(boundary: ArrayLike) -> FloatArray:
    """Return diagonal lumped boundary mass weights."""

    points = as_points(boundary)
    previous_lengths = np.linalg.norm(points - np.roll(points, 1, axis=0), axis=1)
    next_lengths = np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)
    mass = 0.5 * (previous_lengths + next_lengths)
    if np.any(mass <= 0.0):
        raise ValueError("boundary contains duplicate adjacent samples")
    return mass.astype(np.float64)


@dataclass(frozen=True)
class StarFanMesh:
    """Radial fan P1 mesh sharing the supplied boundary samples."""

    nodes: FloatArray
    triangles: NDArray[np.int64]
    boundary_nodes: NDArray[np.int64]
    center: FloatArray
    radial_levels: int

    @property
    def node_count(self) -> int:
        return int(len(self.nodes))

    @property
    def triangle_count(self) -> int:
        return int(len(self.triangles))


def build_star_fan_mesh(boundary: ArrayLike, *, radial_levels: int = 32) -> StarFanMesh:
    """Build a conforming radial fan mesh for star-shaped sampled domains."""

    if radial_levels < 2:
        raise ValueError("radial_levels must be at least two")
    points = as_points(boundary)
    if polygon_area(points) < 0.0:
        points = points[::-1].copy()
    center = star_fan_center(points)
    n = len(points)
    nodes: list[tuple[float, float]] = [(float(center[0]), float(center[1]))]
    for level in range(1, radial_levels + 1):
        fraction = level / radial_levels
        ring = center + fraction * (points - center)
        nodes.extend((float(row[0]), float(row[1])) for row in ring)

    def node_index(level: int, boundary_index: int) -> int:
        if level == 0:
            return 0
        return 1 + (level - 1) * n + (boundary_index % n)

    triangles: list[tuple[int, int, int]] = []
    for boundary_index in range(n):
        triangles.append(
            (
                0,
                node_index(1, boundary_index),
                node_index(1, boundary_index + 1),
            )
        )
    for level in range(1, radial_levels):
        for boundary_index in range(n):
            lower_left = node_index(level, boundary_index)
            lower_right = node_index(level, boundary_index + 1)
            upper_left = node_index(level + 1, boundary_index)
            upper_right = node_index(level + 1, boundary_index + 1)
            triangles.append((lower_left, upper_left, upper_right))
            triangles.append((lower_left, upper_right, lower_right))

    boundary_nodes = np.array([node_index(radial_levels, index) for index in range(n)], dtype=np.int64)
    return StarFanMesh(
        nodes=np.asarray(nodes, dtype=np.float64),
        triangles=np.asarray(triangles, dtype=np.int64),
        boundary_nodes=boundary_nodes,
        center=center.astype(np.float64),
        radial_levels=radial_levels,
    )


def assemble_p1_stiffness(mesh: StarFanMesh) -> csr_matrix:
    """Assemble the scalar Laplace P1 stiffness matrix."""

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    skipped = 0
    for triangle in mesh.triangles:
        x = mesh.nodes[triangle, 0]
        y = mesh.nodes[triangle, 1]
        signed_area = 0.5 * ((x[1] - x[0]) * (y[2] - y[0]) - (x[2] - x[0]) * (y[1] - y[0]))
        area = abs(float(signed_area))
        if area <= 1.0e-30:
            skipped += 1
            continue
        b = [float(y[1] - y[2]), float(y[2] - y[0]), float(y[0] - y[1])]
        c = [float(x[2] - x[1]), float(x[0] - x[2]), float(x[1] - x[0])]
        for local_row in range(3):
            for local_col in range(3):
                rows.append(int(triangle[local_row]))
                cols.append(int(triangle[local_col]))
                data.append((b[local_row] * b[local_col] + c[local_row] * c[local_col]) / (4.0 * area))
    if skipped:
        raise ValueError(f"radial fan mesh has {skipped} degenerate triangles")
    return coo_matrix((data, (rows, cols)), shape=(mesh.node_count, mesh.node_count)).tocsr()


@dataclass(frozen=True)
class FEMBoundaryDtN:
    """Boundary Schur-complement DtN baseline from a volumetric P1 mesh."""

    mesh: StarFanMesh
    boundary_mass: FloatArray
    schur: FloatArray
    eigenvalues: FloatArray
    modes: FloatArray
    build_ms: float
    factor_ms: float
    eig_ms: float

    @property
    def boundary_count(self) -> int:
        return int(len(self.boundary_mass))

    def apply_dtn(self, values: Iterable[complex]) -> ComplexArray:
        vector = np.asarray(tuple(values), dtype=np.complex128)
        self._check_vector(vector)
        return (self.schur @ vector) / self.boundary_mass

    def solve_boundary_problem(
        self,
        problem: str,
        values: Iterable[complex],
        **parameters: float,
    ) -> ComplexArray:
        if problem == "laplace_dtn":
            return self.apply_dtn(values)
        if problem == "heat":
            time = float(parameters.get("time", 0.2))
            return self.apply_function(values, lambda lam: math.exp(-time * max(lam, 0.0)))
        if problem == "poisson":
            mass = float(parameters.get("mass", 0.25))
            if mass < 0.0:
                raise ValueError("mass must be non-negative")
            return self.apply_function(
                values,
                lambda lam: 0.0 if lam + mass == 0.0 else 1.0 / (lam + mass),
            )
        if problem == "helmholtz":
            wavenumber = float(parameters.get("wavenumber", 3.7))
            damping = float(parameters.get("damping", 1.0e-3))
            if damping <= 0.0:
                raise ValueError("damping must be positive")
            return self.apply_function(
                values,
                lambda lam: 1.0 / (lam * lam - wavenumber * wavenumber + 1j * damping),
            )
        if problem == "wave":
            time = float(parameters.get("time", 0.8))
            return self.apply_function(values, lambda lam: math.cos(time * math.sqrt(max(lam, 0.0))))
        raise ValueError(f"unknown boundary PDE problem: {problem}")

    def apply_function(
        self,
        values: Iterable[complex],
        function: Callable[[float], complex],
    ) -> ComplexArray:
        vector = np.asarray(tuple(values), dtype=np.complex128)
        self._check_vector(vector)
        sqrt_mass = np.sqrt(self.boundary_mass)
        inv_sqrt_mass = 1.0 / sqrt_mass
        factors = np.asarray([function(float(value)) for value in self.eigenvalues], dtype=np.complex128)
        coefficients = self.modes.T @ (sqrt_mass * vector)
        return inv_sqrt_mass * (self.modes @ (factors * coefficients))

    def _check_vector(self, vector: ComplexArray) -> None:
        if vector.ndim != 1 or len(vector) != self.boundary_count:
            raise ValueError("values length must match FEM boundary count")


def build_fem_boundary_dtn(boundary: ArrayLike, *, radial_levels: int = 32) -> FEMBoundaryDtN:
    """Assemble a radial-fan P1 FEM boundary DtN baseline."""

    mesh, mesh_ms = _timed(lambda: build_star_fan_mesh(boundary, radial_levels=radial_levels))
    stiffness = assemble_p1_stiffness(mesh)
    boundary_nodes = mesh.boundary_nodes
    boundary_mask = np.zeros(mesh.node_count, dtype=bool)
    boundary_mask[boundary_nodes] = True
    interior_nodes = np.where(~boundary_mask)[0]
    if len(interior_nodes) == 0:
        raise ValueError("mesh has no interior nodes")

    def factor_and_schur() -> tuple[FloatArray, float]:
        interior_stiffness = stiffness[interior_nodes][:, interior_nodes].tocsc()
        interior_boundary = stiffness[interior_nodes][:, boundary_nodes]
        boundary_interior = stiffness[boundary_nodes][:, interior_nodes]
        boundary_boundary = stiffness[boundary_nodes][:, boundary_nodes]
        factor_start = perf_counter()
        factor = splu(interior_stiffness)
        factor_ms = 1000.0 * (perf_counter() - factor_start)
        solved = factor.solve(interior_boundary.toarray())
        schur = boundary_boundary.toarray() - boundary_interior.toarray() @ solved
        return 0.5 * (schur + schur.T), factor_ms

    (schur, factor_ms), schur_ms = _timed(factor_and_schur)
    mass = boundary_lumped_mass(boundary)

    def eigensolve() -> tuple[FloatArray, FloatArray]:
        sqrt_mass = np.sqrt(mass)
        scaled = schur / sqrt_mass[:, None] / sqrt_mass[None, :]
        scaled = 0.5 * (scaled + scaled.T)
        eigenvalues, modes = np.linalg.eigh(scaled)
        eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
        eigenvalues[np.abs(eigenvalues) < 1.0e-10] = 0.0
        eigenvalues = np.maximum(eigenvalues, 0.0)
        return eigenvalues, np.asarray(modes, dtype=np.float64)

    (eigenvalues, modes), eig_ms = _timed(eigensolve)
    return FEMBoundaryDtN(
        mesh=mesh,
        boundary_mass=mass,
        schur=schur.astype(np.float64),
        eigenvalues=eigenvalues,
        modes=modes,
        build_ms=mesh_ms + schur_ms + eig_ms,
        factor_ms=factor_ms,
        eig_ms=eig_ms,
    )


def relative_weighted_l2(
    value: Iterable[complex],
    reference: Iterable[complex],
    mass: Iterable[float],
) -> float:
    """Return mass-weighted relative L2 error."""

    lhs = np.asarray(tuple(value), dtype=np.complex128)
    rhs = np.asarray(tuple(reference), dtype=np.complex128)
    weights = np.asarray(tuple(mass), dtype=np.float64)
    diff = lhs - rhs
    numerator = math.sqrt(float(np.sum(weights * np.abs(diff) ** 2)))
    denominator = max(math.sqrt(float(np.sum(weights * np.abs(rhs) ** 2))), 1.0e-14)
    return numerator / denominator


def relative_inf(value: Iterable[complex], reference: Iterable[complex]) -> float:
    """Return relative infinity-norm error."""

    lhs = np.asarray(tuple(value), dtype=np.complex128)
    rhs = np.asarray(tuple(reference), dtype=np.complex128)
    return float(np.max(np.abs(lhs - rhs)) / max(float(np.max(np.abs(rhs))), 1.0e-14))


def best_weighted_scalar(
    value: Iterable[complex],
    reference: Iterable[complex],
    mass: Iterable[float],
) -> complex:
    """Return scalar alpha minimizing ||alpha * value - reference||_M."""

    lhs = np.asarray(tuple(value), dtype=np.complex128)
    rhs = np.asarray(tuple(reference), dtype=np.complex128)
    weights = np.asarray(tuple(mass), dtype=np.float64)
    denominator = np.sum(weights * np.conjugate(lhs) * lhs)
    if abs(denominator) <= 1.0e-30:
        return 0.0 + 0.0j
    return complex(np.sum(weights * np.conjugate(lhs) * rhs) / denominator)


__all__ = [
    "FEMBoundaryDtN",
    "StarFanMesh",
    "assemble_p1_stiffness",
    "best_weighted_scalar",
    "boundary_lumped_mass",
    "build_fem_boundary_dtn",
    "build_star_fan_mesh",
    "relative_inf",
    "relative_weighted_l2",
    "star_fan_center",
]
