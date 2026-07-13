"""Public production API for matrix-free three-dimensional QJets.

The discrete surface operator is

    (Q_p f)_i = sum_{j != i} w_j (f_i - f_j) / |X_i - X_j|**p.

``p=2`` is the scale-invariant inverse-square discriminant operator.  For a
two-dimensional surface in three dimensions, ``Q_3 / (2*pi)`` has the local
principal normalization of the Laplace Dirichlet-to-Neumann map.  The backend
stores a fair-split tree and fixed-order Cartesian source jets.  It raises
``NearLinearContractError`` if a compiled work budget is exceeded; it does not
fall back to a dense or quadratic production path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from inverse_shape.arbitrary_surface import triangle_lumped_vertex_weights
from inverse_shape.axisymmetric3d import (
    AxisymmetricSurfaceQJet,
    conic_qjet as _conic_qjet,
    radial_profile_qjet as _radial_profile_qjet,
    spheroid_qjet as _spheroid_qjet,
    torus_qjet as _torus_qjet,
)
from inverse_shape.conic_pencil_surface import ConicPencilSurfaceQJet
from inverse_shape.polyhedral_kondratiev import (
    CertifiedPolyhedralSurfaceQJet,
    EdgeMellinPencil,
    MellinKondratievRepayment,
    MellinThreeJetChannel,
    PolyhedralMeshTopology,
    VertexMellinPencil,
)
from inverse_shape.quadrature import PI
from inverse_shape.reversible_cad_qjet import (
    CadArchiveError,
    ExactMesh,
    MeshPart,
    RoundTripAudit,
    archive_header,
    audit_round_trip,
    decode_mesh,
    encode_mesh,
    load_binary_stl_assembly,
    load_ifc_tessellation,
    write_archive,
)
from inverse_shape.riesz_near_linear import (
    NearLinearContractError,
    NearLinearRieszEvaluation,
    ProductionRieszQJet,
)


Point3 = tuple[float, float, float]


@dataclass(frozen=True)
class SurfaceQConfig:
    """Fixed production parameters for a three-dimensional surface QJet."""

    kernel_power: float = 3.0
    tolerance: float = 5.0e-13
    maximum_order: int = 16
    leaf_size: int = 8
    work_budget_factor: int = 96


@dataclass(frozen=True)
class SurfaceQEvaluation:
    """Normalized surface result with the backend certificate attached."""

    values: tuple[complex | float, ...]
    compression_inf_bound: float
    ledger: object
    stats: dict[str, object]


class ProductionSurfaceQEngine:
    """Public wrapper around the hard-no-quadratic surface QJet."""

    def __init__(
        self,
        points: Iterable[Iterable[float]],
        weights: Iterable[float],
        *,
        config: SurfaceQConfig | None = None,
    ) -> None:
        self.config = config or SurfaceQConfig()
        self._qjet = ProductionRieszQJet(
            tuple(tuple(float(component) for component in point) for point in points),
            tuple(float(weight) for weight in weights),
            kernel_power=self.config.kernel_power,
            tolerance=self.config.tolerance,
            maximum_order=self.config.maximum_order,
            leaf_size=self.config.leaf_size,
            work_budget_factor=self.config.work_budget_factor,
        )

    @classmethod
    def from_triangle_mesh(
        cls,
        vertices: Iterable[Iterable[float]],
        triangles: Iterable[Iterable[int]],
        *,
        config: SurfaceQConfig | None = None,
    ) -> "ProductionSurfaceQEngine":
        points = tuple(tuple(float(value) for value in vertex) for vertex in vertices)
        faces = tuple(tuple(int(index) for index in face) for face in triangles)
        return cls(
            points,
            triangle_lumped_vertex_weights(points, faces),
            config=config,
        )

    @property
    def n(self) -> int:
        return self._qjet.n

    @property
    def points(self) -> tuple[Point3, ...]:
        return self._qjet.points

    @property
    def weights(self) -> tuple[float, ...]:
        return self._qjet.weights

    def apply(self, values: Iterable[complex]) -> NearLinearRieszEvaluation:
        """Apply the configured inverse-distance graph and return its ledger."""

        return self._qjet.evaluate(tuple(values))

    def apply_dtn_principal(self, values: Iterable[complex]) -> SurfaceQEvaluation:
        """Apply ``Q_3/(2*pi)``, the normalized local DtN principal operator."""

        if self.config.kernel_power != 3.0:
            raise ValueError("the DtN principal normalization requires kernel_power=3")
        raw = self.apply(values)
        scale = 1.0 / (2.0 * PI)
        stats = dict(raw.stats)
        stats.update(
            {
                "public_api": "gulati_quadrature.ProductionSurfaceQEngine",
                "continuum_principal_symbol": "|xi|_g",
                "dtn_normalization": "Q_3 / (2*pi)",
            }
        )
        return SurfaceQEvaluation(
            values=tuple(scale * complex(value) for value in raw.values),
            compression_inf_bound=scale * raw.compression_inf_bound,
            ledger=raw.ledger,
            stats=stats,
        )

    def stats(self) -> dict[str, object]:
        """Return implementation, complexity, and memory-contract metadata."""

        result = dict(self._qjet.stats())
        result.update(
            {
                "public_api": "gulati_quadrature.ProductionSurfaceQEngine",
                "dense_q_matrix_stored": False,
                "pair_table_stored": False,
                "kernel_power": self.config.kernel_power,
            }
        )
        return result


def build_surface_engine(
    points: Iterable[Iterable[float]],
    weights: Iterable[float],
    *,
    config: SurfaceQConfig | None = None,
) -> ProductionSurfaceQEngine:
    """Build a production QJet from weighted surface nodes."""

    return ProductionSurfaceQEngine(points, weights, config=config)


def build_mesh_engine(
    vertices: Iterable[Iterable[float]],
    triangles: Iterable[Iterable[int]],
    *,
    config: SurfaceQConfig | None = None,
) -> ProductionSurfaceQEngine:
    """Build a production QJet using lumped triangle-area weights."""

    return ProductionSurfaceQEngine.from_triangle_mesh(
        vertices,
        triangles,
        config=config,
    )


def build_axisymmetric_engine(
    geometry: AxisymmetricSurfaceQJet,
    *,
    config: SurfaceQConfig | None = None,
) -> ProductionSurfaceQEngine:
    """Lower an axisymmetric geometry generator into the production backend."""

    points = tuple(
        geometry.cartesian_point(ring, phase)
        for ring in range(geometry.n_rings)
        for phase in range(geometry.n_theta)
    )
    weights = tuple(
        geometry.node_area_weights[ring]
        for ring in range(geometry.n_rings)
        for _phase in range(geometry.n_theta)
    )
    return build_surface_engine(points, weights, config=config)


def build_spheroid_engine(
    equatorial_radius: float,
    polar_radius: float,
    n_meridian: int,
    n_theta: int,
    *,
    config: SurfaceQConfig | None = None,
) -> ProductionSurfaceQEngine:
    """Build a production QJet for a sphere or spheroid."""

    cfg = config or SurfaceQConfig()
    geometry = _spheroid_qjet(
        equatorial_radius,
        polar_radius,
        n_meridian,
        n_theta,
        kernel_power=cfg.kernel_power,
    )
    return build_axisymmetric_engine(geometry, config=cfg)


def build_torus_engine(
    major_radius: float,
    minor_radius: float,
    n_meridian: int,
    n_theta: int,
    *,
    config: SurfaceQConfig | None = None,
) -> ProductionSurfaceQEngine:
    """Build a production QJet for a genus-one torus."""

    cfg = config or SurfaceQConfig()
    geometry = _torus_qjet(
        major_radius,
        minor_radius,
        n_meridian,
        n_theta,
        kernel_power=cfg.kernel_power,
    )
    return build_axisymmetric_engine(geometry, config=cfg)


def build_radial_profile_engine(
    base_radius: float,
    cosine_coefficients: Iterable[float],
    n_meridian: int,
    n_theta: int,
    *,
    config: SurfaceQConfig | None = None,
) -> ProductionSurfaceQEngine:
    """Build a production QJet for a smooth axisymmetric radial profile."""

    cfg = config or SurfaceQConfig()
    geometry = _radial_profile_qjet(
        base_radius,
        tuple(cosine_coefficients),
        n_meridian,
        n_theta,
        kernel_power=cfg.kernel_power,
    )
    return build_axisymmetric_engine(geometry, config=cfg)


def build_axisymmetric_conic_engine(
    radius_start: float,
    radius_stop: float,
    z_start: float,
    z_stop: float,
    n_meridian: int,
    n_theta: int,
    *,
    config: SurfaceQConfig | None = None,
) -> ProductionSurfaceQEngine:
    """Build a production QJet for an open cylindrical or conic surface."""

    cfg = config or SurfaceQConfig()
    geometry = _conic_qjet(
        radius_start,
        radius_stop,
        z_start,
        z_stop,
        n_meridian,
        n_theta,
        kernel_power=cfg.kernel_power,
    )
    return build_axisymmetric_engine(geometry, config=cfg)


def build_conic_pencil_engine(
    geometry: ConicPencilSurfaceQJet,
    *,
    config: SurfaceQConfig | None = None,
) -> ProductionSurfaceQEngine:
    """Lower a sparse moving-conic atlas into the production Riesz backend."""

    nodes = geometry.generate_nodes()
    return build_surface_engine(nodes.points, nodes.weights, config=config)


def build_polyhedral_engine(
    vertices: Iterable[Iterable[float]],
    triangles: Iterable[Iterable[int]],
    *,
    corner_channels: Iterable[MellinThreeJetChannel] = (),
    config: SurfaceQConfig | None = None,
) -> CertifiedPolyhedralSurfaceQJet:
    """Build the graph backend plus explicit Mellin-Kondratiev corner channels."""

    cfg = config or SurfaceQConfig()
    return CertifiedPolyhedralSurfaceQJet(
        tuple(vertices),
        tuple(triangles),
        corner_channels=tuple(corner_channels),
        kernel_power=cfg.kernel_power,
        tolerance=cfg.tolerance,
        maximum_order=cfg.maximum_order,
        leaf_size=cfg.leaf_size,
        work_budget_factor=cfg.work_budget_factor,
    )


__all__ = [
    "CadArchiveError",
    "CertifiedPolyhedralSurfaceQJet",
    "EdgeMellinPencil",
    "ExactMesh",
    "MellinKondratievRepayment",
    "MellinThreeJetChannel",
    "MeshPart",
    "NearLinearContractError",
    "PolyhedralMeshTopology",
    "ProductionRieszQJet",
    "ProductionSurfaceQEngine",
    "RoundTripAudit",
    "SurfaceQConfig",
    "SurfaceQEvaluation",
    "VertexMellinPencil",
    "archive_header",
    "audit_round_trip",
    "build_axisymmetric_conic_engine",
    "build_axisymmetric_engine",
    "build_conic_pencil_engine",
    "build_mesh_engine",
    "build_polyhedral_engine",
    "build_radial_profile_engine",
    "build_spheroid_engine",
    "build_surface_engine",
    "build_torus_engine",
    "decode_mesh",
    "encode_mesh",
    "load_binary_stl_assembly",
    "load_ifc_tessellation",
    "triangle_lumped_vertex_weights",
    "write_archive",
]
