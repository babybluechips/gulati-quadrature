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
from typing import TYPE_CHECKING, Iterable

from inverse_shape.arbitrary_surface import triangle_lumped_vertex_weights
from inverse_shape.axisymmetric3d import (
    AxisymmetricSurfaceQJet,
    conic_qjet as _conic_qjet,
    radial_profile_qjet as _radial_profile_qjet,
    spheroid_qjet as _spheroid_qjet,
    torus_qjet as _torus_qjet,
)
from inverse_shape.cad_surface_compiler import (
    CompiledCadPanelSurface,
    CompiledCadSurface,
    compile_cad_panels,
    compile_cad_surface,
    load_compiled_cad_panels,
    load_compiled_cad_surface,
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
from inverse_shape.quadrature import BorrowComputeRepayLedger, PI, _cos, _sin
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
from inverse_shape.surface_repayment import (
    HarmonicMomentRepayment3D,
    HelmholtzMomentRepayment3D,
    MeshDifferentialGeometry,
)

if TYPE_CHECKING:
    from gulati_quadrature.surface_pde import SurfacePDEResult


Point3 = tuple[float, float, float]
_HELMHOLTZ_DIRECTIONS: tuple[Point3, ...] = (
    (1.0, 0.0, 0.0),
    (-1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, 0.0, -1.0),
    (1.0, 1.0, 1.0),
    (1.0, 1.0, -1.0),
    (1.0, -1.0, 1.0),
    (-1.0, 1.0, 1.0),
)


@dataclass(frozen=True)
class SurfaceQConfig:
    """Fixed production parameters for a three-dimensional surface QJet."""

    kernel_power: float = 3.0
    tolerance: float = 5.0e-13
    maximum_order: int = 16
    leaf_size: int = 8
    work_budget_factor: int = 96
    continuum_repayment: bool = True
    singular_cell_order: int = 3
    harmonic_moment_degree: int = 3
    self_adjoint_moment_repayment: bool = False
    orthogonalization_tolerance: float = 1.0e-11

    def __post_init__(self) -> None:
        if self.singular_cell_order < 0 or self.singular_cell_order > 3:
            raise ValueError("singular_cell_order must lie between zero and three")
        if self.harmonic_moment_degree < 0:
            raise ValueError("harmonic_moment_degree must be nonnegative")
        if self.orthogonalization_tolerance <= 0.0:
            raise ValueError("orthogonalization_tolerance must be positive")


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
        normals: Iterable[Iterable[float]] | None = None,
        faces: Iterable[Iterable[int]] | None = None,
        config: SurfaceQConfig | None = None,
    ) -> None:
        self.config = config or SurfaceQConfig()
        point_rows = tuple(
            tuple(float(component) for component in point) for point in points
        )
        weight_rows = tuple(float(weight) for weight in weights)
        self._qjet = ProductionRieszQJet(
            point_rows,
            weight_rows,
            kernel_power=self.config.kernel_power,
            tolerance=self.config.tolerance,
            maximum_order=self.config.maximum_order,
            leaf_size=self.config.leaf_size,
            work_budget_factor=self.config.work_budget_factor,
        )
        self._mesh_geometry = None
        if faces is not None:
            self._mesh_geometry = MeshDifferentialGeometry(
                point_rows,
                tuple(tuple(int(index) for index in face) for face in faces),
                weight_rows,
                normals=normals,
            )
        if normals is not None:
            self._normals = tuple(
                tuple(float(component) for component in normal) for normal in normals
            )
        elif self._mesh_geometry is not None:
            self._normals = self._mesh_geometry.normals
        else:
            self._normals = None
        if self._normals is not None and len(self._normals) != len(point_rows):
            raise ValueError("normals must contain one vector per surface node")
        self._harmonic_repayment: HarmonicMomentRepayment3D | None = None
        self._harmonic_repayment_compiled = False
        self._helmholtz_repayments: dict[
            tuple[float, tuple[Point3, ...]], HelmholtzMomentRepayment3D
        ] = {}
        self._maximum_compilation_compression_bound = 0.0

    @classmethod
    def from_triangle_mesh(
        cls,
        vertices: Iterable[Iterable[float]],
        triangles: Iterable[Iterable[int]],
        *,
        normals: Iterable[Iterable[float]] | None = None,
        config: SurfaceQConfig | None = None,
    ) -> "ProductionSurfaceQEngine":
        points = tuple(tuple(float(value) for value in vertex) for vertex in vertices)
        faces = tuple(tuple(int(index) for index in face) for face in triangles)
        return cls(
            points,
            triangle_lumped_vertex_weights(points, faces),
            normals=normals,
            faces=faces,
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

    def _base_dtn_values(
        self,
        values: tuple[complex, ...],
        raw: NearLinearRieszEvaluation | None = None,
    ) -> tuple[tuple[complex, ...], NearLinearRieszEvaluation]:
        evaluation = raw or self.apply(values)
        scale = 1.0 / (2.0 * PI)
        output = tuple(scale * complex(value) for value in evaluation.values)
        if (
            self.config.continuum_repayment
            and self.config.singular_cell_order > 0
            and self._mesh_geometry is not None
            and not bool(evaluation.stats.get("constant_shortcut", False))
        ):
            correction = self._mesh_geometry.singular_cell_correction(
                values,
                order=self.config.singular_cell_order,
            )
            output = tuple(
                value + delta
                for value, delta in zip(output, correction, strict=True)
            )
        return output, evaluation

    def _effective_harmonic_degree(self) -> int:
        requested = self.config.harmonic_moment_degree
        degree = 0
        while (degree + 1) * (degree + 3) <= self.n - 1:
            degree += 1
        return min(requested, degree)

    def _ensure_harmonic_repayment(self) -> HarmonicMomentRepayment3D | None:
        if self._harmonic_repayment_compiled:
            return self._harmonic_repayment
        self._harmonic_repayment_compiled = True
        degree = self._effective_harmonic_degree()
        if (
            not self.config.continuum_repayment
            or degree == 0
            or self._normals is None
        ):
            return None

        def base_apply(values: Iterable[complex]) -> tuple[complex, ...]:
            vector = tuple(complex(value) for value in values)
            output, evaluation = self._base_dtn_values(vector)
            self._maximum_compilation_compression_bound = max(
                self._maximum_compilation_compression_bound,
                float(evaluation.compression_inf_bound),
            )
            return output

        while degree > 0:
            try:
                self._harmonic_repayment = HarmonicMomentRepayment3D(
                    self.points,
                    self.weights,
                    self._normals,
                    base_apply,
                    degree=degree,
                    self_adjoint=self.config.self_adjoint_moment_repayment,
                    orthogonalization_tolerance=(
                        self.config.orthogonalization_tolerance
                    ),
                )
                break
            except RuntimeError as error:
                if "lost an expected trace mode" not in str(error):
                    raise
                degree -= 1
        return self._harmonic_repayment

    @property
    def dtn_self_adjoint(self) -> bool:
        """Whether the active discrete DtN path is weighted self-adjoint."""

        has_one_sided_moments = (
            self.config.continuum_repayment
            and self._effective_harmonic_degree() > 0
            and self._normals is not None
            and not self.config.self_adjoint_moment_repayment
        )
        return not has_one_sided_moments

    def apply_dtn_principal(self, values: Iterable[complex]) -> SurfaceQEvaluation:
        """Apply the normalized Q3 principal operator and continuum repayment."""

        if self.config.kernel_power != 3.0:
            raise ValueError("the DtN principal normalization requires kernel_power=3")
        vector = tuple(complex(value) for value in values)
        raw = self.apply(vector)
        scale = 1.0 / (2.0 * PI)
        output, _evaluation = self._base_dtn_values(vector, raw)
        repayment = None
        if not bool(raw.stats.get("constant_shortcut", False)):
            repayment = self._ensure_harmonic_repayment()
        if repayment is not None:
            output = repayment.apply(vector, output)
        stats = dict(raw.stats)
        stats.update(
            {
                "public_api": "gulati_quadrature.ProductionSurfaceQEngine",
                "continuum_principal_symbol": "|xi|_g",
                "dtn_normalization": "Q_3 / (2*pi)",
                "continuum_repayment": bool(self.config.continuum_repayment),
                "singular_cell_order": self.config.singular_cell_order
                if self._mesh_geometry is not None
                else 0,
                "harmonic_moment_degree": repayment.degree
                if repayment is not None
                else 0,
                "dtn_self_adjoint": self.dtn_self_adjoint,
                "maximum_compilation_compression_bound": (
                    self._maximum_compilation_compression_bound
                ),
            }
        )
        if self._mesh_geometry is not None:
            stats.update(self._mesh_geometry.stats())
        if repayment is not None:
            stats.update(repayment.stats())
        ledger = BorrowComputeRepayLedger(
            borrowed=(
                "fixed-order Q3 generating jets",
                "surface measure and geometry jets",
            ),
            computed=(
                "matrix-free inverse-cube graph application",
                "normalized boundary principal operator",
            ),
            repaid=(
                "1/(2*pi) principal-symbol normalization",
                "curvature-adjusted singular-cell Laplace--Beltrami series"
                if self._mesh_geometry is not None
                and self.config.singular_cell_order > 0
                else "no topology-aware singular-cell channel available",
                "fixed-rank solid-harmonic normal-flux moments"
                if repayment is not None
                else "no harmonic normal-flux channel available",
            ),
            residuals=(
                ("qjet_compression_inf_bound", raw.compression_inf_bound),
                (
                    "qjet_compilation_compression_inf_bound",
                    self._maximum_compilation_compression_bound,
                ),
            ),
            residual_norm=max(
                raw.compression_inf_bound,
                self._maximum_compilation_compression_bound,
            ),
            status="borrowed_repaid",
            notes=(
                "Compression and manufactured-moment residuals do not bound "
                "held-out continuum discretization error."
            ),
        )
        return SurfaceQEvaluation(
            values=output,
            compression_inf_bound=scale * raw.compression_inf_bound,
            ledger=ledger,
            stats=stats,
        )

    def apply_helmholtz_dtn(
        self,
        values: Iterable[complex],
        *,
        wavenumber: float,
        directions: Iterable[Iterable[float]] | None = None,
    ) -> SurfaceQEvaluation:
        """Apply the frequency-dependent DtN QJet with plane-wave repayment."""

        if self._normals is None:
            raise ValueError("Helmholtz DtN repayment requires surface normals")
        wave_number = float(wavenumber)
        if wave_number <= 0.0:
            raise ValueError("Helmholtz DtN requires a positive wavenumber")
        direction_rows = tuple(
            tuple(float(component) for component in direction)
            for direction in (directions or _HELMHOLTZ_DIRECTIONS)
        )
        key = (wave_number, direction_rows)
        repayment = self._helmholtz_repayments.get(key)
        if repayment is None:

            def base_apply(row: Iterable[complex]) -> tuple[complex, ...]:
                return tuple(self.apply_dtn_principal(row).values)

            repayment = HelmholtzMomentRepayment3D(
                self.points,
                self.weights,
                self._normals,
                base_apply,
                wavenumber=wave_number,
                directions=direction_rows,
                orthogonalization_tolerance=(
                    self.config.orthogonalization_tolerance
                ),
            )
            self._helmholtz_repayments[key] = repayment
        vector = tuple(complex(value) for value in values)
        base = self.apply_dtn_principal(vector)
        output = repayment.apply(vector, base.values)
        stats = dict(base.stats)
        stats.update(repayment.stats())
        stats.update(
            {
                "operator_role": "interior Helmholtz DtN with fixed Herglotz moments",
                "helmholtz_direction_count": len(direction_rows),
            }
        )
        ledger = BorrowComputeRepayLedger(
            borrowed=(
                "continuum-repaid Laplace DtN QJet",
                "fixed plane-wave trace QJets",
            ),
            computed=("matrix-free frequency-dependent boundary flux",),
            repaid=(
                "exact plane-wave normal-flux channels at the requested wavenumber",
            ),
            residuals=(
                ("qjet_compression_inf_bound", base.compression_inf_bound),
            ),
            residual_norm=base.compression_inf_bound,
            status="borrowed_repaid",
            notes=(
                "Exact reproduction is certified on the retained plane-wave "
                "subspace; held-out continuum error is a separate quantity."
            ),
        )
        return SurfaceQEvaluation(
            values=output,
            compression_inf_bound=base.compression_inf_bound,
            ledger=ledger,
            stats=stats,
        )

    def harmonic_modal_basis(
        self,
    ) -> tuple[tuple[tuple[float, ...], tuple[complex, ...]], ...]:
        """Return retained trace/flux QJets for fixed-rank direct solves."""

        repayment = self._ensure_harmonic_repayment()
        if repayment is None:
            return tuple()
        return tuple(
            (mode.trace, mode.target_flux) for mode in repayment.modes
        )

    def solve(
        self,
        problem: str,
        values: Iterable[complex],
        **parameters: object,
    ) -> SurfacePDEResult:
        """Solve a matrix-free boundary PDE generated by ``Q_3/(2*pi)``.

        The import is local to keep the surface-operator module independent of
        the higher-level PDE facade.
        """

        from gulati_quadrature.surface_pde import solve_surface_pde

        return solve_surface_pde(self, values, problem, **parameters)

    def stats(self) -> dict[str, object]:
        """Return implementation, complexity, and memory-contract metadata."""

        result = dict(self._qjet.stats())
        result.update(
            {
                "public_api": "gulati_quadrature.ProductionSurfaceQEngine",
                "dense_q_matrix_stored": False,
                "pair_table_stored": False,
                "kernel_power": self.config.kernel_power,
                "continuum_repayment": self.config.continuum_repayment,
                "singular_cell_order": self.config.singular_cell_order
                if self._mesh_geometry is not None
                else 0,
                "harmonic_moment_degree": self._effective_harmonic_degree()
                if self._normals is not None
                else 0,
                "self_adjoint_moment_repayment": (
                    self.config.self_adjoint_moment_repayment
                ),
                "harmonic_repayment_compiled": self._harmonic_repayment_compiled,
                "helmholtz_repayment_cache_entries": len(
                    self._helmholtz_repayments
                ),
                "dtn_self_adjoint": self.dtn_self_adjoint,
            }
        )
        if self._mesh_geometry is not None:
            result.update(self._mesh_geometry.stats())
        if self._harmonic_repayment is not None:
            result.update(self._harmonic_repayment.stats())
        return result


def build_surface_engine(
    points: Iterable[Iterable[float]],
    weights: Iterable[float],
    *,
    normals: Iterable[Iterable[float]] | None = None,
    config: SurfaceQConfig | None = None,
) -> ProductionSurfaceQEngine:
    """Build a production QJet from weighted surface nodes."""

    return ProductionSurfaceQEngine(points, weights, normals=normals, config=config)


def build_mesh_engine(
    vertices: Iterable[Iterable[float]],
    triangles: Iterable[Iterable[int]],
    *,
    normals: Iterable[Iterable[float]] | None = None,
    config: SurfaceQConfig | None = None,
) -> ProductionSurfaceQEngine:
    """Build a production QJet using lumped triangle-area weights."""

    return ProductionSurfaceQEngine.from_triangle_mesh(
        vertices,
        triangles,
        normals=normals,
        config=config,
    )


def build_compiled_cad_engine(
    surface: CompiledCadSurface | CompiledCadPanelSurface,
    *,
    config: SurfaceQConfig | None = None,
) -> ProductionSurfaceQEngine:
    """Build a production engine from a complete-mesh CAD audit compiler."""

    if isinstance(surface, CompiledCadSurface):
        return ProductionSurfaceQEngine(
            surface.vertices,
            surface.weights,
            normals=surface.normals,
            faces=surface.faces,
            config=config,
        )
    if isinstance(surface, CompiledCadPanelSurface):
        return build_surface_engine(
            surface.points,
            surface.weights,
            normals=surface.normals,
            config=config,
        )
    raise TypeError("surface must be a compiled CAD mesh or panel atlas")


def _axisymmetric_nodes(
    geometry: AxisymmetricSurfaceQJet,
) -> tuple[tuple[Point3, ...], tuple[float, ...]]:
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
    return points, weights


def _revolve_profile_normals(
    geometry: AxisymmetricSurfaceQJet,
    profile: Iterable[tuple[float, float]],
) -> tuple[Point3, ...]:
    rows = tuple((float(radial), float(vertical)) for radial, vertical in profile)
    if len(rows) != geometry.n_rings:
        raise ValueError("profile normals must match the meridional ring count")
    output = []
    for radial, vertical in rows:
        length = (radial * radial + vertical * vertical) ** 0.5
        if length <= 1.0e-30:
            raise ValueError("axisymmetric profile normal cannot vanish")
        radial /= length
        vertical /= length
        for phase in range(geometry.n_theta):
            angle = geometry.theta(phase)
            output.append((radial * _cos(angle), radial * _sin(angle), vertical))
    return tuple(output)


def build_axisymmetric_engine(
    geometry: AxisymmetricSurfaceQJet,
    *,
    config: SurfaceQConfig | None = None,
) -> ProductionSurfaceQEngine:
    """Lower an axisymmetric geometry generator into the production backend."""

    points, weights = _axisymmetric_nodes(geometry)
    profile_area = 0.0
    for index in range(geometry.n_rings):
        following = (index + 1) % geometry.n_rings
        if not geometry.meridian_periodic and index + 1 == geometry.n_rings:
            break
        profile_area += (
            geometry.radii[index] * geometry.z_values[following]
            - geometry.radii[following] * geometry.z_values[index]
        )
    orientation = 1.0 if profile_area >= 0.0 else -1.0
    profile_normals = []
    for ring in range(geometry.n_rings):
        if geometry.meridian_periodic:
            previous = (ring - 1) % geometry.n_rings
            following = (ring + 1) % geometry.n_rings
        else:
            previous = max(ring - 1, 0)
            following = min(ring + 1, geometry.n_rings - 1)
        dr = geometry.radii[following] - geometry.radii[previous]
        dz = geometry.z_values[following] - geometry.z_values[previous]
        radial = orientation * dz
        vertical = -orientation * dr
        profile_normals.append((radial, vertical))
    return build_surface_engine(
        points,
        weights,
        normals=_revolve_profile_normals(geometry, profile_normals),
        config=config,
    )


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
    points, weights = _axisymmetric_nodes(geometry)
    normals = []
    for ring in range(geometry.n_rings):
        radial = geometry.radii[ring] / (equatorial_radius * equatorial_radius)
        vertical = geometry.z_values[ring] / (polar_radius * polar_radius)
        normals.append((radial, vertical))
    return build_surface_engine(
        points,
        weights,
        normals=_revolve_profile_normals(geometry, normals),
        config=cfg,
    )


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
    points, weights = _axisymmetric_nodes(geometry)
    normals = tuple(
        (geometry.radii[ring] - major_radius, geometry.z_values[ring])
        for ring in range(geometry.n_rings)
    )
    return build_surface_engine(
        points,
        weights,
        normals=_revolve_profile_normals(geometry, normals),
        config=cfg,
    )


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
    coefficients = tuple(float(value) for value in cosine_coefficients)
    geometry = _radial_profile_qjet(
        base_radius,
        coefficients,
        n_meridian,
        n_theta,
        kernel_power=cfg.kernel_power,
    )
    points, weights = _axisymmetric_nodes(geometry)
    profile_normals = []
    for ring in range(geometry.n_rings):
        u_value = PI * (ring + 0.5) / geometry.n_rings
        radius = float(base_radius)
        derivative = 0.0
        for mode, coefficient in enumerate(coefficients, start=1):
            radius += coefficient * _cos(mode * u_value)
            derivative -= mode * coefficient * _sin(mode * u_value)
        sine = _sin(u_value)
        cosine = _cos(u_value)
        dr = derivative * sine + radius * cosine
        dz = derivative * cosine - radius * sine
        profile_normals.append((-dz, dr))
    return build_surface_engine(
        points,
        weights,
        normals=_revolve_profile_normals(geometry, profile_normals),
        config=cfg,
    )


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
    points, weights = _axisymmetric_nodes(geometry)
    dr = float(radius_stop) - float(radius_start)
    dz = float(z_stop) - float(z_start)
    direction = 1.0 if dz >= 0.0 else -1.0
    normal = (direction * dz, -direction * dr)
    return build_surface_engine(
        points,
        weights,
        normals=_revolve_profile_normals(
            geometry,
            (normal for _ring in range(geometry.n_rings)),
        ),
        config=cfg,
    )


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
    "CompiledCadPanelSurface",
    "CompiledCadSurface",
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
    "build_compiled_cad_engine",
    "build_mesh_engine",
    "build_polyhedral_engine",
    "build_radial_profile_engine",
    "build_spheroid_engine",
    "build_surface_engine",
    "build_torus_engine",
    "compile_cad_panels",
    "compile_cad_surface",
    "decode_mesh",
    "encode_mesh",
    "load_binary_stl_assembly",
    "load_ifc_tessellation",
    "load_compiled_cad_panels",
    "load_compiled_cad_surface",
    "triangle_lumped_vertex_weights",
    "write_archive",
]
