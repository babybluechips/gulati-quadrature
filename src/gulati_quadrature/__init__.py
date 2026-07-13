"""Public API for the production Gulati Q quadrature pipeline.

The facade keeps the paper-facing path narrow:

* ordered boundary samples generate QJets;
* DtN/PDE solves use harmonic moment and zeta-tail repayment by default;
* near-boundary log-layer quadrature uses multipole/zeta Q;
* arithmetic certificates are exposed without forming dense Q matrices.

The older :mod:`inverse_shape` package remains available for reproducibility
and for lower-level research scripts.
"""

from gulati_quadrature.api import (
    ProductionQConfig,
    ProductionQEngine,
    PreparedBoundary,
    build_engine,
    cosine_trace,
    cycle_certificate,
    inverse_square_chord_from_scale_phase,
    integrate_log_layer,
    prepare_boundary,
    scale_phase_chord_squared,
    scale_phase_point,
    solve_pde,
    star_boundary,
)
from gulati_quadrature.surface_pde import (
    SurfacePDEConfig,
    SurfacePDENonconvergenceError,
    SurfacePDEResult,
    SurfacePDESolver,
    build_surface_pde_solver,
    solve_surface_pde,
)
from gulati_quadrature.three_d import (
    CadArchiveError,
    CertifiedPolyhedralSurfaceQJet,
    ExactMesh,
    MeshPart,
    NearLinearContractError,
    ProductionRieszQJet,
    ProductionSurfaceQEngine,
    SurfaceQConfig,
    SurfaceQEvaluation,
    archive_header,
    audit_round_trip,
    build_axisymmetric_conic_engine,
    build_axisymmetric_engine,
    build_conic_pencil_engine,
    build_mesh_engine,
    build_polyhedral_engine,
    build_radial_profile_engine,
    build_spheroid_engine,
    build_surface_engine,
    build_torus_engine,
    decode_mesh,
    encode_mesh,
    load_binary_stl_assembly,
    load_ifc_tessellation,
    triangle_lumped_vertex_weights,
    write_archive,
)

__all__ = [
    "PreparedBoundary",
    "ProductionQConfig",
    "ProductionQEngine",
    "CadArchiveError",
    "CertifiedPolyhedralSurfaceQJet",
    "ExactMesh",
    "MeshPart",
    "NearLinearContractError",
    "ProductionRieszQJet",
    "ProductionSurfaceQEngine",
    "SurfaceQConfig",
    "SurfaceQEvaluation",
    "SurfacePDEConfig",
    "SurfacePDENonconvergenceError",
    "SurfacePDEResult",
    "SurfacePDESolver",
    "archive_header",
    "audit_round_trip",
    "build_axisymmetric_conic_engine",
    "build_axisymmetric_engine",
    "build_conic_pencil_engine",
    "build_engine",
    "build_mesh_engine",
    "build_polyhedral_engine",
    "build_radial_profile_engine",
    "build_spheroid_engine",
    "build_surface_engine",
    "build_surface_pde_solver",
    "build_torus_engine",
    "cosine_trace",
    "cycle_certificate",
    "decode_mesh",
    "encode_mesh",
    "inverse_square_chord_from_scale_phase",
    "integrate_log_layer",
    "load_binary_stl_assembly",
    "load_ifc_tessellation",
    "prepare_boundary",
    "scale_phase_chord_squared",
    "scale_phase_point",
    "solve_pde",
    "solve_surface_pde",
    "star_boundary",
    "triangle_lumped_vertex_weights",
    "write_archive",
]
