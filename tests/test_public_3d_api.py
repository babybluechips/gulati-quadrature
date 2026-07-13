import ast
import math
from pathlib import Path

from gulati_quadrature import (
    ExactMesh,
    MeshPart,
    SurfaceQConfig,
    audit_round_trip,
    build_mesh_engine,
    build_spheroid_engine,
    decode_mesh,
    encode_mesh,
)


VERTICES = (
    (1.0, 0.0, 0.0),
    (-1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, 0.0, -1.0),
)
FACES = (
    (0, 2, 4),
    (2, 1, 4),
    (1, 3, 4),
    (3, 0, 4),
    (2, 0, 5),
    (1, 2, 5),
    (3, 1, 5),
    (0, 3, 5),
)


def test_public_mesh_engine_has_normalized_dtn_and_no_dense_fallback() -> None:
    engine = build_mesh_engine(
        VERTICES,
        FACES,
        config=SurfaceQConfig(kernel_power=3.0, leaf_size=2),
    )
    values = tuple(x + 0.2 * y - 0.1 * z for x, y, z in VERTICES)
    raw = engine.apply(values)
    dtn = engine.apply_dtn_principal(values)
    scale = 1.0 / (2.0 * math.pi)

    assert raw.ledger.status == "borrowed_repaid"
    assert raw.stats["constant_shortcut"] is False
    assert all(
        abs(complex(right) - scale * complex(left)) < 2.0e-15
        for left, right in zip(raw.values, dtn.values, strict=True)
    )
    stats = engine.stats()
    assert stats["hard_no_quadratic_contract"] is True
    assert stats["quadratic_fallback"] is False
    assert stats["dense_q_matrix_stored"] is False
    assert stats["pair_table_stored"] is False
    assert stats["constant_shortcut"] is False
    assert stats["apply_complexity"] == "O(N log N) for fixed order in 3D"


def test_public_mesh_engine_kills_constants_exactly() -> None:
    engine = build_mesh_engine(VERTICES, FACES)
    result = engine.apply((1.0,) * len(VERTICES))
    assert result.values == (0.0,) * len(VERTICES)
    assert result.stats["constant_shortcut"] is True


def test_axisymmetric_factory_lowers_into_same_production_backend() -> None:
    engine = build_spheroid_engine(1.0, 0.8, 4, 8)
    values = tuple(x - 0.3 * z for x, _y, z in engine.points)
    result = engine.apply_dtn_principal(values)
    assert len(result.values) == 32
    assert result.ledger.status == "borrowed_repaid"
    assert engine.stats()["method"] == "fixed_order_symmetric_gegenbauer_riesz_wspd"
    assert engine.stats()["quadratic_fallback"] is False


def test_public_qcad_archive_round_trip_is_bitwise_exact() -> None:
    mesh = ExactMesh(
        vertices=VERTICES,
        faces=FACES,
        parts=(MeshPart("octahedron", 0, len(FACES), "test"),),
        scalar_bits=64,
        name="octahedron",
    )
    archive = encode_mesh(mesh)
    decoded = decode_mesh(archive)
    assert audit_round_trip(mesh, decoded).exact


def test_public_3d_facade_has_no_numpy_or_scipy_import() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "gulati_quadrature"
        / "three_d.py"
    ).read_text(encoding="utf-8")
    imports = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
    assert not any(name == "numpy" or name.startswith("numpy.") for name in imports)
    assert not any(name == "scipy" or name.startswith("scipy.") for name in imports)
