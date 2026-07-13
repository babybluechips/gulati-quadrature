import math
import struct
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inverse_shape.reversible_cad_qjet import (  # noqa: E402
    CadArchiveError,
    ExactMesh,
    MeshPart,
    archive_header,
    audit_round_trip,
    decode_mesh,
    encode_mesh,
    float_to_ordered_key,
    load_binary_stl_assembly,
    load_ifc_tessellation,
    ordered_key_to_float,
    third_difference_decode,
    third_difference_encode,
)


SOURCES = ROOT / "benchmarks" / "cad_invertibility" / "sources"


def _bits(value: float, scalar_bits: int) -> int:
    if scalar_bits == 32:
        return struct.unpack("<I", struct.pack("<f", value))[0]
    return struct.unpack("<Q", struct.pack("<d", value))[0]


@pytest.mark.parametrize("scalar_bits", [32, 64])
def test_ordered_ieee_key_is_bitwise_invertible_and_monotone(scalar_bits: int) -> None:
    values = (-1000.25, -1.0, -0.0, 0.0, 1.0e-12, 1.0, 1000.25)
    keys = [float_to_ordered_key(value, scalar_bits) for value in values]
    assert keys == sorted(keys)
    for value, key in zip(values, keys, strict=True):
        recovered = ordered_key_to_float(key, scalar_bits)
        assert _bits(recovered, scalar_bits) == _bits(value, scalar_bits)


def test_third_difference_transform_is_exact_for_large_signed_integers() -> None:
    values = (
        0,
        2**64 - 1,
        2**63,
        -2**90,
        7,
        10**80,
        -(10**80),
        41,
    )
    jets = third_difference_encode(values)
    assert third_difference_decode(jets) == values
    assert len(jets) == len(values)


def test_three_power_sums_alone_are_not_an_injective_geometry_code() -> None:
    left = (-2.0, -1.0, 1.0, 2.0)
    right = (-math.sqrt(3.0), -math.sqrt(2.0), math.sqrt(2.0), math.sqrt(3.0))
    for power in (1, 2, 3):
        assert sum(value**power for value in left) == pytest.approx(
            sum(value**power for value in right), abs=1.0e-14
        )
    assert left != right


def test_archive_recovers_coordinate_bits_connectivity_and_parts() -> None:
    mesh = ExactMesh(
        vertices=(
            (-0.0, 0.0, 1.25),
            (2.0, -3.5, 4.0),
            (-5.0, 6.25, -7.0),
            (8.5, 9.0, -10.0),
        ),
        faces=((0, 1, 2), (1, 3, 2)),
        parts=(MeshPart("two triangles", 0, 2, "synthetic"),),
        scalar_bits=64,
        name="synthetic audit mesh",
    )
    archive = encode_mesh(mesh)
    decoded = decode_mesh(archive)
    audit = audit_round_trip(mesh, decoded)
    assert audit.exact
    assert archive == encode_mesh(mesh)
    header = archive_header(archive)
    assert header["stored_integer_count"] == 3 * len(mesh.vertices) + 3 * len(mesh.faces)
    assert header["no_dense_matrix"] is True
    assert header["no_pair_table"] is True


def test_corruption_is_rejected_before_returning_geometry() -> None:
    mesh = ExactMesh(
        vertices=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
        faces=((0, 1, 2),),
        parts=(MeshPart("triangle", 0, 1),),
        scalar_bits=64,
    )
    damaged = bytearray(encode_mesh(mesh))
    damaged[-1] ^= 0x08
    with pytest.raises(CadArchiveError):
        decode_mesh(bytes(damaged))


def test_public_binary_stl_assembly_round_trips_exactly() -> None:
    directory = SOURCES / "sofia"
    paths = (
        directory / "Nose_section.stl",
        directory / "Fuselage_top.stl",
        directory / "Left_wing.stl",
        directory / "Right_wing.stl",
        directory / "Tail_section.stl",
        directory / "Telescope_cavity_closed.stl",
        directory / "Instrument.stl",
    )
    mesh = load_binary_stl_assembly(paths, name="NASA SOFIA aircraft")
    assert len(mesh.vertices) == 45_838
    assert len(mesh.faces) == 92_974
    assert len(mesh.parts) == 7
    assert mesh.scalar_bits == 32
    assert audit_round_trip(mesh, decode_mesh(encode_mesh(mesh))).exact


def test_high_resolution_public_vehicle_mesh_round_trips_exactly() -> None:
    mesh = load_binary_stl_assembly(
        (SOURCES / "curiosity_assembled" / "Curiosity_200uM.stl",),
        name="NASA Curiosity single-file print layout",
    )
    assert len(mesh.vertices) == 192_301
    assert len(mesh.faces) == 384_942
    assert mesh.scalar_bits == 32
    assert audit_round_trip(mesh, decode_mesh(encode_mesh(mesh))).exact


def test_public_ifc_bridge_applies_recursive_placements_and_round_trips() -> None:
    mesh = load_ifc_tessellation(
        SOURCES / "bridge" / "Infra-Bridge.ifc",
        name="buildingSMART IFC bridge",
        product_filter=lambda value: "bridge" in value.lower(),
    )
    assert len(mesh.vertices) == 6_436
    assert len(mesh.faces) == 12_468
    assert len(mesh.parts) == 47
    minimum, maximum = mesh.bounds
    assert maximum[0] - minimum[0] > 30_000.0
    assert maximum[1] - minimum[1] > 30_000.0
    assert maximum[2] - minimum[2] > 10_000.0
    assert audit_round_trip(mesh, decode_mesh(encode_mesh(mesh))).exact


def test_codec_source_has_no_numpy_or_quadratic_pair_storage() -> None:
    source = (ROOT / "src" / "inverse_shape" / "reversible_cad_qjet.py").read_text(
        encoding="utf-8"
    )
    assert "import numpy" not in source
    assert "from numpy" not in source
    assert "distance_matrix =" not in source
    assert "pair_table =" not in source
