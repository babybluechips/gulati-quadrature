"""Reversible sparse 3-jet archives for tessellated CAD geometry.

The analytic moments used by a fast QJet operator are not injective: finitely
many moments cannot determine an arbitrary mesh.  This module therefore keeps
the operator hierarchy and the lossless geometry channel conceptually
separate.  The latter is a triangular third-difference transform of ordered
IEEE coordinate keys together with exact triangle connectivity.

For a coordinate sequence ``q`` the stored residual jets are

    j[i] = q[i]                                      for i < 3,
    j[i] = q[i] - 3*q[i-1] + 3*q[i-2] - q[i-3]      for i >= 3.

The transform is unit triangular over the integers, hence exactly invertible.
Zig-zag varints and DEFLATE reduce storage but do not change that algebraic
fact.  Archive construction and decoding use O(V + F) work and memory.  No
distance matrix, pair table, or numerical-rank expansion is constructed.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence


MAGIC = b"QCAD3J1\0"
_HEADER_LENGTH = struct.Struct("<I")
_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"
_TRIPLE_RE = re.compile(
    rf"\(\s*({_NUMBER})\s*,\s*({_NUMBER})\s*,\s*({_NUMBER})\s*\)"
)
_INTEGER_TRIPLE_RE = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)")
_ENTITY_RE = re.compile(r"^#(\d+)\s*=\s*([A-Z0-9_]+)\s*\((.*)\)$", re.DOTALL)
_REFERENCE_RE = re.compile(r"#(\d+)")


class CadArchiveError(ValueError):
    """Raised when a CAD archive or source tessellation is invalid."""


@dataclass(frozen=True)
class MeshPart:
    """A named contiguous range of triangle faces."""

    name: str
    face_start: int
    face_count: int
    source: str = ""


@dataclass(frozen=True)
class ExactMesh:
    """A canonical triangle mesh with explicit IEEE coordinate precision."""

    vertices: tuple[tuple[float, float, float], ...]
    faces: tuple[tuple[int, int, int], ...]
    parts: tuple[MeshPart, ...]
    scalar_bits: int
    name: str = "mesh"

    def __post_init__(self) -> None:
        if self.scalar_bits not in (32, 64):
            raise ValueError("mesh scalar precision must be 32 or 64 bits")
        if any(len(vertex) != 3 for vertex in self.vertices):
            raise ValueError("mesh vertices must have three coordinates")
        if any(not math.isfinite(value) for vertex in self.vertices for value in vertex):
            raise ValueError("mesh coordinates must be finite")
        vertex_count = len(self.vertices)
        for face in self.faces:
            if len(face) != 3:
                raise ValueError("only triangle faces are supported")
            if any(index < 0 or index >= vertex_count for index in face):
                raise ValueError("face index is outside the vertex array")
        expected = 0
        for part in self.parts:
            if part.face_start != expected or part.face_count < 0:
                raise ValueError("mesh parts must partition faces contiguously")
            expected += part.face_count
        if expected != len(self.faces):
            raise ValueError("mesh parts do not cover every face")

    @property
    def bounds(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        if not self.vertices:
            zero = (0.0, 0.0, 0.0)
            return zero, zero
        return (
            tuple(min(vertex[axis] for vertex in self.vertices) for axis in range(3)),
            tuple(max(vertex[axis] for vertex in self.vertices) for axis in range(3)),
        )

    @property
    def mesh_sha256(self) -> str:
        digest = hashlib.sha256()
        digest.update(struct.pack("<BQQ", self.scalar_bits, len(self.vertices), len(self.faces)))
        scalar_format = "<f" if self.scalar_bits == 32 else "<d"
        for vertex in self.vertices:
            for value in vertex:
                digest.update(struct.pack(scalar_format, value))
        for face in self.faces:
            digest.update(struct.pack("<QQQ", *face))
        for part in self.parts:
            encoded = part.name.encode("utf-8")
            digest.update(struct.pack("<I", len(encoded)))
            digest.update(encoded)
            digest.update(struct.pack("<QQ", part.face_start, part.face_count))
        return digest.hexdigest()


@dataclass(frozen=True)
class RoundTripAudit:
    """Exact comparison between a compiler input mesh and its decoded mesh."""

    mesh_sha256: str
    decoded_sha256: str
    vertex_mismatches: int
    face_mismatches: int
    part_mismatches: int
    maximum_absolute_error: float

    @property
    def exact(self) -> bool:
        return (
            self.mesh_sha256 == self.decoded_sha256
            and self.vertex_mismatches == 0
            and self.face_mismatches == 0
            and self.part_mismatches == 0
            and self.maximum_absolute_error == 0.0
        )


def third_difference_encode(values: Sequence[int]) -> tuple[int, ...]:
    """Apply the unit-triangular third-difference transform."""

    output = []
    for index, value in enumerate(values):
        integer = int(value)
        if index < 3:
            output.append(integer)
        else:
            output.append(
                integer
                - 3 * int(values[index - 1])
                + 3 * int(values[index - 2])
                - int(values[index - 3])
            )
    return tuple(output)


def third_difference_decode(jets: Sequence[int]) -> tuple[int, ...]:
    """Invert :func:`third_difference_encode` exactly over the integers."""

    output: list[int] = []
    for index, jet in enumerate(jets):
        integer = int(jet)
        if index < 3:
            output.append(integer)
        else:
            output.append(
                integer
                + 3 * output[index - 1]
                - 3 * output[index - 2]
                + output[index - 3]
            )
    return tuple(output)


def _zigzag_encode(value: int) -> int:
    integer = int(value)
    return 2 * integer if integer >= 0 else -2 * integer - 1


def _zigzag_decode(value: int) -> int:
    integer = int(value)
    return integer // 2 if integer % 2 == 0 else -(integer // 2) - 1


def _append_varint(output: bytearray, value: int) -> None:
    unsigned = int(value)
    if unsigned < 0:
        raise ValueError("varints require nonnegative integers")
    while unsigned >= 0x80:
        output.append((unsigned & 0x7F) | 0x80)
        unsigned >>= 7
    output.append(unsigned)


def _read_varint(payload: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while offset < len(payload):
        byte = payload[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7
        if shift > 256:
            raise CadArchiveError("varint exceeds the archive integer limit")
    raise CadArchiveError("truncated varint")


def _float_bits(value: float, scalar_bits: int) -> int:
    if scalar_bits == 32:
        return struct.unpack("<I", struct.pack("<f", value))[0]
    return struct.unpack("<Q", struct.pack("<d", value))[0]


def _bits_float(value: int, scalar_bits: int) -> float:
    if scalar_bits == 32:
        return struct.unpack("<f", struct.pack("<I", value))[0]
    return struct.unpack("<d", struct.pack("<Q", value))[0]


def float_to_ordered_key(value: float, scalar_bits: int) -> int:
    """Map a finite IEEE float to a monotone unsigned integer, exactly."""

    bits = _float_bits(value, scalar_bits)
    sign = 1 << (scalar_bits - 1)
    mask = (1 << scalar_bits) - 1
    return ((~bits) & mask) if bits & sign else bits ^ sign


def ordered_key_to_float(key: int, scalar_bits: int) -> float:
    """Invert :func:`float_to_ordered_key` bit for bit."""

    sign = 1 << (scalar_bits - 1)
    mask = (1 << scalar_bits) - 1
    integer = int(key)
    if integer < 0 or integer > mask:
        raise CadArchiveError("ordered float key is outside its declared precision")
    bits = integer ^ sign if integer & sign else (~integer) & mask
    return _bits_float(bits, scalar_bits)


def _encode_signed_sequence(output: bytearray, values: Sequence[int]) -> None:
    for value in values:
        _append_varint(output, _zigzag_encode(value))


def _decode_signed_sequence(
    payload: bytes, offset: int, count: int
) -> tuple[tuple[int, ...], int]:
    output = []
    for _ in range(count):
        encoded, offset = _read_varint(payload, offset)
        output.append(_zigzag_decode(encoded))
    return tuple(output), offset


def encode_mesh(mesh: ExactMesh, compression_level: int = 9) -> bytes:
    """Compile a mesh into a deterministic sparse QCAD3J archive."""

    payload = bytearray()
    for axis in range(3):
        keys = tuple(
            float_to_ordered_key(vertex[axis], mesh.scalar_bits)
            for vertex in mesh.vertices
        )
        _encode_signed_sequence(payload, third_difference_encode(keys))

    previous = 0
    face_deltas = []
    for face in mesh.faces:
        for index in face:
            face_deltas.append(index - previous)
            previous = index
    _encode_signed_sequence(payload, face_deltas)

    raw_payload = bytes(payload)
    header = {
        "archive_version": 1,
        "coordinate_transform": "ordered-ieee-third-difference",
        "connectivity_transform": "flattened-index-delta",
        "face_count": len(mesh.faces),
        "mesh_name": mesh.name,
        "mesh_sha256": mesh.mesh_sha256,
        "no_dense_matrix": True,
        "no_pair_table": True,
        "parts": [
            {
                "face_count": part.face_count,
                "face_start": part.face_start,
                "name": part.name,
                "source": part.source,
            }
            for part in mesh.parts
        ],
        "payload_sha256": hashlib.sha256(raw_payload).hexdigest(),
        "scalar_bits": mesh.scalar_bits,
        "stored_integer_count": 3 * len(mesh.vertices) + 3 * len(mesh.faces),
        "vertex_count": len(mesh.vertices),
    }
    header_bytes = json.dumps(
        header, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    compressed = zlib.compress(raw_payload, level=int(compression_level))
    return MAGIC + _HEADER_LENGTH.pack(len(header_bytes)) + header_bytes + compressed


def archive_header(archive: bytes) -> dict[str, object]:
    """Read and minimally validate a QCAD3J archive header."""

    minimum = len(MAGIC) + _HEADER_LENGTH.size
    if len(archive) < minimum or archive[: len(MAGIC)] != MAGIC:
        raise CadArchiveError("not a QCAD3J archive")
    (header_length,) = _HEADER_LENGTH.unpack_from(archive, len(MAGIC))
    start = minimum
    end = start + header_length
    if end > len(archive):
        raise CadArchiveError("truncated QCAD3J header")
    try:
        header = json.loads(archive[start:end].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CadArchiveError("invalid QCAD3J header") from error
    if header.get("archive_version") != 1:
        raise CadArchiveError("unsupported QCAD3J archive version")
    return header


def decode_mesh(archive: bytes) -> ExactMesh:
    """Decode and checksum-verify a sparse QCAD3J archive."""

    header = archive_header(archive)
    header_length = _HEADER_LENGTH.unpack_from(archive, len(MAGIC))[0]
    compressed_offset = len(MAGIC) + _HEADER_LENGTH.size + header_length
    try:
        payload = zlib.decompress(archive[compressed_offset:])
    except zlib.error as error:
        raise CadArchiveError("corrupt compressed QCAD3J payload") from error
    if hashlib.sha256(payload).hexdigest() != header.get("payload_sha256"):
        raise CadArchiveError("QCAD3J payload checksum mismatch")

    vertex_count = int(header["vertex_count"])
    face_count = int(header["face_count"])
    scalar_bits = int(header["scalar_bits"])
    if vertex_count < 0 or face_count < 0 or scalar_bits not in (32, 64):
        raise CadArchiveError("invalid QCAD3J mesh dimensions")

    axes = []
    offset = 0
    for _axis in range(3):
        jets, offset = _decode_signed_sequence(payload, offset, vertex_count)
        keys = third_difference_decode(jets)
        axes.append(tuple(ordered_key_to_float(key, scalar_bits) for key in keys))

    deltas, offset = _decode_signed_sequence(payload, offset, 3 * face_count)
    if offset != len(payload):
        raise CadArchiveError("QCAD3J payload has unconsumed trailing bytes")
    flat_faces = []
    previous = 0
    for delta in deltas:
        previous += delta
        flat_faces.append(previous)
    faces = tuple(
        tuple(flat_faces[index : index + 3])
        for index in range(0, len(flat_faces), 3)
    )
    vertices = tuple(
        (axes[0][index], axes[1][index], axes[2][index])
        for index in range(vertex_count)
    )
    parts = tuple(
        MeshPart(
            name=str(item["name"]),
            face_start=int(item["face_start"]),
            face_count=int(item["face_count"]),
            source=str(item.get("source", "")),
        )
        for item in header["parts"]
    )
    mesh = ExactMesh(
        vertices=vertices,
        faces=faces,
        parts=parts,
        scalar_bits=scalar_bits,
        name=str(header["mesh_name"]),
    )
    if mesh.mesh_sha256 != header.get("mesh_sha256"):
        raise CadArchiveError("decoded QCAD3J mesh checksum mismatch")
    return mesh


def write_archive(path: str | Path, mesh: ExactMesh) -> bytes:
    """Encode ``mesh``, write it to ``path``, and return the archive bytes."""

    archive = encode_mesh(mesh)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(archive)
    return archive


def audit_round_trip(original: ExactMesh, decoded: ExactMesh) -> RoundTripAudit:
    """Compare coordinate bit patterns, connectivity, metadata, and hashes."""

    vertex_mismatches = abs(len(original.vertices) - len(decoded.vertices))
    maximum_error = 0.0
    for left, right in zip(original.vertices, decoded.vertices, strict=False):
        mismatch = any(
            _float_bits(left[axis], original.scalar_bits)
            != _float_bits(right[axis], decoded.scalar_bits)
            for axis in range(3)
        )
        vertex_mismatches += int(mismatch)
        maximum_error = max(
            maximum_error,
            *(abs(left[axis] - right[axis]) for axis in range(3)),
        )
    face_mismatches = abs(len(original.faces) - len(decoded.faces)) + sum(
        left != right for left, right in zip(original.faces, decoded.faces, strict=False)
    )
    part_mismatches = abs(len(original.parts) - len(decoded.parts)) + sum(
        left != right for left, right in zip(original.parts, decoded.parts, strict=False)
    )
    return RoundTripAudit(
        mesh_sha256=original.mesh_sha256,
        decoded_sha256=decoded.mesh_sha256,
        vertex_mismatches=vertex_mismatches,
        face_mismatches=face_mismatches,
        part_mismatches=part_mismatches,
        maximum_absolute_error=maximum_error,
    )


def _read_binary_stl_triangles(path: Path) -> Iterator[tuple[bytes, bytes, bytes]]:
    data = path.read_bytes()
    if len(data) < 84:
        raise CadArchiveError(f"{path} is too short to be a binary STL")
    triangle_count = struct.unpack_from("<I", data, 80)[0]
    expected = 84 + 50 * triangle_count
    if len(data) != expected:
        raise CadArchiveError(
            f"{path} is not a canonical binary STL: expected {expected} bytes, got {len(data)}"
        )
    for triangle in range(triangle_count):
        offset = 84 + 50 * triangle + 12
        yield data[offset : offset + 12], data[offset + 12 : offset + 24], data[
            offset + 24 : offset + 36
        ]


def load_binary_stl_assembly(
    paths: Sequence[str | Path], name: str = "STL assembly"
) -> ExactMesh:
    """Load one or more aligned binary STL parts without changing float32 bits."""

    if not paths:
        raise ValueError("at least one STL path is required")
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    parts: list[MeshPart] = []
    vertex_lookup: dict[bytes, int] = {}
    for source in (Path(path) for path in paths):
        face_start = len(faces)
        for triangle in _read_binary_stl_triangles(source):
            face = []
            for raw_vertex in triangle:
                index = vertex_lookup.get(raw_vertex)
                if index is None:
                    index = len(vertices)
                    vertex_lookup[raw_vertex] = index
                    vertices.append(struct.unpack("<fff", raw_vertex))
                face.append(index)
            faces.append(tuple(face))
        parts.append(
            MeshPart(
                name=source.stem.replace("_", " "),
                face_start=face_start,
                face_count=len(faces) - face_start,
                source=source.name,
            )
        )
    return ExactMesh(
        vertices=tuple(vertices),
        faces=tuple(faces),
        parts=tuple(parts),
        scalar_bits=32,
        name=name,
    )


@dataclass(frozen=True)
class _StepEntity:
    kind: str
    body: str


def _iter_step_records(text: str) -> Iterator[str]:
    start = text.find("DATA;")
    if start < 0:
        raise CadArchiveError("IFC file has no DATA section")
    record: list[str] = []
    in_string = False
    index = start + len("DATA;")
    while index < len(text):
        character = text[index]
        if character == "'":
            if in_string and index + 1 < len(text) and text[index + 1] == "'":
                record.extend(("'", "'"))
                index += 2
                continue
            in_string = not in_string
        if character == ";" and not in_string:
            value = "".join(record).strip()
            record = []
            if value == "ENDSEC":
                return
            if value.startswith("#"):
                yield value
        else:
            record.append(character)
        index += 1
    raise CadArchiveError("unterminated IFC DATA section")


def _split_step_args(body: str) -> tuple[str, ...]:
    output = []
    start = 0
    depth = 0
    in_string = False
    index = 0
    while index < len(body):
        character = body[index]
        if character == "'":
            if in_string and index + 1 < len(body) and body[index + 1] == "'":
                index += 2
                continue
            in_string = not in_string
        elif not in_string:
            if character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
                if depth < 0:
                    raise CadArchiveError("unbalanced STEP argument list")
            elif character == "," and depth == 0:
                output.append(body[start:index].strip())
                start = index + 1
        index += 1
    if depth != 0 or in_string:
        raise CadArchiveError("unterminated STEP argument list")
    output.append(body[start:].strip())
    return tuple(output)


def _references(value: str) -> tuple[int, ...]:
    return tuple(int(match) for match in _REFERENCE_RE.findall(value))


def _reference(value: str) -> int | None:
    match = re.fullmatch(r"#(\d+)", value.strip())
    return int(match.group(1)) if match else None


def _parse_entities(text: str) -> dict[int, _StepEntity]:
    entities = {}
    for record in _iter_step_records(text):
        match = _ENTITY_RE.match(record)
        if not match:
            raise CadArchiveError(f"unsupported IFC entity syntax: {record[:80]}")
        identifier = int(match.group(1))
        entities[identifier] = _StepEntity(match.group(2), match.group(3))
    return entities


def _parse_triple(value: str) -> tuple[float, float, float]:
    match = _TRIPLE_RE.fullmatch(value.strip())
    if not match:
        raise CadArchiveError(f"expected a numeric triple, got {value[:80]}")
    return tuple(float(match.group(index)) for index in range(1, 4))


def _unit(value: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(sum(component * component for component in value))
    if length == 0.0:
        raise CadArchiveError("IFC placement direction has zero length")
    return tuple(component / length for component in value)


def _cross(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


_IDENTITY = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0, 0.0),
)


def _compose(left, right):
    rows = []
    for row in range(3):
        rows.append(
            (
                sum(left[row][inner] * right[inner][column] for inner in range(3))
                for column in range(3)
            )
        )
        rows[-1] = tuple(rows[-1]) + (
            left[row][3]
            + sum(left[row][inner] * right[inner][3] for inner in range(3)),
        )
    return tuple(rows)


def _transform(matrix, point):
    return tuple(
        sum(matrix[row][axis] * point[axis] for axis in range(3)) + matrix[row][3]
        for row in range(3)
    )


def _ifc_text(value: str) -> str:
    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] == "'" and stripped[-1] == "'":
        return stripped[1:-1].replace("''", "'")
    return stripped


def load_ifc_tessellation(
    path: str | Path,
    name: str = "IFC tessellation",
    product_filter: Callable[[str], bool] | None = None,
) -> ExactMesh:
    """Load placed ``IfcTriangulatedFaceSet`` geometry from an IFC-SPF file.

    This is intentionally a focused tessellation reader, not a general IFC
    implementation.  It follows product-definition shapes and recursive local
    placements and retains the resulting triangle geometry and product names.
    Semantic properties outside that surface representation are not archived.
    """

    source = Path(path)
    entities = _parse_entities(source.read_text(encoding="utf-8"))
    arguments = {
        identifier: _split_step_args(entity.body)
        for identifier, entity in entities.items()
    }

    cartesian_points = {}
    directions = {}
    point_lists = {}
    face_sets = {}
    for identifier, entity in entities.items():
        args = arguments[identifier]
        if entity.kind == "IFCCARTESIANPOINT":
            cartesian_points[identifier] = _parse_triple(args[0])
        elif entity.kind == "IFCDIRECTION":
            directions[identifier] = _parse_triple(args[0])
        elif entity.kind == "IFCCARTESIANPOINTLIST3D":
            point_lists[identifier] = tuple(
                tuple(float(match.group(index)) for index in range(1, 4))
                for match in _TRIPLE_RE.finditer(args[0])
            )
        elif entity.kind == "IFCTRIANGULATEDFACESET":
            if len(args) < 4:
                raise CadArchiveError("IfcTriangulatedFaceSet has too few arguments")
            point_reference = _reference(args[0])
            if point_reference is None:
                raise CadArchiveError("IfcTriangulatedFaceSet has no point-list reference")
            faces = tuple(
                tuple(int(match.group(index)) - 1 for index in range(1, 4))
                for match in _INTEGER_TRIPLE_RE.finditer(args[3])
            )
            face_sets[identifier] = (point_reference, faces)

    axis_cache = {}

    def axis_matrix(identifier: int):
        if identifier in axis_cache:
            return axis_cache[identifier]
        entity = entities.get(identifier)
        if entity is None or entity.kind != "IFCAXIS2PLACEMENT3D":
            raise CadArchiveError("IFC local placement does not reference Axis2Placement3D")
        args = arguments[identifier]
        location_reference = _reference(args[0])
        if location_reference not in cartesian_points:
            raise CadArchiveError("IFC axis placement has no Cartesian location")
        origin = cartesian_points[location_reference]
        z_reference = _reference(args[1]) if len(args) > 1 else None
        x_reference = _reference(args[2]) if len(args) > 2 else None
        z_axis = _unit(directions[z_reference]) if z_reference else (0.0, 0.0, 1.0)
        x_seed = _unit(directions[x_reference]) if x_reference else (1.0, 0.0, 0.0)
        projection = sum(x_seed[index] * z_axis[index] for index in range(3))
        x_axis = _unit(tuple(x_seed[index] - projection * z_axis[index] for index in range(3)))
        y_axis = _unit(_cross(z_axis, x_axis))
        x_axis = _unit(_cross(y_axis, z_axis))
        matrix = tuple(
            (x_axis[row], y_axis[row], z_axis[row], origin[row]) for row in range(3)
        )
        axis_cache[identifier] = matrix
        return matrix

    placement_cache = {}
    active_placements = set()

    def placement_matrix(identifier: int):
        if identifier in placement_cache:
            return placement_cache[identifier]
        if identifier in active_placements:
            raise CadArchiveError("cyclic IFC local placement")
        active_placements.add(identifier)
        entity = entities.get(identifier)
        if entity is None or entity.kind != "IFCLOCALPLACEMENT":
            raise CadArchiveError("product placement is not IfcLocalPlacement")
        args = arguments[identifier]
        parent_reference = _reference(args[0])
        relative_reference = _reference(args[1])
        if relative_reference is None:
            raise CadArchiveError("IfcLocalPlacement has no relative placement")
        relative = axis_matrix(relative_reference)
        parent = placement_matrix(parent_reference) if parent_reference else _IDENTITY
        result = _compose(parent, relative)
        placement_cache[identifier] = result
        active_placements.remove(identifier)
        return result

    shape_representations = {}
    product_shapes = {}
    for identifier, entity in entities.items():
        refs = _references(entity.body)
        if entity.kind == "IFCSHAPEREPRESENTATION":
            shape_representations[identifier] = tuple(ref for ref in refs if ref in face_sets)
        elif entity.kind == "IFCPRODUCTDEFINITIONSHAPE":
            product_shapes[identifier] = tuple(ref for ref in refs if ref in entities)

    product_rows = []
    for identifier, entity in entities.items():
        args = arguments[identifier]
        direct_refs = tuple(_reference(arg) for arg in args)
        placement_refs = [
            ref
            for ref in direct_refs
            if ref in entities and entities[ref].kind == "IFCLOCALPLACEMENT"
        ]
        shape_refs = [ref for ref in direct_refs if ref in product_shapes]
        if not placement_refs or not shape_refs:
            continue
        label = _ifc_text(args[2]) if len(args) > 2 else f"{entity.kind} #{identifier}"
        searchable = f"{label} {entity.body}"
        if product_filter is not None and not product_filter(searchable):
            continue
        selected_face_sets = []
        for shape_reference in shape_refs:
            for representation_reference in product_shapes[shape_reference]:
                selected_face_sets.extend(shape_representations.get(representation_reference, ()))
        if selected_face_sets:
            product_rows.append(
                (identifier, label, placement_refs[0], tuple(selected_face_sets))
            )

    if not product_rows:
        raise CadArchiveError("no placed IFC tessellation products matched the selection")

    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    parts: list[MeshPart] = []
    vertex_lookup: dict[bytes, int] = {}
    for product_identifier, label, placement_reference, selected_face_sets in product_rows:
        face_start = len(faces)
        matrix = placement_matrix(placement_reference)
        for face_set_reference in selected_face_sets:
            point_reference, local_faces = face_sets[face_set_reference]
            local_points = point_lists.get(point_reference)
            if local_points is None:
                raise CadArchiveError("IFC face set references a missing point list")
            local_to_global = []
            for point in local_points:
                placed = _transform(matrix, point)
                raw = struct.pack("<ddd", *placed)
                vertex_index = vertex_lookup.get(raw)
                if vertex_index is None:
                    vertex_index = len(vertices)
                    vertex_lookup[raw] = vertex_index
                    vertices.append(placed)
                local_to_global.append(vertex_index)
            for face in local_faces:
                if any(index < 0 or index >= len(local_to_global) for index in face):
                    raise CadArchiveError("IFC triangle index is outside its point list")
                faces.append(tuple(local_to_global[index] for index in face))
        parts.append(
            MeshPart(
                name=label,
                face_start=face_start,
                face_count=len(faces) - face_start,
                source=f"{source.name}#{product_identifier}",
            )
        )

    return ExactMesh(
        vertices=tuple(vertices),
        faces=tuple(faces),
        parts=tuple(parts),
        scalar_bits=64,
        name=name,
    )


def source_sha256(paths: Iterable[str | Path]) -> str:
    """Hash source files in supplied order for a reproducible provenance key."""

    digest = hashlib.sha256()
    for path in paths:
        source = Path(path)
        encoded_name = source.name.encode("utf-8")
        digest.update(struct.pack("<I", len(encoded_name)))
        digest.update(encoded_name)
        with source.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
    return digest.hexdigest()
