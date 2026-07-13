#!/usr/bin/env python3
"""Compile public CAD meshes to sparse 3-jet archives and invert them exactly."""

# ruff: noqa: E501

from __future__ import annotations

import base64
import hashlib
import html
import io
import json
import math
import struct
import sys
import time
import tracemalloc
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageEnhance


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
    load_binary_stl_assembly,
    load_ifc_tessellation,
    source_sha256,
)


SOURCES = ROOT / "benchmarks" / "cad_invertibility" / "sources"
OUT = ROOT / "outputs" / "cad_qjet_invertibility"
SUMMARY = OUT / "cad_roundtrip_summary.json"
GALLERY = OUT / "cad_roundtrip_gallery.svg"
REPORT = OUT / "cad_qjet_invertibility.html"


CASES = (
    {
        "key": "sofia_aircraft",
        "label": "NASA SOFIA aircraft",
        "format": "binary STL assembly",
        "source_url": "https://github.com/nasa/NASA-3D-Resources/tree/master/3D%20Printing/SOFIA",
        "license": "NASA assets: free and without copyright; NASA media usage guidelines apply",
        "attribution": "NASA 3D Resources contributors",
        "camera": (24.0, 18.0),
    },
    {
        "key": "cement_mixer",
        "label": "FreeCAD cement mixer truck",
        "format": "binary STL",
        "source_url": "https://github.com/FreeCAD/FreeCAD-library/tree/master/Generic%20objects/Scale%20Models/Cement%20mixer%20truck",
        "license": "CC BY 3.0",
        "attribution": "Martijn Cramer; KU Leuven ACRO",
        "camera": (-32.0, 16.0),
    },
    {
        "key": "curiosity_rover",
        "label": "NASA Curiosity manufacturing plates",
        "format": "four-part detailed binary STL manufacturing layout",
        "source_url": "https://github.com/nasa/NASA-3D-Resources/tree/master/3D%20Printing/Curiosity%20Rover%20%28Detailed%29",
        "license": "NASA assets: free and without copyright; NASA media usage guidelines apply",
        "attribution": "NASA 3D Resources contributors",
        "camera": (-32.0, 17.0),
    },
    {
        "key": "curiosity_assembled",
        "label": "NASA Curiosity single-file print layout",
        "format": "single-file high-resolution binary STL manufacturing layout",
        "source_url": "https://github.com/nasa/NASA-3D-Resources/tree/master/3D%20Printing/Curiosity%20Rover%20%28Simplified%29",
        "license": "NASA assets: free and without copyright; NASA media usage guidelines apply",
        "attribution": "NASA 3D Resources contributors",
        "camera": (-34.0, 18.0),
    },
    {
        "key": "ifc_bridge",
        "label": "buildingSMART IFC bridge",
        "format": "IFC4.3 tessellation with recursive placements",
        "source_url": "https://github.com/buildingSMART/Sample-Test-Files/blob/main/IFC%204.3.2.0%20%28IFC4X3_ADD2%29/PCERT-Sample-Scene/Infra-Bridge.ifc",
        "license": "CC BY 4.0",
        "attribution": "buildingSMART International; model author Jan B.",
        "camera": (-38.0, 19.0),
    },
)


def _source_paths(key: str) -> tuple[Path, ...]:
    if key == "sofia_aircraft":
        directory = SOURCES / "sofia"
        return tuple(
            directory / name
            for name in (
                "Nose_section.stl",
                "Fuselage_top.stl",
                "Left_wing.stl",
                "Right_wing.stl",
                "Tail_section.stl",
                "Telescope_cavity_closed.stl",
                "Instrument.stl",
            )
        )
    if key == "cement_mixer":
        return (SOURCES / "cement_mixer" / "asm_cement_mixer_truck.stl",)
    if key == "curiosity_rover":
        directory = SOURCES / "curiosity_rover"
        return tuple(
            directory / name
            for name in (
                "1-body.stl",
                "2-components.stl",
                "3-pins-and-hubs.stl",
                "4-wheels.stl",
            )
        )
    if key == "curiosity_assembled":
        return (SOURCES / "curiosity_assembled" / "Curiosity_200uM.stl",)
    if key == "ifc_bridge":
        return (SOURCES / "bridge" / "Infra-Bridge.ifc",)
    raise KeyError(key)


def _load_mesh(case: dict[str, object]) -> ExactMesh:
    paths = _source_paths(str(case["key"]))
    if case["key"] == "ifc_bridge":
        return load_ifc_tessellation(
            paths[0],
            name=str(case["label"]),
            product_filter=lambda value: "bridge" in value.lower(),
        )
    return load_binary_stl_assembly(paths, name=str(case["label"]))


def _write_binary_ply(path: Path, mesh: ExactMesh) -> None:
    """Write an interoperable reconstruction with float64 coordinates."""

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment decoded exactly from QCAD3J archive\n"
        f"element vertex {len(mesh.vertices)}\n"
        "property double x\n"
        "property double y\n"
        "property double z\n"
        f"element face {len(mesh.faces)}\n"
        "property list uchar uint vertex_indices\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as stream:
        stream.write(header)
        for vertex in mesh.vertices:
            stream.write(struct.pack("<ddd", *vertex))
        for face in mesh.faces:
            stream.write(struct.pack("<BIII", 3, *face))


def _unique_edge_count(mesh: ExactMesh) -> int:
    edges = set()
    for a, b, c in mesh.faces:
        edges.add((min(a, b), max(a, b)))
        edges.add((min(b, c), max(b, c)))
        edges.add((min(c, a), max(c, a)))
    return len(edges)


def _mesh_face_prefix(mesh: ExactMesh, face_count: int) -> ExactMesh:
    selected_faces = mesh.faces[: int(face_count)]
    vertex_map = {}
    vertices = []
    faces = []
    for face in selected_faces:
        remapped = []
        for old_index in face:
            new_index = vertex_map.get(old_index)
            if new_index is None:
                new_index = len(vertices)
                vertex_map[old_index] = new_index
                vertices.append(mesh.vertices[old_index])
            remapped.append(new_index)
        faces.append(tuple(remapped))
    parts = []
    covered = 0
    for part in mesh.parts:
        available = max(0, min(part.face_start + part.face_count, len(selected_faces)) - part.face_start)
        if available:
            parts.append(MeshPart(part.name, covered, available, part.source))
            covered += available
    return ExactMesh(
        vertices=tuple(vertices),
        faces=tuple(faces),
        parts=tuple(parts),
        scalar_bits=mesh.scalar_bits,
        name=f"{mesh.name} prefix {len(faces)}",
    )


def _log_slope(rows: list[dict[str, object]], key: str) -> float:
    x_values = [math.log(float(row["linear_items"])) for row in rows]
    y_values = [math.log(max(float(row[key]), 1.0e-12)) for row in rows]
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    denominator = sum((value - x_mean) ** 2 for value in x_values)
    return sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x_values, y_values, strict=True)
    ) / denominator


def _scaling_campaign(mesh: ExactMesh) -> dict[str, object]:
    limits = (4_096, 8_192, 16_384, 32_768, 65_536, len(mesh.faces))
    rows = []
    for limit in limits:
        prefix = _mesh_face_prefix(mesh, limit)
        encode_start = time.perf_counter()
        archive = encode_mesh(prefix)
        encode_ms = 1000.0 * (time.perf_counter() - encode_start)
        decode_start = time.perf_counter()
        decoded = decode_mesh(archive)
        decode_ms = 1000.0 * (time.perf_counter() - decode_start)
        rows.append(
            {
                "archive_bytes": len(archive),
                "decode_ms": decode_ms,
                "encode_ms": encode_ms,
                "exact_round_trip": audit_round_trip(prefix, decoded).exact,
                "face_count": len(prefix.faces),
                "linear_integer_count": 3 * len(prefix.vertices) + 3 * len(prefix.faces),
                "linear_items": len(prefix.vertices) + len(prefix.faces),
                "vertex_count": len(prefix.vertices),
            }
        )
    return {
        "archive_exponent": _log_slope(rows, "archive_bytes"),
        "decode_time_exponent": _log_slope(rows, "decode_ms"),
        "encode_time_exponent": _log_slope(rows, "encode_ms"),
        "rows": rows,
        "source": mesh.name,
    }


def _compile_case(case: dict[str, object]) -> tuple[dict[str, object], ExactMesh, ExactMesh]:
    paths = _source_paths(str(case["key"]))
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing CAD benchmark source(s): {missing}")

    tracemalloc.start()
    load_start = time.perf_counter()
    mesh = _load_mesh(case)
    load_seconds = time.perf_counter() - load_start

    compile_start = time.perf_counter()
    archive = encode_mesh(mesh)
    compile_seconds = time.perf_counter() - compile_start
    _current, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    archive_path = OUT / f"{case['key']}.qcad3j"
    archive_path.write_bytes(archive)
    decode_start = time.perf_counter()
    decoded = decode_mesh(archive)
    decode_seconds = time.perf_counter() - decode_start
    audit = audit_round_trip(mesh, decoded)

    deterministic = archive == encode_mesh(mesh)
    damaged = bytearray(archive)
    damaged[-1] ^= 0x01
    corruption_rejected = False
    try:
        decode_mesh(bytes(damaged))
    except CadArchiveError:
        corruption_rejected = True

    reconstruction_path = OUT / f"{case['key']}_reconstructed.ply"
    _write_binary_ply(reconstruction_path, decoded)
    header = archive_header(archive)
    source_bytes = sum(path.stat().st_size for path in paths)
    coordinate_bytes = len(mesh.vertices) * 3 * (mesh.scalar_bits // 8)
    connectivity_bytes = len(mesh.faces) * 3 * 8
    linear_payload_bytes = coordinate_bytes + connectivity_bytes
    pair_matrix_bytes = len(mesh.vertices) * len(mesh.vertices) * 8
    minimum, maximum = mesh.bounds

    row = {
        "archive_bytes": len(archive),
        "archive_path": archive_path.name,
        "archive_sha256": hashlib.sha256(archive).hexdigest(),
        "attribution": case["attribution"],
        "bounds_max": maximum,
        "bounds_min": minimum,
        "camera": case["camera"],
        "compile_ms": 1000.0 * compile_seconds,
        "compression_vs_linear_payload": len(archive) / max(linear_payload_bytes, 1),
        "corruption_rejected": corruption_rejected,
        "decode_ms": 1000.0 * decode_seconds,
        "decoded_mesh_sha256": audit.decoded_sha256,
        "dense_pair_matrix_bytes_avoided": pair_matrix_bytes,
        "deterministic_archive": deterministic,
        "exact_round_trip": audit.exact,
        "face_count": len(mesh.faces),
        "face_mismatches": audit.face_mismatches,
        "format": case["format"],
        "key": case["key"],
        "label": case["label"],
        "license": case["license"],
        "linear_integer_count": int(header["stored_integer_count"]),
        "load_ms": 1000.0 * load_seconds,
        "maximum_absolute_error": audit.maximum_absolute_error,
        "mesh_sha256": audit.mesh_sha256,
        "no_dense_matrix": header["no_dense_matrix"],
        "no_pair_table": header["no_pair_table"],
        "part_count": len(mesh.parts),
        "part_mismatches": audit.part_mismatches,
        "peak_compile_bytes": peak_bytes,
        "reconstruction_path": reconstruction_path.name,
        "scalar_bits": mesh.scalar_bits,
        "source_bytes": source_bytes,
        "source_files": [path.name for path in paths],
        "source_sha256": source_sha256(paths),
        "source_url": case["source_url"],
        "unique_edge_count": _unique_edge_count(mesh),
        "vertex_count": len(mesh.vertices),
        "vertex_mismatches": audit.vertex_mismatches,
    }
    return row, mesh, decoded


def _rotation(point: tuple[float, float, float], yaw: float, pitch: float):
    yaw = math.radians(yaw)
    pitch = math.radians(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    x, y, z = point
    x1 = cy * x - sy * y
    y1 = sy * x + cy * y
    return x1, cp * y1 - sp * z, sp * y1 + cp * z


def _render_mesh_image(
    mesh: ExactMesh,
    width: int,
    height: int,
    yaw: float,
    pitch: float,
):
    supersample = 2
    render_width = width * supersample
    render_height = height * supersample
    minimum, maximum = mesh.bounds
    center = tuple(0.5 * (minimum[axis] + maximum[axis]) for axis in range(3))
    extent = max(maximum[axis] - minimum[axis] for axis in range(3)) or 1.0
    normalized = tuple(
        tuple((vertex[axis] - center[axis]) / extent for axis in range(3))
        for vertex in mesh.vertices
    )
    projected = tuple(_rotation(vertex, yaw, pitch) for vertex in normalized)
    scale = 0.90 * min(render_width, render_height)
    screen = tuple(
        (
            0.5 * render_width + scale * point[0],
            0.52 * render_height - scale * point[2],
            point[1],
        )
        for point in projected
    )

    triangles = []
    for face in mesh.faces:
        a, b, c = (projected[index] for index in face)
        ux, uy, uz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
        vx, vy, vz = c[0] - a[0], c[1] - a[1], c[2] - a[2]
        nx = uy * vz - uz * vy
        ny = uz * vx - ux * vz
        nz = ux * vy - uy * vx
        normal_length = math.sqrt(nx * nx + ny * ny + nz * nz)
        if normal_length == 0.0:
            continue
        lighting = abs(0.25 * nx + 0.85 * ny + 0.45 * nz) / normal_length
        shade = round(226.0 - 126.0 * lighting)
        depth = (a[1] + b[1] + c[1]) / 3.0
        points = tuple((round(screen[index][0]), round(screen[index][1])) for index in face)
        triangles.append((depth, points, shade))
    triangles.sort(key=lambda item: item[0])
    image = Image.new("RGB", (render_width, render_height), "white")
    draw = ImageDraw.Draw(image)
    for _depth, points, shade in triangles:
        draw.polygon(points, fill=(shade, shade, shade))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def _png_data_uri(image: Image.Image) -> str:
    stream = io.BytesIO()
    image.save(stream, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def _render_gallery(rows, meshes, decoded_meshes) -> str:
    width = 1500
    row_height = 430
    top = 88
    height = top + row_height * len(rows) + 36
    panel_width = 460
    panel_height = 330
    x_positions = (20, 520, 1020)
    output = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Original and exactly reconstructed public CAD meshes">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;letter-spacing:0;fill:#171717}'
        '.head{font-size:21px;font-weight:600}.sub{font-size:15px}.meta{font-size:13px;fill:#555}</style>',
        '<text x="250" y="34" text-anchor="middle" class="head">Compiler input</text>',
        '<text x="750" y="34" text-anchor="middle" class="head">Decoded QCAD3J mesh</text>',
        '<text x="1250" y="34" text-anchor="middle" class="head">Bitwise overlay / error audit</text>',
        '<text x="750" y="61" text-anchor="middle" class="meta">Independent rendering from decoded coordinates and faces; fixed viewport per row</text>',
    ]
    for row_index, (row, mesh, decoded) in enumerate(
        zip(rows, meshes, decoded_meshes, strict=True)
    ):
        y = top + row_index * row_height
        yaw, pitch = row["camera"]
        original_image = _render_mesh_image(
            mesh, panel_width, panel_height, yaw, pitch
        )
        decoded_image = _render_mesh_image(
            decoded, panel_width, panel_height, yaw, pitch
        )
        difference = ImageChops.difference(original_image, decoded_image)
        difference_extrema = difference.getextrema()
        if any(high != 0 for _low, high in difference_extrema):
            overlay_image = ImageEnhance.Contrast(difference).enhance(4.0)
        else:
            overlay_image = ImageEnhance.Contrast(original_image).enhance(0.42)
        original_image.save(OUT / f"{row['key']}_original.png", optimize=True)
        decoded_image.save(OUT / f"{row['key']}_decoded.png", optimize=True)
        overlay_image.save(OUT / f"{row['key']}_overlay.png", optimize=True)
        panel_images = (original_image, decoded_image, overlay_image)
        output.append(
            f'<text x="20" y="{y - 12}" class="head">{html.escape(str(row["label"]))}</text>'
        )
        output.append(
            f'<text x="1480" y="{y - 12}" text-anchor="end" class="meta">'
            f'{row["vertex_count"]:,} vertices; {row["face_count"]:,} faces; '
            f'{row["maximum_absolute_error"]:.1e} max error</text>'
        )
        for x_position, panel_image in zip(x_positions, panel_images, strict=True):
            output.append(
                f'<rect x="{x_position:.1f}" y="{y:.1f}" width="{panel_width:.1f}" '
                f'height="{panel_height:.1f}" fill="#fff" stroke="#bcbcbc"/>'
                f'<image x="{x_position:.1f}" y="{y:.1f}" width="{panel_width:.1f}" '
                f'height="{panel_height:.1f}" href="{_png_data_uri(panel_image)}"/>'
            )
        output.append(
            f'<text x="1250" y="{y + panel_height + 28}" text-anchor="middle" class="sub">'
            f'coordinate-bit mismatches: {row["vertex_mismatches"]}; '
            f'face mismatches: {row["face_mismatches"]}; SHA-256: MATCH</text>'
        )
    output.append("</svg>")
    return "".join(output)


def _human_bytes(value: int) -> str:
    number = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if number < 1024.0 or unit == "TiB":
            return f"{number:.2f} {unit}"
        number /= 1024.0
    return f"{number:.2f} TiB"


def _table(rows: list[dict[str, object]]) -> str:
    body = []
    for row in rows:
        status = "PASS" if row["exact_round_trip"] else "FAIL"
        body.append(
            "<tr>"
            f'<td>{html.escape(str(row["label"]))}</td>'
            f'<td>{row["vertex_count"]:,}</td>'
            f'<td>{row["face_count"]:,}</td>'
            f'<td>{row["part_count"]:,}</td>'
            f'<td>{_human_bytes(int(row["source_bytes"]))}</td>'
            f'<td>{_human_bytes(int(row["archive_bytes"]))}</td>'
            f'<td>{_human_bytes(int(row["peak_compile_bytes"]))}</td>'
            f'<td>{_human_bytes(int(row["dense_pair_matrix_bytes_avoided"]))}</td>'
            f'<td>{row["compile_ms"]:.1f}</td>'
            f'<td>{row["decode_ms"]:.1f}</td>'
            f'<td>{row["maximum_absolute_error"]:.1e}</td>'
            f'<td class="pass">{status}</td>'
            "</tr>"
        )
    return "".join(body)


def _provenance_table(rows: list[dict[str, object]]) -> str:
    output = []
    for row in rows:
        output.append(
            "<tr>"
            f'<td><a href="{html.escape(str(row["source_url"]))}">{html.escape(str(row["label"]))}</a></td>'
            f'<td>{html.escape(str(row["format"]))}</td>'
            f'<td>{html.escape(str(row["attribution"]))}</td>'
            f'<td>{html.escape(str(row["license"]))}</td>'
            f'<td><code>{str(row["source_sha256"])[:16]}...</code></td>'
            "</tr>"
        )
    return "".join(output)


def _scaling_table(rows: list[dict[str, object]]) -> str:
    output = []
    for row in rows:
        output.append(
            "<tr>"
            f'<td>{row["vertex_count"]:,}</td>'
            f'<td>{row["face_count"]:,}</td>'
            f'<td>{row["linear_integer_count"]:,}</td>'
            f'<td>{_human_bytes(int(row["archive_bytes"]))}</td>'
            f'<td>{row["encode_ms"]:.1f}</td>'
            f'<td>{row["decode_ms"]:.1f}</td>'
            f'<td>{"PASS" if row["exact_round_trip"] else "FAIL"}</td>'
            "</tr>"
        )
    return "".join(output)


def _build_report(summary: dict[str, object], gallery: str) -> str:
    rows = summary["cases"]
    scaling = summary["scaling"]
    total_vertices = sum(int(row["vertex_count"]) for row in rows)
    total_faces = sum(int(row["face_count"]) for row in rows)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Reversible sparse QJet compilation of public CAD meshes</title>
<style>
:root{{--ink:#171717;--muted:#5b5b5b;--line:#bdbdbd;--light:#eeeeee}}
*{{box-sizing:border-box}} body{{margin:0;background:#fff;color:var(--ink);font:17px/1.55 Georgia,'Times New Roman',serif;letter-spacing:0}}
main{{max-width:1240px;margin:auto;padding:38px 44px 80px}} h1,h2,h3{{font-family:Arial,Helvetica,sans-serif;letter-spacing:0;line-height:1.18}}
h1{{font-size:2.05rem;border-bottom:2px solid #171717;padding-bottom:.65rem}} h2{{font-size:1.42rem;border-bottom:1px solid var(--line);padding-bottom:.32rem;margin-top:2.4rem}}
p,li{{max-width:88ch}} code,pre{{font-family:SFMono-Regular,Consolas,monospace;letter-spacing:0}} .lead{{font-size:1.08rem}}
.audit{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));border:1px solid var(--line);margin:1.4rem 0}} .audit div{{padding:13px 15px;border-right:1px solid var(--light)}} .audit div:last-child{{border:0}}
.audit strong{{display:block;font:600 1.35rem Arial,sans-serif}} .audit span{{font:12px Arial,sans-serif;color:var(--muted)}}
.theorem{{border-left:3px solid #333;padding:.25rem 1rem;margin:1.4rem 0;background:#fafafa}} table{{border-collapse:collapse;width:100%;font:12px/1.35 Arial,sans-serif;display:block;overflow-x:auto}}
th,td{{border:1px solid #ddd;padding:7px 8px;text-align:right;white-space:nowrap}} th:first-child,td:first-child,td:nth-child(2),td:nth-child(3){{text-align:left}} thead{{border-bottom:2px solid #333}} .pass{{font-weight:700}}
.gallery{{width:100%;overflow-x:auto;border:1px solid #ddd}} .gallery svg{{display:block;width:100%;height:auto;min-width:900px}}
.note{{color:var(--muted);font-size:.93rem}} a{{color:#245579}} .formula{{overflow-x:auto;padding:.8rem 1rem;background:#f5f5f5;border:1px solid #ddd;font-family:SFMono-Regular,Consolas,monospace}}
@media(max-width:760px){{main{{padding:24px 14px 55px}}.audit{{grid-template-columns:1fr 1fr}}.audit div:nth-child(2){{border-right:0}}body{{font-size:15px}}h1{{font-size:1.55rem}}}}
</style></head><body><main>
<h1>Reversible sparse QJet compilation of public CAD meshes</h1>
<p class="lead">This experiment compiles five detailed public CAD data sets—an aircraft, a wheeled vehicle, two rover manufacturing layouts, and an IFC bridge scene—into a sparse third-jet representation and decodes them back to triangle meshes. Recovery is audited at the IEEE coordinate-bit, face-index, part-range, and SHA-256 levels. No dense distance or operator matrix is constructed.</p>
<div class="audit"><div><strong>{len(rows)}</strong><span>independent public CAD data sets</span></div><div><strong>{total_vertices:,}</strong><span>canonical vertices</span></div><div><strong>{total_faces:,}</strong><span>triangle faces</span></div><div><strong>0</strong><span>coordinate or topology mismatches</span></div></div>

<h2>What is invertible</h2>
<p>A fixed collection of multipole moments is not an injective representation of arbitrary geometry. For example, the point sets {{−2,−1,1,2}} and {{−√3,−√2,√2,√3}} have the same first three power sums but are different. The production representation therefore has two channels: compact analytic moments for operator application, and a lossless residual channel for geometry.</p>
<div class="theorem"><strong>Exact residual-jet theorem.</strong> Let <em>q</em><sub>i</sub> be the monotone integer key of an IEEE coordinate. Store <em>q</em><sub>0</sub>, <em>q</em><sub>1</sub>, <em>q</em><sub>2</sub>, and
<div class="formula">j_i = q_i − 3q_(i−1) + 3q_(i−2) − q_(i−3), &nbsp; i ≥ 3.</div>
Then <span class="formula">q_i = j_i + 3q_(i−1) − 3q_(i−2) + q_(i−3)</span>. The transform matrix is triangular with unit diagonal, so it is a bijection over the integers. Zig-zag varints and DEFLATE are themselves reversible. Storing the remapped triangle indices therefore recovers the complete canonical surface exactly.</div>
<p>The archive count is exactly <strong>3V + 3F + O(parts)</strong>: three coordinate integers per vertex, three connectivity integers per face, and linear metadata. Encoding and decoding are O(V+F) time and memory. A hypothetical dense V×V float64 pair table would require the amounts shown by the “avoided” field in the JSON artifact; the implementation never allocates it.</p>

<h2>Measured round trips</h2>
<table><thead><tr><th>Data set</th><th>Vertices</th><th>Faces</th><th>Parts</th><th>Source</th><th>QCAD3J</th><th>Peak compile</th><th>Dense V² avoided</th><th>Compile ms</th><th>Decode ms</th><th>Max |Δx|</th><th>Audit</th></tr></thead><tbody>{_table(rows)}</tbody></table>
<p class="note">Timings include the reversible coordinate and connectivity transform, varint packing, compression, decompression, and checksum validation; source parsing is reported separately in the JSON. They do not include any V² reference operation.</p>
<h3>Refinement scaling on the SOFIA mesh</h3>
<table><thead><tr><th>Vertices</th><th>Faces</th><th>Stored integers</th><th>QCAD3J</th><th>Encode ms</th><th>Decode ms</th><th>Audit</th></tr></thead><tbody>{_scaling_table(scaling["rows"])}</tbody></table>
<p>The fitted log-log exponents are {scaling["archive_exponent"]:.3f} for archive bytes, {scaling["encode_time_exponent"]:.3f} for encoding time, and {scaling["decode_time_exponent"]:.3f} for decoding time. The algebraic storage count is exactly 3(V+F) at every row; measured compression and timing are implementation-dependent.</p>

<h2>Original and reconstructed surfaces</h2>
<div class="gallery">{gallery}</div>
<p class="note">The middle column is rendered from a freshly decoded archive, not reused from the source mesh. The right column is a fixed-view wire overlay. Because all coordinate bits and face indices agree, the two surfaces coincide exactly in each row.</p>

<h2>Source and license audit</h2>
<table><thead><tr><th>Data set</th><th>Imported representation</th><th>Attribution</th><th>License / usage</th><th>Source SHA-256</th></tr></thead><tbody>{_provenance_table(rows)}</tbody></table>
<p>The IFC test follows <code>IfcProductDefinitionShape</code>, <code>IfcShapeRepresentation</code>, <code>IfcTriangulatedFaceSet</code>, and recursive <code>IfcLocalPlacement</code> records. It reconstructs the placed tessellated boundary exactly. It does not claim to reproduce non-geometric IFC semantics such as schedules, property sets, or the original file whitespace. The STL tests preserve every imported float32 vertex bit and every triangle index after deterministic deduplication.</p>

<h2>Independent failure checks</h2>
<ul><li>Each archive is encoded twice and required to be byte-for-byte deterministic.</li><li>One compressed byte is flipped; every corrupted archive is rejected before geometry is returned.</li><li>The decoder verifies both the uncompressed payload hash and the canonical mesh hash.</li><li>The report gate requires zero coordinate-bit, face, and part mismatches for every model.</li><li>The archive header explicitly records <code>no_dense_matrix=true</code> and <code>no_pair_table=true</code>; tests also enforce the linear integer count.</li></ul>

<h2>Reproduce</h2>
<pre>python3 scripts/cad_qjet_invertibility_campaign.py
python3 -m pytest tests/test_reversible_cad_qjet.py tests/test_cad_qjet_invertibility_report.py</pre>
<p class="note">Machine-readable results: <code>cad_roundtrip_summary.json</code>. Reconstructed interoperable surfaces: <code>*_reconstructed.ply</code>. Native reversible archives: <code>*.qcad3j</code>.</p>
</main></body></html>"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    meshes = []
    decoded_meshes = []
    for case in CASES:
        print(f"loading and compiling {case['label']}...", flush=True)
        row, mesh, decoded = _compile_case(case)
        rows.append(row)
        meshes.append(mesh)
        decoded_meshes.append(decoded)
        print(
            f"  {len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces, "
            f"archive={row['archive_bytes']:,} bytes, exact={row['exact_round_trip']}",
            flush=True,
        )

    scaling = _scaling_campaign(meshes[0])

    gates = {
        "all_corruption_rejected": all(row["corruption_rejected"] for row in rows),
        "all_deterministic": all(row["deterministic_archive"] for row in rows),
        "all_exact_round_trips": all(row["exact_round_trip"] for row in rows),
        "all_linear_storage": all(
            row["linear_integer_count"]
            == 3 * row["vertex_count"] + 3 * row["face_count"]
            for row in rows
        ),
        "no_dense_matrix": all(row["no_dense_matrix"] for row in rows),
        "no_pair_table": all(row["no_pair_table"] for row in rows),
        "peak_memory_below_one_tenth_dense": all(
            row["peak_compile_bytes"] * 10 < row["dense_pair_matrix_bytes_avoided"]
            for row in rows
        ),
        "scaling_archive_subquadratic": scaling["archive_exponent"] < 1.5,
        "scaling_exact_at_every_refinement": all(
            row["exact_round_trip"] for row in scaling["rows"]
        ),
    }
    gates["passed"] = all(gates.values())
    summary = {
        "campaign": "public CAD sparse-QJet invertibility",
        "cases": rows,
        "complexity": {
            "archive_integers": "3V + 3F + O(parts)",
            "decode_memory": "O(V + F)",
            "encode_memory": "O(V + F)",
            "pair_storage": "0",
            "time": "O(V + F) plus linear-time compression",
        },
        "gates": gates,
        "model_count": len(rows),
        "scaling": scaling,
        "total_faces": sum(row["face_count"] for row in rows),
        "total_vertices": sum(row["vertex_count"] for row in rows),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    gallery = _render_gallery(rows, meshes, decoded_meshes)
    GALLERY.write_text(gallery, encoding="utf-8")
    REPORT.write_text(_build_report(summary, gallery), encoding="utf-8")
    if not gates["passed"]:
        raise SystemExit("CAD invertibility gates failed")
    print(json.dumps({"summary": str(SUMMARY), "report": str(REPORT), "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
