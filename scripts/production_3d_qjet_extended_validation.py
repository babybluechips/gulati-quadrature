#!/usr/bin/env python3
"""Extended no-dense validation campaign for the production 3D Riesz QJet."""

from __future__ import annotations

import csv
import html
import json
import math
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inverse_shape.arbitrary_surface import triangle_lumped_vertex_weights  # noqa: E402
from inverse_shape.conic_pencil_surface import (  # noqa: E402
    aircraft_conic_bundle_qjet,
    bent_conic_tube_qjet,
    toroidal_conic_bundle_qjet,
    twisted_ellipse_tube_qjet,
)
from inverse_shape.riesz_near_linear import ProductionRieszQJet  # noqa: E402
from inverse_shape.testing.reference_pairwise import (  # noqa: E402
    reference_weighted_distance_graph,
)


OUT = ROOT / "outputs" / "production_3d_qjet_html"
PI = math.pi
TAU = 2.0 * PI
TOLERANCE = 3.0e-13
MAXIMUM_ORDER = 16
LEAF_SIZE = 4
STANDARD_ERROR_GATE = 1.0e-12
STRESS_ERROR_GATE = 2.0e-12
TRANSFORM_GATE = 2.0e-12


def cross(left, right):
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def norm(value):
    return math.sqrt(sum(component * component for component in value))


def sphere(latitude_count=8, longitude_count=12, scales=(1.0, 1.0, 1.0)):
    points = []
    weights = []
    a, b, c = (float(value) for value in scales)
    dtheta = PI / latitude_count
    dphase = TAU / longitude_count
    for latitude in range(latitude_count):
        theta = PI * (latitude + 0.5) / latitude_count
        sine = math.sin(theta)
        cosine = math.cos(theta)
        for longitude in range(longitude_count):
            phase = TAU * longitude / longitude_count
            cp, sp = math.cos(phase), math.sin(phase)
            points.append((a * sine * cp, b * sine * sp, c * cosine))
            d_theta = (a * cosine * cp, b * cosine * sp, -c * sine)
            d_phase = (-a * sine * sp, b * sine * cp, 0.0)
            weights.append(norm(cross(d_theta, d_phase)) * dtheta * dphase)
    return tuple(points), tuple(weights)


def torus(major_count=10, minor_count=6):
    points = []
    weights = []
    major_radius = 1.4
    minor_radius = 0.35
    du = TAU / major_count
    dv = TAU / minor_count
    for major in range(major_count):
        u = TAU * major / major_count
        for minor in range(minor_count):
            v = TAU * minor / minor_count
            radial = major_radius + minor_radius * math.cos(v)
            points.append(
                (
                    radial * math.cos(u),
                    radial * math.sin(u),
                    minor_radius * math.sin(v),
                )
            )
            weights.append(minor_radius * radial * du * dv)
    return tuple(points), tuple(weights)


def folded_sheet(nx=10, ny=6):
    points = []
    weights = []
    dx = 2.0 / nx
    dy = 1.6 / ny
    for ix in range(nx):
        x = -1.0 + 2.0 * (ix + 0.5) / nx
        for iy in range(ny):
            y = -0.8 + 1.6 * (iy + 0.5) / ny
            phase = 2.3 * y + 0.4 * x
            z = 0.35 * math.sin(1.7 * x) + 0.18 * math.sin(phase)
            zx = 0.595 * math.cos(1.7 * x) + 0.072 * math.cos(phase)
            zy = 0.414 * math.cos(phase)
            points.append((x, y, z))
            weights.append(math.sqrt(1.0 + zx * zx + zy * zy) * dx * dy)
    return tuple(points), tuple(weights)


def mobius(around=12, across=5):
    points = []
    weights = []
    half_width = 0.28
    du = TAU / around
    dv = 2.0 * half_width / across
    for along in range(around):
        u = TAU * along / around
        cu, su = math.cos(u), math.sin(u)
        ch, sh = math.cos(0.5 * u), math.sin(0.5 * u)
        for transverse in range(across):
            v = half_width * (2.0 * (transverse + 0.5) / across - 1.0)
            radial = 1.2 + v * ch
            points.append((radial * cu, radial * su, v * sh))
            du_vector = (
                -0.5 * v * sh * cu - radial * su,
                -0.5 * v * sh * su + radial * cu,
                0.5 * v * ch,
            )
            dv_vector = (ch * cu, ch * su, sh)
            weights.append(norm(cross(du_vector, dv_vector)) * du * dv)
    return tuple(points), tuple(weights)


def star_surface():
    points, weights = sphere()
    output = []
    for x, y, z in points:
        radial = math.sqrt(max(x * x + y * y, 0.0))
        scale = 1.0 + 0.16 * (4.0 * z * z - 1.0) * radial
        output.append((scale * x, scale * y, scale * z))
    return tuple(output), weights


def spherical_spiral(count=128):
    points = []
    golden_angle = PI * (3.0 - math.sqrt(5.0))
    for index in range(count):
        z = 1.0 - 2.0 * (index + 0.5) / count
        radius = math.sqrt(max(1.0 - z * z, 0.0))
        phase = golden_angle * index
        points.append((radius * math.cos(phase), radius * math.sin(phase), z))
    return tuple(points), (4.0 * PI / count,) * count


def faceted_octahedron(refinements=2):
    vertices = [
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 0.0, -1.0),
    ]
    faces = [
        (0, 2, 4),
        (2, 1, 4),
        (1, 3, 4),
        (3, 0, 4),
        (2, 0, 5),
        (1, 2, 5),
        (3, 1, 5),
        (0, 3, 5),
    ]
    for _ in range(refinements):
        midpoint_cache = {}

        def midpoint(left, right):
            edge = (min(left, right), max(left, right))
            if edge not in midpoint_cache:
                a, b = vertices[left], vertices[right]
                midpoint_cache[edge] = len(vertices)
                vertices.append(tuple(0.5 * (a[axis] + b[axis]) for axis in range(3)))
            return midpoint_cache[edge]

        refined = []
        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            refined.extend(
                ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))
            )
        faces = refined
    points = tuple(vertices)
    triangles = tuple(faces)
    return points, triangle_lumped_vertex_weights(points, triangles)


def generated_conic(factory):
    nodes = factory.generate_nodes()
    return nodes.points, nodes.weights


def airplane_assembly():
    fuselage_points, fuselage_weights = generated_conic(
        aircraft_conic_bundle_qjet(4.0, 8, 10)
    )
    points = [(z, x, y) for x, y, z in fuselage_points]
    weights = list(fuselage_weights)

    def add_tapered_wing(
        root_x, root_span, span, root_chord, tip_chord, span_count, chord_count
    ):
        ds = 1.0 / span_count
        dt = 1.0 / chord_count
        for side in (-1.0, 1.0):
            for face_height in (-0.026, 0.026):
                for span_index in range(span_count):
                    s = (span_index + 0.5) * ds
                    chord = root_chord + s * (tip_chord - root_chord)
                    chord_slope = tip_chord - root_chord
                    for chord_index in range(chord_count):
                        t = (chord_index + 0.5) * dt - 0.5
                        points.append(
                            (
                                root_x + t * chord,
                                side * (root_span + span * s),
                                face_height,
                            )
                        )
                        tangent_s = (t * chord_slope, side * span, 0.0)
                        tangent_t = (chord, 0.0, 0.0)
                        weights.append(norm(cross(tangent_s, tangent_t)) * ds * dt)

    add_tapered_wing(-0.10, 0.42, 1.85, 1.45, 0.62, 4, 4)
    add_tapered_wing(-1.48, 0.24, 0.72, 0.72, 0.30, 2, 3)

    ds = 1.0 / 3.0
    dt = 1.0 / 3.0
    for side in (-1.0, 1.0):
        for height_index in range(3):
            s = (height_index + 0.5) * ds
            for chord_index in range(3):
                t = (chord_index + 0.5) * dt
                chord = 0.62 * (1.0 - 0.45 * s)
                points.append((-1.72 + chord * t, side * 0.018, 0.12 + 0.72 * s))
                weights.append(chord * 0.72 * ds * dt)
    return tuple(points), tuple(weights)


def car_assembly(body_u=6, body_v=12, wheel_u=4, wheel_v=4):
    points = []
    weights = []
    du = PI / body_u
    dv = TAU / body_v

    def body_point(u, v):
        longitudinal = 2.35 * math.cos(u)
        radius = math.sin(u)
        width = (0.70 + 0.08 * radius * radius) * radius
        side = math.cos(v)
        vertical = math.sin(v)
        cabin = (
            0.52
            * math.exp(-((longitudinal + 0.05) / 1.08) ** 4)
            * (0.5 + 0.5 * vertical) ** 2
            * radius
        )
        return (
            longitudinal,
            width * side,
            -0.12 + 0.39 * radius * vertical + cabin,
        )

    difference_step = 1.0e-5
    for u_index in range(body_u):
        u = PI * (u_index + 0.5) / body_u
        for v_index in range(body_v):
            v = TAU * v_index / body_v
            point = body_point(u, v)
            up = body_point(u + difference_step, v)
            down = body_point(u - difference_step, v)
            right = body_point(u, v + difference_step)
            left = body_point(u, v - difference_step)
            tangent_u = tuple(
                (up[axis] - down[axis]) / (2.0 * difference_step)
                for axis in range(3)
            )
            tangent_v = tuple(
                (right[axis] - left[axis]) / (2.0 * difference_step)
                for axis in range(3)
            )
            points.append(point)
            weights.append(norm(cross(tangent_u, tangent_v)) * du * dv)

    wheel_radius = 0.28
    tube_radius = 0.075
    da = TAU / wheel_u
    db = TAU / wheel_v
    for center_x in (-1.25, 1.25):
        for center_y in (-0.77, 0.77):
            for around_index in range(wheel_u):
                a = TAU * around_index / wheel_u
                for tube_index in range(wheel_v):
                    b = TAU * tube_index / wheel_v
                    radial = wheel_radius + tube_radius * math.cos(b)
                    points.append(
                        (
                            center_x + radial * math.cos(a),
                            center_y + tube_radius * math.sin(b),
                            -0.34 + radial * math.sin(a),
                        )
                    )
                    weights.append(tube_radius * radial * da * db)
    return tuple(points), tuple(weights)


def append_box(points, weights, center, size, counts):
    cx, cy, cz = center
    sx, sy, sz = size
    nx, ny, nz = counts
    for sign in (-1.0, 1.0):
        for iy in range(ny):
            y = cy + sy * ((iy + 0.5) / ny - 0.5)
            for iz in range(nz):
                z = cz + sz * ((iz + 0.5) / nz - 0.5)
                points.append((cx + sign * 0.5 * sx, y, z))
                weights.append(sy * sz / (ny * nz))
        for ix in range(nx):
            x = cx + sx * ((ix + 0.5) / nx - 0.5)
            for iz in range(nz):
                z = cz + sz * ((iz + 0.5) / nz - 0.5)
                points.append((x, cy + sign * 0.5 * sy, z))
                weights.append(sx * sz / (nx * nz))
        for ix in range(nx):
            x = cx + sx * ((ix + 0.5) / nx - 0.5)
            for iy in range(ny):
                y = cy + sy * ((iy + 0.5) / ny - 0.5)
                points.append((x, y, cz + sign * 0.5 * sz))
                weights.append(sx * sy / (nx * ny))


def bridge_assembly():
    points = []
    weights = []
    append_box(points, weights, (0.0, 0.0, 0.38), (6.0, 1.4, 0.18), (6, 2, 1))
    for tower_x in (-1.75, 1.75):
        for tower_y in (-0.56, 0.56):
            append_box(
                points,
                weights,
                (tower_x, tower_y, 1.18),
                (0.18, 0.18, 1.78),
                (1, 1, 3),
            )

    along_count = 8
    around_count = 4
    dx = 6.0 / along_count
    dphase = TAU / around_count
    tube_radius = 0.035
    for cable_y in (-0.70, 0.70):
        for along_index in range(along_count):
            x = -3.0 + (along_index + 0.5) * dx
            z = 0.98 + 0.14 * x * x
            slope = 0.28 * x
            tangent_length = math.sqrt(1.0 + slope * slope)
            normal = (-slope / tangent_length, 0.0, 1.0 / tangent_length)
            for around_index in range(around_count):
                phase = TAU * around_index / around_count
                points.append(
                    (
                        x + tube_radius * math.sin(phase) * normal[0],
                        cable_y + tube_radius * math.cos(phase),
                        z + tube_radius * math.sin(phase) * normal[2],
                    )
                )
                weights.append(tube_radius * dphase * tangent_length * dx)
    return tuple(points), tuple(weights)


def double_sheet(side=8, gap=0.025):
    points = []
    weights = []
    step = 2.0 / side
    area = math.sqrt(1.0 + 0.01 * 0.01 * 2.0) * step * step
    for height in (-0.5 * gap, 0.5 * gap):
        for ix in range(side):
            x = -1.0 + 2.0 * (ix + 0.5) / side
            for iy in range(side):
                y = -1.0 + 2.0 * (iy + 0.5) / side
                points.append((x, y, height + 0.01 * x * y))
                weights.append(area)
    return tuple(points), tuple(weights)


def multiscale_cluster(count=96):
    points = []
    for index in range(count):
        scale = 2.0 ** (-(index % 18))
        branch = index // 18
        points.append(
            (
                scale,
                (branch + 1) * 2.0e-3 * scale,
                (index + 1) * 2.0e-6 * scale,
            )
        )
    return tuple(points), (1.0 / count,) * count


def fields(points):
    return (
        tuple(
            x + 0.2 * y - 0.1 * z * z + 0.03j * (y + z)
            for x, y, z in points
        ),
        tuple(
            math.sin(2.1 * x) + 0.4 * math.cos(1.7 * y) - 0.2 * z
            for x, y, z in points
        ),
        tuple(
            math.exp(-((x - 0.3) ** 2 + (y + 0.2) ** 2 + (z - 0.1) ** 2))
            + 0.05j * math.sin(x + y + z)
            for x, y, z in points
        ),
    )


def relative_l2(reference, candidate):
    numerator = sum(
        abs(complex(left) - complex(right)) ** 2
        for left, right in zip(reference, candidate, strict=True)
    )
    denominator = sum(abs(complex(value)) ** 2 for value in reference)
    return math.sqrt(numerator / max(denominator, 1.0e-300))


def compile_qjet(points, weights, power):
    return ProductionRieszQJet(
        points,
        weights,
        kernel_power=power,
        tolerance=TOLERANCE,
        maximum_order=MAXIMUM_ORDER,
        leaf_size=LEAF_SIZE,
    )


def validate_case(name, category, geometry, stress=False):
    points, weights = geometry
    test_fields = fields(points)
    rows = []
    for power in (2.0, 3.0):
        started = time.perf_counter()
        qjet = compile_qjet(points, weights, power)
        compile_ms = 1000.0 * (time.perf_counter() - started)
        started = time.perf_counter()
        candidates = qjet.apply_fields(test_fields)
        apply_ms_per_field = (
            1000.0 * (time.perf_counter() - started) / len(test_fields)
        )
        direct_ms = 0.0
        relative_errors = []
        certificate_ratios = []
        certificate_passed = True
        for values, candidate in zip(test_fields, candidates, strict=True):
            started = time.perf_counter()
            reference = reference_weighted_distance_graph(
                points,
                weights,
                values,
                power,
            )
            direct_ms += 1000.0 * (time.perf_counter() - started)
            relative_errors.append(relative_l2(reference, candidate))
            actual_inf = max(
                abs(complex(left) - complex(right))
                for left, right in zip(reference, candidate, strict=True)
            )
            bound = qjet.compression_inf_bound(values)
            reference_scale = max(
                1.0,
                max(abs(complex(value)) for value in reference),
            )
            allowance = 2.0e-12 * reference_scale
            certificate_passed &= actual_inf <= bound + allowance
            certificate_ratios.append(actual_inf / max(bound + allowance, 1.0e-300))

        left = tuple(x - 0.1 * z for x, _y, z in points)
        right = tuple(y + 0.2 * z for _x, y, z in points)
        q_left = qjet.apply(left)
        q_right = qjet.apply(right)
        lhs = sum(
            weights[index] * left[index] * complex(q_right[index]).real
            for index in range(len(points))
        )
        rhs = sum(
            weights[index] * complex(q_left[index]).real * right[index]
            for index in range(len(points))
        )
        adjoint_residual = abs(lhs - rhs) / max(1.0, abs(lhs), abs(rhs))
        energy = sum(
            weights[index] * left[index] * complex(q_left[index]).real
            for index in range(len(points))
        )
        stats = qjet.stats()
        gate = STRESS_ERROR_GATE if stress else STANDARD_ERROR_GATE
        passed = all(
            (
                max(relative_errors) <= gate,
                certificate_passed,
                qjet.constant_residual() == 0.0,
                adjoint_residual <= 2.0e-11,
                energy >= -2.0e-11 * max(1.0, abs(energy)),
                stats["hard_no_quadratic_contract"],
                not stats["quadratic_fallback"],
                not stats["stored_dense_distance_matrix"],
                not stats["stored_dense_operator_matrix"],
                stats["temporary_pair_table_entries"] == 0,
                stats["adaptive_rank"] == 0,
                stats["pair_partition_residual"] == 0,
                stats["near_field_pairs"] <= stats["near_field_pair_budget"],
                stats["analytic_apply_units"] <= stats["analytic_apply_budget"],
            )
        )
        rows.append(
            {
                "shape": name,
                "category": category,
                "stress": stress,
                "kernel_power": int(power),
                "nodes": len(points),
                "fields": len(test_fields),
                "compile_ms": compile_ms,
                "apply_ms_per_field": apply_ms_per_field,
                "direct_ms_per_field": direct_ms / len(test_fields),
                "maximum_relative_error": max(relative_errors),
                "maximum_certificate_ratio": max(certificate_ratios),
                "constant_residual": qjet.constant_residual(),
                "adjoint_residual": adjoint_residual,
                "energy": energy,
                "analytic_pair_fraction": stats["analytic_pair_fraction"],
                "maximum_block_order": stats["maximum_block_order"],
                "near_field_pairs": stats["near_field_pairs"],
                "analytic_blocks": stats["analytic_blocks"],
                "persistent_moment_entries": stats["persistent_moment_entries"],
                "hard_no_quadratic_contract": stats["hard_no_quadratic_contract"],
                "quadratic_fallback": stats["quadratic_fallback"],
                "stored_dense_matrix": stats["stored_dense_operator_matrix"],
                "temporary_pair_table_entries": stats["temporary_pair_table_entries"],
                "passed": passed,
            }
        )
    return rows


def transform_residual(reference, candidate):
    return relative_l2(reference, candidate)


def validate_transformations():
    points, weights = torus(10, 6)
    values = fields(points)[0]
    rows = []
    angle = 0.73
    ca, sa = math.cos(angle), math.sin(angle)
    transformed_geometries = {
        "translation": (
            tuple((x + 3.4, y - 1.7, z + 0.8) for x, y, z in points),
            weights,
            values,
            1.0,
            None,
        ),
        "rotation": (
            tuple((ca * x - sa * y, sa * x + ca * y, z) for x, y, z in points),
            weights,
            values,
            1.0,
            None,
        ),
        "scale": (
            tuple((2.3 * x, 2.3 * y, 2.3 * z) for x, y, z in points),
            tuple(2.3**2 * weight for weight in weights),
            values,
            None,
            None,
        ),
    }
    permutation = tuple(reversed(range(len(points))))
    transformed_geometries["permutation"] = (
        tuple(points[index] for index in permutation),
        tuple(weights[index] for index in permutation),
        tuple(values[index] for index in permutation),
        1.0,
        permutation,
    )

    for power in (2.0, 3.0):
        baseline = compile_qjet(points, weights, power).apply(values)
        for name, (new_points, new_weights, new_values, multiplier, order) in (
            transformed_geometries.items()
        ):
            candidate = compile_qjet(new_points, new_weights, power).apply(new_values)
            if name == "scale":
                multiplier = 2.3 ** (2.0 - power)
            if order is not None:
                restored = [0.0j for _ in points]
                for new_index, old_index in enumerate(order):
                    restored[old_index] = candidate[new_index]
                candidate = tuple(restored)
            expected = tuple(multiplier * complex(value) for value in baseline)
            residual = transform_residual(expected, candidate)
            rows.append(
                {
                    "transformation": name,
                    "kernel_power": int(power),
                    "relative_residual": residual,
                    "gate": TRANSFORM_GATE,
                    "passed": residual <= TRANSFORM_GATE,
                }
            )
    return rows


def project(point, yaw=0.68, pitch=0.48):
    x, y, z = point
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    first, second = cy * x - sy * y, sy * x + cy * y
    return first, cp * z - sp * second, sp * z + cp * second


def write_gallery(path, geometries):
    columns = 3
    panel_width = 320
    panel_height = 235
    rows = math.ceil(len(geometries) / columns)
    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{columns * panel_width}' "
        f"height='{rows * panel_height}' viewBox='0 0 {columns * panel_width} "
        f"{rows * panel_height}'>",
        "<rect width='100%' height='100%' fill='white'/>",
    ]
    for index, (name, _category, geometry, _stress) in enumerate(geometries):
        points, _weights = geometry
        column = index % columns
        row = index // columns
        ox, oy = column * panel_width, row * panel_height
        projected = [project(point) for point in points]
        xs = [value[0] for value in projected]
        ys = [value[1] for value in projected]
        span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0e-14)
        scale = 160.0 / span
        center_x = 0.5 * (max(xs) + min(xs))
        center_y = 0.5 * (max(ys) + min(ys))
        parts.append(
            f"<text x='{ox + 18}' y='{oy + 24}' font-family='serif' "
            f"font-size='16'>{html.escape(name)}</text>"
        )
        parts.append(
            f"<text x='{ox + 18}' y='{oy + 43}' font-family='serif' "
            f"font-size='12' fill='#555'>N = {len(points)}</text>"
        )
        for px, py, depth in sorted(projected, key=lambda value: value[2]):
            sx = ox + 0.5 * panel_width + scale * (px - center_x)
            sy_value = oy + 140.0 - scale * (py - center_y)
            shade = max(35, min(195, int(118 - 25 * depth / max(span, 0.2))))
            parts.append(
                f"<circle cx='{sx:.2f}' cy='{sy_value:.2f}' r='2.1' "
                f"fill='rgb({shade},{shade},{shade})'/>"
            )
        parts.append(
            f"<line x1='{ox + 18}' y1='{oy + 218}' x2='{ox + 302}' "
            f"y2='{oy + 218}' stroke='#ddd'/>"
        )
    parts.append("</svg>")
    path.write_text("".join(parts), encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    geometries = [
        ("sphere", "closed smooth genus 0", sphere(), False),
        ("triaxial ellipsoid", "high aspect smooth", sphere(scales=(2.8, 0.35, 0.18)), False),
        ("torus", "closed smooth genus 1", torus(), False),
        ("star surface", "nonconvex smooth", star_surface(), False),
        ("spherical spiral", "unstructured point surface", spherical_spiral(), False),
        ("folded sheet", "open smooth", folded_sheet(), False),
        ("Mobius strip", "open nonorientable", mobius(), False),
        ("faceted octahedron", "closed polyhedral", faceted_octahedron(), False),
        (
            "bent elliptic tube",
            "curved conic atlas",
            generated_conic(bent_conic_tube_qjet(2.4, 1.2, 0.42, 0.25, 8, 10)),
            False,
        ),
        (
            "twisted elliptic tube",
            "twisted conic atlas",
            generated_conic(twisted_ellipse_tube_qjet(2.8, 0.5, 0.22, 1.1, 8, 10)),
            False,
        ),
        (
            "toroidal conic bundle",
            "periodic curved conic",
            generated_conic(toroidal_conic_bundle_qjet(1.5, 0.28, 0.18, 8, 10)),
            False,
        ),
        ("airplane assembly", "airplane composite conic surface", airplane_assembly(), False),
        ("car assembly", "automotive composite surface", car_assembly(), False),
        ("suspension bridge", "architectural composite surface", bridge_assembly(), False),
        ("close double sheet", "near-collision stress", double_sheet(), True),
        ("multiscale cluster", "dynamic-range stress", multiscale_cluster(), True),
    ]

    cases = []
    for name, category, geometry, stress in geometries:
        cases.extend(validate_case(name, category, geometry, stress))
    transformations = validate_transformations()
    standard_rows = [row for row in cases if not row["stress"]]
    stress_rows = [row for row in cases if row["stress"]]
    categories = sorted({row["category"] for row in cases})
    summary = {
        "method": "fixed_order_symmetric_gegenbauer_riesz_wspd",
        "configuration": {
            "tolerance": TOLERANCE,
            "maximum_order": MAXIMUM_ORDER,
            "leaf_size": LEAF_SIZE,
            "kernel_powers": [2, 3],
            "fields_per_case": 3,
            "standard_error_gate": STANDARD_ERROR_GATE,
            "stress_error_gate": STRESS_ERROR_GATE,
            "transformation_gate": TRANSFORM_GATE,
        },
        "shape_count": len(geometries),
        "case_count": len(cases),
        "field_comparisons": len(cases) * 3,
        "categories": categories,
        "maximum_standard_relative_error": max(
            row["maximum_relative_error"] for row in standard_rows
        ),
        "maximum_stress_relative_error": max(
            row["maximum_relative_error"] for row in stress_rows
        ),
        "maximum_certificate_ratio": max(
            row["maximum_certificate_ratio"] for row in cases
        ),
        "maximum_adjoint_residual": max(row["adjoint_residual"] for row in cases),
        "maximum_transformation_residual": max(
            row["relative_residual"] for row in transformations
        ),
        "all_constant_residuals_zero": all(
            row["constant_residual"] == 0.0 for row in cases
        ),
        "all_hard_no_quadratic": all(
            row["hard_no_quadratic_contract"] for row in cases
        ),
        "no_quadratic_fallback": not any(row["quadratic_fallback"] for row in cases),
        "no_dense_matrix": not any(row["stored_dense_matrix"] for row in cases),
        "no_pair_table": not any(
            row["temporary_pair_table_entries"] for row in cases
        ),
        "cases": cases,
        "transformations": transformations,
    }
    summary["gates"] = {
        "all_case_gates": all(row["passed"] for row in cases),
        "all_transformation_gates": all(row["passed"] for row in transformations),
        "standard_accuracy": summary["maximum_standard_relative_error"]
        <= STANDARD_ERROR_GATE,
        "stress_accuracy": summary["maximum_stress_relative_error"]
        <= STRESS_ERROR_GATE,
        "certificate_bounds": summary["maximum_certificate_ratio"] <= 1.0,
        "weighted_adjoint": summary["maximum_adjoint_residual"] <= 2.0e-11,
        "constant_nullspace": summary["all_constant_residuals_zero"],
        "hard_no_quadratic": summary["all_hard_no_quadratic"],
        "no_quadratic_fallback": summary["no_quadratic_fallback"],
        "no_dense_matrix": summary["no_dense_matrix"],
        "no_pair_table": summary["no_pair_table"],
    }
    summary["gates"]["passed"] = all(summary["gates"].values())
    write_csv(OUT / "extended_validation_cases.csv", cases)
    write_csv(OUT / "transformation_checks.csv", transformations)
    write_gallery(OUT / "extended_shape_gallery.svg", geometries)
    (OUT / "validation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    if not summary["gates"]["passed"]:
        failed = ", ".join(
            name for name, passed in summary["gates"].items() if not passed
        )
        raise RuntimeError(f"extended 3D QJet validation failed: {failed}")
    print(
        json.dumps(
            {
                "output": str(OUT),
                "shape_count": summary["shape_count"],
                "case_count": summary["case_count"],
                "field_comparisons": summary["field_comparisons"],
                "maximum_standard_relative_error": summary[
                    "maximum_standard_relative_error"
                ],
                "maximum_stress_relative_error": summary[
                    "maximum_stress_relative_error"
                ],
                "maximum_transformation_residual": summary[
                    "maximum_transformation_residual"
                ],
                "passed": True,
            },
            indent=2,
        )
    )
    return summary


if __name__ == "__main__":
    main()
