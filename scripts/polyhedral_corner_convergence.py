#!/usr/bin/env python3
"""Continuum convergence campaign for 3D edge/vertex Mellin channels."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from inverse_shape.polyhedral_kondratiev import (  # noqa: E402
    EdgeMellinPencil,
    MellinKondratievRepayment,
    MellinThreeJetChannel,
    PolyhedralMeshTopology,
    SparseSphericalDirichletPencil,
    VertexMellinPencil,
)
from inverse_shape.quadrature import (  # noqa: E402
    PI,
    _abs,
    _log,
    _sqrt,
)


OUT = ROOT / "outputs" / "polyhedral_corner_convergence"
FICHERA_EXPONENT = 0.45417371533061


def _voxel_surface(cells):
    cells = frozenset(tuple(cell) for cell in cells)
    vertices = []
    lookup = {}
    triangles = []

    def vertex(point):
        if point not in lookup:
            lookup[point] = len(vertices)
            vertices.append(tuple(float(value) for value in point))
        return lookup[point]

    directions = (
        ((-1, 0, 0), lambda x, y, z: ((x, y, z), (x, y, z + 1), (x, y + 1, z + 1), (x, y + 1, z))),
        (
            (1, 0, 0),
            lambda x, y, z: (
                (x + 1, y, z),
                (x + 1, y + 1, z),
                (x + 1, y + 1, z + 1),
                (x + 1, y, z + 1),
            ),
        ),
        ((0, -1, 0), lambda x, y, z: ((x, y, z), (x + 1, y, z), (x + 1, y, z + 1), (x, y, z + 1))),
        (
            (0, 1, 0),
            lambda x, y, z: (
                (x, y + 1, z),
                (x, y + 1, z + 1),
                (x + 1, y + 1, z + 1),
                (x + 1, y + 1, z),
            ),
        ),
        ((0, 0, -1), lambda x, y, z: ((x, y, z), (x, y + 1, z), (x + 1, y + 1, z), (x + 1, y, z))),
        (
            (0, 0, 1),
            lambda x, y, z: (
                (x, y, z + 1),
                (x + 1, y, z + 1),
                (x + 1, y + 1, z + 1),
                (x, y + 1, z + 1),
            ),
        ),
    )
    for x, y, z in sorted(cells):
        for direction, quad_builder in directions:
            neighbor = (x + direction[0], y + direction[1], z + direction[2])
            if neighbor in cells:
                continue
            quad = tuple(vertex(point) for point in quad_builder(x, y, z))
            triangles.append((quad[0], quad[1], quad[2]))
            triangles.append((quad[0], quad[2], quad[3]))
    return tuple(vertices), tuple(triangles)


def fichera_surface():
    return _voxel_surface(
        (x, y, z)
        for x in (-1, 0)
        for y in (-1, 0)
        for z in (-1, 0)
        if (x, y, z) != (0, 0, 0)
    )


def _spherical_octant_union(refinement, signs):
    n = int(refinement)
    points = []
    lookup = {}
    triangles = []

    def global_vertex(sign, i, j):
        k = n - i - j
        key = (
            0 if i == 0 else sign[0] * i,
            0 if j == 0 else sign[1] * j,
            0 if k == 0 else sign[2] * k,
        )
        if key not in lookup:
            norm = _sqrt(sum(value * value for value in key))
            lookup[key] = len(points)
            points.append(tuple(value / norm for value in key))
        return lookup[key]

    for sign in signs:
        local = {
            (i, j): global_vertex(sign, i, j)
            for i in range(n + 1)
            for j in range(n + 1 - i)
        }
        for i in range(n):
            for j in range(n - i):
                triangles.append(
                    (local[(i, j)], local[(i + 1, j)], local[(i, j + 1)])
                )
                if i + j <= n - 2:
                    triangles.append(
                        (
                            local[(i + 1, j)],
                            local[(i + 1, j + 1)],
                            local[(i, j + 1)],
                        )
                    )
    keys = {index: key for key, index in lookup.items()}
    return tuple(points), tuple(triangles), keys


def fichera_spherical_link(refinement):
    signs = tuple(
        (sx, sy, sz)
        for sx in (-1, 1)
        for sy in (-1, 1)
        for sz in (-1, 1)
        if (sx, sy, sz) != (1, 1, 1)
    )
    points, triangles, keys = _spherical_octant_union(refinement, signs)
    boundary = tuple(
        index
        for index, key in keys.items()
        if all(value >= 0 for value in key) and any(value == 0 for value in key)
    )
    return SparseSphericalDirichletPencil(points, triangles, boundary)


def _adaptive_simpson(function, left, right, tolerance=2.0e-14, depth=28):
    middle = 0.5 * (left + right)
    f_left = function(left)
    f_middle = function(middle)
    f_right = function(right)
    whole = (right - left) * (f_left + 4.0 * f_middle + f_right) / 6.0

    def recurse(a, b, fa, fm, fb, parent, local_tolerance, remaining):
        center = 0.5 * (a + b)
        left_center = 0.5 * (a + center)
        right_center = 0.5 * (center + b)
        f_left_center = function(left_center)
        f_right_center = function(right_center)
        left_value = (center - a) * (fa + 4.0 * f_left_center + fm) / 6.0
        right_value = (b - center) * (fm + 4.0 * f_right_center + fb) / 6.0
        combined = left_value + right_value
        defect = combined - parent
        if remaining <= 0 or _abs(defect) <= 15.0 * local_tolerance:
            return combined + defect / 15.0
        return recurse(
            a,
            center,
            fa,
            f_left_center,
            fm,
            left_value,
            0.5 * local_tolerance,
            remaining - 1,
        ) + recurse(
            center,
            b,
            fm,
            f_right_center,
            fb,
            right_value,
            0.5 * local_tolerance,
            remaining - 1,
        )

    return recurse(
        left,
        right,
        f_left,
        f_middle,
        f_right,
        whole,
        tolerance,
        depth,
    )


def _asinh(value):
    return _log(value + _sqrt(value * value + 1.0))


def _midpoint(function, count):
    step = 1.0 / count
    return step * sum(function((index + 0.5) * step) for index in range(count))


def _convergence_order(rows, key, tail=3, floor=2.0e-15):
    usable = tuple(row for row in rows if row[key] > floor)
    selected = usable[-tail:]
    if len(selected) < 2:
        return None
    x_values = tuple(_log(float(row["radial_nodes"])) for row in selected)
    y_values = tuple(_log(float(row[key])) for row in selected)
    mean_x = sum(x_values) / len(x_values)
    mean_y = sum(y_values) / len(y_values)
    numerator = sum(
        (x - mean_x) * (y - mean_y)
        for x, y in zip(x_values, y_values, strict=True)
    )
    denominator = sum((x - mean_x) ** 2 for x in x_values)
    return -numerator / denominator


def _multiply_jets(left, right):
    return tuple(
        sum(left[offset] * right[degree - offset] for offset in range(degree + 1))
        for degree in range(4)
    )


def edge_layer_case(topology):
    reentrant = tuple(edge for edge in topology.edges if edge.is_reentrant)
    if len(reentrant) != 3:
        raise RuntimeError("Fichera topology did not expose three re-entrant edges")
    pencil = EdgeMellinPencil.from_edge(reentrant[0])
    exponent = pencil.exponent
    distance = 0.08
    tangent_offset = 0.23
    edge_length = 1.0

    def integrated_kernel(radius):
        radial_distance = _sqrt(distance * distance + radius * radius)
        return (
            _asinh((edge_length + tangent_offset) / radial_distance)
            - _asinh(tangent_offset / radial_distance)
        ) / (4.0 * PI)

    j0 = integrated_kernel(0.0)
    upper = edge_length + tangent_offset
    lower = tangent_offset
    j1 = (
        upper / (distance * distance * _sqrt(distance * distance + upper * upper))
        - lower / (distance * distance * _sqrt(distance * distance + lower * lower))
    ) / (4.0 * PI)
    kernel_jet = (j0, 0.0, -0.5 * j1, 0.0)
    cutoff_jet = (1.0, -8.0, 28.0, -56.0)
    coefficients = tuple(
        exponent * value for value in _multiply_jets(kernel_jet, cutoff_jet)
    )
    channel = MellinThreeJetChannel(
        pencil,
        coefficients,
        label="fichera_reentrant_edge",
    )
    repayment = MellinKondratievRepayment((channel,))

    def integrand(radius):
        if radius == 0.0:
            return 0.0
        return (
            exponent
            * radius ** (exponent - 1.0)
            * (1.0 - radius) ** 8
            * integrated_kernel(radius)
        )

    def transformed_reference(transform_power):
        def transformed(value):
            if value == 0.0:
                return 0.0
            radius = value**transform_power
            return (
                integrand(radius)
                * transform_power
                * value ** (transform_power - 1)
            )

        return _adaptive_simpson(transformed, 0.0, 1.0)

    reference_a = transformed_reference(6)
    reference_b = transformed_reference(9)
    reference = reference_b
    reference_discrepancy = _abs(reference_a - reference_b)
    rows = []
    for count in (16, 32, 64, 128, 256, 512):
        raw = _midpoint(integrand, count)
        evaluation = repayment.repay(raw, 1.0 / count)
        corrected = float(complex(evaluation.value).real)
        rows.append(
            {
                "case": "fichera_reentrant_edge_laplace_slp",
                "radial_nodes": count,
                "step": 1.0 / count,
                "reference": reference,
                "raw": raw,
                "corrected": corrected,
                "raw_abs_error": _abs(raw - reference),
                "corrected_abs_error": _abs(corrected - reference),
                "correction": float(complex(evaluation.correction).real),
                "hurwitz_evaluator_next_term_estimate": (
                    evaluation.ledger.residual_norm
                ),
            }
        )
    return {
        "name": "fichera_reentrant_edge_laplace_slp",
        "geometry": "one planar face of a 270-degree Fichera prism edge",
        "kernel": "3D Laplace single layer, tangential coordinate integrated exactly",
        "target_distance": distance,
        "opening_angle": pencil.opening_angle,
        "kondratiev_exponent": exponent,
        "boundary_quadrature_exponent": pencil.boundary_quadrature_exponent,
        "three_jet_coefficients": coefficients,
        "reference": reference,
        "reference_method": "adaptive Simpson after independent r=t^6 and r=t^9 substitutions",
        "reference_crosscheck_abs": reference_discrepancy,
        "raw_fitted_order": _convergence_order(rows, "raw_abs_error"),
        "corrected_fitted_order": _convergence_order(
            rows,
            "corrected_abs_error",
            floor=2.0e-14,
        ),
        "rows": rows,
        "stored_dense_matrix": False,
        "corner_apply_complexity": "O(1) for this retained edge channel",
    }


def vertex_layer_case():
    pencil = VertexMellinPencil.from_exponent(
        FICHERA_EXPONENT,
        "fichera_dirichlet_vertex",
    )
    exponent = pencil.exponent
    distance = 0.075
    kernel_jet = (
        1.0 / (4.0 * PI * distance),
        0.0,
        -1.0 / (8.0 * PI * distance**3),
        0.0,
    )
    cutoff_jet = (1.0, -8.0, 28.0, -56.0)
    coefficients = _multiply_jets(kernel_jet, cutoff_jet)
    channel = MellinThreeJetChannel(
        pencil,
        coefficients,
        label="fichera_reentrant_vertex",
    )
    repayment = MellinKondratievRepayment((channel,))

    def integrand(radius):
        return (
            radius**exponent
            * (1.0 - radius) ** 8
            / (4.0 * PI * _sqrt(distance * distance + radius * radius))
        )

    def transformed_reference(transform_power):
        def transformed(value):
            if value == 0.0:
                return 0.0
            radius = value**transform_power
            return (
                integrand(radius)
                * transform_power
                * value ** (transform_power - 1)
            )

        return _adaptive_simpson(transformed, 0.0, 1.0)

    reference_a = transformed_reference(8)
    reference_b = transformed_reference(11)
    reference = reference_b
    reference_discrepancy = _abs(reference_a - reference_b)
    rows = []
    for count in (16, 32, 64, 128, 256, 512):
        raw = _midpoint(integrand, count)
        evaluation = repayment.repay(raw, 1.0 / count)
        corrected = float(complex(evaluation.value).real)
        rows.append(
            {
                "case": "fichera_vertex_face_laplace_slp",
                "radial_nodes": count,
                "step": 1.0 / count,
                "reference": reference,
                "raw": raw,
                "corrected": corrected,
                "raw_abs_error": _abs(raw - reference),
                "corrected_abs_error": _abs(corrected - reference),
                "correction": float(complex(evaluation.correction).real),
                "hurwitz_evaluator_next_term_estimate": (
                    evaluation.ledger.residual_norm
                ),
            }
        )
    return {
        "name": "fichera_vertex_face_laplace_slp",
        "geometry": "planar face sector incident to the Fichera vertex",
        "kernel": "3D Laplace single layer at a face-normal target",
        "target_distance": distance,
        "published_kondratiev_exponent": FICHERA_EXPONENT,
        "angular_eigenvalue": pencil.angular_eigenvalue,
        "boundary_quadrature_exponent": pencil.boundary_quadrature_exponent,
        "three_jet_coefficients": coefficients,
        "reference": reference,
        "reference_method": "adaptive Simpson after independent r=t^8 and r=t^11 substitutions",
        "reference_crosscheck_abs": reference_discrepancy,
        "raw_fitted_order": _convergence_order(rows, "raw_abs_error"),
        "corrected_fitted_order": _convergence_order(
            rows,
            "corrected_abs_error",
            floor=2.0e-14,
        ),
        "rows": rows,
        "stored_dense_matrix": False,
        "corner_apply_complexity": "O(1) for this retained vertex channel",
    }


def spherical_pencil_convergence():
    rows = []
    for refinement in (4, 6, 8, 12, 16):
        link = fichera_spherical_link(refinement)
        start = perf_counter()
        pencil = VertexMellinPencil.from_spherical_link(
            link,
            tolerance=2.0e-10,
        )
        elapsed_ms = 1000.0 * (perf_counter() - start)
        eigenpair = pencil.spherical_link_eigenpair
        rows.append(
            {
                "refinement": refinement,
                "spherical_nodes": len(link.points),
                "interior_dofs": len(link.interior_nodes),
                "sparse_nnz": link.stiffness_nnz + link.mass_nnz,
                "angular_eigenvalue": pencil.angular_eigenvalue,
                "kondratiev_exponent": pencil.exponent,
                "exponent_abs_error": _abs(pencil.exponent - FICHERA_EXPONENT),
                "relative_residual": eigenpair.residual,
                "inverse_iterations": eigenpair.inverse_iterations,
                "cg_iterations": eigenpair.cg_iterations,
                "solve_ms": elapsed_ms,
                "stored_dense_matrix": False,
            }
        )
    return rows


def _three_level_power_limit(rows):
    selected = rows[-3:]
    nodes = tuple(float(row["refinement"]) for row in selected)
    values = tuple(float(row["kondratiev_exponent"]) for row in selected)
    ratio = (values[0] - values[1]) / (values[1] - values[2])

    def residual(power):
        return (
            (nodes[0] ** (-power) - nodes[1] ** (-power))
            / (nodes[1] ** (-power) - nodes[2] ** (-power))
            - ratio
        )

    lower = 0.1
    upper = 5.0
    if residual(lower) * residual(upper) > 0.0:
        raise RuntimeError("three-level pencil extrapolation did not bracket a power")
    for _iteration in range(90):
        middle = 0.5 * (lower + upper)
        if residual(lower) * residual(middle) <= 0.0:
            upper = middle
        else:
            lower = middle
    power = 0.5 * (lower + upper)
    coefficient = (values[0] - values[1]) / (
        nodes[0] ** (-power) - nodes[1] ** (-power)
    )
    limit = values[0] - coefficient * nodes[0] ** (-power)
    return power, limit


def couple_vertex_pencil(vertex_case, pencil_rows):
    raw = vertex_case["rows"][-1]["raw"]
    step = vertex_case["rows"][-1]["step"]
    reference = vertex_case["reference"]
    coefficients = vertex_case["three_jet_coefficients"]
    for row in pencil_rows:
        channel = MellinThreeJetChannel(
            VertexMellinPencil.from_exponent(row["kondratiev_exponent"]),
            coefficients,
        )
        candidate = MellinKondratievRepayment((channel,)).repay(raw, step).value
        row["coupled_vertex_abs_error_at_512"] = _abs(candidate - reference)
    power, extrapolated = _three_level_power_limit(pencil_rows)
    channel = MellinThreeJetChannel(
        VertexMellinPencil.from_exponent(extrapolated),
        coefficients,
    )
    candidate = MellinKondratievRepayment((channel,)).repay(raw, step).value
    return {
        "levels": tuple(row["refinement"] for row in pencil_rows[-3:]),
        "fitted_power": power,
        "extrapolated_exponent": extrapolated,
        "exponent_abs_error": _abs(extrapolated - FICHERA_EXPONENT),
        "coupled_vertex_abs_error_at_512": _abs(candidate - reference),
    }


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _write_report(summary):
    edge = summary["edge_layer_case"]
    vertex = summary["vertex_layer_case"]
    lines = [
        "# 3D Mellin--Kondratiev corner convergence",
        "",
        "This campaign separates the continuum corner error from the discrete "
        "surface-graph compression error. The references are independent "
        "adaptive integrals after a radial power substitution that removes the "
        "known singular exponent.",
        "",
        "## Pencil audit",
        "",
        "| link h-level | spherical nodes | interior dofs | lambda_h | "
        "|lambda_h-lambda_*| | coupled error at 512 | residual | solve ms |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["spherical_pencil_convergence"]:
        lines.append(
            f"| {row['refinement']} | {row['spherical_nodes']} | "
            f"{row['interior_dofs']} | {row['kondratiev_exponent']:.12f} | "
            f"{row['exponent_abs_error']:.3e} | "
            f"{row['coupled_vertex_abs_error_at_512']:.3e} | "
            f"{row['relative_residual']:.3e} | "
            f"{row['solve_ms']:.3f} |"
        )
    extrapolation = summary["spherical_pencil_extrapolation"]
    finest_coupled_error = summary["spherical_pencil_convergence"][-1][
        "coupled_vertex_abs_error_at_512"
    ]
    lines.extend(
        [
            "",
            "The reference exponent is `0.45417371533061`. The local angular "
            "problem is assembled as sparse P1 stiffness and mass rows; no dense "
            "eigenmatrix is formed.",
            "The raw level-16 exponent leaves a `"
            f"{finest_coupled_error:.3e}` "
            "coupled radial error. A three-level power extrapolation gives "
            f"`lambda={extrapolation['extrapolated_exponent']:.12f}` and "
            f"`{extrapolation['coupled_vertex_abs_error_at_512']:.3e}` error. "
            "The machine-scale vertex row below uses the held-out published "
            "exponent, not the finite P1 estimate.",
            "",
            "The two continuum reference values are each recomputed after two "
            "different radial substitutions. Their absolute discrepancies are "
            f"`{edge['reference_crosscheck_abs']:.3e}` (edge) and "
            f"`{vertex['reference_crosscheck_abs']:.3e}` (vertex).",
            "",
            "## Layer-potential convergence",
            "",
            "| case | expected raw power | fitted raw | expected 3-jet power | "
            "fitted corrected | finest raw error | finest corrected error |",
            "|---|---:|---:|---:|---:|---:|---:|",
            f"| re-entrant edge | {edge['boundary_quadrature_exponent']:.6f} | "
            f"{edge['raw_fitted_order']:.3f} | "
            f"{edge['boundary_quadrature_exponent'] + 4.0:.6f} | "
            f"{edge['corrected_fitted_order']:.3f} | "
            f"{edge['rows'][-1]['raw_abs_error']:.3e} | "
            f"{edge['rows'][-1]['corrected_abs_error']:.3e} |",
            f"| Fichera vertex | {vertex['boundary_quadrature_exponent']:.6f} | "
            f"{vertex['raw_fitted_order']:.3f} | "
            f"{vertex['boundary_quadrature_exponent'] + 4.0:.6f} | "
            f"{vertex['corrected_fitted_order']:.3f} | "
            f"{vertex['rows'][-1]['raw_abs_error']:.3e} | "
            f"{vertex['rows'][-1]['corrected_abs_error']:.3e} |",
            "",
            "The edge test is a 3D Laplace single-layer contribution on one "
            "face of a 270-degree prism; its tangential integral is evaluated "
            "in closed form. The vertex test is a face-sector contribution at "
            "the Fichera exponent, including the required surface-measure shift "
            "from `lambda` to `lambda+1`. Both use a localized eighth-order "
            "cutoff so the displayed slope isolates the corner endpoint.",
            "",
            "Each correction stores four amplitude coefficients and has `O(1)` "
            "apply cost per retained edge or vertex mode. The global smooth "
            "surface backend has separate complexity accounting.",
            "",
        ]
    )
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run_campaign(write_outputs=True):
    topology = PolyhedralMeshTopology(*fichera_surface())
    edge = edge_layer_case(topology)
    vertex = vertex_layer_case()
    pencil_rows = spherical_pencil_convergence()
    pencil_extrapolation = couple_vertex_pencil(vertex, pencil_rows)
    summary = {
        "method": "sparse_3d_mellin_kondratiev_three_jet",
        "scope": "continuum corner convergence for localized Laplace layer-potential channels",
        "fichera_reference_exponent": FICHERA_EXPONENT,
        "topology": topology.stats(),
        "spherical_pencil_convergence": pencil_rows,
        "spherical_pencil_extrapolation": pencil_extrapolation,
        "edge_layer_case": edge,
        "vertex_layer_case": vertex,
        "complexity": {
            "corner_compile": "O(local spherical-link triangles * sparse iterations)",
            "corner_apply": "O(number of retained edge/vertex three-jets)",
            "corner_storage": "O(local topology + retained three-jets)",
            "dense_global_matrix": False,
        },
        "claim_boundary": (
            "This verifies the edge and vertex endpoint channels. It does not "
            "turn the separate arbitrary-surface far-field hierarchy into an "
            "unconditional near-linear algorithm."
        ),
    }
    if write_outputs:
        OUT.mkdir(parents=True, exist_ok=True)
        _write_csv(OUT / "edge_layer_convergence.csv", edge["rows"])
        _write_csv(OUT / "vertex_layer_convergence.csv", vertex["rows"])
        _write_csv(OUT / "spherical_pencil_convergence.csv", pencil_rows)
        (OUT / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        _write_report(summary)
    return summary


def main():
    print(json.dumps(run_campaign(), indent=2))


if __name__ == "__main__":
    main()
