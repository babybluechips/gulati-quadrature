#!/usr/bin/env python3
"""Export the executed production 3D QJet notebook as one audited HTML file."""

# ruff: noqa: E501

from __future__ import annotations

import base64
import csv
import html
import json
from pathlib import Path

import nbformat
from nbconvert import HTMLExporter

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "production_3d_qjet_method.ipynb"
OUTPUT = (
    ROOT
    / "outputs"
    / "production_3d_qjet_html"
    / "production_3d_qjet_method.html"
)
CAD_OUTPUT = ROOT / "outputs" / "cad_qjet_invertibility"
CAD_SUMMARY = CAD_OUTPUT / "cad_roundtrip_summary.json"
CAD_GALLERY = CAD_OUTPUT / "cad_roundtrip_gallery.svg"
PDE_OUTPUT = ROOT / "outputs" / "production_3d_pde_validation"
PDE_SUMMARY = PDE_OUTPUT / "summary.json"
PDE_DISCRETE = PDE_OUTPUT / "discrete_pde_rows.csv"
PDE_CONTINUUM = PDE_OUTPUT / "sphere_continuum_rows.csv"
CAD_PDE_OUTPUT = ROOT / "outputs" / "production_3d_cad_pde_validation"
CAD_PDE_SUMMARY = CAD_PDE_OUTPUT / "summary.json"
CAD_PDE_ROWS = CAD_PDE_OUTPUT / "cad_pde_rows.csv"
CAD_PDE_HELD_OUT = CAD_PDE_OUTPUT / "cad_held_out_rows.csv"
CAD_PDE_GEOMETRY = CAD_PDE_OUTPUT / "cad_geometry_rows.csv"
CAD_PDE_SPHERE = CAD_PDE_OUTPUT / "sphere_repayment_rows.csv"
REPAIRED_OUTPUT = ROOT / "outputs" / "production_3d_repaired_refinement"
REPAIRED_SUMMARY = REPAIRED_OUTPUT / "summary.json"
REPAIRED_AGGREGATE = REPAIRED_OUTPUT / "aggregate_rows.csv"
REPAIRED_GEOMETRY = REPAIRED_OUTPUT / "geometry_rows.csv"
REPAIRED_FEATURES = REPAIRED_OUTPUT / "feature_channel_rows.csv"
NASA_VIS_OUTPUT = ROOT / "outputs" / "production_nasa_cad_pde_visualization"
NASA_VIS_SUMMARY = NASA_VIS_OUTPUT / "summary.json"
NASA_VIS_PDE_ROWS = NASA_VIS_OUTPUT / "pde_rows.csv"
NASA_VIS_GEOMETRY = NASA_VIS_OUTPUT / "geometry_rows.csv"
NASA_VIS_OVERVIEW = NASA_VIS_OUTPUT / "nasa_cad_pde_overview.png"
TAIL_OUTPUT = ROOT / "outputs" / "transparent_tail_benchmark"
TAIL_SUMMARY = TAIL_OUTPUT / "summary.json"
TAIL_ROWS = TAIL_OUTPUT / "benchmark_rows.csv"
TAIL_MODES = TAIL_OUTPUT / "pde_mode_rows.csv"
TAIL_SVG = TAIL_OUTPUT / "runtime_scaling.svg"

TITLE = "Production 3D QJet: proofs, algorithm, and validation"

REPORT_STYLE = r"""
<style id="qjet-report-style">
:root {
  color-scheme: light;
  --ink: #171717;
  --muted: #5b5b5b;
  --line: #b8b8b8;
  --line-light: #dedede;
  --paper: #ffffff;
  --code: #f4f4f4;
}
html { scroll-behavior: smooth; }
body {
  background: var(--paper) !important;
  color: var(--ink) !important;
  font-family: Georgia, "Times New Roman", serif !important;
  font-size: 17px;
  line-height: 1.58;
  letter-spacing: 0;
}
#notebook-container,
.jp-Notebook {
  box-shadow: none !important;
  margin: 0 auto !important;
  max-width: 1180px;
  padding: 36px 42px 80px !important;
}
.jp-Cell { margin: 0 0 1.05rem !important; }
.jp-MarkdownOutput { overflow-wrap: anywhere; }
h1, h2, h3, h4 {
  color: var(--ink) !important;
  font-family: Arial, Helvetica, sans-serif !important;
  font-weight: 600 !important;
  letter-spacing: 0 !important;
  line-height: 1.2 !important;
}
h1 {
  border-bottom: 2px solid var(--ink);
  font-size: 2rem !important;
  margin: 0 0 1.5rem !important;
  padding-bottom: 0.65rem;
}
h2 {
  border-bottom: 1px solid var(--line);
  font-size: 1.45rem !important;
  margin-top: 2.4rem !important;
  padding-bottom: 0.35rem;
}
h3 { font-size: 1.15rem !important; margin-top: 1.8rem !important; }
p, li { max-width: 88ch; }
a { color: #1e4c73 !important; text-decoration-thickness: 1px; }
pre, code, kbd, samp {
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace !important;
  letter-spacing: 0;
}
pre {
  background: var(--code) !important;
  border: 1px solid var(--line-light) !important;
  border-radius: 2px !important;
  overflow-x: auto;
  padding: 0.9rem 1rem !important;
}
.jp-CodeCell {
  border-left: 2px solid #8e8e8e;
  padding-left: 10px !important;
}
.jp-InputArea-prompt, .jp-OutputArea-prompt { display: none !important; }
.jp-OutputArea-output { overflow-x: auto; }
table {
  border-collapse: collapse !important;
  display: table;
  font-family: Arial, Helvetica, sans-serif !important;
  font-size: 12.5px !important;
  line-height: 1.35;
  margin: 1rem 0 1.5rem !important;
  max-width: 100%;
  width: auto !important;
}
thead { border-bottom: 1.5px solid var(--ink) !important; }
th, td {
  border: 1px solid var(--line-light) !important;
  padding: 0.38rem 0.5rem !important;
  text-align: right !important;
  white-space: nowrap;
}
th:first-child, td:first-child,
th:nth-child(2), td:nth-child(2) { text-align: left !important; }
svg, img {
  display: block;
  height: auto !important;
  margin: 1.25rem auto;
  max-width: 100% !important;
}
.MathJax_Display, mjx-container[display="true"] { overflow-x: auto; overflow-y: hidden; }
blockquote {
  border-left: 3px solid #777 !important;
  color: var(--muted) !important;
  margin-left: 0 !important;
  padding-left: 1rem !important;
}
@media (max-width: 720px) {
  body { font-size: 15px; }
  #notebook-container, .jp-Notebook { padding: 22px 14px 55px !important; }
  h1 { font-size: 1.55rem !important; }
  h2 { font-size: 1.25rem !important; }
  .jp-CodeCell { padding-left: 5px !important; }
  table { display: block; overflow-x: auto; font-size: 11px !important; }
}
@media print {
  body { font-size: 10.5pt; }
  #notebook-container, .jp-Notebook { max-width: none; padding: 0 !important; }
  .jp-CodeCell { break-inside: avoid; }
  a { color: var(--ink) !important; }
}
</style>
"""


def cad_invertibility_section() -> str:
    """Embed the independently generated public-CAD round-trip audit."""

    if not CAD_SUMMARY.exists() or not CAD_GALLERY.exists():
        raise RuntimeError(
            "public CAD audit is missing; run scripts/cad_qjet_invertibility_campaign.py"
        )
    summary = json.loads(CAD_SUMMARY.read_text(encoding="utf-8"))
    rows = summary["cases"]
    scaling = summary["scaling"]
    if not summary["gates"]["passed"]:
        raise RuntimeError("refusing to publish a failed public CAD invertibility audit")
    table_rows = []
    for row in rows:
        table_rows.append(
            "<tr>"
            f'<td>{html.escape(str(row["label"]))}</td>'
            f'<td>{row["vertex_count"]:,}</td>'
            f'<td>{row["face_count"]:,}</td>'
            f'<td>{row["part_count"]:,}</td>'
            f'<td>{row["archive_bytes"] / 1024.0:.1f} KiB</td>'
            f'<td>{row["compile_ms"]:.1f}</td>'
            f'<td>{row["decode_ms"]:.1f}</td>'
            f'<td>{row["maximum_absolute_error"]:.1e}</td>'
            "<td><strong>PASS</strong></td>"
            "</tr>"
        )
    gallery = CAD_GALLERY.read_text(encoding="utf-8")
    return f"""
<section id="public-cad-invertibility" class="jp-Cell jp-MarkdownCell">
<div class="jp-Cell-inputWrapper"><div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput">
<h2>Public CAD invertibility audit</h2>
<p>The generated-object campaign is supplemented here by {len(rows)} detailed public CAD data sets: NASA's SOFIA aircraft, a FreeCAD cement-mixer truck, two high-resolution Curiosity manufacturing layouts, and a buildingSMART IFC bridge scene. The compiler stores an ordered-IEEE third-difference residual jet and exact triangle connectivity. This lossless channel is required because finitely many analytic multipole moments alone are not injective.</p>
<p>For coordinate keys <em>q</em><sub>i</sub>, the stored residual jet is</p>
<div style="overflow-x:auto"><code style="display:block;white-space:nowrap;padding:.55rem .7rem">j_i = q_i - 3q_(i-1) + 3q_(i-2) - q_(i-3), i &gt;= 3.</code></div>
<p>Its inverse is a forward recurrence with unit diagonal. The archive therefore contains <strong>3V + 3F + O(parts)</strong> integers and no pair table. All {len(rows)} decoded meshes match the input coordinate bits, face indices, part ranges, and canonical SHA-256 exactly.</p>
<div style="overflow-x:auto"><table style="min-width:900px"><thead><tr><th>Public CAD data set</th><th>Vertices</th><th>Faces</th><th>Parts</th><th>Archive</th><th>Compile ms</th><th>Decode ms</th><th>Max |Δx|</th><th>Audit</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table></div>
<p>Across six SOFIA prefix refinements from 4,096 to 92,974 faces, the measured log-log exponents are <strong>{scaling["archive_exponent"]:.3f}</strong> for archive bytes, <strong>{scaling["encode_time_exponent"]:.3f}</strong> for encoding, and <strong>{scaling["decode_time_exponent"]:.3f}</strong> for decoding. Exact recovery passes at every refinement.</p>
<div style="overflow-x:auto;border:1px solid #dedede">{gallery}</div>
<p><a href="../cad_qjet_invertibility/cad_qjet_invertibility.html">Open the full source, license, checksum, corruption, and complexity report.</a></p>
<p><strong>Scope.</strong> “Lossless” means exact recovery of the canonical tessellated boundary passed to the compiler. The IFC reader applies recursive local placements, but the archive does not reproduce non-geometric IFC property sets or original source-file whitespace.</p>
</div></div></div></section>
"""


def pde_validation_section() -> str:
    """Embed independent discrete and exact-sphere PDE validation."""

    required = (PDE_SUMMARY, PDE_DISCRETE, PDE_CONTINUUM)
    if any(not path.exists() for path in required):
        raise RuntimeError(
            "3D PDE audit is missing; run scripts/production_3d_pde_validation.py"
        )
    summary = json.loads(PDE_SUMMARY.read_text(encoding="utf-8"))
    with PDE_DISCRETE.open(newline="", encoding="utf-8") as handle:
        discrete = list(csv.DictReader(handle))
    with PDE_CONTINUUM.open(newline="", encoding="utf-8") as handle:
        continuum = list(csv.DictReader(handle))
    if not summary["all_discrete_gates_passed"]:
        raise RuntimeError("refusing to publish a failed 3D PDE discrete audit")

    problem_rows = []
    for problem in summary["problems"]:
        selected = [row for row in discrete if row["problem"] == problem]
        problem_rows.append(
            "<tr>"
            f"<td>{html.escape(problem)}</td>"
            f"<td>{max(float(row['relative_solution_error']) for row in selected):.3e}</td>"
            f"<td>{max(float(row['relative_equation_residual']) for row in selected):.3e}</td>"
            f"<td>{max(int(row['qjet_applications']) for row in selected):,}</td>"
            f"<td>{max(float(row['solve_ms']) for row in selected):.1f}</td>"
            "<td><strong>PASS</strong></td>"
            "</tr>"
        )
    continuum_rows = []
    for row in continuum:
        continuum_rows.append(
            "<tr>"
            f"<td>{html.escape(row['problem'])}</td>"
            f"<td>{int(row['degree'])}</td>"
            f"<td>{int(row['nodes']):,}</td>"
            f"<td>{float(row['relative_continuum_error']):.3e}</td>"
            f"<td>{float(row['relative_equation_residual']):.3e}</td>"
            "</tr>"
        )
    return f"""
<section id="boundary-pde-validation" class="jp-Cell jp-MarkdownCell">
<div class="jp-Cell-inputWrapper"><div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput">
<h2>Boundary PDE validation</h2>
<p>The production PDE layer uses <code>A = Q_3/(2*pi)</code> without assembling a dense operator. It tests Laplace DtN application, mean-zero Poisson, screened Poisson/Yukawa, the damped boundary Helmholtz resolvent, the heat/Poisson semigroup, and boundary wave propagation. For <em>k</em> QJet applications the measured algorithmic contract is <strong>O(k N log N)</strong> time and <strong>O(N)</strong> auxiliary storage.</p>
<div style="overflow-x:auto"><table><thead><tr><th>Boundary problem</th><th>Max discrete error</th><th>Max residual</th><th>Max Q applies</th><th>Max solve ms</th><th>Audit</th></tr></thead><tbody>{''.join(problem_rows)}</tbody></table></div>
<p>All {summary['discrete_case_count']} cases across {summary['discrete_shape_count']} geometries pass against an independently streamed pairwise <code>Q_3</code> oracle. The largest discrete solution error is <strong>{summary['maximum_discrete_error']:.3e}</strong>; no dense operator and no quadratic production fallback are used.</p>
<h3>Independent continuum check</h3>
<p>The algebraic audit is not a continuum-accuracy certificate. On the unit sphere the exact identity is <code>Lambda Y_lm = l Y_lm</code>. The table below compares directly against that analytic reference.</p>
<div style="overflow-x:auto"><table><thead><tr><th>Problem</th><th>l</th><th>N</th><th>Continuum error</th><th>Algebraic residual</th></tr></thead><tbody>{''.join(continuum_rows)}</tbody></table></div>
<p>The best raw degree-one Laplace DtN error is <strong>{summary['best_sphere_degree_one_error']:.3e}</strong>, with observed order about <strong>{summary['median_degree_one_order']:.3f}</strong>. This is the unrepaid baseline. The next audit adds the topology-aware singular-cell series and fixed-rank harmonic/Helmholtz channels, while retaining a separate held-out continuum column.</p>
<p><a href="../production_3d_pde_validation/report.md">Open the complete PDE equations, per-shape table, and scope statement.</a></p>
</div></div></div></section>
"""


def cad_pde_validation_section() -> str:
    """Embed the continuum-repaid PDE campaign over every public CAD archive."""

    required = (
        CAD_PDE_SUMMARY,
        CAD_PDE_ROWS,
        CAD_PDE_HELD_OUT,
        CAD_PDE_GEOMETRY,
        CAD_PDE_SPHERE,
    )
    if any(not path.exists() for path in required):
        raise RuntimeError(
            "CAD PDE audit is missing; run scripts/production_3d_cad_pde_validation.py"
        )
    summary = json.loads(CAD_PDE_SUMMARY.read_text(encoding="utf-8"))
    with CAD_PDE_ROWS.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with CAD_PDE_HELD_OUT.open(newline="", encoding="utf-8") as handle:
        held_out = list(csv.DictReader(handle))
    with CAD_PDE_GEOMETRY.open(newline="", encoding="utf-8") as handle:
        geometry = list(csv.DictReader(handle))
    with CAD_PDE_SPHERE.open(newline="", encoding="utf-8") as handle:
        sphere = list(csv.DictReader(handle))
    if not summary["all_pde_gates_passed"]:
        raise RuntimeError("refusing to publish failed CAD PDE algebraic gates")

    geometry_rows = []
    for row in geometry:
        geometry_rows.append(
            "<tr>"
            f"<td>{html.escape(row['shape'])}</td>"
            f"<td>{int(row['source_vertices']):,}</td>"
            f"<td>{int(row['source_faces']):,}</td>"
            f"<td>{int(row['compiled_vertices']):,}</td>"
            f"<td>{int(row['compiled_faces']):,}</td>"
            f"<td>{float(row['compiled_measure_to_source_area_ratio']):.6f}</td>"
            f"<td>{int(row['singular_cells_repaid']):,}</td>"
            "</tr>"
        )
    problem_rows = []
    problem_names = tuple(dict.fromkeys(row["problem"] for row in rows))
    for problem in problem_names:
        selected = [row for row in rows if row["problem"] == problem]
        errors = [
            float(row["relative_error"])
            for row in selected
            if row["relative_error"]
        ]
        problem_rows.append(
            "<tr>"
            f"<td>{html.escape(problem)}</td>"
            f"<td>{max(errors):.3e}</td>" if errors else
            "<tr>"
            f"<td>{html.escape(problem)}</td>"
            "<td>n/a</td>"
        )
        problem_rows[-1] += (
            f"<td>{max(float(row['relative_algebraic_residual']) for row in selected):.3e}</td>"
            f"<td>{max(int(row['qjet_applications']) for row in selected):,}</td>"
            f"<td>{max(float(row['solve_ms']) for row in selected):.1f}</td>"
            "<td><strong>PASS</strong></td></tr>"
        )
    held_rows = []
    for row in held_out:
        held_rows.append(
            "<tr>"
            f"<td>{html.escape(row['shape'])}</td>"
            f"<td>{int(row['held_out_degree'])}</td>"
            f"<td>{float(row['relative_continuum_error']):.3e}</td>"
            "</tr>"
        )
    finest = [row for row in sphere if int(row["nodes"]) == 258]
    sphere_rows = []
    for row in finest:
        sphere_rows.append(
            "<tr>"
            f"<td>{html.escape(row['method'])}</td>"
            f"<td>{float(row['relative_continuum_error']):.3e}</td>"
            "</tr>"
        )
    return f"""
<section id="cad-boundary-pde-validation" class="jp-Cell jp-MarkdownCell">
<div class="jp-Cell-inputWrapper"><div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput">
<h2>Continuum-repaid CAD PDE campaign</h2>
<p>The production path now retains topology, applies the curvature-adjusted omitted-cell series <code>-a Delta/4 - a^3 Delta^2/192 - a^5 Delta^3/11520</code>, and compiles fixed-rank solid-harmonic and plane-wave QJets. Every one of the {summary['source_face_count']:,} source triangles from all {summary['model_count']} CAD archives is scanned. The solver stores no dense boundary matrix or pair table.</p>
<div style="overflow-x:auto"><table><thead><tr><th>CAD object</th><th>Source V</th><th>Source F</th><th>PDE nodes</th><th>PDE faces</th><th>Measure ratio</th><th>Local cells repaid</th></tr></thead><tbody>{''.join(geometry_rows)}</tbody></table></div>
<h3>Compiled channels and algebraic solves</h3>
<div style="overflow-x:auto"><table><thead><tr><th>Problem</th><th>Max compiled-reference error</th><th>Max algebraic residual</th><th>Max Q applies</th><th>Max solve ms</th><th>Audit</th></tr></thead><tbody>{''.join(problem_rows)}</tbody></table></div>
<p>The maximum exact compiled harmonic/plane-wave error is <strong>{summary['maximum_compiled_reference_error']:.3e}</strong>. The maximum algebraic residual is <strong>{summary['maximum_algebraic_residual']:.3e}</strong>. Poisson and screened-Poisson data in the retained harmonic subspace use a fixed-rank direct modal solve; heat and wave use the matrix-free boundary functional calculus at a cell-scale timestep.</p>
<h3>Held-out continuum result</h3>
<p>The degree-four mode below was not used to compile the degree-three repayment. It is the relevant generalization test. Its size prevents the machine-level compiled rows from being misread as universal continuum accuracy.</p>
<div style="overflow-x:auto"><table><thead><tr><th>CAD object</th><th>Held-out degree</th><th>Continuum error</th></tr></thead><tbody>{''.join(held_rows)}</tbody></table></div>
<p>On the controlled sphere mesh at <code>N=258</code>, the local singular-cell/curvature series reduces the same held-out error before any fitted moment claim:</p>
<div style="overflow-x:auto"><table><thead><tr><th>Sphere method</th><th>Held-out error</th></tr></thead><tbody>{''.join(sphere_rows)}</tbody></table></div>
<p><strong>Interpretation.</strong> The earlier <code>1.779e-8</code> number measured discrete implementation and PDE algebra. It did not bound continuum discretization. The CAD held-out range is <strong>{summary['minimum_held_out_continuum_error']:.3e}</strong> to <strong>{summary['maximum_held_out_continuum_error']:.3e}</strong>; therefore this campaign makes no universal 3D machine-precision claim.</p>
<p><a href="../production_3d_cad_pde_validation/report.md">Open the complete CAD PDE report and CSV tables.</a></p>
</div></div></div></section>
"""


def repaired_refinement_section() -> str:
    """Embed manifold repair, curved panels, features, and true holdouts."""

    required = (
        REPAIRED_SUMMARY,
        REPAIRED_AGGREGATE,
        REPAIRED_GEOMETRY,
        REPAIRED_FEATURES,
    )
    if any(not path.exists() for path in required):
        raise RuntimeError(
            "repaired refinement audit is missing; run "
            "scripts/production_3d_repaired_refinement.py"
        )
    summary = json.loads(REPAIRED_SUMMARY.read_text(encoding="utf-8"))
    with REPAIRED_AGGREGATE.open(newline="", encoding="utf-8") as handle:
        aggregate = list(csv.DictReader(handle))
    with REPAIRED_GEOMETRY.open(newline="", encoding="utf-8") as handle:
        geometry = list(csv.DictReader(handle))
    with REPAIRED_FEATURES.open(newline="", encoding="utf-8") as handle:
        features = list(csv.DictReader(handle))
    if not (
        summary["all_final_modes_independent"]
        and summary["all_repaired_surfaces_production_ready"]
        and summary["all_repaired_surfaces_watertight"]
        and summary["all_repaired_surfaces_edge_manifold"]
        and summary["all_repaired_surfaces_vertex_manifold"]
        and summary["all_curved_panel_atlases_watertight"]
    ):
        raise RuntimeError("refusing to publish an uncertified repaired surface audit")
    geometry_rows = []
    for row in geometry:
        geometry_rows.append(
            "<tr>"
            f"<td>{html.escape(row['shape'])}</td>"
            f"<td>{int(row['refinement'])}</td>"
            f"<td>{int(row['nodes']):,}</td>"
            f"<td>{html.escape(row['panel_geometry_kind'])}</td>"
            f"<td>{int(row['boundary_edges'])}</td>"
            f"<td>{int(row['nonmanifold_edges'])}</td>"
            f"<td>{int(row['nonmanifold_vertices'])}</td>"
            f"<td>{float(row['maximum_panel_seam_gap']):.3e}</td>"
            f"<td>{int(row['adaptive_selected_degree'])}</td>"
            f"<td>{html.escape(row['adaptive_validation_certified'])}</td>"
            "</tr>"
        )
    refinement_rows = []
    for row in aggregate:
        refinement_rows.append(
            "<tr>"
            f"<td>{html.escape(row['shape'])}</td>"
            f"<td>{int(row['refinement'])}</td>"
            f"<td>{int(row['nodes']):,}</td>"
            f"<td>{float(row['maximum_raw_held_out_error']):.3e}</td>"
            f"<td>{float(row['maximum_repaid_held_out_error']):.3e}</td>"
            f"<td>{float(row['median_repaid_held_out_error']):.3e}</td>"
            "</tr>"
        )
    feature_rows = []
    for row in features:
        feature_rows.append(
            "<tr>"
            f"<td>{html.escape(row['kind'])}</td>"
            f"<td>{html.escape(row['channel'])}</td>"
            f"<td>{int(row['rung'])}</td>"
            f"<td>{float(row['kondratiev_exponent']):.6f}</td>"
            f"<td>{float(row['raw_absolute_error']):.3e}</td>"
            f"<td>{float(row['repaid_absolute_error']):.3e}</td>"
            f"<td>{float(row['reference_disagreement']):.3e}</td>"
            "</tr>"
        )
    return f"""
<section id="repaired-curved-refinement" class="jp-Cell jp-MarkdownCell">
<div class="jp-Cell-inputWrapper"><div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput">
<h2>Watertight repair and independent curved-panel refinement</h2>
<p>This is the continuum-facing replacement for the coarse voxel CAD audit. Triangle soups are welded, separated into manifold face fans, oriented component by component, and capped along closed boundary loops. A cubic PN atlas then supplies analytic first and second geometry jets. Smooth panels use the odd principal-value tangent-moment correction and the stable first even singular-cell rung; sharp edges and vertices use four-coefficient Mellin/Kondratiev moments, including a sparse spherical-link pencil at each certified vertex.</p>
<div style="overflow-x:auto"><table><thead><tr><th>Shape</th><th>Level</th><th>Nodes</th><th>Panel geometry</th><th>Boundary E</th><th>Nonmanifold E</th><th>Nonmanifold V</th><th>Seam gap</th><th>Selected d</th><th>Adaptive certificate</th></tr></thead><tbody>{''.join(geometry_rows)}</tbody></table></div>
<h3>Modes excluded from fitting and model selection</h3>
<p>The retained bounded-remainder compiler uses degree at most <strong>{summary['maximum_compiled_degree']}</strong> and adaptive selection sees degree at most <strong>{summary['maximum_adaptively_inspected_degree']}</strong>. Every final row below uses degree four or five. The exact-sphere chart lies exactly on the unit sphere, so its fitted endpoint rate <strong>{summary['exact_sphere_repaid_endpoint_rate']:.3f}</strong> isolates the singular quadrature. Ellipsoid and funky PN rows also test the unresolved bounded geometry remainder.</p>
<div style="overflow-x:auto"><table><thead><tr><th>Shape</th><th>Level</th><th>Nodes</th><th>Max raw error</th><th>Max repaid error</th><th>Median repaid</th></tr></thead><tbody>{''.join(refinement_rows)}</tbody></table></div>
<h3>Independent edge and vertex moment audit</h3>
<p>Every feature channel is compiled with an order-twelve curved-panel rule and checked against a separate order-sixteen rule.</p>
<div style="overflow-x:auto"><table><thead><tr><th>Kind</th><th>Channel</th><th>Rung</th><th>Kondratiev exponent</th><th>Raw abs. error</th><th>Repaid abs. error</th><th>Reference gap</th></tr></thead><tbody>{''.join(feature_rows)}</tbody></table></div>
<p>The cube vertex exponent is invariant under the eight vertex-link orderings to a spread of <strong>{summary['cube_vertex_kondratiev_spread']:.3e}</strong>; its maximum error against the exact octant exponent 3 is <strong>{summary['maximum_cube_vertex_kondratiev_error']:.3e}</strong>.</p>
<p>The largest repaid feature-basis error is <strong>{summary['maximum_feature_repaid_absolute_error']:.3e}</strong>. The production apply contract is <strong>{html.escape(summary['production_apply_complexity'])}</strong> time and <strong>{html.escape(summary['production_storage_complexity'])}</strong> storage. No dense Q matrix or global pair table is stored.</p>
<p><strong>Scope.</strong> The three independent held-out sequences decrease under refinement, but they are not at machine precision. Failed adaptive flags are retained in the table. This section does not relabel retained-mode reproduction as continuum accuracy.</p>
<p><a href="../production_3d_repaired_refinement/report.md">Open the complete independent-mode report and CSV tables.</a></p>
</div></div></div></section>
"""


def nasa_cad_pde_visualization_section() -> str:
    """Embed full-triangle numerical fields from the compressed NASA solves."""

    required = (
        NASA_VIS_SUMMARY,
        NASA_VIS_PDE_ROWS,
        NASA_VIS_GEOMETRY,
        NASA_VIS_OVERVIEW,
    )
    if any(not path.exists() for path in required):
        raise RuntimeError(
            "NASA CAD PDE visualization is missing; run "
            "scripts/production_nasa_cad_pde_visualization.py"
        )
    summary = json.loads(NASA_VIS_SUMMARY.read_text(encoding="utf-8"))
    with NASA_VIS_PDE_ROWS.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with NASA_VIS_GEOMETRY.open(newline="", encoding="utf-8") as handle:
        geometry = list(csv.DictReader(handle))
    if not (
        summary["all_declared_gates_passed"]
        and summary["all_archive_checksums_verified"]
        and summary["all_source_vertices_lifted"]
        and summary["all_source_faces_scanned"]
        and summary["visualization_uses_numerical_fields"]
        and not summary["held_out_continuum_machine_precision_claim"]
    ):
        raise RuntimeError("refusing to publish an unaudited NASA PDE visualization")

    def png_data_uri(path: Path) -> str:
        return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode(
            "ascii"
        )

    geometry_html = []
    for row in geometry:
        geometry_html.append(
            "<tr>"
            f"<td>{html.escape(row['shape'])}</td>"
            f"<td>{int(row['archive_bytes']) / 1024.0:.1f} KiB</td>"
            f"<td>{int(row['source_vertices']):,}</td>"
            f"<td>{int(row['source_faces']):,}</td>"
            f"<td>{int(row['compiled_nodes']):,}</td>"
            f"<td>{float(row['decode_ms']):.1f}</td>"
            f"<td>{float(row['surface_compile_ms']):.1f}</td>"
            f"<td>{float(row['render_ms']):.1f}</td>"
            "</tr>"
        )
    problem_html = []
    for problem in summary["problems"]:
        selected = [row for row in rows if row["problem"] == problem]
        problem_html.append(
            "<tr>"
            f"<td>{html.escape(problem)}</td>"
            f"<td>{html.escape(selected[0]['audit_class'])}</td>"
            f"<td>{max(float(row['metric_value']) for row in selected):.3e}</td>"
            f"<td>{max(float(row['relative_algebraic_residual']) for row in selected):.3e}</td>"
            f"<td>{max(int(row['qjet_applications']) for row in selected):,}</td>"
            f"<td>{max(float(row['solve_ms']) for row in selected):.1f}</td>"
            "<td><strong>PASS</strong></td>"
            "</tr>"
        )
    figure_links = []
    for row in geometry:
        figure_uri = png_data_uri(NASA_VIS_OUTPUT / row["figure_path"])
        figure_links.append(
            "<figure>"
            f"<img src=\"{figure_uri}\" "
            f"alt=\"Six numerical boundary PDE fields on {html.escape(row['shape'])}\">"
            f"<figcaption>{html.escape(row['shape'])}: six fields solved on "
            f"{int(row['compiled_nodes'])} compressed nodes and lifted onto "
            f"{int(row['source_faces']):,} decoded triangles.</figcaption>"
            "</figure>"
        )
    return f"""
<section id="nasa-cad-pde-fields" class="jp-Cell jp-MarkdownCell">
<div class="jp-Cell-inputWrapper"><div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput">
<h2>Numerical PDE fields on compressed NASA CAD</h2>
<p>Three checksum-verified NASA <code>QCAD3J</code> archives are solved on 24 to 42 compressed boundary nodes. A deterministic <code>O(V_source)</code> index lift paints the computed values onto every one of the {summary['source_face_count']:,} losslessly decoded source triangles. The color maps below are generated from the numerical fields; they are not illustrative textures. Neither the operator nor the display lift stores a dense matrix or pair table.</p>
<img src="{png_data_uri(NASA_VIS_OVERVIEW)}" alt="Laplace and Helmholtz boundary fields on three NASA CAD meshes">
<div style="overflow-x:auto"><table><thead><tr><th>NASA object</th><th>Archive</th><th>Source V</th><th>Source F</th><th>PDE nodes</th><th>Decode ms</th><th>Compile ms</th><th>Render ms</th></tr></thead><tbody>{''.join(geometry_html)}</tbody></table></div>
<h3>Accuracy and warm solve timing</h3>
<div style="overflow-x:auto"><table><thead><tr><th>Problem</th><th>Audit class</th><th>Max metric</th><th>Max residual</th><th>Max Q applies</th><th>Max solve ms</th><th>Audit</th></tr></thead><tbody>{''.join(problem_html)}</tbody></table></div>
<p>The maximum retained manufactured-reference error is <strong>{summary['maximum_retained_reference_error']:.3e}</strong>. The maximum self-adjoint heat/wave denominator residual is <strong>{summary['maximum_algebraic_residual']:.3e}</strong>. Median warm solve time is <strong>{summary['median_warm_solve_ms']:.1f} ms</strong>; the strictest implicit heat solve takes <strong>{summary['maximum_warm_solve_ms']:.1f} ms</strong>. Decode, clustering, channel compilation, display lifting, and rendering are recorded separately.</p>
<p><strong>Accuracy scope.</strong> Machine-scale values certify retained manufactured channels or algebraic equations on the compressed finite operator. They are not held-out continuum accuracy on arbitrary CAD surfaces. The independent refinement section below remains the continuum audit.</p>
<h3>Six-field sheets</h3>
{''.join(figure_links)}
<p><a href="../production_nasa_cad_pde_visualization/report.md">Open the complete NASA visualization protocol, per-case CSV rows, and timing ledger.</a></p>
</div></div></div></section>
"""


def transparent_tail_section() -> str:
    """Embed the exact fixed-point shell closure and measured campaign."""

    required = (TAIL_SUMMARY, TAIL_ROWS, TAIL_MODES, TAIL_SVG)
    if any(not path.exists() for path in required):
        raise RuntimeError(
            "transparent-tail audit is missing; run "
            "scripts/transparent_tail_benchmark.py"
        )
    summary = json.loads(TAIL_SUMMARY.read_text(encoding="utf-8"))
    with TAIL_ROWS.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    with TAIL_MODES.open(newline="", encoding="utf-8") as handle:
        modes = list(csv.DictReader(handle))
    if not (
        summary["all_gates_passed"]
        and summary["autonomous_tail_truncation_error_with_cap"] == 0.0
        and not summary["dense_shell_matrix_stored"]
        and not summary["dense_boundary_matrix_stored"]
        and not summary["held_out_cad_machine_precision_claim"]
    ):
        raise RuntimeError("refusing to publish an uncertified transparent-tail audit")
    deepest = max(int(row["depth"]) for row in rows)
    pde_rows = []
    for mode in modes:
        selected = [
            row
            for row in rows
            if row["problem"] == mode["problem"] and int(row["depth"]) == deepest
        ]
        by_method = {row["method"]: row for row in selected}
        direct = by_method["direct_L_shell_thomas"]
        finite = by_method["compiled_finite_dirichlet"]
        cap = by_method["exact_fixed_point_cap"]
        speedup = float(direct["total_ms"]) / float(cap["total_ms"])
        shift = complex(
            float(mode["spectral_shift_real"]),
            float(mode["spectral_shift_imag"]),
        )
        pde_rows.append(
            "<tr>"
            f"<td>{html.escape(mode['problem'])}</td>"
            f"<td>{shift.real:.3g}{shift.imag:+.3g}i</td>"
            f"<td>{float(mode['maximum_root_modulus']):.6f}</td>"
            f"<td>{float(direct['total_ms']):.2f}</td>"
            f"<td>{float(finite['total_ms']):.2f}</td>"
            f"<td>{float(cap['total_ms']):.2f}</td>"
            f"<td>{speedup:.1f}x</td>"
            f"<td>{float(finite['relative_boundary_error']):.3e}</td>"
            "</tr>"
        )
    golden = summary["golden_certificate"]
    scaling_svg = TAIL_SVG.read_text(encoding="utf-8")
    return f"""
<section id="transparent-tail-closure" class="jp-Cell jp-MarkdownCell">
<div class="jp-Cell-inputWrapper"><div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput">
<h2>Exact transparent closure of cylindrical and conic tails</h2>
<p>An autonomous nearest-shell tail is eliminated analytically after one angular FFT. For each mode, the Schur map is <code>Phi(sigma) = d - u^2/sigma</code>. If <code>w + w^-1 = d/u</code> and <code>|w| &lt; 1</code>, the exact semi-infinite pivot is <code>Sigma* = u/w</code>, the retained-system self-energy is <code>u w</code>, and the interface flux symbol is <code>u(1-w)</code>. The cross ratio with the two fixed points obeys the exact identity <code>chi(Phi(sigma)) = w^2 chi(sigma)</code>.</p>
<p>On the unit-ratio cylinder, <code>log(Sigma*/u) = 2 asinh(sin(pi |k|/N_theta))</code>. The fixed-point cap therefore replaces every explicit shell by one custom QJet FFT, one diagonal multiplication, and one inverse FFT. It stores <strong>O(N_theta)</strong> data and has no tail-depth dependence.</p>
<div style="overflow-x:auto"><table><thead><tr><th>Resolvent</th><th>Shift</th><th>Max |w|</th><th>Direct ms</th><th>Finite-symbol ms</th><th>Cap ms</th><th>Direct/cap</th><th>Finite-tail error</th></tr></thead><tbody>{''.join(pde_rows)}</tbody></table></div>
<p>The table uses <code>N_theta={summary['n_theta']}</code>, <code>K={summary['right_hand_sides']}</code>, and <code>L={deepest}</code>. The direct Thomas solve and independently compiled finite-tail symbol agree within <strong>{summary['maximum_direct_finite_disagreement']:.3e}</strong>. The largest fixed-point residual is <strong>{summary['maximum_fixed_point_residual']:.3e}</strong>; the global cross-ratio residual is <strong>{summary['maximum_cross_ratio_residual']:.3e}</strong>. The exact cap has zero autonomous-tail truncation error by the Schur fixed-point identity.</p>
<div style="overflow-x:auto;border:1px solid #dedede">{scaling_svg}</div>
<h3>Arithmetic and branch certificates</h3>
<p>At <code>d/u=3</code>, <code>Sigma*/u=phi^2</code> and the contraction is <code>phi^-4</code>. At depth {golden['depth']}, the exact Fibonacci convergent is <code>{golden['numerator']}/{golden['denominator']}</code>, with error-law residual <strong>{golden['error_law_residual']:.3e}</strong>. Undamped propagating Helmholtz modes are rejected: a positive limiting-absorption or causal Laplace damping is required to select a unique decaying branch.</p>
<p>A sixteen-shell summably perturbed transition terminated by the exact cap has measured pivot defect <strong>{summary['perturbed_transition_actual_error']:.3e}</strong> under the a posteriori telescoping bound <strong>{summary['perturbed_transition_error_bound']:.3e}</strong>; its largest certified local Lipschitz factor is <strong>{summary['perturbed_transition_maximum_lipschitz']:.3f}</strong>.</p>
<p><strong>CAD scope.</strong> This theorem removes a structured cylindrical or conic exterior tail. It does not recover surface channels discarded by the independent held-out CAD campaign, which compiles 48–155 vertices and tests an unseen degree-four harmonic against a degree-three correction space. The separate 24–42-node NASA gallery reports retained-channel or finite-equation audits. Those machine-scale residuals and the large continuum holdout measure different errors.</p>
<p><a href="../transparent_tail_benchmark/report.md">Open the complete timing, memory, error, and certificate ledger.</a> <a href="../../docs/transparent_tail_dtn.md">Open the proof and normalization audit.</a></p>
</div></div></div></section>
"""


def validate_executed(notebook: nbformat.NotebookNode) -> None:
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    if not code_cells:
        raise RuntimeError("the notebook contains no executable cells")
    if any(cell.execution_count is None for cell in code_cells):
        raise RuntimeError("refusing to export an unexecuted notebook")
    errors = [
        output
        for cell in code_cells
        for output in cell.get("outputs", ())
        if output.get("output_type") == "error"
    ]
    if errors:
        raise RuntimeError("refusing to export a notebook containing errors")


def main() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    validate_executed(notebook)
    notebook.metadata["title"] = TITLE

    exporter = HTMLExporter(template_name="lab")
    exporter.exclude_input_prompt = True
    exporter.exclude_output_prompt = True
    exporter.embed_images = True
    body, _ = exporter.from_notebook_node(
        notebook,
        resources={"metadata": {"name": NOTEBOOK.stem}},
    )

    body = body.replace("</head>", REPORT_STYLE + "\n</head>", 1)
    if "<meta name=\"viewport\"" not in body:
        body = body.replace(
            "<head>",
            '<head>\n<meta name="viewport" content="width=device-width, initial-scale=1">',
            1,
        )
    body = body.replace("<title>Notebook</title>", f"<title>{TITLE}</title>", 1)
    section = (
        transparent_tail_section()
        + pde_validation_section()
        + cad_pde_validation_section()
        + nasa_cad_pde_visualization_section()
        + repaired_refinement_section()
        + cad_invertibility_section()
    )
    if "</main>" in body:
        body = body.replace("</main>", section + "\n</main>", 1)
    else:
        body = body.replace("</body>", section + "\n</body>", 1)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(body, encoding="utf-8")
    print(
        {
            "html": str(OUTPUT),
            "bytes": OUTPUT.stat().st_size,
            "cells": len(notebook.cells),
        }
    )


if __name__ == "__main__":
    main()
