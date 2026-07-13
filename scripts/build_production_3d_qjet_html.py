#!/usr/bin/env python3
"""Export the executed production 3D QJet notebook as one audited HTML file."""

# ruff: noqa: E501

from __future__ import annotations

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
<p>The best tested degree-one Laplace DtN error is <strong>{summary['best_sphere_degree_one_error']:.3e}</strong>, with observed order about <strong>{summary['median_degree_one_order']:.3f}</strong>. Thus the current lumped-node singular quadrature does not provide machine-precision continuum 3D PDE solves. A high-order tangent-cell/curvature repayment is still required; true bulk Poisson/heat additionally needs a volume-source channel, and the full Helmholtz DtN map needs its frequency-dependent lower-order operator.</p>
<p><a href="../production_3d_pde_validation/report.md">Open the complete PDE equations, per-shape table, and scope statement.</a></p>
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
    section = pde_validation_section() + cad_invertibility_section()
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
