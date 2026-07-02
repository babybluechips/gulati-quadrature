#!/usr/bin/env python3
"""Create a self-contained Jupyter notebook for the final Q pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "final_q_machine_precision_pipeline.py"
NOTEBOOK = ROOT / "notebooks" / "q_machine_precision_pipeline.ipynb"
CELL_COUNTER = 0


def next_cell_id(prefix: str) -> str:
    global CELL_COUNTER
    CELL_COUNTER += 1
    return f"{prefix}-{CELL_COUNTER:02d}"


def markdown_cell(text: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "id": next_cell_id("md"),
        "metadata": {},
        "source": dedent(text).strip("\n").splitlines(keepends=True),
    }


def code_cell(text: str, metadata: dict[str, object] | None = None) -> dict[str, object]:
    return {
        "cell_type": "code",
        "id": next_cell_id("code"),
        "execution_count": None,
        "metadata": {} if metadata is None else metadata,
        "outputs": [],
        "source": dedent(text).strip("\n").splitlines(keepends=True),
    }


def main() -> None:
    global CELL_COUNTER
    CELL_COUNTER = 0
    source = SCRIPT.read_text(encoding="utf-8")
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    cells = [
        markdown_cell(
            """
            # Self-contained machine-precision Q pipeline

            This notebook is the executable companion to the final Q pipeline. It embeds the
            production script source directly inside the notebook, executes that embedded source
            as an isolated module, and then audits the numerical outputs.

            The computational path is intentionally narrow: custom radix-two QJet FFT, final
            endpoint-repaid production Q symbols, finite-Laurent pullbacks, BGK/zeta repayment,
            and boundary PDE production-Q checks. It does not import NumPy, SciPy, or any FFT
            package, and it never stores a dense Q matrix.
            """
        ),
        markdown_cell(
            """
            ## What this notebook verifies

            1. The embedded source defines the production QJet FFT kernel and does not import
               NumPy/SciPy.
            2. The final quadrature/BGK suite passes the `1e-12` machine-precision gate.
            3. The boundary-only PDE suite covers Laplace DtN, heat, Poisson, Helmholtz, and wave
               over all finite-Laurent benchmark shapes.
            4. Direct competitors are benchmarked on the same shape/PDE grid with explicit shared
               solve versus per-shape repayment timing and Big-O storage notation.
            5. FEM, QBX, and structurally different quadrature benchmark assets are summarized as
               external repository competitor suites, never as ground-truth authorities.
            6. The matrix-free contract is explicit: generated QJets and spectra are stored, not
               an `n x n` dense Q matrix.
            7. The generated CSV/JSON/SVG artifacts are hashed and listed for reproducibility.
            """
        ),
        code_cell(
            "SCRIPT_SOURCE = " + repr(source) + "\n"
            "import sys, types\n"
            "from pathlib import Path\n"
            "PROJECT_ROOT = Path.cwd()\n"
            "if not (PROJECT_ROOT / 'scripts' / 'final_q_machine_precision_pipeline.py').exists() and (PROJECT_ROOT.parent / 'scripts' / 'final_q_machine_precision_pipeline.py').exists():\n"
            "    PROJECT_ROOT = PROJECT_ROOT.parent\n"
            "module = types.ModuleType('embedded_final_q_pipeline')\n"
            "namespace = module.__dict__\n"
            "sys.modules[module.__name__] = module\n"
            "exec(compile(SCRIPT_SOURCE, 'embedded_final_q_machine_precision_pipeline.py', 'exec'), namespace)\n"
            "namespace['ROOT'] = PROJECT_ROOT\n"
            "namespace['OUT'] = PROJECT_ROOT / 'outputs' / 'final_q_machine_precision_pipeline'\n"
            "print('Project root:', PROJECT_ROOT)\n"
            "print('Embedded functions:', ', '.join(name for name in ['fft', 'q_split_derivatives', 'run_quadrature_and_bgk', 'run_pde', 'main'] if name in namespace))\n"
            "print('Embedded source characters:', len(SCRIPT_SOURCE))\n",
            metadata={"tags": ["hide-input"], "jupyter": {"source_hidden": True}},
        ),
        code_cell(
            """
            import base64
            import csv
            import hashlib
            import html
            import json
            import math
            import re
            from pathlib import Path

            try:
                from IPython.display import HTML, SVG, display
            except Exception:
                HTML = SVG = display = None

            def show_html(fragment):
                if display is not None and HTML is not None:
                    display(HTML(fragment))
                else:
                    print(fragment)

            def show_svg(path, alt=None):
                path = Path(path)
                alt_text = html.escape(alt or path.stem.replace("_", " "))
                encoded = base64.b64encode(path.read_bytes()).decode("ascii")
                fragment = f"<img alt='{alt_text}' src='data:image/svg+xml;base64,{encoded}' style='max-width:100%;height:auto'/>"
                if display is not None and HTML is not None:
                    display(HTML(fragment))
                else:
                    print(path)

            def fmt(value, digits=3):
                if isinstance(value, str):
                    return value
                value = float(value)
                if value == 0:
                    return "0"
                if abs(value) < 1.0e-3 or abs(value) >= 1.0e4:
                    return f"{value:.{digits}e}"
                return f"{value:.{digits}f}"

            def read_csv_rows(path):
                with Path(path).open(newline="", encoding="utf-8") as handle:
                    return list(csv.DictReader(handle))

            def median(values):
                clean = sorted(float(value) for value in values if value is not None and value != "")
                if not clean:
                    return float("nan")
                mid = len(clean) // 2
                if len(clean) % 2:
                    return clean[mid]
                return 0.5 * (clean[mid - 1] + clean[mid])

            def load_json_if_exists(path):
                path = Path(path)
                if not path.exists():
                    return None
                return json.loads(path.read_text(encoding="utf-8"))

            def html_table(headers, rows, max_rows=None):
                shown = rows if max_rows is None else rows[:max_rows]
                head = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
                body = []
                for row in shown:
                    body.append("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>")
                if max_rows is not None and len(rows) > max_rows:
                    body.append(f"<tr><td colspan='{len(headers)}'>... {len(rows) - max_rows} more rows</td></tr>")
                return (
                    "<table style='border-collapse:collapse;font-family:serif;font-size:13px'>"
                    "<thead><tr style='border-bottom:1px solid #111'>" + head + "</tr></thead>"
                    "<tbody>" + "".join(body) + "</tbody></table>"
                )

            print("Notebook helpers loaded.")
            """
        ),
        markdown_cell(
            """
            ## Source and contract audit

            This cell checks the embedded source before any numerical run. The audit is deliberately
            syntactic for import bans, then semantic for the callable kernel and no-dense-matrix
            payload flag after execution.
            """
        ),
        code_cell(
            """
            banned_imports = []
            for pattern in (
                r"^\\s*import\\s+numpy\\b",
                r"^\\s*import\\s+scipy\\b",
                r"^\\s*from\\s+numpy\\b",
                r"^\\s*from\\s+scipy\\b",
                r"np\\.",
            ):
                if re.search(pattern, SCRIPT_SOURCE, flags=re.MULTILINE):
                    banned_imports.append(pattern)

            required_symbols = ["fft", "ifft", "q_split_derivatives", "taylor_repay", "run_quadrature_and_bgk", "run_pde", "main"]
            missing_symbols = [name for name in required_symbols if name not in namespace]

            assert not banned_imports, banned_imports
            assert not missing_symbols, missing_symbols
            assert "dense_q_matrix_stored" in SCRIPT_SOURCE
            assert "def fft(" in SCRIPT_SOURCE and "def ifft(" in SCRIPT_SOURCE

            audit_rows = [
                ["NumPy/SciPy import patterns", "none"],
                ["custom FFT definitions", "fft and ifft present"],
                ["dense matrix storage flag", "present and checked after run"],
                ["embedded source sha256", hashlib.sha256(SCRIPT_SOURCE.encode("utf-8")).hexdigest()[:16]],
            ]
            show_html(html_table(["audit item", "result"], audit_rows))
            """
        ),
        markdown_cell(
            """
            ## Execute the production pipeline

            This runs the embedded `main()` function. It writes the same JSON, CSV, and SVG artifacts
            as the standalone script, then returns a pass/fail payload.
            """
        ),
        code_cell(
            """
            payload = namespace["main"]()
            assert payload["passed"] is True
            assert payload["dense_q_matrix_stored"] is False
            payload
            """
        ),
        markdown_cell(
            """
            ## Machine-precision gate summary

            The gate is the maximum of the split quadrature error, BGK-8 repayment error, and the
            endpoint-repaid production-Q PDE residual over the included benchmark suite. The raw
            finite-cycle symbol is reported separately as a diagnostic only.
            """
        ),
        code_cell(
            """
            summary_rows = [
                ["pass gate", str(payload["passed"]).lower()],
                ["machine tolerance", fmt(payload["machine_tol"])],
                ["max split relative error", fmt(payload["max_split_rel_error"])],
                ["max BGK-8 relative error", fmt(payload["max_bgk8_rel_error"])],
                ["max production-Q PDE residual", fmt(payload["max_pde_production_q_residual"])],
                ["max production-Q continuum error", fmt(payload["max_pde_continuum_rel_error"])],
                ["max raw finite-cycle diagnostic error", fmt(payload["max_raw_cycle_diagnostic_rel_error"])],
                ["core quadrature shapes", payload["core_quadrature_shape_count"]],
                ["extended shapes", payload["extended_shape_count"]],
                ["total PDE/competitor shapes", payload["shape_count"]],
                ["PDE case count", payload["pde_case_count"]],
                ["direct competitor methods", payload["competitor_method_count"]],
                ["direct competitor cases", payload["competitor_pde_case_count"]],
                ["direct competitor n", payload["competitor_n"]],
                ["mean shared circle solve ms", fmt(payload["pde_mean_shared_circle_solve_ms"])],
                ["mean standalone shape/PDE ms", fmt(payload["pde_mean_standalone_shape_pde_ms"])],
                ["mean batched shape/PDE ms", fmt(payload["pde_mean_batched_problem_shape_ms"])],
                ["mean full-suite amortized ms", fmt(payload["pde_mean_full_suite_amortized_ms"])],
                ["FFT kernel", payload["fft_kernel"]],
                ["dense Q matrix stored", str(payload["dense_q_matrix_stored"]).lower()],
            ]
            show_html(html_table(["quantity", "value"], summary_rows))
            assert max(payload["max_split_rel_error"], payload["max_bgk8_rel_error"], payload["max_pde_production_q_residual"], payload["max_pde_continuum_rel_error"]) <= payload["machine_tol"]
            """
        ),
        markdown_cell(
            """
            ## Benchmark shape suite

            The first ten shapes are the analytic machine-precision quadrature gate. The remaining
            shapes expand the PDE and competitor benchmark grid with smoothed polygonal, star,
            airfoil, stealth-like, and GWW-pair surrogate geometries. Exact-corner polygon and GWW
            claims are kept separate below; the Laurent list here is the smooth pullback catalogue
            used by the boundary-only PDE benchmark.
            """
        ),
        code_cell(
            """
            shape_rows = []
            core_names = {shape.name for shape in namespace["core_shapes"]()}
            for shape in namespace["shapes"]():
                coeff_text = ", ".join(f"{power}:{coeff.real:+.3f}{coeff.imag:+.3f}i" for power, coeff in shape.coeffs)
                gate = "quadrature+PDE" if shape.name in core_names else "PDE/competitor"
                shape_rows.append([shape.name, shape.family, gate, len(shape.coeffs), coeff_text])
            show_html(html_table(["shape", "family", "coverage", "terms", "Laurent coefficients"], shape_rows, max_rows=40))
            """
        ),
        code_cell(
            """
            def make_shape_gallery_svg(path, samples=192):
                shapes = list(namespace["shapes"]())
                cols = 3
                panel_w, panel_h = 330, 205
                pad = 36
                width = cols * panel_w
                rows = math.ceil(len(shapes) / cols)
                height = rows * panel_h
                parts = [
                    f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
                    "<rect width='100%' height='100%' fill='white'/>",
                ]
                unit = namespace["unit"]
                tau = namespace["TAU"]
                for index, shape in enumerate(shapes):
                    col = index % cols
                    row = index // cols
                    ox, oy = col * panel_w, row * panel_h
                    pts = [shape.psi(unit(tau * k / samples)) for k in range(samples + 1)]
                    min_x, max_x = min(p.real for p in pts), max(p.real for p in pts)
                    min_y, max_y = min(p.imag for p in pts), max(p.imag for p in pts)
                    span = max(max_x - min_x, max_y - min_y, 1.0e-12)
                    scale = min((panel_w - 2 * pad) / span, (panel_h - 2 * pad) / span)
                    cx, cy = 0.5 * (min_x + max_x), 0.5 * (min_y + max_y)

                    def map_point(z):
                        x = ox + panel_w / 2 + (z.real - cx) * scale
                        y = oy + panel_h / 2 - (z.imag - cy) * scale
                        return x, y

                    x_axis_a = map_point(complex(min_x, 0.0))
                    x_axis_b = map_point(complex(max_x, 0.0))
                    y_axis_a = map_point(complex(0.0, min_y))
                    y_axis_b = map_point(complex(0.0, max_y))
                    parts.append(f"<line x1='{x_axis_a[0]:.2f}' y1='{x_axis_a[1]:.2f}' x2='{x_axis_b[0]:.2f}' y2='{x_axis_b[1]:.2f}' stroke='#dddddd'/>")
                    parts.append(f"<line x1='{y_axis_a[0]:.2f}' y1='{y_axis_a[1]:.2f}' x2='{y_axis_b[0]:.2f}' y2='{y_axis_b[1]:.2f}' stroke='#dddddd'/>")
                    path_data = []
                    for k, z in enumerate(pts):
                        x, y = map_point(z)
                        path_data.append(("M" if k == 0 else "L") + f"{x:.2f},{y:.2f}")
                    parts.append(f"<path d='{' '.join(path_data)} Z' fill='none' stroke='#111111' stroke-width='1.8'/>")
                    parts.append(f"<text x='{ox + 18}' y='{oy + 24}' font-family='serif' font-size='15'>{html.escape(shape.name)}</text>")
                    parts.append(f"<text x='{ox + 18}' y='{oy + 43}' font-family='serif' font-size='11' fill='#555'>{html.escape(shape.family)}</text>")
                parts.append("</svg>")
                Path(path).write_text("\\n".join(parts), encoding="utf-8")
                return Path(path)

            shape_gallery = Path(payload["figure"]).with_name("notebook_shape_gallery.svg")
            make_shape_gallery_svg(shape_gallery)
            show_svg(shape_gallery, "Finite Laurent benchmark shape gallery")
            """
        ),
        markdown_cell(
            """
            ## Quadrature and BGK/zeta repayment

            The raw shifted target records the endpoint defect. The BGK-8 column applies the
            Taylor/zeta repayment ledger and is the value used in the pass gate.
            """
        ),
        code_cell(
            """
            quad_rows = read_csv_rows(payload["quadrature_rows"])
            quad_display = [
                [
                    row["shape"],
                    row["family"],
                    row["n"],
                    fmt(row["split_rel_error"]),
                    fmt(row["raw_shift_rel_error"]),
                    fmt(row["bgk8_rel_error"]),
                    row["dense_q_matrix_stored"],
                ]
                for row in quad_rows
            ]
            show_html(html_table(["shape", "family", "n", "split err", "raw shift err", "BGK-8 err", "dense stored"], quad_display))
            worst_quad = max(quad_rows, key=lambda row: float(row["bgk8_rel_error"]))
            print("Worst BGK-8 row:", worst_quad)
            assert all(float(row["bgk8_rel_error"]) <= payload["machine_tol"] for row in quad_rows)
            assert all(row["dense_q_matrix_stored"] == "False" for row in quad_rows)
            """
        ),
        markdown_cell(
            """
            ## Boundary-only PDE production-Q checks

            The PDE rows use one endpoint-repaid production-Q circle solve per PDE/boundary datum,
            then repay the conformal metric on each finite-Laurent shape. Timing is therefore split
            into the shared circle solve, the per-shape pullback, the per-row repayment/error
            bookkeeping, and two derived totals: standalone shape/PDE time and batched per-shape
            time when all shapes share the same circle solve.
            """
        ),
        code_cell(
            """
            pde_rows = read_csv_rows(payload["pde_rows"])
            problem_names = sorted({row["problem"] for row in pde_rows})
            pde_summary = []
            for problem in problem_names:
                subset = [row for row in pde_rows if row["problem"] == problem]
                worst_generated = max(subset, key=lambda row: float(row["generated_q_residual"]))
                worst_continuum = max(subset, key=lambda row: float(row["continuum_rel_error"]))
                worst_raw_cycle = max(subset, key=lambda row: float(row["raw_cycle_diagnostic_rel_error"]))
                mean_shared = sum(float(row["shared_circle_solve_ms"]) for row in subset) / len(subset)
                mean_pullback = sum(float(row["shape_pullback_ms"]) for row in subset) / len(subset)
                mean_repay = sum(float(row["shape_repay_ms"]) for row in subset) / len(subset)
                mean_standalone = sum(float(row["standalone_shape_pde_ms"]) for row in subset) / len(subset)
                mean_batched = sum(float(row["batched_problem_shape_ms"]) for row in subset) / len(subset)
                pde_summary.append([
                    problem,
                    len(subset),
                    worst_generated["shape"],
                    fmt(worst_generated["generated_q_residual"]),
                    worst_continuum["shape"],
                    fmt(worst_continuum["continuum_rel_error"]),
                    fmt(worst_raw_cycle["raw_cycle_diagnostic_rel_error"]),
                    fmt(mean_shared),
                    fmt(mean_pullback),
                    fmt(mean_repay),
                    fmt(mean_standalone),
                    fmt(mean_batched),
                ])
            show_html(html_table([
                "problem",
                "cases",
                "worst production shape",
                "max production residual",
                "worst continuum shape",
                "max continuum err",
                "max raw-cycle diagnostic",
                "mean shared solve ms",
                "mean pullback ms",
                "mean repay ms",
                "mean standalone ms",
                "mean batched ms",
            ], pde_summary))
            assert len(pde_rows) == payload["pde_case_count"] == payload["shape_count"] * len(problem_names)
            assert all(float(row["generated_q_residual"]) <= payload["machine_tol"] for row in pde_rows)
            assert all(float(row["continuum_rel_error"]) <= payload["machine_tol"] for row in pde_rows)
            assert all(row["dense_q_matrix_stored"] == "False" for row in pde_rows)
            """
        ),
        markdown_cell(
            """
            ## Direct competitor benchmark on the same shape/PDE grid

            This section benchmarks every direct competitor on the same 24-shape by 5-PDE grid.
            The benchmark grid is intentionally smaller than the final accuracy gate (`n=256`) so
            the deliberately naive `O(n^2)` direct DFT competitor can run inside the notebook. The
            timing columns keep the shared circle solve separate from the shape-dependent repayment.
            The comparison is still direct: same shapes, same PDEs, same boundary mode, same conformal
            metric repayment, and the same output norms.
            """
        ),
        code_cell(
            """
            competitor_rows = read_csv_rows(payload["competitor_pde_rows"])
            methods = sorted({row["method"] for row in competitor_rows})
            direct_summary = []
            for method in methods:
                subset = [row for row in competitor_rows if row["method"] == method]
                first = subset[0]
                direct_summary.append([
                    method,
                    first["label"],
                    first["time_big_o"],
                    first["storage_big_o"],
                    len(subset),
                    len({row["shape"] for row in subset}),
                    len({row["problem"] for row in subset}),
                    fmt(median(row["shared_circle_solve_ms"] for row in subset)),
                    fmt(median(row["standalone_shape_pde_ms"] for row in subset)),
                    fmt(median(row["batched_problem_shape_ms"] for row in subset)),
                    fmt(max(float(row["relative_l2_vs_generated_q"]) for row in subset)),
                    fmt(max(float(row["relative_l2_vs_continuum"]) for row in subset)),
                    first["dense_matrix_stored"],
                ])
            show_html(html_table([
                "method",
                "description",
                "time",
                "storage",
                "cases",
                "shapes",
                "PDEs",
                "median shared solve ms",
                "median standalone ms",
                "median batched ms",
                "max rel. vs production Q",
                "max rel. vs continuum",
                "dense stored",
            ], direct_summary))
            assert len(competitor_rows) == payload["competitor_pde_case_count"]
            assert len(competitor_rows) == payload["competitor_method_count"] * payload["shape_count"] * len(problem_names)
            assert all(row["dense_matrix_stored"] == "False" for row in competitor_rows)
            """
        ),
        code_cell(
            """
            by_problem_rows = []
            for problem in problem_names:
                for method in methods:
                    subset = [row for row in competitor_rows if row["problem"] == problem and row["method"] == method]
                    worst_q = max(subset, key=lambda row: float(row["relative_l2_vs_generated_q"]))
                    worst_c = max(subset, key=lambda row: float(row["relative_l2_vs_continuum"]))
                    by_problem_rows.append([
                        problem,
                        method,
                        subset[0]["time_big_o"],
                        fmt(median(row["shared_circle_solve_ms"] for row in subset)),
                        fmt(median(row["batched_problem_shape_ms"] for row in subset)),
                        worst_q["shape"],
                        fmt(worst_q["relative_l2_vs_generated_q"]),
                        worst_c["shape"],
                        fmt(worst_c["relative_l2_vs_continuum"]),
                    ])
            show_html(html_table([
                "PDE",
                "method",
                "time",
                "median shared solve ms",
                "median batched ms",
                "worst production-Q shape",
                "max rel. vs production Q",
                "worst continuum shape",
                "max rel. vs continuum",
            ], by_problem_rows))
            """
        ),
        code_cell(
            """
            by_shape_rows = []
            shape_names = sorted({row["shape"] for row in competitor_rows})
            for shape in shape_names:
                for method in methods:
                    subset = [row for row in competitor_rows if row["shape"] == shape and row["method"] == method]
                    by_shape_rows.append([
                        shape,
                        subset[0]["family"],
                        method,
                        subset[0]["time_big_o"],
                        fmt(median(row["shape_pullback_ms"] for row in subset)),
                        fmt(median(row["shape_repay_ms"] for row in subset)),
                        fmt(median(row["batched_problem_shape_ms"] for row in subset)),
                        fmt(max(float(row["relative_l2_vs_generated_q"]) for row in subset)),
                        fmt(max(float(row["relative_l2_vs_continuum"]) for row in subset)),
                    ])
            show_html(html_table([
                "shape",
                "family",
                "method",
                "time",
                "median pullback ms",
                "median repay ms",
                "median batched ms",
                "max rel. vs production Q",
                "max rel. vs continuum",
            ], by_shape_rows, max_rows=80))
            """
        ),
        markdown_cell(
            """
            ## External competitor suites in the repository

            The direct table above is the same-grid notebook benchmark. The repository also contains
            heavier competitor suites for FEM, QBX, and structurally different quadrature methods.
            These are loaded here as audited artifacts and summarized separately because they use
            their own meshes, reference resolutions, or local-expansion protocols. FEM is a
            volumetric baseline, not the reference truth; Q/FEM norms in pairwise suites are
            disagreement diagnostics unless an analytic or independently overresolved reference is
            present. Headline ground-truth claims must cite the held-out benchmark registry.
            """
        ),
        code_cell(
            """
            external_assets = {
                "final production Q / FEM pairwise": PROJECT_ROOT / "outputs/final_q_vs_fem_funky/final_q_vs_fem_funky.json",
                "funky Q/FEM pairwise": PROJECT_ROOT / "docs/assets/q_dtn_funky_fem_head_to_head.json",
                "disk/manufactured analytic reference with FEM baseline": PROJECT_ROOT / "docs/assets/q_dtn_vs_fem_benchmark.json",
                "QBX head-to-head": PROJECT_ROOT / "docs/assets/qbx_head_to_head_benchmark.json",
                "structural quadrature methods": PROJECT_ROOT / "docs/assets/structural_quadrature_methods_benchmark.json",
                "GWW Q convergence": PROJECT_ROOT / "outputs/gww_isospectral_section11/section11_gww_q_convergence.csv",
                "held-out benchmark registry": PROJECT_ROOT / "outputs/standard_scientific_benchmarks/benchmark_registry.json",
            }
            external_rows = []
            registry = load_json_if_exists(external_assets["held-out benchmark registry"])
            if registry:
                external_rows.append([
                    "held-out benchmark registry",
                    registry["case_count"],
                    "cited external standards",
                    "defines what may be called ground truth",
                    "",
                    "",
                    "",
                    "FEM/QBX/local manufactured rows are diagnostics unless tied to a registry id",
                ])

            final_q_fem = load_json_if_exists(external_assets["final production Q / FEM pairwise"])
            if final_q_fem:
                summary = final_q_fem["summary"]
                external_rows.append([
                    "final production Q / FEM pairwise",
                    summary["case_count"],
                    f"{summary['shape_count']} shapes x {summary['problem_count']} PDEs",
                    "final custom-FFT production Q, O(n log n), paired with FEM O(n^2) apply",
                    fmt(summary["median_q_apply_ms"]),
                    fmt(summary["median_fem_apply_ms"]),
                    fmt(summary["median_fem_cold_ms"]),
                    f"Q cold faster {summary.get('q_cold_faster_count', summary['q_cold_win_count'])}/{summary['case_count']}; FEM apply faster {summary.get('fem_apply_faster_count', summary['fem_apply_win_count'])}/{summary['case_count']}; median pairwise L2 {fmt(summary.get('median_best_scaled_pairwise_l2_disagreement', summary['median_best_scaled_relative_l2_vs_fem']))}",
                ])

            fem_funky = load_json_if_exists(external_assets["funky Q/FEM pairwise"])
            if fem_funky:
                summary = fem_funky["summary"]
                external_rows.append([
                    "funky Q/FEM pairwise",
                    summary["case_count"],
                    f"{summary['shape_count']} shapes x {summary['problem_count']} PDEs",
                    "Q apply O(n log n)+low-rank; FEM sparse volumetric Schur",
                    fmt(summary["median_q_apply_ms"]),
                    fmt(summary["median_fem_apply_ms"]),
                    fmt(summary["median_fem_cold_ms"]),
                    fmt(summary.get("median_best_scaled_pairwise_l2_disagreement", summary["median_best_scaled_relative_l2_vs_fem"])),
                ])

            fem_disk = load_json_if_exists(external_assets["disk/manufactured analytic reference with FEM baseline"])
            if fem_disk:
                summary = fem_disk["summary"]
                external_rows.append([
                    "cited disk modal reference with FEM baseline",
                    summary["case_count"],
                    f"{summary['mode_count']} modes x {summary['problem_count']} PDEs",
                    "Q formula O(1) per mode; cited Steklov/DLMF reference; FEM volumetric baseline",
                    fmt(summary["median_q_formula_ms"]),
                    fmt(summary["median_fem_ms"]),
                    "",
                    fmt(summary["median_q_formula_relative_error"]),
                ])

            qbx = load_json_if_exists(external_assets["QBX head-to-head"])
            if qbx:
                summary = qbx["summary"]
                external_rows.append([
                    "QBX head-to-head",
                    summary["case_count"],
                    "near-boundary quadrature cases",
                    "Q bridge/multipole-zeta vs QBX local expansion",
                    "",
                    "",
                    "",
                    f"max QBX refined {fmt(summary['max_qbx_refined_relative_error'])}; max bridge {fmt(summary['max_bridge_relative_error'])}",
                ])

            structural = load_json_if_exists(external_assets["structural quadrature methods"])
            if structural:
                summary = structural["summary"]
                methods_count = len(summary.get("methods", {}))
                external_rows.append([
                    "structural quadrature methods",
                    summary["case_count"],
                    f"{methods_count} methods",
                    "trap, subtraction, adaptive panel, QBX, Q bridge, multipole-zeta Q",
                    "",
                    "",
                    "",
                    f"families {len(summary.get('families', {}))}",
                ])

            gww_path = external_assets["GWW Q convergence"]
            if gww_path.exists():
                gww_rows = read_csv_rows(gww_path)
                last = gww_rows[-1]
                external_rows.append([
                    "GWW Q convergence",
                    len(gww_rows),
                    "isospectral polygon pair refinements",
                    "projected chord-Q Ritz witness",
                    "",
                    "",
                    "",
                    f"n={last['n']}, split={fmt(last['relative_split_first6'])}",
                ])

            show_html(html_table([
                "suite",
                "cases",
                "coverage",
                "competitor/model",
                "Q median ms",
                "competitor median ms",
                "competitor cold ms",
                "headline diagnostic/result",
            ], external_rows))
            """
        ),
        code_cell(
            """
            final_q_fem_path = PROJECT_ROOT / "outputs/final_q_vs_fem_funky/final_q_vs_fem_funky.json"
            final_q_fem = load_json_if_exists(final_q_fem_path)
            if final_q_fem:
                by_problem_rows = []
                for problem, stats in sorted(final_q_fem["summary"]["by_problem"].items()):
                    by_problem_rows.append([
                        problem,
                        stats["rows"],
                        "O(n log n)",
                        "O(n^2) apply after Schur/eigensolve build",
                        fmt(stats["median_q_apply_ms"]),
                        fmt(stats["median_fem_apply_ms"]),
                        fmt(stats["median_q_cold_ms"]),
                        fmt(stats["median_fem_cold_ms"]),
                        f"{stats.get('q_cold_faster_count', stats['q_cold_win_count'])}/{stats['rows']}",
                        f"{stats.get('fem_apply_faster_count', stats['fem_apply_win_count'])}/{stats['rows']}",
                        fmt(stats.get("median_best_scaled_pairwise_l2_disagreement", stats["median_best_scaled_relative_l2_vs_fem"])),
                        fmt(stats.get("max_best_scaled_pairwise_l2_disagreement", stats["max_best_scaled_relative_l2_vs_fem"])),
                    ])
                show_html(html_table([
                    "PDE",
                    "cases",
                    "Q cost",
                    "FEM apply cost",
                    "Q apply ms",
                    "FEM apply ms",
                    "Q cold ms",
                    "FEM cold ms",
                    "Q cold faster",
                    "FEM apply faster",
                    "median Q-FEM pairwise L2",
                    "max Q-FEM pairwise L2",
                ], by_problem_rows))
            """
        ),
        code_cell(
            """
            if final_q_fem:
                by_shape_rows = []
                for shape, stats in sorted(final_q_fem["summary"]["by_shape"].items()):
                    by_shape_rows.append([
                        shape,
                        stats["family"],
                        stats["rows"],
                        fmt(stats["median_q_apply_ms"]),
                        fmt(stats["median_fem_apply_ms"]),
                        fmt(stats.get("median_best_scaled_pairwise_l2_disagreement", stats["median_best_scaled_relative_l2_vs_fem"])),
                        fmt(stats.get("max_best_scaled_pairwise_l2_disagreement", stats["max_best_scaled_relative_l2_vs_fem"])),
                    ])
                show_html(html_table([
                    "shape",
                    "family",
                    "PDE cases",
                    "Q apply ms",
                    "FEM apply ms",
                    "median Q-FEM pairwise L2",
                    "max Q-FEM pairwise L2",
                ], by_shape_rows, max_rows=40))
            """
        ),
        code_cell(
            """
            structural_path = PROJECT_ROOT / "docs/assets/structural_quadrature_methods_benchmark.json"
            structural_detail = load_json_if_exists(structural_path)
            if structural_detail:
                structural_shape_rows = []
                for shape in sorted({row["shape"] for row in structural_detail["rows"]}):
                    subset = [row for row in structural_detail["rows"] if row["shape"] == shape]
                    structural_shape_rows.append([
                        shape,
                        subset[0]["family"],
                        len(subset),
                        ", ".join(sorted({row["target_mode"] for row in subset})),
                        fmt(median(row["trapezoid_relative_error"] for row in subset)),
                        fmt(median(row["gulati_q_bridge_relative_error"] for row in subset)),
                        fmt(median(row["multipole_zeta_q_relative_error"] for row in subset)),
                        fmt(median(row["qbx_refined_relative_error"] for row in subset)),
                    ])
                show_html(html_table([
                    "external shape",
                    "family",
                    "cases",
                    "target modes",
                    "trap median err",
                    "Q bridge median err",
                    "multipole-zeta Q median err",
                    "QBX refined median err",
                ], structural_shape_rows))
            """
        ),
        code_cell(
            """
            gww_path = PROJECT_ROOT / "outputs/gww_isospectral_section11/section11_gww_q_convergence.csv"
            if gww_path.exists():
                gww_rows = read_csv_rows(gww_path)
                gww_display = []
                for row in gww_rows:
                    gww_display.append([
                        row["n"],
                        fmt(row["relative_split_first6"]),
                        ", ".join(fmt(row[f"left_ritz_{idx}"], 4) for idx in range(1, 4)),
                        ", ".join(fmt(row[f"right_ritz_{idx}"], 4) for idx in range(1, 4)),
                    ])
                show_html(html_table(["n", "relative split first 6", "left Ritz 1-3", "right Ritz 1-3"], gww_display))
            else:
                print("GWW convergence CSV not found.")
            """
        ),
        markdown_cell(
            """
            ## Residual figure

            The figure below is generated by the embedded production script. The left panel shows
            BGK-8 quadrature errors across the shape suite; the right panel shows worst production-Q
            PDE residuals by equation.
            """
        ),
        code_cell(
            """
            residual_svg = Path(payload["figure"])
            assert residual_svg.exists()
            show_svg(residual_svg, "Final Q pipeline residual figure")
            """
        ),
        markdown_cell(
            """
            ## Cost accounting with Big-O notation

            The computational path is two custom FFTs plus diagonal spectral multipliers. For a batch
            of `S` shapes using the same pulled-back boundary datum, the circle solve is paid once and
            the shape-dependent repayment is linear in `S n`. Dense storage is included only as a
            counterfactual baseline.
            """
        ),
        code_cell(
            """
            big_o_rows = [
                ["Q shared circle solve", "O(n log n)", "O(n)", "paid once per PDE/boundary datum"],
                ["Q metric repayment for S shapes", "O(S n)", "O(n) per streamed shape", "shape pullback and divide by |psi'|"],
                ["Q amortized per shape in S-shape batch", "O((n log n)/S + n)", "O(n)", "fair per-shape timing for this harness"],
                ["direct naive DFT generated spectrum", "O(n^2)", "O(n)", "same generated operator, slow transform"],
                ["finite-difference sqrt-Laplacian surrogate", "O(n log n)", "O(n)", "local-symbol competitor"],
                ["continuum disk symbol oracle", "O(n log n)", "O(n)", "accuracy oracle for disk symbol"],
                ["dense Q matrix apply", "O(n^2)", "O(n^2)", "counterfactual; not stored by this pipeline"],
                ["FEM volumetric DtN baseline", "superlinear sparse build + boundary apply", "mesh-dependent; often O(N_mesh)", "competitor only; not ground truth"],
                ["QBX local expansion", "O(n p) to O(n log n + n p) with acceleration", "O(n p)", "external benchmark asset"],
            ]
            show_html(html_table(["method/model", "time complexity", "storage complexity", "role"], big_o_rows))

            cost_rows = []
            for n in (512, namespace["N"], 4096, namespace["REFERENCE_N"]):
                fft_work = 2 * n * int(math.log2(n))
                dense_entries = n * n
                dense_bytes_complex = dense_entries * 16
                symbol_bytes_complex = n * 16
                cost_rows.append([
                    n,
                    f"~{fft_work:,}",
                    f"{dense_entries:,}",
                    f"{dense_bytes_complex / (1024**2):.1f} MiB",
                    f"{symbol_bytes_complex / 1024:.1f} KiB",
                    f"{dense_entries / max(n, 1):.0f}x entries",
                ])
            show_html(html_table(["n", "two-FFT work units", "dense entries avoided", "dense complex storage", "Q symbol storage", "entry ratio"], cost_rows))
            """
        ),
        markdown_cell(
            """
            ## Reproducibility manifest

            These are the files produced by this notebook run. The hashes make it clear which
            numerical artifacts were inspected.
            """
        ),
        code_cell(
            """
            manifest_paths = [
                Path(payload["quadrature_rows"]),
                Path(payload["pde_rows"]),
                Path(payload["competitor_pde_rows"]),
                PROJECT_ROOT / "outputs/final_q_vs_fem_funky/final_q_vs_fem_funky.json",
                PROJECT_ROOT / "outputs/final_q_vs_fem_funky/final_q_vs_fem_funky_rows.csv",
                Path(payload["figure"]),
                shape_gallery,
                Path(payload["figure"]).with_name("final_q_machine_precision_summary.json"),
                PROJECT_ROOT / "outputs/gww_isospectral_section11/section11_gww_q_convergence.csv",
            ]
            manifest_rows = []
            for path in manifest_paths:
                if not path.exists():
                    continue
                data = path.read_bytes()
                manifest_rows.append([path.name, path.stat().st_size, hashlib.sha256(data).hexdigest()[:16], str(path)])
            show_html(html_table(["artifact", "bytes", "sha256 prefix", "path"], manifest_rows))
            """
        ),
        markdown_cell(
            """
            ## Result

            The notebook has executed the same production source it embeds, audited the core
            invariants, run all quadrature and PDE checks, benchmarked direct competitors across the
            expanded shape/PDE grid, summarized FEM/QBX/GWW external competitor artifacts without
            treating FEM as truth, displayed
            the shape and residual figures, and recorded a reproducibility manifest. Passing this
            notebook means the final Q pipeline satisfies the matrix-free machine-precision gate for
            the analytic core suite and reports the broader extended-geometry benchmark suite.
            """
        ),
    ]
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
    print(json.dumps({"notebook": str(NOTEBOOK), "embedded_script": str(SCRIPT)}, indent=2))


if __name__ == "__main__":
    main()
