#!/usr/bin/env python3
"""Build the Q+BGK/PDE benchmark LaTeX report.

The report intentionally consumes benchmark artifacts rather than adding a new
numerical method.  The one fresh computation is a matrix-free scaling sweep of
the current exact-Laurent pullback pipeline.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "machine_precision_pipeline_report"
FIG = OUT / "figures"
PDF_TEXT = OUT / "pdf_text"
SIZES = (256, 512, 1024, 2048, 4096)
SPECTRUM_MODES = (0, 1, 2, 4, 7, 8, 16, 32)
PDE_PROBLEMS = {
    "laplace_dtn": {},
    "heat": {"time": 0.17},
    "poisson": {"mass": 0.35},
    "helmholtz": {"wavenumber": 3.7, "damping": 0.02},
    "wave": {"time": 0.8},
}
KEYWORD_GROUPS = {
    "Q/kernel": ("quadrature", "kernel", "QBX", "spectral", "machine precision"),
    "DtN/PDE": ("Dirichlet-to-Neumann", "DtN", "Laplace", "Helmholtz", "heat", "Poisson", "wave"),
    "BGK/zeta": ("BGK", "zeta", "endpoint", "Euler", "Spitzer"),
    "shape/corner": ("corner", "cusp", "ellipse", "Joukowski", "polygon", "meshless", "shape optimization"),
}
REFERENCED_PDFS = (
    (
        "optimal_quadrature_merged (26).pdf",
        Path("/Users/rick/Downloads/optimal_quadrature_merged (26).pdf"),
        "early optimal-Q source; benchmark claim cross-check",
    ),
    (
        "optimal_quadrature_merged (37).pdf",
        Path("/Users/rick/Downloads/optimal_quadrature_merged (37).pdf"),
        "later merged optimal-Q source; benchmark claim cross-check",
    ),
    (
        "optimal_quadrature_v63_public (11).pdf",
        Path("/Users/rick/Library/Mobile Documents/com~apple~CloudDocs/optimal_quadrature_v63_public (11).pdf"),
        "kernel split, circle pullback, QBX/corner claim boundary",
    ),
    (
        "isl_hessian_v5 (6) copy.pdf",
        Path("/Users/rick/FREEFORM/oo/isl_hessian_v5 (6) copy.pdf"),
        "ellipse/Hessian/context audit; not a numerical benchmark source",
    ),
    (
        "paper.pdf",
        Path("/Users/rick/MESHLESS/paper.pdf"),
        "meshless shape-optimization/autograd context audit",
    ),
    (
        "cauchy_transport_calculus_trim (13).pdf",
        Path("/Users/rick/Downloads/cauchy_transport_calculus_trim (13).pdf"),
        "Cauchy transport early source; pullback notation audit",
    ),
    (
        "cauchy_transport_calculus_trim (41).pdf",
        Path("/Users/rick/Downloads/cauchy_transport_calculus_trim (41).pdf"),
        "corner/Joukowski/Kondratiev source audit",
    ),
    (
        "cauchy_transport_calculus_trim (47).pdf",
        Path("/Users/rick/Downloads/cauchy_transport_calculus_trim (47).pdf"),
        "complex scale-phase pullback source audit",
    ),
    (
        "cauchy_transport_calculus_trim (49).pdf",
        Path("/Users/rick/Downloads/cauchy_transport_calculus_trim (49).pdf"),
        "ellipse/de Moivre correction source audit",
    ),
    (
        "cauchy_transport_calculus_trim (73).pdf",
        Path("/Users/rick/Downloads/cauchy_transport_calculus_trim (73).pdf"),
        "latest complex pullback and de Moivre source audit",
    ),
    (
        "golden_branch_free_tetration_outline.pdf",
        Path("/Users/rick/Library/Mobile Documents/com~apple~CloudDocs/golden_branch_free_tetration_outline.pdf"),
        "golden ellipse/narrative source audit; not a benchmark source",
    ),
    (
        "dtn_bgk_merge (9).pdf",
        Path("/Users/rick/Downloads/dtn_bgk_merge (9).pdf"),
        "DtN/BGK/zeta endpoint correction source",
    ),
    (
        "harmonic_sp (41).pdf",
        Path("/Users/rick/Downloads/harmonic_sp (41).pdf"),
        "harmonic sampling and DtN spectrum source",
    ),
    (
        "geometry_of_money (16).pdf",
        Path("/Users/rick/Library/Mobile Documents/com~apple~CloudDocs/geometry_of_money (16).pdf"),
        "BGK continuity correction and chord-arc bookkeeping source",
    ),
)


def load_python(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def fit_power(x_values: list[float], y_values: list[float]) -> float:
    xs = [math.log(x) for x in x_values]
    ys = [math.log(max(y, 1.0e-300)) for y in y_values]
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    denom = sum((x - mx) ** 2 for x in xs)
    if denom == 0.0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / denom


def fmt(value: float, digits: int = 3) -> str:
    if value == 0.0:
        return "0"
    if abs(value) >= 1.0e4 or abs(value) < 1.0e-3:
        return f"{value:.{digits}e}"
    return f"{value:.{digits}f}"


def tex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def make_booktabs_table(headers: list[str], rows: list[list[object]], align: str | None = None) -> str:
    spec = align or ("l" + "r" * (len(headers) - 1))
    lines = [rf"\begin{{tabular}}{{{spec}}}", r"\toprule"]
    lines.append(" & ".join(tex_escape(header) for header in headers) + r" \\")
    lines.append(r"\midrule")
    for row in rows:
        lines.append(" & ".join(tex_escape(item) for item in row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def fit_table(table: str) -> str:
    return "\n".join((r"\par\noindent\resizebox{\linewidth}{!}{%", table, r"}\par"))


def latex_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def copy_figures() -> dict[str, str]:
    FIG.mkdir(parents=True, exist_ok=True)
    copies = {
        "exterior_split": ROOT / "outputs" / "exterior_kernel_split_qrule" / "exterior_kernel_split_qrule_benchmark.png",
        "bgk_pipeline": ROOT / "outputs" / "exterior_q_bgk_pipeline" / "exterior_q_bgk_pipeline.png",
        "funky_pde": ROOT / "outputs" / "funky_shape_q_bgk_pde_suite" / "funky_shape_q_bgk_pde_suite.png",
        "polygon_continuity": ROOT / "outputs" / "polygon_smooth_arc_continuity" / "polygon_smooth_arc_continuity_benchmark.png",
        "qbx_failures": ROOT / "docs" / "assets" / "qbx_failure_examples.png",
        "gww": ROOT / "outputs" / "gww_isospectral_section11" / "section11_gww_q_convergence.png",
        "spectrum_gallery": ROOT / "outputs" / "discriminant_curvature_shape_gallery" / "q_spectrum_shape_gallery.png",
        "bgk_bookkeeping": ROOT / "outputs" / "bgk_endpoint_bookkeeping" / "bgk_endpoint_bookkeeping.png",
    }
    out: dict[str, str] = {}
    for key, source in copies.items():
        if not source.exists():
            continue
        target = FIG / f"{key}{source.suffix}"
        shutil.copy2(source, target)
        out[key] = f"figures/{target.name}"
    return out


def file_sha256_prefix(path: Path, length: int = 12) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:length]


def pdf_page_count(path: Path) -> int | None:
    pdfinfo = shutil.which("pdfinfo")
    if pdfinfo is None or not path.exists():
        return None
    result = subprocess.run([pdfinfo, str(path)], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        if line.startswith("Pages:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except ValueError:
                return None
    return None


def extract_pdf_text(path: Path, target: Path) -> str:
    if target.exists():
        return target.read_text(encoding="utf-8", errors="replace")
    pdftotext = shutil.which("pdftotext")
    if pdftotext is None or not path.exists():
        return ""
    target.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run([pdftotext, str(path), str(target)], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0 or not target.exists():
        return ""
    return target.read_text(encoding="utf-8", errors="replace")


def keyword_hits(text: str) -> str:
    lowered = text.lower()
    groups = []
    for group, terms in KEYWORD_GROUPS.items():
        count = sum(1 for term in terms if term.lower() in lowered)
        if count:
            groups.append(f"{group}:{count}")
    return ", ".join(groups) if groups else "no configured keyword hits"


def build_referenced_pdf_audit() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    PDF_TEXT.mkdir(parents=True, exist_ok=True)
    for index, (name, path, role) in enumerate(REFERENCED_PDFS, start=1):
        exists = path.exists()
        text_target = PDF_TEXT / f"{index:02d}_{name.replace('/', '_').replace(' ', '_')}.txt"
        text = extract_pdf_text(path, text_target) if exists else ""
        rows.append(
            {
                "source": name,
                "exists": exists,
                "pages": pdf_page_count(path) if exists else "",
                "sha256_prefix": file_sha256_prefix(path) if exists else "",
                "extracted_words": len(text.split()) if text else 0,
                "keyword_hits": keyword_hits(text),
                "role_in_report": role,
                "path": str(path),
            }
        )
    write_csv(OUT / "referenced_pdf_audit.csv", rows)
    return rows


def run_scaling_sweep() -> tuple[list[dict[str, object]], list[dict[str, object]], Path]:
    funky = load_python(ROOT / "examples" / "pullback" / "funky_shape_q_bgk_pde_suite.py", "funky_shape_suite_for_report")
    rows: list[dict[str, object]] = []
    reference_n = 32768
    true_offset = 1.0e-3
    monitor_h = 2.0**-26
    shift = funky.BGK_BETA * math.sqrt(monitor_h)
    shapes = funky.shape_suite()
    references = {
        shape.name: funky.q_split_derivatives(shape, 1.0 + true_offset, reference_n, 0)[0]
        for shape in shapes
    }

    for shape in shapes:
        boundary_map = funky.qjet_map(shape)
        for n in SIZES:
            q_times: list[float] = []
            corrected = 0.0
            for _ in range(3):
                start = perf_counter()
                derivatives = funky.q_split_derivatives(shape, 1.0 + true_offset + shift, n, 4)
                corrected = funky.taylor_repay(derivatives, shift, 4)
                q_times.append(1000.0 * (perf_counter() - start))
            reference = references[shape.name]
            q_rel_error = funky.relative_error(corrected, reference)

            start = perf_counter()
            boundary = funky.build_boundary_pullback_qjet(n, boundary_map)
            build_ms = 1000.0 * (perf_counter() - start)
            values = funky.cosine_mode(n, 7)
            batch_ms = 0.0
            max_generated_error = 0.0
            for problem, params in PDE_PROBLEMS.items():
                start = perf_counter()
                result = boundary.solve_boundary_problem(problem, values, **params)
                solve_ms = 1000.0 * (perf_counter() - start)
                batch_ms += solve_ms
                output = [complex(value) for value in result.values]
                if problem == "laplace_dtn":
                    generated = [
                        funky.cycle_dtn_eigenvalue(7, n) * math.cos(funky.TAU * 7 * index / n) / boundary.speeds[index]
                        for index in range(n)
                    ]
                    generated_error = funky.relative_l2(output, [complex(value) for value in generated])
                else:
                    amplitude = funky.projected_cos_amplitude(output, 7)
                    generated = funky.generated_q_amplitude(problem, 7, n, params)
                    generated_error = funky.relative_error(amplitude, generated)
                max_generated_error = max(max_generated_error, generated_error)
                rows.append(
                    {
                        "shape": shape.name,
                        "family": shape.family,
                        "n": n,
                        "operation": f"pde_{problem}",
                        "ms": solve_ms,
                        "build_ms": build_ms,
                        "work_units_reported": result.work_units,
                        "fft_work_units_estimate": int(n * math.log2(n)),
                        "generated_q_residual": generated_error,
                        "dense_q_matrix_stored": False,
                    }
                )

            rows.append(
                {
                    "shape": shape.name,
                    "family": shape.family,
                    "n": n,
                    "operation": "q_bgk4",
                    "ms": median(q_times),
                    "build_ms": 0.0,
                    "work_units_reported": int(n * math.log2(n)),
                    "fft_work_units_estimate": int(n * math.log2(n)),
                    "generated_q_residual": q_rel_error,
                    "dense_q_matrix_stored": False,
                }
            )
            rows.append(
                {
                    "shape": shape.name,
                    "family": shape.family,
                    "n": n,
                    "operation": "pde_batch_all5",
                    "ms": batch_ms,
                    "build_ms": build_ms,
                    "work_units_reported": int(n * len(PDE_PROBLEMS)),
                    "fft_work_units_estimate": int(len(PDE_PROBLEMS) * n * math.log2(n)),
                    "generated_q_residual": max_generated_error,
                    "dense_q_matrix_stored": False,
                }
            )

    summary: list[dict[str, object]] = []
    for operation in ("q_bgk4", "pde_laplace_dtn", "pde_heat", "pde_helmholtz", "pde_batch_all5"):
        medians_by_n = []
        residuals_by_n = []
        for n in SIZES:
            subset = [row for row in rows if row["operation"] == operation and int(row["n"]) == n]
            medians_by_n.append((n, median([float(row["ms"]) for row in subset])))
            residuals_by_n.append((n, max(float(row["generated_q_residual"]) for row in subset)))
        summary.append(
            {
                "operation": operation,
                "fit_time_power": fit_power([x for x, _ in medians_by_n], [y for _, y in medians_by_n]),
                "median_ms_n256": medians_by_n[0][1],
                "median_ms_n512": medians_by_n[1][1],
                "median_ms_n1024": medians_by_n[2][1],
                "median_ms_n2048": medians_by_n[3][1],
                "median_ms_n4096": medians_by_n[4][1],
                "max_generated_q_residual": max(value for _, value in residuals_by_n),
                "dense_q_matrix_stored": False,
            }
        )

    write_csv(OUT / "matrix_free_scaling_rows.csv", rows)
    write_csv(OUT / "matrix_free_scaling_summary.csv", summary)
    figure = FIG / "matrix_free_scaling.png"
    plot_scaling(summary, figure)
    return rows, summary, figure


def plot_scaling(summary: list[dict[str, object]], path: Path) -> None:
    fig, axis = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
    styles = {
        "q_bgk4": ("0.0", "o"),
        "pde_laplace_dtn": ("0.25", "s"),
        "pde_heat": ("0.45", "^"),
        "pde_helmholtz": ("0.60", "D"),
        "pde_batch_all5": ("0.10", "x"),
    }
    for row in summary:
        operation = str(row["operation"])
        color, marker = styles.get(operation, ("0.5", "o"))
        values = [
            float(row["median_ms_n256"]),
            float(row["median_ms_n512"]),
            float(row["median_ms_n1024"]),
            float(row["median_ms_n2048"]),
            float(row["median_ms_n4096"]),
        ]
        label = f"{operation.replace('_', ' ')} (p={float(row['fit_time_power']):.2f})"
        axis.loglog(SIZES, values, color=color, marker=marker, linewidth=1.1, label=label)
    axis.set_xlabel("boundary samples n")
    axis.set_ylabel("median wall time (ms)")
    axis.grid(True, which="both", color="0.88", linewidth=0.5)
    axis.legend(frameon=False, fontsize=7)
    axis.set_title("Matrix-free generated QJet scaling")
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_cycle_spectrum() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for n in SIZES:
        for mode in SPECTRUM_MODES:
            if mode >= n:
                continue
            q_lambda = mode * (n - mode) / 2.0
            dtn_mu = mode * (n - mode) / n
            rows.append(
                {
                    "n": n,
                    "mode": mode,
                    "q_eigenvalue_lambda_m": q_lambda,
                    "normalized_dtn_mu_m": dtn_mu,
                    "continuum_limit": mode,
                    "relative_dtn_symbol_error": 0.0 if mode == 0 else abs(dtn_mu - mode) / mode,
                }
            )
    write_csv(OUT / "exact_cycle_q_spectrum.csv", rows)
    return rows


def exterior_split_summary() -> list[list[object]]:
    rows = read_csv(ROOT / "outputs" / "exterior_kernel_split_qrule" / "exterior_kernel_split_qrule_benchmark.csv")
    selected: list[list[object]] = []
    shapes = sorted({row["shape"] for row in rows})
    max_n = max(int(row["n"]) for row in rows)
    for shape in shapes:
        subset = [row for row in rows if row["shape"] == shape and int(row["n"]) == max_n]
        if not subset:
            continue
        worst_direct = max(float(row["direct_rel_error"]) for row in subset)
        worst_split = max(float(row["split_rel_error"]) for row in subset)
        smallest_offset = min(float(row["conformal_offset"]) for row in subset)
        selected.append([shape, f"n={max_n}, offset={fmt(smallest_offset)}", fmt(worst_direct), fmt(worst_split)])
    return selected


def funky_summary_table() -> list[list[object]]:
    bgk = read_csv(ROOT / "outputs" / "funky_shape_q_bgk_pde_suite" / "funky_shape_bgk_ladder_summary.csv")
    diagnostics = {
        row["shape"]: row
        for row in read_csv(ROOT / "outputs" / "funky_shape_q_bgk_pde_suite" / "funky_shape_diagnostics.csv")
    }
    pde = read_csv(ROOT / "outputs" / "funky_shape_q_bgk_pde_suite" / "funky_shape_pde_rows.csv")
    out: list[list[object]] = []
    for row in bgk:
        shape = row["shape"]
        subset = [p for p in pde if p["shape"] == shape]
        out.append(
            [
                shape,
                row["family"],
                fmt(float(diagnostics[shape]["anisotropy"])),
                fmt(float(row["bgk4_rel_error"])),
                fmt(max(float(p["continuum_rel_error"]) for p in subset)),
                fmt(max(float(p["generated_rel_error"]) for p in subset)),
            ]
        )
    return out


def pde_problem_table() -> list[list[object]]:
    rows = read_csv(ROOT / "outputs" / "funky_shape_q_bgk_pde_suite" / "funky_shape_pde_rows.csv")
    out: list[list[object]] = []
    for problem in PDE_PROBLEMS:
        subset = [row for row in rows if row["problem"] == problem]
        out.append(
            [
                problem,
                fmt(max(float(row["continuum_rel_error"]) for row in subset)),
                fmt(max(float(row["generated_rel_error"]) for row in subset)),
                fmt(median([float(row["solve_ms"]) for row in subset])),
                str(int(median([float(row["work_units"]) for row in subset]))),
            ]
        )
    return out


def structural_method_table() -> list[list[object]]:
    data = json.loads((ROOT / "docs" / "assets" / "structural_quadrature_methods_benchmark.json").read_text(encoding="utf-8"))
    out: list[list[object]] = []
    for method, stats in data["summary"]["methods"].items():
        out.append(
            [
                method,
                fmt(float(stats["median_relative_error"])),
                fmt(float(stats["median_ms"])),
                fmt(float(stats["median_work_units"])),
                int(stats["failure_count"]),
                fmt(float(stats["median_improvement_vs_trap"])),
            ]
        )
    return out


def qbx_cusp_table() -> list[list[object]]:
    data = json.loads((ROOT / "docs" / "assets" / "qbx_gulati_cusp_benchmark.json").read_text(encoding="utf-8"))
    summary = data["summary"]
    return [
        ["QBX failure rows", summary["failure_count"]],
        ["QBX median ms", fmt(float(summary["median_qbx_refined_ms"]))],
        ["QBX median work", fmt(float(summary["median_qbx_refined_work_units"]))],
        ["multipole/zeta Q median ms", fmt(float(summary["median_multipole_zeta_q_ms"]))],
        ["multipole/zeta Q single-target work", fmt(float(summary["median_multipole_zeta_q_single_target_work_units"]))],
        ["tip multipole/zeta improvement", fmt(float(summary["tip_median_multipole_zeta_q_improvement"]))],
        ["near-tip multipole/zeta improvement", fmt(float(summary["near_tip_median_multipole_zeta_q_improvement"]))],
    ]


def helmholtz_caveat_table() -> list[list[object]]:
    disk = json.loads((ROOT / "docs" / "assets" / "q_dtn_helmholtz_ground_truth.json").read_text(encoding="utf-8"))["summary"]
    resonance = json.loads((ROOT / "docs" / "assets" / "q_dtn_hard_helmholtz_resonance.json").read_text(encoding="utf-8"))["summary"]
    return [
        ["disk modal exact", f"{disk['disk_q_spectral_win_count']} Q wins / {disk['disk_fem_win_count']} FEM wins"],
        ["manufactured exact", f"{disk['manufactured_q_helmholtz_win_count']} Q wins / {disk['manufactured_fem_win_count']} FEM wins"],
        ["hard resonance sweep", f"{resonance['q65536_q_win_count']} Q wins / {resonance['q65536_fem_win_count']} FEM wins"],
        ["median Q manufactured err", fmt(float(disk["manufactured_median_q_relative_l2_to_exact"]))],
        ["median FEM manufactured err", fmt(float(disk["manufactured_median_fem_relative_l2_to_exact"]))],
        ["max Q/FEM error ratio in FEM wins", fmt(float(resonance["q65536_max_fem_win_ratio"]))],
    ]


def gww_convergence_table() -> list[list[object]]:
    rows = read_csv(ROOT / "outputs" / "gww_isospectral_section11" / "section11_gww_q_convergence.csv")
    out: list[list[object]] = []
    for row in rows:
        left = ", ".join(fmt(float(row[f"left_ritz_{idx}"]), 6) for idx in range(1, 7))
        right = ", ".join(fmt(float(row[f"right_ritz_{idx}"]), 6) for idx in range(1, 7))
        out.append([row["n"], fmt(float(row["relative_split_first6"]), 6), left, right])
    return out


def q_spectrum_gallery_table() -> list[list[object]]:
    rows = read_csv(ROOT / "outputs" / "discriminant_curvature_shape_gallery" / "q_spectrum_shape_gallery_summary.csv")
    out: list[list[object]] = []
    for row in rows:
        out.append(
            [
                row["shape"],
                row["family"],
                fmt(float(row["lambda_1"])),
                fmt(float(row["lambda_2"])),
                fmt(float(row["lambda_4"])),
                fmt(float(row["largest_ritz"])),
                fmt(float(row["condition_proxy"])),
            ]
        )
    return out


def scaling_summary_table(summary: list[dict[str, object]]) -> list[list[object]]:
    return [
        [
            row["operation"],
            fmt(float(row["fit_time_power"])),
            fmt(float(row["median_ms_n256"])),
            fmt(float(row["median_ms_n1024"])),
            fmt(float(row["median_ms_n4096"])),
            fmt(float(row["max_generated_q_residual"])),
        ]
        for row in summary
    ]


def cycle_spectrum_table(rows: list[dict[str, object]]) -> list[list[object]]:
    selected = [row for row in rows if int(row["n"]) in (256, 2048, 4096) and int(row["mode"]) in (0, 1, 2, 7, 16, 32)]
    return [
        [
            row["n"],
            row["mode"],
            fmt(float(row["q_eigenvalue_lambda_m"]), 6),
            fmt(float(row["normalized_dtn_mu_m"]), 9),
            fmt(float(row["relative_dtn_symbol_error"])),
        ]
        for row in selected
    ]


def source_audit_table() -> list[list[object]]:
    return [
        [
            "optimal_quadrature_v63_public (11).pdf",
            "kernel split / Q rule",
            "Analytic-boundary pullback to the circle; circle term by Fourier Q primitive; corners only algebraic.",
        ],
        [
            "dtn_bgk_merge (9).pdf",
            "DtN + BGK",
            "Chord Hessian is a matrix-free DtN discretization; half-integer zeta endpoint is the BGK ladder.",
        ],
        [
            "geometry_of_money (16).pdf",
            "BGK constant",
            "The continuity correction constant is beta equals negative zeta(1/2)/sqrt(2 pi).",
        ],
        [
            "cauchy_transport_calculus_trim (73).pdf",
            "complex pullback",
            "Use z = exp(rho + i theta); de Moivre modes diagonalize the circle operator and repay metric factors.",
        ],
        [
            "harmonic_sp (41).pdf",
            "harmonic sampling",
            "Cycle spectrum and endpoint defects are used as the harmonic/DtN bookkeeping channel.",
        ],
    ]


def referenced_pdf_table(pdf_rows: list[dict[str, object]]) -> list[list[object]]:
    def display_name(name: object) -> str:
        text = str(name)
        return (
            text.replace("optimal_quadrature_merged", "optimal_quad_merged")
            .replace("optimal_quadrature_v63_public", "optimal_q_v63_public")
            .replace("cauchy_transport_calculus_trim", "cauchy_transport_trim")
            .replace("golden_branch_free_tetration_outline.pdf", "golden_branch_outline.pdf")
        )

    table: list[list[object]] = []
    for row in pdf_rows:
        table.append(
            [
                display_name(row["source"]),
                row["pages"] if row["pages"] != "" else "missing",
                row["sha256_prefix"],
                row["keyword_hits"],
                row["role_in_report"],
            ]
        )
    return table


def write_latex(
    figures: dict[str, str],
    scaling_summary: list[dict[str, object]],
    cycle_rows: list[dict[str, object]],
    pdf_audit_rows: list[dict[str, object]],
) -> Path:
    tex = OUT / "machine_precision_pipeline_report.tex"
    lines: list[str] = []
    lines.extend(
        [
            r"\documentclass[10pt]{article}",
            r"\usepackage[margin=0.7in]{geometry}",
            r"\usepackage{amsmath,amssymb}",
            r"\usepackage{booktabs}",
            r"\usepackage{array}",
            r"\usepackage{graphicx}",
            r"\usepackage{float}",
            r"\usepackage{longtable}",
            r"\usepackage{xcolor}",
            r"\usepackage{hyperref}",
            r"\usepackage{caption}",
            r"\captionsetup{font=small,labelfont=bf}",
            r"\hypersetup{colorlinks=true,linkcolor=black,urlcolor=black,citecolor=black}",
            r"\setlength{\parindent}{0pt}",
            r"\setlength{\parskip}{5pt}",
            r"\begin{document}",
            r"\title{Machine-Precision Exterior Q, BGK Repayment, and Boundary PDE Benchmarks}",
            r"\author{Generated benchmark report}",
            r"\date{June 2026}",
            r"\maketitle",
            r"\begin{abstract}",
            "This report consolidates the new exterior-map Q split, the BGK/zeta endpoint repayment ladder, the matrix-free boundary PDE pipeline, polygon continuity corrections, QBX comparisons, and Q-spectral diagnostics.  The implementation stores generating QJets, Laurent coefficients, boundary samples, and low-rank correction modes.  It does not store a dense Q matrix.  The exact analytic pullback cases reach the fp64 floor after BGK repayment; the PDE pipeline reaches machine residual against the generated Q spectrum while retaining a visible continuum discretization error against the limiting DtN operator.  Head-to-head method tables and hard Helmholtz caveats are kept separate so the claim boundary is auditable.",
            r"\end{abstract}",
            r"\tableofcontents",
            r"\newpage",
            r"\section{Claim Boundary}",
            "There are three different statements in the data.  First, for analytic finite-Laurent exterior maps, the kernel split plus BGK Taylor repayment reaches the floating-point floor.  Second, for boundary PDEs, the implementation residual against the generated Q spectrum is at machine precision, while the continuum error is the finite-n spectral error of the generated DtN symbol.  Third, polygon and cusp cases require extra repayment channels; the polygon continuity benchmark is an algebraic local replacement ledger, not proof that unsmoothed corner conformal maps become analytic.",
            "",
            "The strongest defensible formulation is: the current Q engine is a matrix-free generated-operator pipeline whose exact analytic pullback path is fp64 accurate, whose PDE path is load-bearing for the generated DtN correspondence, and whose spectrum identifies the active error channel.  FEM is a useful engineering baseline, but it is not automatically the ground truth on hard boundaries or near resonant Helmholtz cases.",
            r"\section{Source Audit}",
            fit_table(
                make_booktabs_table(
                    ["source", "role", "used here"],
                    source_audit_table(),
                    align=r"p{0.26\linewidth}p{0.17\linewidth}p{0.47\linewidth}",
                )
            ),
            r"\par\medskip",
            "The table below is the mechanical audit for every user-referenced local PDF found in this thread.  Each file was inspected with \\texttt{pdfinfo} and extracted with \\texttt{pdftotext}; the keyword hits are only a routing check, not a citation substitute.",
            r"\begingroup\small",
            fit_table(
                make_booktabs_table(
                    ["referenced PDF", "pages", "sha", "keyword hits", "role/boundary"],
                    referenced_pdf_table(pdf_audit_rows),
                    align=r"p{0.25\linewidth}rp{0.10\linewidth}p{0.22\linewidth}p{0.27\linewidth}",
                )
            ),
            r"\endgroup",
            r"\section{Implementation Protocol}",
            r"\begin{align*}",
            r"z &= \psi(w), \qquad w = \rho e^{i\theta} = \exp(\log \rho + i\theta),\\",
            r"\log|\psi(w)-\psi(z_j)| &= \log|w-z_j| + \log|G(w,z_j)|,\\",
            r"I_p &= \sum_{r=0}^{p}\frac{(-\beta\sqrt{h})^r}{r!}\,\partial_\rho^r I(\rho+\beta\sqrt{h}),",
            r"\qquad \beta=-\zeta(1/2)/\sqrt{2\pi}.",
            r"\end{align*}",
            r"The borrow-compute-repay sequence is: borrow the circle coordinate, compute the singular log term using de Moivre Fourier characters and the generated cycle Q spectrum, repay the analytic quotient term, then repay endpoint displacement through the BGK/zeta Taylor ladder.  For Laplace normal flux the metric repayment is multiplication by $|d\psi/d\theta|^{-1}$.",
            r"\section{Exact Cycle Spectrum}",
            "For the regular cycle QJet, the stored generator is the spectrum, not a matrix:",
            r"\[",
            r"\lambda_m(Q_n)=\frac{m(n-m)}{2},\qquad \mu_m=(h/\pi)\lambda_m=\frac{m(n-m)}{n},\qquad h=2\pi/n.",
            r"\]",
            make_booktabs_table(["n", "mode", "Q eigenvalue", "normalized DtN", "symbol rel. err"], cycle_spectrum_table(cycle_rows)),
        ]
    )

    if "exterior_split" in figures:
        lines.extend(
            [
                r"\section{Exterior Kernel Split}",
                "The direct trapezoidal rule sees the near singularity.  The split Q path removes the singular circle term exactly and integrates only the analytic quotient remainder.",
                fit_table(make_booktabs_table(["shape", "case", "worst direct err", "worst split err"], exterior_split_summary())),
                r"\begin{figure}[H]\centering",
                rf"\includegraphics[width=0.95\linewidth]{{{figures['exterior_split']}}}",
                r"\caption{Exterior kernel split benchmark.  Black/gray styling is used throughout for paper-style reproduction.}",
                r"\end{figure}",
            ]
        )

    if "bgk_pipeline" in figures:
        bgk_rows = read_csv(ROOT / "outputs" / "exterior_q_bgk_pipeline" / "exterior_q_bgk_pipeline_summary.csv")
        bgk_table = [
            [
                row["shape"],
                fmt(float(row["raw_error_power_in_h"])),
                fmt(float(row["order1_power_in_h"])),
                fmt(float(row["order2_power_in_h"])),
                fmt(float(row["finest_raw_rel_error"])),
                fmt(float(row["finest_order4_rel_error"])),
            ]
            for row in bgk_rows
        ]
        lines.extend(
            [
                r"\section{BGK/Zeta Repayment}",
                "The BGK layer is not cosmetic.  It removes the discrete endpoint displacement with the same half-integer zeta defect that appears in the DtN spectral endpoint ledger.",
                fit_table(make_booktabs_table(["shape", "raw power", "BGK-1 power", "BGK-2 power", "raw err", "BGK-4 err"], bgk_table)),
                r"\begin{figure}[H]\centering",
                rf"\includegraphics[width=0.95\linewidth]{{{figures['bgk_pipeline']}}}",
                r"\caption{Raw endpoint error has the square-root law; BGK-1 and BGK-2 shift the slope; BGK-4 reaches fp64 floor on exact Laurent maps.}",
                r"\end{figure}",
            ]
        )

    if "funky_pde" in figures:
        lines.extend(
            [
                r"\section{Funky Shapes and PDE Pipeline}",
                "The same exact-Laurent pullback suite is run through quadrature and boundary-only PDE operators: Laplace DtN, heat, Poisson/Steklov, Helmholtz resolvent, and wave.  Generated-Q residual is the implementation check; continuum error is the finite-n approximation to the limiting operator.",
                fit_table(
                    make_booktabs_table(
                        ["shape", "family", "anisotropy", "BGK-4 err", "max PDE continuum err", "max generated-Q residual"],
                        funky_summary_table(),
                        align="llrrrr",
                    )
                ),
                "",
                fit_table(make_booktabs_table(["problem", "max continuum err", "max generated-Q residual", "median solve ms", "median work"], pde_problem_table())),
                r"\begin{figure}[H]\centering",
                rf"\includegraphics[width=0.95\linewidth]{{{figures['funky_pde']}}}",
                r"\caption{Ten structurally different exact-Laurent domains.  The PDE bar chart shows continuum finite-n error; generated-Q residual is reported in the tables.}",
                r"\end{figure}",
            ]
        )

    lines.extend(
        [
            r"\section{Measured Matrix-Free Scaling}",
            r"This fresh sweep uses the same generated QJet path on all ten shapes for $n=256,\ldots,4096$.  The fitted exponent is measured from wall time.  The expected model is near $O(n\log n)$ for the FFT-generated path, not dense $O(n^2)$ storage.",
            fit_table(make_booktabs_table(["operation", "fit power", "ms n=256", "ms n=1024", "ms n=4096", "max residual"], scaling_summary_table(scaling_summary))),
            r"\begin{figure}[H]\centering",
            r"\includegraphics[width=0.82\linewidth]{figures/matrix_free_scaling.png}",
            r"\caption{Measured matrix-free scaling.  No dense Q matrix is stored in this sweep.}",
            r"\end{figure}",
        ]
    )

    if "polygon_continuity" in figures:
        polygon_rows = read_csv(ROOT / "outputs" / "polygon_smooth_arc_continuity" / "polygon_smooth_arc_continuity_benchmark.csv")
        table_rows = []
        best: dict[tuple[str, str], dict[str, str]] = {}
        for row in polygon_rows:
            key = (row["shape"], row["target_kind"])
            if key not in best or float(row["corrected_rel_error"]) < float(best[key]["corrected_rel_error"]):
                best[key] = row
        for _, row in sorted(best.items()):
            table_rows.append(
                [
                    row["shape"],
                    row["target_kind"],
                    row["panels"],
                    fmt(float(row["smooth_rel_error"])),
                    fmt(float(row["corrected_rel_error"])),
                ]
            )
        lines.extend(
            [
                r"\section{Polygon Smooth-Arc Continuity Ledger}",
                "Polygonal inputs are represented as smooth local arcs for the compute step and then corrected by replacing the arc contribution with exact polygon edge stubs.  The machine-precision entries are therefore a local continuity repayment result, not a free analytic conformal map through corners.",
                fit_table(make_booktabs_table(["shape", "target", "panels", "smooth arc err", "corrected err"], table_rows)),
                r"\begin{figure}[H]\centering",
                rf"\includegraphics[width=0.90\linewidth]{{{figures['polygon_continuity']}}}",
                r"\caption{Polygon continuity correction as a corner repayment layer.}",
                r"\end{figure}",
            ]
        )

    lines.extend(
        [
            r"\section{Head-to-Head Against QBX and Other Methods}",
            "The structural method benchmark still shows multipole/zeta Q as the preferred robust quadrature path in this corpus.  QBX is strong where its expansion geometry is valid, but it has explicit source-free disk failures at cusp tips.",
            fit_table(make_booktabs_table(["method", "median err", "median ms", "work", "failures", "improvement"], structural_method_table())),
            "",
            make_booktabs_table(["quantity", "value"], qbx_cusp_table(), align="lr"),
        ]
    )
    if "qbx_failures" in figures:
        lines.extend(
            [
                r"\begin{figure}[H]\centering",
                rf"\includegraphics[width=0.85\linewidth]{{{figures['qbx_failures']}}}",
                r"\caption{QBX expansion-disk failures at one-cusp and two-cusp geometries.}",
                r"\end{figure}",
            ]
        )

    lines.extend(
        [
            r"\section{Where FEM Still Beats Q}",
            "The Helmholtz data should be positioned carefully.  Against manufactured exact boundary data, Q wins almost everywhere.  Near resonant volumetric Helmholtz comparisons, FEM can still win in a nontrivial subset.  That is not contradictory: the Q path is a boundary spectral model, while the FEM Schur comparison includes a volumetric discretization and resonance conditioning.",
            make_booktabs_table(["suite", "result"], helmholtz_caveat_table(), align="ll"),
        ]
    )

    if "gww" in figures:
        lines.extend(
            [
                r"\section{GWW Isospectral Pair: Exact Q Ritz Values}",
                "The continuum Dirichlet spectra of the GWW pair are classically equal.  The experiment below is deliberately narrower: at the discretized chord-operator level, the projected Q spectra separate robustly under refinement.",
                r"\begin{verbatim}",
                "left raw clockwise:  [(-3,-3),(-3,-1),(1,3),(1,1),(3,1),(1,-1),(-1,-1),(-1,-3)]",
                "right raw clockwise: [(-3,1),(1,1),(1,3),(3,1),(1,-1),(-1,-1),(-1,-3),(-3,-1)]",
                "orientation: raw lists are reversed to counterclockwise, then centered and scaled to unit area.",
                "nodes: equal arclength s_k=((k+alpha) mod n)L/n with alpha=1/2 midpoint sampling.",
                "corners: exact polygon vertices are kept; midpoint nodes avoid duplicate corner nodes.",
                "normalization: (Lambda_Q,n f)_i=(L/(pi n)) sum_{j!=i}(f_i-f_j)/|x_i-x_j|^2.",
                r"\end{verbatim}",
                r"\begingroup\small",
                make_booktabs_table(
                    ["n", "relative split", "left Ritz 1-6", "right Ritz 1-6"],
                    gww_convergence_table(),
                    align=r"rp{0.12\linewidth}p{0.34\linewidth}p{0.34\linewidth}",
                ),
                r"\endgroup",
                r"\begin{figure}[H]\centering",
                rf"\includegraphics[width=0.92\linewidth]{{{figures['gww']}}}",
                r"\caption{GWW projected Q-spectral separation and refinement.}",
                r"\end{figure}",
            ]
        )

    if "spectrum_gallery" in figures:
        lines.extend(
            [
                r"\section{Q Spectrum and Error Type}",
                "The Q spectrum is reported as a diagnostic, not just a convergence plot.  High condition proxies and split low modes correlate with corner/cusp channels where multipole/zeta or Mellin corrections are needed.",
                r"\begingroup\small",
                fit_table(
                    make_booktabs_table(
                        ["shape", "family", "lambda1", "lambda2", "lambda4", "largest", "cond proxy"],
                        q_spectrum_gallery_table(),
                        align="llrrrrr",
                    )
                ),
                r"\endgroup",
                r"\begin{figure}[H]\centering",
                rf"\includegraphics[width=0.95\linewidth]{{{figures['spectrum_gallery']}}}",
                r"\caption{Exact computed Q Ritz spectrum gallery.}",
                r"\end{figure}",
            ]
        )

    if "bgk_bookkeeping" in figures:
        lines.extend(
            [
                r"\section{BGK Endpoint Bookkeeping}",
                "The endpoint ledger identifies the same constant in Monte Carlo barrier crossing, DtN spectral endpoint sums, and the chord-to-arc correction channel.  This is the conceptual reason the BGK layer belongs in the Q pipeline.",
                r"\begin{figure}[H]\centering",
                rf"\includegraphics[width=0.90\linewidth]{{{figures['bgk_bookkeeping']}}}",
                r"\caption{BGK endpoint bookkeeping: Spitzer, zeta, and Q/DtN endpoints.}",
                r"\end{figure}",
            ]
        )

    lines.extend(
        [
            r"\section{Reproducibility}",
            r"\begin{verbatim}",
            "python3 examples/pullback/exterior_kernel_split_qrule_benchmark.py",
            "python3 examples/pullback/exterior_q_bgk_correction_pipeline.py",
            "python3 examples/pullback/funky_shape_q_bgk_pde_suite.py",
            "python3 examples/corners/polygon_smooth_arc_continuity_benchmark.py",
            "python3 examples/isospectral/gww_section11_audit.py",
            "python3 examples/q_spectrum_shape_gallery.py",
            "python3 scripts/build_machine_precision_pipeline_report.py",
            r"\end{verbatim}",
            "Generated artifacts live under \\texttt{outputs/machine\\_precision\\_pipeline\\_report}.",
            r"\end{document}",
        ]
    )
    tex.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tex


def compile_latex(tex: Path) -> Path | None:
    pdflatex = shutil.which("pdflatex")
    if pdflatex is None:
        return None
    for _ in range(2):
        subprocess.run(
            [pdflatex, "-interaction=nonstopmode", "-halt-on-error", tex.name],
            cwd=tex.parent,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    pdf = tex.with_suffix(".pdf")
    return pdf if pdf.exists() else None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    figures = copy_figures()
    pdf_audit_rows = build_referenced_pdf_audit()
    scaling_rows, scaling_summary, scaling_figure = run_scaling_sweep()
    figures["scaling"] = f"figures/{scaling_figure.name}"
    cycle_rows = write_cycle_spectrum()
    tex = write_latex(figures, scaling_summary, cycle_rows, pdf_audit_rows)
    pdf = compile_latex(tex)
    payload = {
        "tex": str(tex),
        "pdf": None if pdf is None else str(pdf),
        "referenced_pdf_audit": str(OUT / "referenced_pdf_audit.csv"),
        "scaling_rows": str(OUT / "matrix_free_scaling_rows.csv"),
        "scaling_summary": str(OUT / "matrix_free_scaling_summary.csv"),
        "cycle_spectrum": str(OUT / "exact_cycle_q_spectrum.csv"),
        "figures": figures,
        "scaling_operation_count": len(scaling_rows),
        "dense_q_matrix_stored": False,
    }
    (OUT / "machine_precision_pipeline_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
