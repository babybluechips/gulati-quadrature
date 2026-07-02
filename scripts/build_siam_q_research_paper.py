#!/usr/bin/env python3
"""Build a SIAM-style research paper for the Q/BGK pipeline."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "siam_q_research_paper"
FIG = OUT / "figures"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def fmt(value: float, digits: int = 3) -> str:
    if value == 0:
        return "0"
    if abs(value) < 1.0e-3 or abs(value) >= 1.0e4:
        return f"{value:.{digits}e}"
    return f"{value:.{digits}f}"


def tex_escape(value: object) -> str:
    text = str(value).replace("limaçon", "limacon")
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


def table(headers: list[str], rows: list[list[object]], align: str | None = None, *, resize: bool = True) -> str:
    spec = align or ("l" + "r" * (len(headers) - 1))
    lines = [rf"\begin{{tabular}}{{{spec}}}", r"\toprule"]
    lines.append(" & ".join(tex_escape(header) for header in headers) + r" \\")
    lines.append(r"\midrule")
    for row in rows:
        lines.append(" & ".join(tex_escape(item) for item in row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    body = "\n".join(lines)
    if not resize:
        return body
    return "\n".join((r"\par\noindent\resizebox{\linewidth}{!}{%", body, r"}\par"))


def add_block(lines: list[str], block: str) -> None:
    clean = block.strip("\n")
    if clean:
        lines.extend(clean.splitlines())


def copy_figures() -> dict[str, str]:
    FIG.mkdir(parents=True, exist_ok=True)
    sources = {
        "exterior_split": ROOT / "outputs" / "machine_precision_pipeline_report" / "figures" / "exterior_split.png",
        "bgk_pipeline": ROOT / "outputs" / "machine_precision_pipeline_report" / "figures" / "bgk_pipeline.png",
        "funky_pde": ROOT / "outputs" / "machine_precision_pipeline_report" / "figures" / "funky_pde.png",
        "scaling": ROOT / "outputs" / "machine_precision_pipeline_report" / "figures" / "matrix_free_scaling.png",
        "qbx_failures": ROOT / "outputs" / "machine_precision_pipeline_report" / "figures" / "qbx_failures.png",
        "gww": ROOT / "outputs" / "machine_precision_pipeline_report" / "figures" / "gww.png",
        "spectrum": ROOT / "outputs" / "machine_precision_pipeline_report" / "figures" / "spectrum_gallery.png",
        "bgk_bookkeeping": ROOT / "outputs" / "machine_precision_pipeline_report" / "figures" / "bgk_bookkeeping.png",
    }
    out: dict[str, str] = {}
    for key, source in sources.items():
        if source.exists():
            target = FIG / f"{key}.png"
            shutil.copy2(source, target)
            out[key] = f"figures/{target.name}"
    return out


def scaling_table() -> list[list[object]]:
    rows = read_csv(ROOT / "outputs" / "machine_precision_pipeline_report" / "matrix_free_scaling_summary.csv")
    return [
        [
            row["operation"],
            fmt(float(row["fit_time_power"])),
            fmt(float(row["median_ms_n256"])),
            fmt(float(row["median_ms_n1024"])),
            fmt(float(row["median_ms_n4096"])),
            fmt(float(row["max_generated_q_residual"])),
        ]
        for row in rows
    ]


def bgk_table() -> list[list[object]]:
    rows = read_csv(ROOT / "outputs" / "exterior_q_bgk_pipeline" / "exterior_q_bgk_pipeline_summary.csv")
    return [
        [
            row["shape"],
            fmt(float(row["raw_error_power_in_h"])),
            fmt(float(row["order1_power_in_h"])),
            fmt(float(row["order2_power_in_h"])),
            fmt(float(row["finest_raw_rel_error"])),
            fmt(float(row["finest_order4_rel_error"])),
        ]
        for row in rows
    ]


def pde_problem_table() -> list[list[object]]:
    rows = read_csv(ROOT / "outputs" / "funky_shape_q_bgk_pde_suite" / "funky_shape_pde_rows.csv")
    problems = ("laplace_dtn", "heat", "poisson", "helmholtz", "wave")
    out: list[list[object]] = []
    for problem in problems:
        subset = [row for row in rows if row["problem"] == problem]
        out.append(
            [
                problem,
                fmt(max(float(row["continuum_rel_error"]) for row in subset)),
                fmt(max(float(row["generated_rel_error"]) for row in subset)),
                fmt(sum(float(row["solve_ms"]) for row in subset) / len(subset)),
                str(int(float(subset[0]["work_units"]))),
            ]
        )
    return out


def funky_shape_table() -> list[list[object]]:
    bgk = read_csv(ROOT / "outputs" / "funky_shape_q_bgk_pde_suite" / "funky_shape_bgk_ladder_summary.csv")
    diagnostics = {
        row["shape"]: row
        for row in read_csv(ROOT / "outputs" / "funky_shape_q_bgk_pde_suite" / "funky_shape_diagnostics.csv")
    }
    pde = read_csv(ROOT / "outputs" / "funky_shape_q_bgk_pde_suite" / "funky_shape_pde_rows.csv")
    out: list[list[object]] = []
    for row in bgk:
        shape = row["shape"]
        subset = [item for item in pde if item["shape"] == shape]
        out.append(
            [
                shape,
                row["family"],
                fmt(float(diagnostics[shape]["anisotropy"])),
                fmt(float(row["bgk4_rel_error"])),
                fmt(max(float(item["continuum_rel_error"]) for item in subset)),
                fmt(max(float(item["generated_rel_error"]) for item in subset)),
            ]
        )
    return out


def structural_table() -> list[list[object]]:
    data = json.loads((ROOT / "docs" / "assets" / "structural_quadrature_methods_benchmark.json").read_text())
    rows = []
    for method, stats in data["summary"]["methods"].items():
        rows.append(
            [
                method,
                fmt(float(stats["median_relative_error"])),
                fmt(float(stats["median_ms"])),
                fmt(float(stats["median_work_units"])),
                int(stats["failure_count"]),
                fmt(float(stats["median_improvement_vs_trap"])),
            ]
        )
    return rows


def helmholtz_table() -> list[list[object]]:
    data = json.loads((ROOT / "docs" / "assets" / "q_dtn_helmholtz_ground_truth.json").read_text())["summary"]
    return [
        ["disk modal exact", f"{data['disk_q_spectral_win_count']} Q wins / {data['disk_fem_win_count']} FEM wins"],
        ["manufactured exact", f"{data['manufactured_q_helmholtz_win_count']} Q wins / {data['manufactured_fem_win_count']} FEM wins"],
        ["median Q manufactured error", fmt(float(data["manufactured_median_q_relative_l2_to_exact"]))],
        ["median FEM manufactured error", fmt(float(data["manufactured_median_fem_relative_l2_to_exact"]))],
    ]


def gww_table() -> list[list[object]]:
    rows = read_csv(ROOT / "outputs" / "gww_isospectral_section11" / "section11_gww_q_convergence.csv")
    out = []
    for row in rows:
        left = ", ".join(fmt(float(row[f"left_ritz_{idx}"]), 5) for idx in range(1, 7))
        right = ", ".join(fmt(float(row[f"right_ritz_{idx}"]), 5) for idx in range(1, 7))
        out.append([row["n"], fmt(float(row["relative_split_first6"]), 5), left, right])
    return out


def cycle_spectrum_table() -> list[list[object]]:
    rows = read_csv(ROOT / "outputs" / "machine_precision_pipeline_report" / "exact_cycle_q_spectrum.csv")
    keep = []
    for row in rows:
        if int(row["n"]) in (256, 2048, 4096) and int(row["mode"]) in (1, 2, 7, 16, 32):
            keep.append(
                [
                    row["n"],
                    row["mode"],
                    fmt(float(row["q_eigenvalue_lambda_m"]), 6),
                    fmt(float(row["normalized_dtn_mu_m"]), 8),
                    fmt(float(row["relative_dtn_symbol_error"])),
                ]
            )
    return keep


def pdf_audit_table() -> list[list[object]]:
    rows = read_csv(ROOT / "outputs" / "machine_precision_pipeline_report" / "referenced_pdf_audit.csv")
    out = []
    for row in rows:
        name = (
            row["source"]
            .replace("optimal_quadrature_merged", "optimal_quad_merged")
            .replace("optimal_quadrature_v63_public", "optimal_q_v63_public")
            .replace("cauchy_transport_calculus_trim", "cauchy_transport_trim")
            .replace("golden_branch_free_tetration_outline.pdf", "golden_branch_outline.pdf")
        )
        out.append([name, row["pages"], row["sha256_prefix"], row["extracted_words"], row["role_in_report"]])
    return out


def final_pipeline_table() -> list[list[object]]:
    path = ROOT / "outputs" / "final_q_machine_precision_pipeline" / "final_q_machine_precision_summary.json"
    if not path.exists():
        return [["summary status", "missing; run scripts/final_q_machine_precision_pipeline.py first"]]
    data = json.loads(path.read_text(encoding="utf-8"))
    return [
        ["pass gate", str(bool(data["passed"])).lower()],
        ["machine tolerance", fmt(float(data["machine_tol"]))],
        ["max split rel. error", fmt(float(data["max_split_rel_error"]))],
        ["max BGK-8 rel. error", fmt(float(data["max_bgk8_rel_error"]))],
        ["max generated-Q PDE residual", fmt(float(data["max_pde_generated_q_residual"]))],
        ["shape count", data.get("shape_count", "not recorded")],
        ["PDE case count", data.get("pde_case_count", "not recorded")],
        ["dense Q matrix stored", str(bool(data["dense_q_matrix_stored"])).lower()],
        ["FFT kernel", data["fft_kernel"]],
        ["script", "scripts/final_q_machine_precision_pipeline.py"],
        ["notebook", "notebooks/q_machine_precision_pipeline.ipynb"],
    ]


def build_tex(fig: dict[str, str]) -> str:
    lines: list[str] = []
    add_block(
        lines,
        r"""
\documentclass[10pt,leqno]{article}
\usepackage[margin=0.85in]{geometry}
\usepackage{amsmath,amssymb,amsfonts,amsthm,mathtools}
\usepackage{booktabs,array,graphicx,float,caption,hyperref,microtype}
\hypersetup{colorlinks=true,linkcolor=black,citecolor=black,urlcolor=black}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\numberwithin{equation}{section}
\theoremstyle{plain}
\newtheorem{theorem}{Theorem}[section]
\newtheorem{lemma}[theorem]{Lemma}
\newtheorem{proposition}[theorem]{Proposition}
\newtheorem{corollary}[theorem]{Corollary}
\theoremstyle{definition}
\newtheorem{definition}[theorem]{Definition}
\newtheorem{assumption}[theorem]{Assumption}
\theoremstyle{remark}
\newtheorem{remark}[theorem]{Remark}
\DeclareMathOperator{\dist}{dist}
\DeclareMathOperator{\diag}{diag}
\newcommand{\T}{\mathbb T}
\newcommand{\C}{\mathbb C}
\newcommand{\R}{\mathbb R}
\newcommand{\Z}{\mathbb Z}
\newcommand{\Qop}{Q}
\newcommand{\Lam}{\Lambda}
\newcommand{\dd}{\,d}
\begin{document}
\title{\bf A Matrix-Free QJet Calculus for Near-Singular Quadrature, BGK Endpoint Repayment, and Boundary PDEs}
\author{Rick Gulati Q pipeline research draft}
\date{Generated June 2026}
\maketitle
\begin{abstract}
We give a self-contained SIAM-style account of the machine-precision Q pipeline tested in the accompanying repository.  The central object is a generated inverse-square chord operator: on the regular cycle it has the exact eigenvalues $\lambda_m=m(n-m)/2$, and after the normalization $h/\pi$ it is a finite-dimensional Dirichlet-to-Neumann (DtN) generator with symbol $\mu_{m,n}=m(1-m/n)$.  The implementation stores QJets, Fourier symbols, Laurent coefficients, and low-rank correction modes, never a dense matrix.  We prove the cycle spectrum identity, the normalized DtN error bounds, the exterior conformal kernel split, the BGK/zeta Taylor repayment bound, and boundary-PDE multiplier bounds for Laplace, heat, Poisson, Helmholtz, and wave propagators.  The analysis explains why exact finite-Laurent pullbacks reach the fp64 floor, why polygon and cusp cases need separate corner endpoint ledgers, and why QBX fails when its source-free expansion disk condition is violated.  Numerical evidence is included only after the theorem statements and is labelled as evidence for the discretized generated operator rather than as a continuum theorem beyond the stated hypotheses.
\end{abstract}
\textbf{Keywords.} near-singular quadrature; Dirichlet-to-Neumann map; QJet; BGK correction; spectral zeta; QBX; Helmholtz; boundary integral methods.

\textbf{AMS subject classifications.} 65D32, 65N38, 30C30, 35J25, 41A25.
\tableofcontents
\newpage
""",
    )

    add_block(
        lines,
        r"""
\section{Introduction}
The practical problem is to evaluate near-singular logarithmic layer potentials and boundary PDE propagators on complicated planar curves without forming an $n\times n$ dense matrix.  The method developed here is organized around a simple principle:
\[
\hbox{borrow a circle coordinate, compute in the generated Q spectrum, and repay the geometry.}
\]
On analytic boundaries the repayment is a harmless analytic quotient.  On monitored or polygonal boundaries it is a BGK/zeta or corner continuity correction.  On cusp geometries it is a spectral warning that local expansion methods such as QBX may violate their source-free disk condition.

The paper is intentionally split into two layers.  Sections 2--7 contain the mathematical statements proved under explicit hypotheses.  Sections 8--12 report the repository experiments and show that the implementation follows the theory: exact-Laurent quadrature reaches the floating point floor, generated-Q PDE residuals are at machine precision, and the measured work is near-linear in the sample count.

\subsection{Claim boundary}
The proved results cover the generated regular-cycle operator, analytic finite-Laurent exterior maps with a nonvanishing quotient, and finite-rank or Taylor repayment layers under the stated analyticity assumptions.  For polygons, cusps, and GWW isospectral pairs, the paper proves invariance or validity of the discrete chord-operator experiment, but it does not claim a new continuum Q-spectrum theorem without a separate convergence theorem.
""",
    )

    add_block(
        lines,
        r"""
\section{The Generated Cycle Operator}
\begin{definition}[cycle QJet]
Let $z_j=\exp(2\pi i j/n)$ and let $f=(f_j)_{j=0}^{n-1}$.  The regular-cycle QJet is the circulant operator
\begin{equation}
(\Qop_n f)_j=\sum_{\ell\ne j}\frac{f_j-f_\ell}{|z_j-z_\ell|^2}.
\end{equation}
It is represented in the implementation by its generator $n$ and its eigenvalue formula, not by a dense matrix.
\end{definition}

\begin{lemma}[trigonometric sum identity]\label{lem:cycle-sum}
For $0\le m\le n$,
\begin{equation}
\sum_{d=1}^{n-1}\frac{1-\cos(2\pi m d/n)}{4\sin^2(\pi d/n)}
=\frac{m(n-m)}{2}.
\end{equation}
\end{lemma}
\begin{proof}
Let $\omega=\exp(2\pi i/n)$ and $P_m(z)=1+z+\cdots+z^{m-1}$.  For $d\not\equiv0\pmod n$,
\[
\frac{|1-\omega^{md}|^2}{|1-\omega^d|^2}=|P_m(\omega^d)|^2.
\]
Discrete orthogonality gives $\sum_{d=0}^{n-1}|P_m(\omega^d)|^2=nm$, while the $d=0$ term is $m^2$.  Therefore the sum over $d=1,\ldots,n-1$ is $m(n-m)$.  Since $|1-\omega^{md}|^2=2(1-\cos(2\pi md/n))$ and $|1-\omega^d|^2=4\sin^2(\pi d/n)$, the desired sum is one half of this value.
\end{proof}

\begin{theorem}[exact cycle spectrum]\label{thm:cycle-spectrum}
The Fourier vector $\phi_m(j)=\exp(2\pi i m j/n)$ is an eigenvector of $\Qop_n$ with eigenvalue
\begin{equation}
\lambda_{m,n}=\frac{m(n-m)}{2},\qquad m=0,\ldots,n-1.
\end{equation}
Consequently $\Qop_n{\bf 1}=0$ exactly.
\end{theorem}
\begin{proof}
The circulant structure reduces the eigenvalue to the sum in Lemma~\ref{lem:cycle-sum}.  The constant vector corresponds to $m=0$, for which the numerator vanishes termwise.
\end{proof}

\begin{corollary}[DtN normalization and mode error]\label{cor:dtn-symbol}
Let $h=2\pi/n$ and define $\Lam_n=(h/\pi)\Qop_n$.  For $1\le m<n/2$,
\begin{equation}
\mu_{m,n}:=\frac{h}{\pi}\lambda_{m,n}=m\left(1-\frac{m}{n}\right),
\qquad
0\le m-\mu_{m,n}=\frac{m^2}{n}.
\end{equation}
Thus on modes $m\le M$ the regular-cycle DtN symbol differs from the unit-disk DtN symbol $m$ by at most $M^2/n$.
\end{corollary}

\begin{theorem}[matrix-free functional calculus]\label{thm:matrix-free}
Let $F$ be any scalar multiplier on the cycle spectrum.  The action $f\mapsto F(\Lam_n)f$ can be evaluated with two FFTs and $n$ scalar multiplications, using $O(n)$ storage and $O(n\log n)$ work for radix-two $n$.  No dense Q matrix is required.
\end{theorem}
\begin{proof}
The Fourier basis diagonalizes $\Qop_n$ by Theorem~\ref{thm:cycle-spectrum}.  Applying $F(\Lam_n)$ is therefore the composition
\[
f\mapsto \widehat f\mapsto \{F(\mu_{m,n})\widehat f_m\}_{m=0}^{n-1}\mapsto F(\Lam_n)f.
\]
The first and last maps are the FFT and inverse FFT.  The intermediate map is diagonal storage of $n$ scalars.
\end{proof}
""",
    )

    lines.append(table(["n", "mode", "Q eigenvalue", "normalized DtN", "relative symbol error"], cycle_spectrum_table()))

    add_block(
        lines,
        r"""
\section{Exterior Pullback and Kernel Splitting}
\begin{assumption}[analytic exterior map]\label{ass:analytic-map}
Let $\psi$ be univalent in an exterior annulus $\{1-\epsilon<|w|<R\}$ and suppose $\psi'$ does not vanish there.  For $z$ on the unit circle define
\[
G(w,z)=\frac{\psi(w)-\psi(z)}{w-z}.
\]
Assume $G$ has no zeros on the collar traversed by the target and boundary samples.
\end{assumption}

\begin{theorem}[kernel split]\label{thm:kernel-split}
Under Assumption~\ref{ass:analytic-map},
\begin{equation}
\log|\psi(w)-\psi(z)|=\log|w-z|+\log|G(w,z)|.
\end{equation}
The first term is the universal circle singularity.  The second term is real analytic in the boundary angle and is therefore exponentially convergent under the periodic trapezoidal rule.
\end{theorem}
\begin{proof}
The identity is algebraic away from $w=z$.  At $w=z$ the quotient tends to $\psi'(z)$, nonzero by assumption; hence $\log|G(w,z)|$ extends real analytically to the diagonal and to a complex strip inside the annulus.  The standard trapezoidal theorem for analytic periodic functions gives an error bounded by $C e^{-\tau n}$, where $\tau$ is any strip width free of singularities.
\end{proof}

\begin{proposition}[closed circle primitive]\label{prop:circle-primitive}
For $\rho>1$,
\begin{equation}
\log|\rho e^{i\phi}-e^{i\theta}|
=\log\rho-\sum_{k=1}^\infty \frac{\rho^{-k}}{k}\cos k(\theta-\phi).
\end{equation}
If the density has Fourier coefficients $|\widehat a_k|\le A R^{-|k|}$ with $R>1$, then truncation after $K$ modes has error bounded by
\begin{equation}
E_K\le \frac{2\pi A(\rho R)^{-K-1}}{(K+1)(1-(\rho R)^{-1})}.
\end{equation}
\end{proposition}
\begin{proof}
Write
\[
\rho e^{i\phi}-e^{i\theta}=\rho e^{i\phi}\{1-\rho^{-1}e^{i(\theta-\phi)}\}
\]
and expand $\log(1-\zeta)=-\sum_{k\ge1}\zeta^k/k$ for $|\zeta|<1$.  Taking real parts gives the formula.  Multiplying by $a$ and integrating leaves only matching Fourier modes; the displayed geometric tail follows from $|\widehat a_k|\le A R^{-|k|}$.
\end{proof}

\begin{corollary}[analytic quadrature error]\label{cor:analytic-error}
For an analytic density and a map satisfying Assumption~\ref{ass:analytic-map}, the split quadrature error has the form
\begin{equation}
E_n \le C_1 e^{-\tau n}+C_2(\rho R)^{-n/2}+E_{\rm fp},
\end{equation}
where $E_{\rm fp}$ is floating point roundoff.  The constants depend on the annulus and density bounds but not on the target distance once $\rho>1$ is fixed.
\end{corollary}

\begin{proposition}[golden ellipse quotient]\label{prop:golden-ellipse}
The map
\[
\psi(w)=\frac{3+\sqrt5}{2}w+\frac{3-\sqrt5}{2}w^{-1}
\]
has boundary trace $\psi(e^{i\theta})=3\cos\theta+i\sqrt5\sin\theta$.  Moreover
\[
\frac{\psi(w)-\psi(z)}{w-z}=\frac{3+\sqrt5}{2}-\frac{3-\sqrt5}{2}\frac{1}{wz}.
\]
Since $(3-\sqrt5)/(3+\sqrt5)<1$, the quotient is nonzero for $|wz|$ sufficiently near one on the exterior collar.
\end{proposition}
\begin{proof}
Substituting $w=e^{i\theta}$ gives
\[
\frac{3+\sqrt5}{2}e^{i\theta}+\frac{3-\sqrt5}{2}e^{-i\theta}
=3\cos\theta+i\sqrt5\sin\theta.
\]
For the quotient, use $w^{-1}-z^{-1}=-(w-z)/(wz)$.  The nonvanishing statement follows from the strict coefficient ratio.
\end{proof}
""",
    )

    if "exterior_split" in fig:
        add_block(
            lines,
            rf"""
\begin{{figure}}[H]\centering
\includegraphics[width=0.92\linewidth]{{{fig['exterior_split']}}}
\caption{{Exterior split results: the direct rule sees the near singularity; the Q split evaluates the circle singularity by the closed primitive and leaves an analytic quotient.}}
\end{{figure}}
""",
        )

    add_block(
        lines,
        r"""
\section{BGK/Zeta Endpoint Repayment}
The split above handles analytic geometry.  Discrete monitoring or endpoint displacement introduces a different error: the target is evaluated at a shifted normal coordinate.  The BGK constant is the universal leading endpoint defect.

\begin{theorem}[Taylor repayment bound]\label{thm:taylor-repay}
Let $I(\rho)$ be analytic in $|\rho-\rho_0|\le r$ and let $|I(\rho)|\le M$ there.  Put $\delta=\beta\sqrt h$ and assume $|\delta|<r$.  Define
\[
T_p(h)=\sum_{j=0}^{p}\frac{(-\delta)^j}{j!}I^{(j)}(\rho_0+\delta).
\]
Then
\begin{equation}
|I(\rho_0)-T_p(h)|\le
\frac{M}{1-|\delta|/r}\left(\frac{|\delta|}{r}\right)^{p+1}.
\end{equation}
In particular the order-$p$ repayment error is $O(h^{(p+1)/2})$.
\end{theorem}
\begin{proof}
Taylor expand $I$ about the observed point $\rho_0+\delta$:
\[
I(\rho_0)=\sum_{j=0}^{p}\frac{(-\delta)^j}{j!}I^{(j)}(\rho_0+\delta)+R_{p+1}.
\]
Cauchy's estimate gives $|I^{(j)}(\rho_0+\delta)|/j!\le M r^{-j}$ on any disk of radius $r$ contained in the analyticity domain.  Summing the omitted geometric tail gives the displayed bound.
\end{proof}

\begin{proposition}[BGK constant as endpoint zeta defect]\label{prop:bgk}
Let $(S_k)$ be a centered Gaussian walk with unit variance increments.  Then
\begin{equation}
\lim_{N\to\infty}\left\{\mathbb E\max_{0\le k\le N}S_k-\sqrt{\frac{2N}{\pi}}\right\}
=\frac{\zeta(1/2)}{\sqrt{2\pi}}.
\end{equation}
Thus the standard BGK shift constant is $\beta=-\zeta(1/2)/\sqrt{2\pi}$.
\end{proposition}
\begin{proof}
Spitzer's identity gives
\[
\mathbb E\max_{0\le k\le N}S_k=\sum_{k=1}^N\frac{1}{k}\mathbb E(S_k)^+.
\]
Since $S_k$ is Gaussian with variance $k$, $\mathbb E(S_k)^+=\sqrt{k}/\sqrt{2\pi}$.  Therefore
\[
\mathbb E\max S_k=\frac1{\sqrt{2\pi}}\sum_{k=1}^N k^{-1/2}.
\]
Euler--Maclaurin gives $\sum_{k=1}^N k^{-1/2}=2\sqrt N+\zeta(1/2)+O(N^{-1/2})$.  Subtracting $\sqrt{2N/\pi}=2\sqrt N/\sqrt{2\pi}$ yields the limit.
\end{proof}

\begin{corollary}[combined analytic and BGK bound]\label{cor:combined}
Under Assumption~\ref{ass:analytic-map}, with an analytic density and order-$p$ BGK repayment,
\begin{equation}
E_{n,h,p}\le C_1 e^{-\tau n}+C_2(\rho R)^{-n/2}
C_3 h^{(p+1)/2}+E_{\rm fp}.
\end{equation}
\end{corollary}
""",
    )
    lines.append(table(["shape", "raw power", "BGK-1 power", "BGK-2 power", "raw err", "BGK-4 err"], bgk_table()))
    if "bgk_pipeline" in fig:
        add_block(
            lines,
            rf"""
\begin{{figure}}[H]\centering
\includegraphics[width=0.92\linewidth]{{{fig['bgk_pipeline']}}}
\caption{{BGK/zeta repayment slopes.  The raw error follows the square-root endpoint law; the first and second repayment layers shift the power to one and three-halves; order four reaches fp64 floor in these analytic tests.}}
\end{{figure}}
""",
        )

    add_block(
        lines,
        r"""
\section{Boundary PDEs from the Generated DtN Correspondence}
The finite Q spectrum can be used as a load-bearing operator, not just as a quadrature correction.  On the disk the exact continuum DtN symbol is $m$; the generated symbol is $\mu_{m,n}=m-m^2/n$.

\begin{theorem}[multiplier error bounds]\label{thm:pde-bounds}
Fix $m<n/2$ and let $\delta_m=m^2/n$.  For the generated symbol $\mu=m-\delta_m$:
\begin{align}
|m-\mu|&=\delta_m,\\
|e^{-tm}-e^{-t\mu}|&\le t\delta_m,\qquad t\ge0,\\
\left|\frac1{m+\alpha}-\frac1{\mu+\alpha}\right|&\le
\frac{\delta_m}{(m+\alpha)(\mu+\alpha)},\qquad \alpha>0.
\end{align}
For the damped Helmholtz boundary resolvent $R_k(x)=(x^2-k^2+i\gamma)^{-1}$ with $\gamma>0$,
\begin{equation}
|R_k(m)-R_k(\mu)|
\le \frac{(m+\mu)\delta_m}
{|m^2-k^2+i\gamma|\,|\mu^2-k^2+i\gamma|}.
\end{equation}
For the wave multiplier $W_t(x)=\cos(t\sqrt x)$,
\begin{equation}
|W_t(m)-W_t(\mu)|\le \frac{t\delta_m}{\sqrt m+\sqrt\mu}.
\end{equation}
\end{theorem}
\begin{proof}
The first identity is Corollary~\ref{cor:dtn-symbol}.  The heat bound follows from the mean value theorem and $|\frac{d}{dx}e^{-tx}|\le t$ for $x\ge0$.  The Poisson and Helmholtz inequalities follow by subtracting the two fractions.  For the wave bound, use $|\cos a-\cos b|\le |a-b|$ and $|\sqrt m-\sqrt\mu|=\delta_m/(\sqrt m+\sqrt\mu)$.
\end{proof}

\begin{corollary}[conformal metric repayment]\label{cor:metric-repay}
Let $\Gamma=\psi(S^1)$ with $s_{\min}=\inf_\theta|\psi'(e^{i\theta})|>0$.  If the circle DtN flux has $L^2(d\theta)$ error $\epsilon$, then the physical normal flux after repayment by $|\partial_\theta\psi|^{-1}$ has $L^2(d\theta)$ error at most $s_{\min}^{-1}\epsilon$.
\end{corollary}
\begin{proof}
The physical normal derivative is the circle normal derivative multiplied pointwise by $|\partial_\theta\psi|^{-1}$.  Taking the $L^\infty$ bound on this multiplier gives the result.
\end{proof}
""",
    )
    lines.append(table(["problem", "max continuum err", "max generated-Q residual", "mean solve ms", "work"], pde_problem_table()))

    add_block(
        lines,
        r"""
\section{Corners, Cusps, and Why QBX Can Fail}
The analytic proof does not pass through corners for free.  If a corner has opening angle $\alpha$, the conformal map has local power behavior $(w-w_c)^{\alpha/\pi}$, so Fourier coefficients decay algebraically rather than exponentially.  The correct response is not to pretend analyticity remains; it is to add a corner or endpoint repayment channel.

\begin{proposition}[QBX disk condition]\label{prop:qbx-disk}
A local QBX expansion about a center $c$ can be evaluated at a target $x$ only when $x$ lies in the expansion disk and that disk is free of source geometry.  If the nearest boundary source enters the disk before the target does, the local expansion hypothesis is false and a robust implementation must reject the evaluation.
\end{proposition}
\begin{proof}
The QBX expansion is a local series for the potential in a disk of analyticity.  A source point inside that disk is a singularity of the potential representation and prevents convergence of the source-free local series.  Therefore the condition is necessary for the expansion proof itself, not merely for numerical conditioning.
\end{proof}

\begin{remark}
The cusp failure figures in the experiments are examples of Proposition~\ref{prop:qbx-disk}: the reported margin is negative, so the target is outside the valid source-free expansion disk.  The Q pipeline does not use this local disk as its principal certificate; it uses the global chord spectrum and an endpoint/corner repayment ledger.
\end{remark}
""",
    )
    if "qbx_failures" in fig:
        add_block(
            lines,
            rf"""
\begin{{figure}}[H]\centering
\includegraphics[width=0.84\linewidth]{{{fig['qbx_failures']}}}
\caption{{Cusp configurations where QBX violates its source-free expansion disk condition.}}
\end{{figure}}
""",
        )

    add_block(
        lines,
        r"""
\section{Discrete Spectral Witnesses}
\begin{proposition}[Euclidean covariance of the chord operator]\label{prop:covariance}
Let $\Gamma_n=\{x_j\}_{j=0}^{n-1}$ and define
\[
(\Lam_{\Gamma,n}f)_i=\frac{L}{\pi n}\sum_{j\ne i}\frac{f_i-f_j}{|x_i-x_j|^2}.
\]
Translations and rotations of all $x_j$ leave $\Lam_{\Gamma,n}$ unchanged.  Uniform scaling by $a>0$ changes the operator by the factor $a^{-1}$.
\end{proposition}
\begin{proof}
Translations cancel in the differences $x_i-x_j$, and rotations preserve Euclidean distances.  Under $x_j\mapsto ax_j$, the perimeter $L$ scales by $a$ while squared chord lengths scale by $a^2$, leaving the total factor $a/a^2=a^{-1}$.
\end{proof}

\begin{remark}[GWW scope]
The GWW experiment in this paper is a discretized chord-operator witness.  The classical Dirichlet spectra agree in the continuum, but the finite projected Q Ritz values below are not asserted to prove a new continuum invariant.  They show robust separation at the audited discretized operator level.
\end{remark}
""",
    )
    lines.append(table(["n", "relative split", "left Ritz 1-6", "right Ritz 1-6"], gww_table(), align=r"rlll"))
    if "gww" in fig:
        add_block(
            lines,
            rf"""
\begin{{figure}}[H]\centering
\includegraphics[width=0.88\linewidth]{{{fig['gww']}}}
\caption{{Projected Q Ritz separation for the Gordon--Webb--Wolpert pair under refinement.}}
\end{{figure}}
""",
        )

    add_block(
        lines,
        r"""
\section{Numerical Evidence}
All timings below are generated from the current repository state.  They are evidence for the implementation and for the discretized generated operator; the mathematical convergence statements are those proved above.

\subsection{Funky analytic shapes}
""",
    )
    lines.append(table(["shape", "family", "anisotropy", "BGK-4 err", "max PDE continuum err", "max generated-Q residual"], funky_shape_table()))
    if "funky_pde" in fig:
        add_block(
            lines,
            rf"""
\begin{{figure}}[H]\centering
\includegraphics[width=0.92\linewidth]{{{fig['funky_pde']}}}
\caption{{Ten exact-Laurent shapes used for Q+BGK quadrature and boundary-only PDE tests.}}
\end{{figure}}
""",
        )

    add_block(lines, r"\subsection{Measured matrix-free scaling}")
    lines.append(table(["operation", "fit power", "ms n=256", "ms n=1024", "ms n=4096", "max residual"], scaling_table()))
    if "scaling" in fig:
        add_block(
            lines,
            rf"""
\begin{{figure}}[H]\centering
\includegraphics[width=0.78\linewidth]{{{fig['scaling']}}}
\caption{{Measured matrix-free scaling.  The fits are near-linear over the tested range and no dense Q matrix is stored.}}
\end{{figure}}
""",
        )

    add_block(lines, r"\subsection{Head-to-head methods and Helmholtz caveat}")
    lines.append(table(["method", "median err", "median ms", "work", "failures", "improvement"], structural_table()))
    lines.append(table(["suite", "result"], helmholtz_table(), align="ll"))

    if "spectrum" in fig:
        add_block(
            lines,
            rf"""
\begin{{figure}}[H]\centering
\includegraphics[width=0.92\linewidth]{{{fig['spectrum']}}}
\caption{{Q spectrum gallery.  The spectrum diagnoses whether the active error channel is smooth-tail, corner, cusp, or mixed.}}
\end{{figure}}
""",
        )
    if "bgk_bookkeeping" in fig:
        add_block(
            lines,
            rf"""
\begin{{figure}}[H]\centering
\includegraphics[width=0.88\linewidth]{{{fig['bgk_bookkeeping']}}}
\caption{{BGK endpoint bookkeeping as Spitzer, zeta, and Q/DtN endpoint defect.}}
\end{{figure}}
""",
        )

    add_block(
        lines,
        r"""
\section{Implementation Contract}
The production implementation follows three invariants.
\begin{enumerate}
\item Store QJets, spectra, Laurent coefficients, samples, and low-rank correction modes; do not store a dense Q matrix.
\item Use the custom QJet Fourier kernel for the singular term and repay geometry through analytic quotients, metric factors, or endpoint/corner ledgers.
\item Report both continuum error and generated-Q residual.  The latter verifies the implementation; the former measures finite-$n$ approximation to the limiting operator.
\end{enumerate}

\subsection{Final self-contained artifact}
The repository contains a final script and a companion notebook generated from the same source.  The script uses no NumPy, no SciPy, and no imported FFT package; its radix-two QJet FFT is implemented directly.  Python library calls are used only for ordinary scalar math, timing, and file output.  The notebook embeds the full script source in a code cell rather than delegating to an external module, so the executable numerical path is auditable from a single \texttt{.ipynb} file.
""",
    )
    lines.append(table(["quantity", "value"], final_pipeline_table(), align="ll"))
    add_block(
        lines,
        r"""
The gate in this table is deliberately stricter than the qualitative plots: the final script fails if the maximum of the split quadrature error, the order-eight BGK repayment error, and the generated-Q PDE residual exceeds $10^{-12}$ over the included finite-Laurent shape suite.

\section{Source Audit}
The following local PDFs were inspected mechanically with \texttt{pdfinfo} and \texttt{pdftotext}.  The audit is not a bibliography; it records which local source artifacts were available while drafting the paper.
""",
    )
    lines.append(table(["PDF", "pages", "sha", "words", "role"], pdf_audit_table(), align="lrrrl"))

    add_block(
        lines,
        r"""
\section{Conclusion}
The elegant core is the chain
\[
\frac{1-\cos(2\pi md/n)}{4\sin^2(\pi d/n)}
\longrightarrow
\lambda_{m,n}=\frac{m(n-m)}{2}
\longrightarrow
\mu_{m,n}=m\left(1-\frac mn\right)
\longrightarrow
\Lam_{S^1}=|D|.
\]
This identity supplies the generated Q spectrum.  The exterior map supplies the split
\[
\log|\psi(w)-\psi(z)|=\log|w-z|+\log|G(w,z)|,
\]
and the endpoint calculus supplies the BGK/zeta repayment
\[
\beta=-\zeta(1/2)/\sqrt{2\pi}.
\]
Together these identities explain why the exact analytic path reaches machine precision, why the PDE pipeline can use Q as an operator rather than a post-process, and why the spectrum is the correct diagnostic for deciding whether smooth, corner, cusp, or Helmholtz-specific corrections are required.

\begin{thebibliography}{9}
\bibitem{BGK}
M. Broadie, P. Glasserman, and S. Kou, A continuity correction for discrete barrier options, \emph{Mathematical Finance}, 7 (1997), pp. 325--349.
\bibitem{GWW}
C. Gordon, D. Webb, and S. Wolpert, One cannot hear the shape of a drum, \emph{Bull. Amer. Math. Soc.}, 27 (1992), pp. 134--138.
\bibitem{QBX}
A. Klockner, A. Barnett, L. Greengard, and M. O'Neil, Quadrature by expansion: A new method for the evaluation of layer potentials, \emph{J. Comput. Phys.}, 252 (2013), pp. 332--349.
\bibitem{Spitzer}
F. Spitzer, A combinatorial lemma and its application to probability theory, \emph{Trans. Amer. Math. Soc.}, 82 (1956), pp. 323--339.
\bibitem{TW}
L. N. Trefethen and J. A. C. Weideman, The exponentially convergent trapezoidal rule, \emph{SIAM Review}, 56 (2014), pp. 385--458.
\end{thebibliography}
\end{document}
""",
    )
    return "\n".join(lines) + "\n"


def compile_tex(tex: Path) -> Path | None:
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
    fig = copy_figures()
    tex = OUT / "siam_q_research_paper.tex"
    tex.write_text(build_tex(fig), encoding="utf-8")
    pdf = compile_tex(tex)
    payload = {
        "tex": str(tex),
        "pdf": None if pdf is None else str(pdf),
        "figures": fig,
        "dense_q_matrix_stored": False,
        "style": "self-contained SIAM-like article; SIAM class not installed",
    }
    (OUT / "siam_q_research_paper.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
