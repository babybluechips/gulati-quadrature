#!/usr/bin/env python3
"""Generate the held-out, cited scientific benchmark registry.

The registry is deliberately separate from solver output.  A row may only be
called ground truth when its reference quantity comes from a named external
benchmark or a standard closed-form solution used as such.  Repo-internal
manufactured fields are useful diagnostics, but they are not ground truth unless
they are tied to one of these sources.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "standard_scientific_benchmarks"


def registry() -> list[dict[str, object]]:
    return [
        {
            "id": "unit_disk_steklov_dtn",
            "source": "Girouard and Polterovich, Spectral geometry of the Steklov problem, 2017 survey",
            "url": "https://dms.umontreal.ca/~iossif/steklov_spectral_geometry.pdf",
            "external_status": "standard closed-form DtN/Steklov reference",
            "problem": "Laplace Dirichlet-to-Neumann map / Steklov spectrum",
            "domain": "unit disk",
            "reference_quantity": "Steklov eigenvalues 0,1,1,2,2,... with harmonic traces r^k cos(k theta), r^k sin(k theta)",
            "repo_status": "implemented in the repaid disk DtN rows and exact modal PDE calibration",
            "ground_truth_policy": "authoritative for unit-disk DtN modal rows; raw finite-cycle Q rows remain diagnostics",
        },
        {
            "id": "nistir7668_poly_poisson",
            "source": "William F. Mitchell, A Collection of 2D Elliptic Problems for Testing Adaptive Algorithms, NISTIR 7668, 2010",
            "url": "https://www.nist.gov/publications/collection-2d-elliptic-problems-testing-adaptive-algorithms",
            "external_status": "standard held-out benchmark suite",
            "problem": "Poisson, Dirichlet",
            "domain": "unit square",
            "reference_quantity": "exact polynomial solution and derived forcing/boundary data",
            "repo_status": "registered; should be ported into the Q/FEM comparison before any square-domain accuracy claim",
            "ground_truth_policy": "authoritative after the NIST formula and parameters are implemented verbatim",
        },
        {
            "id": "nistir7668_singular_elliptic_suite",
            "source": "William F. Mitchell, A Collection of 2D Elliptic Problems for Testing Adaptive Algorithms, NISTIR 7668, 2010",
            "url": "https://www.nist.gov/publications/collection-2d-elliptic-problems-testing-adaptive-algorithms",
            "external_status": "standard held-out benchmark suite",
            "problem": "2D elliptic problems with singularities and near singularities",
            "domain": "NISTIR 7668 parametrized domains",
            "reference_quantity": "published exact solutions or stated benchmark data",
            "repo_status": "registered; replace ad-hoc funky manufactured cases when exact NIST formulas are ported",
            "ground_truth_policy": "authoritative only for the externally specified cases, not for lookalike local variants",
        },
        {
            "id": "motz_laplace_corner",
            "source": "Georgiou, Olson, Smyrlis, A singular function boundary integral method for the Laplace equation, Communications in Numerical Methods in Engineering 12, 1996",
            "url": "https://www.mas.ucy.ac.cy/georgios/pdfiles/cnme96.pdf",
            "external_status": "standard singular Laplace benchmark",
            "problem": "Motz problem, Laplace equation with mixed boundary-condition corner singularity",
            "domain": "Motz rectangle with boundary-condition change at the singular point",
            "reference_quantity": "Rosser-Papamichael analytical solution coefficients as cited in the benchmark literature",
            "repo_status": "diagnostic Motz-like corner rows exist; not yet accepted as ground truth until exact Motz geometry and coefficients are ported",
            "ground_truth_policy": "the local Motz-like case is not ground truth; use the published Motz data verbatim",
        },
        {
            "id": "l_shape_laplacian_eigenvalue",
            "source": "Betcke and Trefethen, Reviving the method of particular solutions, SIAM Review 47, 2005",
            "url": "https://eprints.maths.manchester.ac.uk/589/1/MPSfinal.pdf",
            "external_status": "standard Laplacian eigenvalue benchmark",
            "problem": "Dirichlet Laplacian eigenvalue on the L-shaped domain",
            "domain": "standard L-shaped polygon",
            "reference_quantity": "first eigenvalue lambda_1 approximately 9.6397238440219",
            "repo_status": "registered; current corner flux rows are not this eigenvalue benchmark",
            "ground_truth_policy": "authoritative for eigenvalue comparisons once the exact L-shape sampling and eigen-solver are wired",
        },
        {
            "id": "gww_isospectral_drums",
            "source": "Gordon, Webb, Wolpert, One cannot hear the shape of a drum, Inventiones Mathematicae 110, 1992; Driscoll isospectral-drums notes",
            "url": "https://tobydriscoll.net/project/drums/",
            "external_status": "standard isospectral polygon benchmark",
            "problem": "Dirichlet Laplacian isospectrality",
            "domain": "Gordon-Webb-Wolpert eight-sided nonconvex polygon pair",
            "reference_quantity": "Dirichlet Laplacian spectra are identical by transplantation; exact numeric eigenvalues are nontrivial and not the ground-truth claim",
            "repo_status": "Q-spectrum separation experiment exists; must be described as a Q-operator diagnostic, not a contradiction of Dirichlet isospectrality",
            "ground_truth_policy": "the held-out fact is Dirichlet isospectral equality; Q separation is a different operator-level observable",
        },
        {
            "id": "qbx_layer_potential_jcp2013",
            "source": "Klöckner, Barnett, Greengard, O'Neil, Quadrature by Expansion, Journal of Computational Physics 252, 2013",
            "url": "https://arxiv.org/abs/1207.4461",
            "external_status": "standard near-boundary layer-potential quadrature benchmark family",
            "problem": "singular and nearly singular layer-potential evaluation",
            "domain": "smooth and corner curves used in QBX accuracy tests",
            "reference_quantity": "overresolved or analytic layer-potential values for the stated QBX test cases",
            "repo_status": "QBX comparison rows exist; accepted ground truth requires matching the published geometry, density, target protocol, and reference resolution",
            "ground_truth_policy": "QBX is a competitor method; reference must be external analytic/overresolved data, not QBX output itself",
        },
        {
            "id": "dlmf_disk_bessel_helmholtz",
            "source": "NIST Digital Library of Mathematical Functions, Chapter 10 Bessel Functions",
            "url": "https://dlmf.nist.gov/10",
            "external_status": "standard closed-form special-function reference",
            "problem": "disk/circular-cylinder Helmholtz modal DtN",
            "domain": "unit disk or circular cylinder",
            "reference_quantity": "Bessel/Hankel modal ratios such as k J'_m(k)/J_m(k) and k H_m'(k)/H_m(k)",
            "repo_status": "implemented in the disk Helmholtz/DtN scripts; should be labeled as closed-form standard reference, not FEM truth",
            "ground_truth_policy": "authoritative for circular modal cases and useful as a calibration gate, but not evidence for arbitrary domains",
        },
        {
            "id": "helsing_ojala_corner_integral",
            "source": "Helsing and Ojala, Corner singularities for elliptic problems, Journal of Computational Physics 227, 2008",
            "url": "https://www.semanticscholar.org/paper/bcb59bf6097d483ddab38da69e97ec9960bf36dc",
            "external_status": "standard corner-integral-equation benchmark literature",
            "problem": "elliptic boundary integral equations with corner singularities",
            "domain": "piecewise smooth domains with corners",
            "reference_quantity": "published corner singular quadrature and compressed inverse-preconditioning results",
            "repo_status": "registered; current corner-fix demos need exact protocol alignment before becoming ground truth rows",
            "ground_truth_policy": "use as a protocol/source benchmark for corner correction, not as an unlabeled local proxy",
        },
    ]


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Held-Out Scientific Benchmark Registry",
        "",
        "This registry defines what may be called ground truth in the reports. FEM, QBX, and local manufactured cases are not ground truth unless they are compared against one of these cited external references or a verbatim implementation of its formulas/protocol.",
        "",
        "| id | source | problem/domain | accepted reference | repo status |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['id']} | [{row['source']}]({row['url']}) | {row['problem']} on {row['domain']} | "
            f"{row['reference_quantity']} | {row['repo_status']} |"
        )
    lines += [
        "",
        "## Policy",
        "",
        "- Accuracy tables must name the registry id behind their reference.",
        "- Rows without a registry id are diagnostics, not ground truth.",
        "- FEM and QBX may win or lose against a reference, but they are not the reference.",
        "- Internal manufactured examples can remain in stress tests, but the report must not use them for headline claims until the setup is tied to a cited benchmark protocol.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> dict[str, object]:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = registry()
    payload = {
        "policy": "Only cited external benchmarks or verbatim standard closed-form formulas may be called ground truth.",
        "case_count": len(rows),
        "registry": rows,
    }
    (OUT / "benchmark_registry.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(OUT / "benchmark_registry.md", rows)
    print(json.dumps({"output": str(OUT), "case_count": len(rows)}, indent=2, sort_keys=True))
    return payload


if __name__ == "__main__":
    main()
