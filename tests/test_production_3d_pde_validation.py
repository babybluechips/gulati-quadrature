import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "production_3d_pde_validation"


def _rows(name):
    with (OUTPUT / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_discrete_pde_campaign_covers_every_problem_on_every_shape() -> None:
    summary = json.loads((OUTPUT / "summary.json").read_text(encoding="utf-8"))
    rows = _rows("discrete_pde_rows.csv")
    problems = {
        "laplace_dtn",
        "poisson",
        "screened_poisson",
        "helmholtz",
        "heat",
        "wave",
    }
    shapes = {row["shape"] for row in rows}

    assert len(shapes) == 5
    assert len(rows) == len(shapes) * len(problems) == 30
    assert all(
        {row["problem"] for row in rows if row["shape"] == shape} == problems
        for shape in shapes
    )
    assert all(row["passed"] == "True" for row in rows)
    assert all(row["dense_operator_stored"] == "False" for row in rows)
    assert all(row["quadratic_fallback"] == "False" for row in rows)
    assert summary["all_discrete_gates_passed"] is True
    assert summary["maximum_discrete_error"] < 2.0e-8
    assert summary["maximum_discrete_residual"] < 2.0e-9


def test_sphere_continuum_audit_is_kept_separate_from_algebraic_residual() -> None:
    summary = json.loads((OUTPUT / "summary.json").read_text(encoding="utf-8"))
    rows = _rows("sphere_continuum_rows.csv")
    degree_one = [
        row
        for row in rows
        if row["problem"] == "laplace_dtn" and row["degree"] == "1"
    ]
    degree_one.sort(key=lambda row: int(row["nodes"]))
    errors = [float(row["relative_continuum_error"]) for row in degree_one]

    assert len(rows) == 13
    assert all(left > right for left, right in zip(errors, errors[1:], strict=False))
    assert 0.9 < summary["median_degree_one_order"] < 1.2
    assert summary["best_sphere_degree_one_error"] > 1.0e-2
    assert summary["continuum_machine_precision_claim"] is False
    assert summary["volume_source_solver_claim"] is False

    report = (OUTPUT / "report.md").read_text(encoding="utf-8")
    assert "does **not** establish universal machine-precision 3D PDE accuracy" in report
    assert "volume-source channels for bulk Poisson/heat" in report
