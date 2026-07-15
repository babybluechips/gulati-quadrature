import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "transparent_tail_benchmark"


def _summary():
    return json.loads((OUT / "summary.json").read_text(encoding="utf-8"))


def _rows():
    with (OUT / "benchmark_rows.csv").open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_transparent_tail_campaign_passes_exactness_and_storage_gates() -> None:
    summary = _summary()
    assert summary["all_gates_passed"] is True
    assert summary["pde_case_count"] == 5
    assert summary["benchmark_row_count"] == 120
    assert summary["autonomous_tail_truncation_error_with_cap"] == 0.0
    assert summary["maximum_direct_finite_disagreement"] < 3.0e-14
    assert summary["maximum_fixed_point_residual"] < 3.0e-15
    assert summary["maximum_cross_ratio_residual"] < 2.0e-14
    assert summary["cylinder_identity_residual"] < 2.0e-14
    assert summary["perturbed_transition_actual_error"] <= (
        summary["perturbed_transition_error_bound"] + 2.0e-15
    )
    assert summary["perturbed_transition_maximum_lipschitz"] < 1.0
    assert summary["dense_shell_matrix_stored"] is False
    assert summary["dense_boundary_matrix_stored"] is False
    assert summary["held_out_cad_machine_precision_claim"] is False


def test_deep_fixed_point_cap_removes_depth_and_is_faster() -> None:
    rows = _rows()
    deepest = max(int(row["depth"]) for row in rows)
    assert deepest == 512
    problems = {row["problem"] for row in rows}
    for problem in problems:
        selected = [
            row
            for row in rows
            if row["problem"] == problem and int(row["depth"]) == deepest
        ]
        by_method = {row["method"]: row for row in selected}
        direct = by_method["direct_L_shell_thomas"]
        finite = by_method["compiled_finite_dirichlet"]
        cap = by_method["exact_fixed_point_cap"]
        assert float(cap["relative_boundary_error"]) == 0.0
        assert int(cap["shell_iterations"]) == 0
        assert float(direct["total_ms"]) > 50.0 * float(cap["total_ms"])
        assert float(finite["total_ms"]) < float(direct["total_ms"])
        assert direct["dense_matrix_stored"] == "False"
        assert finite["storage_big_o"] == "O(N_theta)"


def test_compiled_and_cap_application_times_have_no_depth_factor() -> None:
    rows = _rows()
    for problem in {row["problem"] for row in rows}:
        cap_rows = [
            row
            for row in rows
            if row["problem"] == problem and row["method"] == "exact_fixed_point_cap"
        ]
        assert len({float(row["apply_ms"]) for row in cap_rows}) == 1
        direct_rows = sorted(
            (
                row
                for row in rows
                if row["problem"] == problem
                and row["method"] == "direct_L_shell_thomas"
            ),
            key=lambda row: int(row["depth"]),
        )
        assert float(direct_rows[-1]["apply_ms"]) > 20.0 * float(
            direct_rows[0]["apply_ms"]
        )


def test_report_states_the_cad_scope_and_exact_normalization() -> None:
    report = (OUT / "report.md").read_text(encoding="utf-8")
    for phrase in (
        "exact fixed-point transparent cap",
        "zero autonomous-tail truncation error",
        "O(K N_theta L + K N_theta log N_theta)",
        "does not reconstruct those discarded surface channels",
        "F_42/F_40",
    ):
        assert phrase in report
    svg = (OUT / "runtime_scaling.svg").read_text(encoding="utf-8")
    assert "Direct L-shell" in svg
    assert "Fixed-point cap" in svg
