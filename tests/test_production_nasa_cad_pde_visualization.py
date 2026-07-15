import csv
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "production_nasa_cad_pde_visualization"


def _rows(name: str) -> list[dict[str, str]]:
    with (OUT / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_nasa_cad_pde_summary_has_machine_scale_declared_audits() -> None:
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    assert summary["shape_count"] == 3
    assert summary["pde_case_count"] == 18
    assert summary["all_declared_gates_passed"] is True
    assert summary["maximum_retained_reference_error"] < 1.0e-14
    assert summary["maximum_algebraic_residual"] < 2.0e-14
    assert summary["maximum_warm_solve_ms"] < 10_000.0
    assert summary["median_warm_solve_ms"] < 100.0
    assert summary["all_archive_checksums_verified"] is True
    assert summary["all_source_vertices_lifted"] is True
    assert summary["all_source_faces_scanned"] is True
    assert summary["source_vertex_count"] == 612_609
    assert summary["source_face_count"] == 1_227_262
    assert summary["rendered_nondegenerate_face_count"] == 1_227_262
    assert summary["dense_q_matrix_stored"] is False
    assert summary["pair_table_stored"] is False
    assert summary["visualization_uses_numerical_fields"] is True
    assert summary["held_out_continuum_machine_precision_claim"] is False


def test_each_nasa_archive_has_six_lifted_numerical_fields() -> None:
    rows = _rows("pde_rows.csv")
    geometry = {row["key"]: row for row in _rows("geometry_rows.csv")}
    expected_problems = {
        "laplace_dtn",
        "poisson_boundary_inverse",
        "screened_poisson_boundary_inverse",
        "helmholtz_dtn",
        "heat_boundary_semigroup",
        "wave_boundary_calculus",
    }
    assert set(geometry) == {
        "sofia_aircraft",
        "curiosity_rover",
        "curiosity_assembled",
    }
    assert {key: int(row["compiled_nodes"]) for key, row in geometry.items()} == {
        "sofia_aircraft": 42,
        "curiosity_rover": 24,
        "curiosity_assembled": 36,
    }
    for key, geometry_row in geometry.items():
        selected = [row for row in rows if row["key"] == key]
        assert {row["problem"] for row in selected} == expected_problems
        assert all(row["passed"] == "True" for row in selected)
        assert all(row["dense_operator_stored"] == "False" for row in selected)
        assert all(row["quadratic_fallback"] == "False" for row in selected)
        assert all(row["continuum_accuracy_claim"] == "False" for row in selected)
        assert all(
            int(row["source_lift_entries"])
            == int(geometry_row["source_vertices"])
            for row in selected
        )


def test_nasa_field_pngs_are_nonblank_multicolor_numerical_renders() -> None:
    rows = _rows("pde_rows.csv")
    for row in rows:
        path = OUT / row["panel_path"]
        assert path.exists()
        with Image.open(path) as image:
            assert image.size == (800, 500)
            colors = image.convert("RGB").getcolors(maxcolors=1_000_000)
            assert colors is not None
            assert len(colors) > 500
    with Image.open(OUT / "nasa_cad_pde_overview.png") as overview:
        assert overview.size == (1294, 1430)
        colors = overview.convert("RGB").getcolors(maxcolors=1_000_000)
        assert colors is not None
        assert len(colors) > 5_000
    for key in ("sofia_aircraft", "curiosity_rover", "curiosity_assembled"):
        with Image.open(OUT / f"{key}_pde_fields.png") as figure:
            assert figure.size == (1654, 1664)


def test_nasa_visual_report_states_timing_and_accuracy_scope() -> None:
    report = (OUT / "report.md").read_text(encoding="utf-8")
    for text in (
        "The heatmaps are not illustrations or interpolated stock textures.",
        "weighted self-adjoint repayment engine",
        "Warm solve timing excludes archive decode",
        "maximum retained-reference error",
        "do not override the independent held-out continuum errors",
    ):
        assert text in report
