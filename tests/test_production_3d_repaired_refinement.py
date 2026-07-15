import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "production_3d_repaired_refinement"


def test_repaired_refinement_artifact_uses_independent_modes() -> None:
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    rows = list(
        csv.DictReader((OUT / "held_out_mode_rows.csv").open(encoding="utf-8"))
    )
    assert summary["all_final_modes_independent"] is True
    assert summary["maximum_compiled_degree"] == 2
    assert summary["maximum_adaptively_inspected_degree"] == 3
    assert all(int(row["held_out_degree"]) in (4, 5) for row in rows)
    assert all(row["used_in_compilation_or_selection"] == "False" for row in rows)
    assert all(
        int(row["held_out_degree"])
        > int(row["maximum_adaptively_inspected_degree"])
        for row in rows
        if row["shape"] != "exact_sphere"
    )


def test_repaired_refinement_is_watertight_linear_storage_evidence() -> None:
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    geometry = list(
        csv.DictReader((OUT / "geometry_rows.csv").open(encoding="utf-8"))
    )
    assert summary["all_repaired_surfaces_production_ready"] is True
    assert summary["all_repaired_surfaces_watertight"] is True
    assert summary["all_repaired_surfaces_edge_manifold"] is True
    assert summary["all_repaired_surfaces_vertex_manifold"] is True
    assert summary["all_curved_panel_atlases_watertight"] is True
    assert summary["maximum_curved_panel_seam_gap"] < 2.0e-12
    assert summary["dense_q_matrix_stored"] is False
    assert summary["pair_table_stored"] is False
    assert all(row["boundary_edges"] == "0" for row in geometry)
    assert all(row["nonmanifold_edges"] == "0" for row in geometry)
    assert all(row["nonmanifold_vertices"] == "0" for row in geometry)
    assert all(
        float(row["maximum_panel_seam_gap"]) < 2.0e-12
        for row in geometry
    )
    assert all(row["dense_q_matrix_stored"] == "False" for row in geometry)
    assert all(row["pair_table_stored"] == "False" for row in geometry)


def test_repaid_holdouts_refine_and_feature_moments_close() -> None:
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    aggregate = list(
        csv.DictReader((OUT / "aggregate_rows.csv").open(encoding="utf-8"))
    )
    assert summary["exact_sphere_repaid_endpoint_rate"] > 0.5
    feature_rows = list(
        csv.DictReader((OUT / "feature_channel_rows.csv").open(encoding="utf-8"))
    )
    assert len(feature_rows) == 4 * (12 + 8)
    assert summary["maximum_feature_repaid_absolute_error"] < 1.0e-5
    assert summary["maximum_cube_vertex_kondratiev_error"] < 0.05
    assert summary["cube_vertex_kondratiev_spread"] < 1.0e-11
    assert all(
        float(row["repaid_absolute_error"])
        < float(row["raw_absolute_error"])
        for row in feature_rows
    )
    assert all(row["compiled_reference_order"] == "12" for row in feature_rows)
    assert all(row["independent_reference_order"] == "16" for row in feature_rows)
    for shape in ("exact_sphere", "exact_ellipsoid", "funky_pn"):
        rows = sorted(
            (row for row in aggregate if row["shape"] == shape),
            key=lambda row: int(row["nodes"]),
        )
        assert float(rows[-1]["maximum_repaid_held_out_error"]) < float(
            rows[0]["maximum_repaid_held_out_error"]
        )
    assert summary["universal_machine_precision_claim"] is False
