import json
from pathlib import Path

from bs4 import BeautifulSoup

from scripts.production_3d_qjet_extended_validation import (
    airplane_assembly,
    bridge_assembly,
    car_assembly,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "production_3d_qjet_html"
HTML_REPORT = OUTPUT / "production_3d_qjet_method.html"
SUMMARY = OUTPUT / "validation_summary.json"
PDE_OUTPUT = ROOT / "outputs" / "production_3d_pde_validation"
PDE_SUMMARY = PDE_OUTPUT / "summary.json"
CAD_PDE_OUTPUT = ROOT / "outputs" / "production_3d_cad_pde_validation"
CAD_PDE_SUMMARY = CAD_PDE_OUTPUT / "summary.json"
REPAIRED_OUTPUT = ROOT / "outputs" / "production_3d_repaired_refinement"
REPAIRED_SUMMARY = REPAIRED_OUTPUT / "summary.json"
NASA_VIS_OUTPUT = ROOT / "outputs" / "production_nasa_cad_pde_visualization"
NASA_VIS_SUMMARY = NASA_VIS_OUTPUT / "summary.json"
TAIL_OUTPUT = ROOT / "outputs" / "transparent_tail_benchmark"
TAIL_SUMMARY = TAIL_OUTPUT / "summary.json"


def load_summary() -> dict[str, object]:
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


def test_extended_validation_covers_distinct_3d_geometry_classes() -> None:
    summary = load_summary()
    assert summary["shape_count"] == 16
    assert summary["case_count"] == 32
    assert summary["field_comparisons"] == 96

    cases = summary["cases"]
    shapes = {case["shape"] for case in cases}
    assert len(shapes) == 16
    assert {"airplane assembly", "car assembly", "suspension bridge"} <= shapes
    assert all(
        {case["kernel_power"] for case in cases if case["shape"] == shape}
        == {2, 3}
        for shape in shapes
    )
    categories = set(summary["categories"])
    for required in (
        "closed smooth genus 0",
        "closed smooth genus 1",
        "closed polyhedral",
        "open nonorientable",
        "twisted conic atlas",
        "airplane composite conic surface",
        "automotive composite surface",
        "architectural composite surface",
        "near-collision stress",
        "dynamic-range stress",
    ):
        assert required in categories


def test_extended_validation_meets_accuracy_and_structural_gates() -> None:
    summary = load_summary()
    assert summary["gates"]["passed"] is True
    assert all(summary["gates"].values())
    assert summary["maximum_standard_relative_error"] < 1.0e-12
    assert summary["maximum_stress_relative_error"] < 2.0e-12
    assert summary["maximum_certificate_ratio"] <= 1.0
    assert summary["maximum_transformation_residual"] < 2.0e-12
    assert summary["all_constant_residuals_zero"] is True
    assert summary["all_hard_no_quadratic"] is True
    assert summary["no_quadratic_fallback"] is True
    assert summary["no_dense_matrix"] is True
    assert summary["no_pair_table"] is True
    assert all(case["passed"] for case in summary["cases"])
    assert all(item["passed"] for item in summary["transformations"])


def test_object_generators_are_genuinely_three_dimensional_weighted_surfaces() -> None:
    for factory in (airplane_assembly, car_assembly, bridge_assembly):
        points, weights = factory()
        assert len(points) == len(weights)
        assert len(points) >= 100
        assert len(set(points)) == len(points)
        assert min(weights) > 0.0
        assert all(
            max(point[axis] for point in points)
            - min(point[axis] for point in points)
            > 0.1
            for axis in range(3)
        )

    summary = load_summary()
    object_rows = [
        row
        for row in summary["cases"]
        if row["shape"]
        in {"airplane assembly", "car assembly", "suspension bridge"}
    ]
    assert len(object_rows) == 6
    assert max(row["maximum_relative_error"] for row in object_rows) < 1.0e-12
    assert all(row["passed"] for row in object_rows)
    assert not any(row["stored_dense_matrix"] for row in object_rows)
    assert not any(row["quadratic_fallback"] for row in object_rows)


def test_html_is_a_complete_executed_single_file_report() -> None:
    assert HTML_REPORT.stat().st_size > 500_000
    source = HTML_REPORT.read_text(encoding="utf-8")
    soup = BeautifulSoup(source, "html.parser")

    assert soup.title is not None
    assert "Production 3D QJet" in soup.title.get_text()
    text = soup.get_text(" ", strip=True)
    for required in (
        "A matrix-free inverse-distance QJet for three-dimensional surfaces",
        "centered graph-difference evaluation",
        "Extended rigorous validation",
        "What has and has not been proved",
        "16 geometries",
        "96 independent field comparisons",
    ):
        assert required in text

    assert soup.find("style", id="qjet-report-style") is not None
    assert len(soup.find_all("table")) >= 7
    embedded_svgs = [
        image
        for image in soup.find_all("img")
        if (image.get("src") or "").startswith("data:image/svg+xml;base64,")
    ]
    assert len(embedded_svgs) >= 3
    assert source.count("$$") >= 40
    raw_display_math = [
        paragraph
        for paragraph in soup.find_all("p")
        if paragraph.get_text(strip=True).startswith("[")
        and "\\" in paragraph.get_text()
    ]
    assert not raw_display_math
    assert "/Users/rick" not in source
    assert "output_type&quot;: &quot;error&quot;" not in source
    assert "Traceback (most recent call last)" not in source
    assert not soup.find_all("img", src=lambda value: value and not value.startswith("data:"))


def test_html_embeds_independent_discrete_and_continuum_pde_audits() -> None:
    summary = json.loads(PDE_SUMMARY.read_text(encoding="utf-8"))
    assert summary["all_discrete_gates_passed"] is True
    assert summary["discrete_case_count"] == 30
    assert summary["dense_operator_stored"] is False
    assert summary["quadratic_fallback"] is False
    assert summary["continuum_machine_precision_claim"] is False

    soup = BeautifulSoup(HTML_REPORT.read_text(encoding="utf-8"), "html.parser")
    section = soup.find("section", id="boundary-pde-validation")
    assert section is not None
    text = section.get_text(" ", strip=True)
    for required in (
        "Boundary PDE validation",
        "Laplace DtN application",
        "screened Poisson/Yukawa",
        "damped boundary Helmholtz resolvent",
        "independently streamed pairwise",
        "unrepaid baseline",
        "separate held-out continuum column",
    ):
        assert required in text
    assert len(section.find_all("table")) == 2


def test_html_embeds_exact_transparent_tail_closure_without_cad_overclaim() -> None:
    summary = json.loads(TAIL_SUMMARY.read_text(encoding="utf-8"))
    assert summary["all_gates_passed"] is True
    assert summary["autonomous_tail_truncation_error_with_cap"] == 0.0
    assert summary["held_out_cad_machine_precision_claim"] is False

    soup = BeautifulSoup(HTML_REPORT.read_text(encoding="utf-8"), "html.parser")
    section = soup.find("section", id="transparent-tail-closure")
    assert section is not None
    text = section.get_text(" ", strip=True)
    for required in (
        "Exact transparent closure of cylindrical and conic tails",
        "chi(Phi(sigma)) = w^2 chi(sigma)",
        "zero autonomous-tail truncation error",
        "Undamped propagating Helmholtz modes are rejected",
        "does not recover surface channels discarded",
    ):
        assert required in text
    assert len(section.find_all("table")) == 1
    assert section.find("svg") is not None


def test_html_embeds_repaid_cad_pde_and_held_out_audits() -> None:
    summary = json.loads(CAD_PDE_SUMMARY.read_text(encoding="utf-8"))
    assert summary["model_count"] == 5
    assert summary["pde_case_count"] == 30
    assert summary["source_face_count"] == 1_261_986
    assert summary["all_source_faces_scanned"] is True
    assert summary["all_pde_gates_passed"] is True
    assert summary["maximum_compiled_reference_error"] < 1.0e-11
    assert summary["maximum_algebraic_residual"] < 1.0e-7
    assert summary["maximum_held_out_continuum_error"] > 1.0
    assert summary["universal_machine_precision_claim"] is False
    assert summary["dense_q_matrix_stored"] is False
    assert summary["pair_table_stored"] is False

    soup = BeautifulSoup(HTML_REPORT.read_text(encoding="utf-8"), "html.parser")
    section = soup.find("section", id="cad-boundary-pde-validation")
    assert section is not None
    text = section.get_text(" ", strip=True)
    for required in (
        "Continuum-repaid CAD PDE campaign",
        "curvature-adjusted omitted-cell series",
        "1,261,986 source triangles",
        "Held-out continuum result",
        "no universal 3D machine-precision claim",
        "The earlier 1.779e-8 number measured discrete implementation and PDE algebra",
    ):
        assert required in text
    assert len(section.find_all("table")) == 4


def test_html_embeds_nasa_compressed_pde_fields_and_scope() -> None:
    summary = json.loads(NASA_VIS_SUMMARY.read_text(encoding="utf-8"))
    assert summary["shape_count"] == 3
    assert summary["pde_case_count"] == 18
    assert summary["all_declared_gates_passed"] is True
    assert summary["maximum_retained_reference_error"] < 1.0e-14
    assert summary["maximum_algebraic_residual"] < 2.0e-14
    assert summary["maximum_warm_solve_ms"] < 10_000.0
    assert summary["dense_q_matrix_stored"] is False
    assert summary["held_out_continuum_machine_precision_claim"] is False

    soup = BeautifulSoup(HTML_REPORT.read_text(encoding="utf-8"), "html.parser")
    section = soup.find("section", id="nasa-cad-pde-fields")
    assert section is not None
    text = section.get_text(" ", strip=True)
    for required in (
        "Numerical PDE fields on compressed NASA CAD",
        "1,227,262 losslessly decoded source triangles",
        "24 to 42 compressed boundary nodes",
        "self-adjoint heat/wave denominator residual",
        "Machine-scale values certify retained manufactured channels",
        "not held-out continuum accuracy",
    ):
        assert required in text
    assert len(section.find_all("table")) == 2
    images = section.find_all("img")
    assert len(images) == 4
    assert all(image["src"].startswith("data:image/png;base64,") for image in images)


def test_html_embeds_repaired_manifold_and_independent_refinement() -> None:
    summary = json.loads(REPAIRED_SUMMARY.read_text(encoding="utf-8"))
    assert summary["all_final_modes_independent"] is True
    assert summary["all_repaired_surfaces_production_ready"] is True
    assert summary["all_repaired_surfaces_watertight"] is True
    assert summary["all_repaired_surfaces_edge_manifold"] is True
    assert summary["all_repaired_surfaces_vertex_manifold"] is True
    assert summary["all_curved_panel_atlases_watertight"] is True
    assert summary["maximum_curved_panel_seam_gap"] < 2.0e-12
    assert summary["maximum_feature_repaid_absolute_error"] < 1.0e-5
    assert summary["maximum_cube_vertex_kondratiev_error"] < 0.05
    assert summary["cube_vertex_kondratiev_spread"] < 1.0e-11
    assert summary["exact_sphere_repaid_endpoint_rate"] > 0.5

    soup = BeautifulSoup(HTML_REPORT.read_text(encoding="utf-8"), "html.parser")
    section = soup.find("section", id="repaired-curved-refinement")
    assert section is not None
    text = section.get_text(" ", strip=True)
    for required in (
        "Watertight repair and independent curved-panel refinement",
        "cubic PN atlas",
        "odd principal-value tangent-moment correction",
        "sparse spherical-link pencil",
        "order-sixteen rule",
        "Modes excluded from fitting and model selection",
        "degree four or five",
        "not at machine precision",
    ):
        assert required in text
    assert len(section.find_all("table")) == 3
