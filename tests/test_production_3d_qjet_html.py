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
