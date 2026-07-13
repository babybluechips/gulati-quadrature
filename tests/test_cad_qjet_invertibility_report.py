import hashlib
import json
from pathlib import Path

from bs4 import BeautifulSoup
from PIL import Image, ImageChops


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "cad_qjet_invertibility"


def _summary() -> dict[str, object]:
    return json.loads((OUT / "cad_roundtrip_summary.json").read_text(encoding="utf-8"))


def test_public_cad_campaign_passes_all_exactness_and_structure_gates() -> None:
    summary = _summary()
    assert summary["model_count"] == 5
    assert summary["total_vertices"] == 629_706
    assert summary["total_faces"] == 1_261_986
    assert summary["gates"]["passed"] is True
    assert all(summary["gates"].values())
    assert {row["key"] for row in summary["cases"]} == {
        "sofia_aircraft",
        "cement_mixer",
        "curiosity_rover",
        "curiosity_assembled",
        "ifc_bridge",
    }
    for row in summary["cases"]:
        assert row["exact_round_trip"] is True
        assert row["maximum_absolute_error"] == 0.0
        assert row["vertex_mismatches"] == 0
        assert row["face_mismatches"] == 0
        assert row["part_mismatches"] == 0
        assert row["deterministic_archive"] is True
        assert row["corruption_rejected"] is True
        assert row["no_dense_matrix"] is True
        assert row["no_pair_table"] is True
        assert row["linear_integer_count"] == 3 * row["vertex_count"] + 3 * row["face_count"]
        assert row["dense_pair_matrix_bytes_avoided"] == 8 * row["vertex_count"] ** 2
        assert row["peak_compile_bytes"] * 10 < row["dense_pair_matrix_bytes_avoided"]

    scaling = summary["scaling"]
    assert 0.9 < scaling["archive_exponent"] < 1.1
    assert scaling["encode_time_exponent"] < 1.5
    assert scaling["decode_time_exponent"] < 1.5
    assert len(scaling["rows"]) == 6
    for row in scaling["rows"]:
        assert row["exact_round_trip"] is True
        assert row["linear_integer_count"] == 3 * row["linear_items"]


def test_original_and_decoded_renders_are_pixel_identical() -> None:
    for key in (
        "sofia_aircraft",
        "cement_mixer",
        "curiosity_rover",
        "curiosity_assembled",
        "ifc_bridge",
    ):
        original = Image.open(OUT / f"{key}_original.png").convert("RGB")
        decoded = Image.open(OUT / f"{key}_decoded.png").convert("RGB")
        assert original.size == (460, 330)
        assert decoded.size == original.size
        difference = ImageChops.difference(original, decoded)
        assert difference.getbbox() is None


def test_reconstructed_meshes_and_native_archives_are_emitted() -> None:
    for row in _summary()["cases"]:
        archive = OUT / row["archive_path"]
        reconstruction = OUT / row["reconstruction_path"]
        assert archive.stat().st_size == row["archive_bytes"]
        assert reconstruction.stat().st_size > archive.stat().st_size
        header = reconstruction.read_bytes()[:256]
        assert header.startswith(b"ply\nformat binary_little_endian 1.0\n")


def test_standalone_html_contains_proofs_sources_results_and_inline_visuals() -> None:
    report = OUT / "cad_qjet_invertibility.html"
    assert report.stat().st_size > 200_000
    source = report.read_text(encoding="utf-8")
    soup = BeautifulSoup(source, "html.parser")
    text = soup.get_text(" ", strip=True)
    for required in (
        "Reversible sparse QJet compilation of public CAD meshes",
        "Exact residual-jet theorem",
        "NASA SOFIA aircraft",
        "FreeCAD cement mixer truck",
        "NASA Curiosity manufacturing plates",
        "NASA Curiosity single-file print layout",
        "buildingSMART IFC bridge",
        "3V + 3F + O(parts)",
        "Refinement scaling on the SOFIA mesh",
        "Max |Δx|",
        "PASS",
    ):
        assert required in text
    assert len(soup.find_all("table")) == 3
    assert len(soup.find_all("svg")) == 1
    inline_images = soup.find_all("image")
    assert len(inline_images) == 15
    assert all(
        (image.get("href") or "").startswith("data:image/png;base64,")
        for image in inline_images
    )
    assert "Traceback (most recent call last)" not in source
    assert "/Users/rick" not in source


def test_source_manifest_matches_every_downloaded_cad_file() -> None:
    benchmark = ROOT / "benchmarks" / "cad_invertibility"
    manifest = json.loads((benchmark / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    assert set(manifest) == {
        "buildingSMART_ifc_bridge",
        "freecad_cement_mixer",
        "nasa_curiosity_rover",
        "nasa_curiosity_rover_single_file",
        "nasa_sofia",
    }
    for source in manifest.values():
        assert source["source"].startswith("https://github.com/")
        assert source["license"]
        assert (benchmark / source["license_file"]).is_file()
        for item in source["files"]:
            path = benchmark / item["path"]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]


def test_main_production_3d_html_embeds_the_public_cad_audit() -> None:
    report = ROOT / "outputs" / "production_3d_qjet_html" / "production_3d_qjet_method.html"
    soup = BeautifulSoup(report.read_text(encoding="utf-8"), "html.parser")
    section = soup.find("section", id="public-cad-invertibility")
    assert section is not None
    text = section.get_text(" ", strip=True)
    for required in (
        "Public CAD invertibility audit",
        "NASA's SOFIA aircraft",
        "FreeCAD cement-mixer truck",
        "two high-resolution Curiosity manufacturing layouts",
        "buildingSMART IFC bridge scene",
        "3V + 3F + O(parts)",
        "canonical SHA-256 exactly",
        "six SOFIA prefix refinements",
        "Exact recovery passes at every refinement",
    ):
        assert required in text
    assert len(section.find_all("image")) == 15
    link = section.find("a", href="../cad_qjet_invertibility/cad_qjet_invertibility.html")
    assert link is not None
