import json
from pathlib import Path

from inverse_shape.arbitrary_surface import CertifiedArbitrarySurfaceQJet
from inverse_shape.riesz_near_linear import ProductionRieszQJet


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "outputs" / "arbitrary_surface" / "summary.json"


def test_public_arbitrary_surface_backend_is_production_wspd() -> None:
    assert CertifiedArbitrarySurfaceQJet is ProductionRieszQJet
    assert not hasattr(ProductionRieszQJet, "apply_direct")


def test_generated_arbitrary_surface_production_gates_pass() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert summary["method"] == "fixed_order_symmetric_gegenbauer_riesz_wspd"
    assert summary["complexity"] == {
        "compile": "O(N log^2 N) for fixed order in 3D",
        "apply": "O(N log N) for fixed order in 3D",
        "storage": "O(N) for fixed order in 3D",
    }
    assert summary["maximum_relative_error"] < 5.0e-14
    assert summary["fits"]["hierarchy_apply_exponent"] < 1.9
    assert summary["fits"]["streamed_direct_exponent"] > 1.9
    assert summary["gates"]["passed"] is True
    assert summary["stored_dense_matrix"] is False
    assert all(
        row["quadratic_fallback"] is False
        and row["dense_matrix_stored"] is False
        for row in (*summary["shape_cases"], *summary["surface_scaling"])
    )
