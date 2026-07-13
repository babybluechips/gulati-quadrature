import ast
import json
from pathlib import Path

from scripts import production_3d_shape_campaign as campaign

from inverse_shape.arbitrary_surface import triangle_lumped_vertex_weights
from inverse_shape.golden_hyperbolic_atlas import GoldenHyperbolicJetAtlas
from inverse_shape.testing.reference_pairwise import (
    reference_weighted_distance_graph,
)


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "outputs" / "production_3d_shape_campaign" / "summary.json"


def test_campaign_has_no_external_numerical_dependency() -> None:
    with open(campaign.__file__, encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    assert "numpy" not in imported
    assert "scipy" not in imported


def test_refined_cube_and_aircraft_meshes_are_closed_and_noncolliding() -> None:
    cube = campaign.refine_triangles(*campaign.cube_mesh(False), 2)
    assert len(cube[0]) == 26
    assert len(cube[1]) == 48
    assert len(set(cube[0])) == len(cube[0])
    assert all(value > 0.0 for value in triangle_lumped_vertex_weights(*cube))

    aircraft = campaign.polyhedral_aircraft_mesh()
    refined = campaign.refine_triangles(*aircraft, 2)
    assert len(set(refined[0])) == len(refined[0])
    assert all(
        value > 0.0 for value in triangle_lumped_vertex_weights(*refined)
    )
    x_values = tuple(point[0] for point in refined[0])
    y_values = tuple(point[1] for point in refined[0])
    assert max(x_values) - min(x_values) > 4.0
    assert max(y_values) - min(y_values) < 0.5


def test_mobius_seam_has_positive_lumped_weights() -> None:
    points, triangles = campaign.mobius_strip(12, 5)
    assert len(set(points)) == len(points)
    weights = triangle_lumped_vertex_weights(points, triangles)
    assert all(value > 0.0 for value in weights)


def test_golden_axisymmetric_small_case_matches_isolated_oracle() -> None:
    coordinates = campaign._axis_coordinates(9, 0.5)
    radius_jets, height_jets = campaign._axisymmetric_jets(
        "sphere_cap",
        coordinates,
    )
    atlas = GoldenHyperbolicJetAtlas(
        coordinates,
        radius_jets,
        height_jets,
    )
    assert len(atlas.patches) == 1
    qjet = atlas.patches[0].qjet(8, 8)
    points, weights = campaign._axisymmetric_points(qjet)
    values = campaign.physical_field(points)
    grid = tuple(
        values[index * qjet.n_theta : (index + 1) * qjet.n_theta]
        for index in range(qjet.n_scale)
    )
    candidate = tuple(value for row in qjet.apply(grid) for value in row)
    reference = reference_weighted_distance_graph(
        points,
        weights,
        values,
        2.0,
    )
    assert campaign.relative_error(reference, candidate) < 2.0e-13
    stats = qjet.stats()
    assert stats["quadratic_fallback"] is False
    assert stats["stored_dense_operator_matrix"] is False


def test_generated_3d_campaign_passes_universal_subquadratic_gate() -> None:
    summary = json.loads(SUMMARY.read_text(encoding="utf-8"))
    assert summary["universal_hard_no_quadratic_contract"] is True
    assert summary["gates"] == {
        "machine_scale_accuracy": True,
        "all_measured_apply_fits_subquadratic": True,
        "hard_no_quadratic_contract": True,
        "no_dense_matrix": True,
        "passed": True,
    }
    assert max(summary["maximum_errors"].values()) < 1.0e-13
    assert all(
        fit["apply"] < 1.9 for fit in summary["scaling_fits"].values()
    )
    production_rows = (
        *summary["axisymmetric_rows"],
        *summary["conic_rows"],
        *summary["polyhedral_rows"],
        *summary["unstructured_curved_rows"],
    )
    assert all(
        row["hard_no_quadratic_contract"]
        and not row["quadratic_fallback"]
        and not row["stored_dense_matrix"]
        for row in production_rows
    )
