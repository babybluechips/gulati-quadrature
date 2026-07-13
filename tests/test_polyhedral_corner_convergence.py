import ast

from scripts import polyhedral_corner_convergence as campaign


def test_corner_campaign_has_no_external_numerical_dependency() -> None:
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


def test_continuum_edge_and_vertex_channels_recover_expected_orders() -> None:
    summary = campaign.run_campaign(write_outputs=False)
    edge = summary["edge_layer_case"]
    vertex = summary["vertex_layer_case"]
    assert summary["topology"]["reentrant_edges"] == 3
    assert 0.62 < edge["raw_fitted_order"] < 0.73
    assert edge["corrected_fitted_order"] > 4.5
    assert edge["rows"][-1]["corrected_abs_error"] < 2.0e-15
    assert 1.38 < vertex["raw_fitted_order"] < 1.52
    assert vertex["corrected_fitted_order"] > 5.2
    assert vertex["rows"][-1]["corrected_abs_error"] < 8.0e-14
    assert edge["reference_crosscheck_abs"] < 3.0e-15
    assert vertex["reference_crosscheck_abs"] < 3.0e-15
    extrapolation = summary["spherical_pencil_extrapolation"]
    assert extrapolation["exponent_abs_error"] < 2.0e-4
    assert extrapolation["coupled_vertex_abs_error_at_512"] < 8.0e-9


def test_fichera_spherical_link_refines_without_dense_matrices() -> None:
    rows = campaign.spherical_pencil_convergence()
    campaign.couple_vertex_pencil(campaign.vertex_layer_case(), rows)
    errors = tuple(row["exponent_abs_error"] for row in rows)
    assert all(
        right < left
        for left, right in zip(errors[:-1], errors[1:], strict=True)
    )
    assert errors[-1] < 3.0e-3
    coupled = tuple(row["coupled_vertex_abs_error_at_512"] for row in rows)
    assert all(
        right < left
        for left, right in zip(coupled[:-1], coupled[1:], strict=True)
    )
    assert coupled[-1] < 1.5e-7
    assert all(row["stored_dense_matrix"] is False for row in rows)
    assert all(row["sparse_nnz"] < 16 * row["spherical_nodes"] for row in rows)
