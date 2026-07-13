import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "production_3d_qjet_method.ipynb"


def test_production_3d_notebook_is_executed_and_covers_the_proofs() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    source = "\n".join(
        "".join(cell.get("source", ())) for cell in notebook["cells"]
    )
    for required in (
        "Proposition 2.1",
        "Proposition 3.1",
        "Theorem 10.1",
        "Exact pair partition",
        "Gegenbauer source jets",
        "A posteriori compression bound",
        "Mellin-Kondrat'ev repayment",
        "What has and has not been proved",
    ):
        assert required in source
    code_cells = [
        cell for cell in notebook["cells"] if cell["cell_type"] == "code"
    ]
    assert code_cells
    assert all(cell["execution_count"] is not None for cell in code_cells)
    assert all("error" not in {output["output_type"] for output in cell["outputs"]} for cell in code_cells)


def test_production_3d_notebook_code_does_not_import_numpy_or_scipy() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    imported = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.append(node.module or "")
    assert not any(name == "numpy" or name.startswith("numpy.") for name in imported)
    assert not any(name == "scipy" or name.startswith("scipy.") for name in imported)
