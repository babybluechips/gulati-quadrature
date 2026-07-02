"""Input/output helpers for CLI workflows."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from inverse_shape.geometry import as_points

FloatArray = NDArray[np.float64]


def load_points_csv(path: str | Path) -> FloatArray:
    arr = np.loadtxt(path, delimiter=",", dtype=np.float64)
    return as_points(arr)


def save_points_csv(path: str | Path, points: ArrayLike) -> None:
    np.savetxt(path, as_points(points), delimiter=",", header="x,y", comments="")


def load_vector_csv(path: str | Path) -> FloatArray:
    arr = np.loadtxt(path, delimiter=",", dtype=np.float64)
    return np.asarray(arr, dtype=np.float64).reshape(-1)


def save_vector_csv(path: str | Path, values: ArrayLike, header: str = "value") -> None:
    np.savetxt(
        path,
        np.asarray(values, dtype=np.float64).reshape(-1),
        delimiter=",",
        header=header,
        comments="",
    )


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("JSON root must be an object")
    return data


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
