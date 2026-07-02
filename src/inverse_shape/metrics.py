"""Alignment and reconstruction diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from inverse_shape.geometry import as_points, hausdorff_distance

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class AlignmentResult:
    """Similarity alignment result."""

    points: FloatArray
    scale: float
    rotation: FloatArray
    translation: FloatArray
    rms_error: float
    hausdorff_error: float


@dataclass(frozen=True)
class RigidRotationSearchResult:
    """Rigid rotation alignment selected by discrete Hausdorff distance."""

    points: FloatArray
    angle: float
    translation: FloatArray
    hausdorff_error: float


def procrustes_align(
    source: ArrayLike,
    target: ArrayLike,
    *,
    allow_reflection: bool = True,
) -> AlignmentResult:
    """Align ``source`` to ``target`` by a least-squares similarity transform."""

    src = as_points(source)
    dst = as_points(target)
    if src.shape != dst.shape:
        raise ValueError("source and target must have the same shape")

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    x = src - src_mean
    y = dst - dst_mean
    norm_x2 = float(np.sum(x * x))
    if norm_x2 <= 0:
        raise ValueError("source has zero variance")

    u, singular, vt = np.linalg.svd(x.T @ y)
    rotation = u @ vt
    if not allow_reflection and np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        singular[-1] *= -1.0
        rotation = u @ vt
    scale = float(np.sum(singular) / norm_x2)
    aligned = scale * (x @ rotation) + dst_mean
    diff = aligned - dst
    rms = float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))
    return AlignmentResult(
        points=aligned,
        scale=scale,
        rotation=rotation,
        translation=dst_mean - scale * (src_mean @ rotation),
        rms_error=rms,
        hausdorff_error=hausdorff_distance(aligned, dst),
    )


def rigid_rotation_hausdorff_align(
    source: ArrayLike,
    target: ArrayLike,
    *,
    rotations: int = 720,
) -> RigidRotationSearchResult:
    """Align by scanning rigid rotations and minimizing Hausdorff distance.

    This is useful for spectrum-only reconstructions where the boundary is
    determined only up to Euclidean motion and point labels have no meaning.
    """

    if rotations < 8:
        raise ValueError("rotations must be at least 8")
    src = as_points(source)
    dst = as_points(target)
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    centered = src - src_mean

    best_points: FloatArray | None = None
    best_angle = 0.0
    best_error = np.inf
    for angle in np.linspace(0.0, 2.0 * np.pi, rotations, endpoint=False):
        c = float(np.cos(angle))
        s = float(np.sin(angle))
        rotation = np.array([[c, -s], [s, c]], dtype=np.float64)
        aligned = centered @ rotation.T + dst_mean
        error = hausdorff_distance(aligned, dst)
        if error < best_error:
            best_error = error
            best_angle = float(angle)
            best_points = aligned

    if best_points is None:
        raise RuntimeError("rotation search failed")
    return RigidRotationSearchResult(
        points=best_points,
        angle=best_angle,
        translation=dst_mean - src_mean,
        hausdorff_error=float(best_error),
    )
