"""Boundary geometry utilities."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def as_points(points: ArrayLike) -> FloatArray:
    """Return points as an ``(n, 2)`` float array."""

    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    if len(arr) < 3:
        raise ValueError("at least three boundary points are required")
    if not np.all(np.isfinite(arr)):
        raise ValueError("points contain non-finite values")
    return arr


def close_curve(points: ArrayLike) -> FloatArray:
    """Return a copy with the first point repeated at the end."""

    pts = as_points(points)
    if np.linalg.norm(pts[0] - pts[-1]) <= 1e-14:
        return pts.copy()
    return np.vstack([pts, pts[0]])


def polygon_area(points: ArrayLike) -> float:
    """Signed area of a closed polygonal boundary."""

    pts = as_points(points)
    x = pts[:, 0]
    y = pts[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def perimeter(points: ArrayLike) -> float:
    """Perimeter of a closed polygonal boundary."""

    pts = as_points(points)
    return float(np.linalg.norm(np.roll(pts, -1, axis=0) - pts, axis=1).sum())


def centroid(points: ArrayLike) -> FloatArray:
    """Area centroid when possible, otherwise arithmetic centroid."""

    pts = as_points(points)
    area = polygon_area(pts)
    if abs(area) < 1e-14:
        return pts.mean(axis=0)
    x = pts[:, 0]
    y = pts[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    cx = np.sum((x + np.roll(x, -1)) * cross) / (6.0 * area)
    cy = np.sum((y + np.roll(y, -1)) * cross) / (6.0 * area)
    return np.array([cx, cy], dtype=np.float64)


def arclength_parameter(points: ArrayLike) -> FloatArray:
    """Cumulative arclength parameter on a closed curve."""

    pts = close_curve(points)
    ds = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(ds)])


def resample_closed_curve(points: ArrayLike, n: int) -> FloatArray:
    """Resample a closed polygonal curve at equally spaced arclength values."""

    if n < 3:
        raise ValueError("n must be at least 3")
    pts = close_curve(points)
    s = arclength_parameter(pts[:-1])
    total = s[-1]
    if total <= 0:
        raise ValueError("degenerate curve has zero length")
    target = np.linspace(0.0, total, n, endpoint=False)
    out = np.empty((n, 2), dtype=np.float64)
    for dim in range(2):
        out[:, dim] = np.interp(target, s, pts[:, dim])
    return out


def curvature_periodic(points: ArrayLike) -> FloatArray:
    """Estimate signed curvature for approximately arclength-spaced samples."""

    pts = as_points(points)
    n = len(pts)
    ds = perimeter(pts) / n
    dx = (np.roll(pts[:, 0], -1) - np.roll(pts[:, 0], 1)) / (2.0 * ds)
    dy = (np.roll(pts[:, 1], -1) - np.roll(pts[:, 1], 1)) / (2.0 * ds)
    ddx = (np.roll(pts[:, 0], -1) - 2.0 * pts[:, 0] + np.roll(pts[:, 0], 1)) / ds**2
    ddy = (np.roll(pts[:, 1], -1) - 2.0 * pts[:, 1] + np.roll(pts[:, 1], 1)) / ds**2
    speed2 = dx * dx + dy * dy
    denom = np.maximum(speed2 ** 1.5, 1e-15)
    return (dx * ddy - dy * ddx) / denom


def hausdorff_distance(a: ArrayLike, b: ArrayLike) -> float:
    """Symmetric discrete Hausdorff distance."""

    pa = as_points(a)
    pb = as_points(b)
    diff = pa[:, None, :] - pb[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    return float(max(dist.min(axis=1).max(), dist.min(axis=0).max()))


@dataclass(frozen=True)
class BoundaryCurve:
    """A sampled closed planar boundary."""

    points: FloatArray

    def __post_init__(self) -> None:
        object.__setattr__(self, "points", as_points(self.points).copy())

    @classmethod
    def from_iterable(cls, points: Iterable[Iterable[float]]) -> BoundaryCurve:
        return cls(np.asarray(list(points), dtype=np.float64))

    @property
    def n(self) -> int:
        return int(len(self.points))

    @property
    def area(self) -> float:
        return polygon_area(self.points)

    @property
    def perimeter(self) -> float:
        return perimeter(self.points)

    @property
    def centroid(self) -> FloatArray:
        return centroid(self.points)

    def resample(self, n: int) -> BoundaryCurve:
        return BoundaryCurve(resample_closed_curve(self.points, n))

    def centered(self) -> BoundaryCurve:
        return BoundaryCurve(self.points - self.centroid)

    def normalized(self, area: float = 1.0) -> BoundaryCurve:
        centered = self.centered().points
        current = abs(polygon_area(centered))
        if current <= 0:
            raise ValueError("cannot normalize a zero-area curve")
        return BoundaryCurve(centered * np.sqrt(area / current))

    def curvature(self) -> FloatArray:
        return curvature_periodic(self.points)


@dataclass(frozen=True)
class StarShapeModel:
    """Low-mode star-shaped radial Fourier model.

    The radial function is

    ``r(theta) = base_radius + sum_k cos[k-1] cos(k theta) + sin[k-1] sin(k theta)``.
    """

    center: FloatArray
    base_radius: float
    cos: FloatArray
    sin: FloatArray

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=np.float64)
        if center.shape != (2,):
            raise ValueError("center must have shape (2,)")
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "cos", np.asarray(self.cos, dtype=np.float64))
        object.__setattr__(self, "sin", np.asarray(self.sin, dtype=np.float64))
        if self.cos.shape != self.sin.shape:
            raise ValueError("cos and sin coefficient arrays must have the same shape")
        if self.base_radius <= 0:
            raise ValueError("base_radius must be positive")

    @property
    def modes(self) -> int:
        return int(len(self.cos))

    def radius(self, theta: ArrayLike) -> FloatArray:
        th = np.asarray(theta, dtype=np.float64)
        r = np.full_like(th, self.base_radius, dtype=np.float64)
        for idx, (a, b) in enumerate(zip(self.cos, self.sin, strict=True), start=1):
            r += a * np.cos(idx * th) + b * np.sin(idx * th)
        return r

    def boundary_points(self, n: int = 256) -> FloatArray:
        theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        r = self.radius(theta)
        if np.any(r <= 0):
            raise ValueError("model is not star-shaped: radial function became non-positive")
        return self.center + np.column_stack([r * np.cos(theta), r * np.sin(theta)])
