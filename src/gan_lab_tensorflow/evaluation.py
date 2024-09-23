from __future__ import annotations

import math

from .data import Point


def squared_distance(a: Point, b: Point) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def rbf_kernel(a: Point, b: Point, sigma: float = 1.0) -> float:
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return math.exp(-squared_distance(a, b) / (2 * sigma * sigma))


def maximum_mean_discrepancy(real: list[Point], generated: list[Point], sigma: float = 1.0) -> float:
    if not real or not generated:
        raise ValueError("real and generated samples must be non-empty")

    def average_kernel(left: list[Point], right: list[Point]) -> float:
        total = 0.0
        for a in left:
            for b in right:
                total += rbf_kernel(a, b, sigma)
        return total / (len(left) * len(right))

    mmd2 = average_kernel(real, real) + average_kernel(generated, generated) - 2 * average_kernel(real, generated)
    return math.sqrt(max(mmd2, 0.0))


def nearest_neighbor_precision(real: list[Point], generated: list[Point], radius: float) -> float:
    if radius <= 0:
        raise ValueError("radius must be positive")
    if not generated:
        raise ValueError("generated samples must be non-empty")
    hits = 0
    radius2 = radius * radius
    for sample in generated:
        if any(squared_distance(sample, real_point) <= radius2 for real_point in real):
            hits += 1
    return hits / len(generated)

