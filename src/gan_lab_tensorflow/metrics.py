from __future__ import annotations

import math

from .data import Point, summarize


def moment_distance(real: list[Point], generated: list[Point]) -> float:
    real_summary = summarize(real)
    gen_summary = summarize(generated)
    dx = real_summary.mean_x - gen_summary.mean_x
    dy = real_summary.mean_y - gen_summary.mean_y
    spread_x = (real_summary.max_x - real_summary.min_x) - (gen_summary.max_x - gen_summary.min_x)
    spread_y = (real_summary.max_y - real_summary.min_y) - (gen_summary.max_y - gen_summary.min_y)
    return math.sqrt(dx * dx + dy * dy + 0.05 * spread_x * spread_x + 0.05 * spread_y * spread_y)


def coverage_ratio(real: list[Point], generated: list[Point], *, bins: int = 12) -> float:
    if bins <= 0:
        raise ValueError("bins must be positive")
    real_summary = summarize(real)
    width = max((real_summary.max_x - real_summary.min_x) / bins, 1e-9)

    def bucket(point: Point) -> int:
        x, _ = point
        return max(0, min(bins - 1, int((x - real_summary.min_x) / width)))

    real_buckets = {bucket(point) for point in real}
    gen_buckets = {bucket(point) for point in generated if real_summary.min_x <= point[0] <= real_summary.max_x}
    return len(gen_buckets & real_buckets) / len(real_buckets)

