from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Iterable


Point = tuple[float, float]


def quadratic_y(x: float) -> float:
    return 10.0 + x * x


def sine_y(x: float) -> float:
    return 4.0 * math.sin(x / 3.0) + 0.25 * x


def sample_curve(
    count: int,
    *,
    scale: float = 20.0,
    noise: float = 0.0,
    fn: Callable[[float], float] = quadratic_y,
    seed: int | None = None,
) -> list[Point]:
    rng = random.Random(seed)
    samples: list[Point] = []
    for _ in range(count):
        x = scale * (rng.random() - 0.5)
        y = fn(x) + rng.gauss(0.0, noise)
        samples.append((x, y))
    return samples


def sample_mixture(count: int, *, seed: int | None = None) -> list[Point]:
    rng = random.Random(seed)
    centers = [(-4.0, -1.5), (0.0, 3.0), (4.0, -1.0)]
    samples: list[Point] = []
    for _ in range(count):
        cx, cy = centers[rng.randrange(len(centers))]
        samples.append((rng.gauss(cx, 0.55), rng.gauss(cy, 0.55)))
    return samples


def sample_ring(
    count: int, *, modes: int = 8, radius: float = 6.0, std: float = 0.35, seed: int | None = None
) -> list[Point]:
    """The classic ``modes``-Gaussian ring benchmark.

    Gaussians spaced evenly on a circle. GANs famously mode-collapse here --
    dropping to one or a few of the ring's blobs -- which makes it the standard
    stress test for mode coverage.
    """
    if modes <= 0:
        raise ValueError("modes must be positive")
    centers = [
        (radius * math.cos(2 * math.pi * k / modes), radius * math.sin(2 * math.pi * k / modes))
        for k in range(modes)
    ]
    rng = random.Random(seed)
    samples: list[Point] = []
    for _ in range(count):
        cx, cy = centers[rng.randrange(modes)]
        samples.append((rng.gauss(cx, std), rng.gauss(cy, std)))
    return samples


def sample_noise(count: int, dim: int, *, seed: int | None = None) -> list[tuple[float, ...]]:
    rng = random.Random(seed)
    return [tuple(rng.uniform(-1.0, 1.0) for _ in range(dim)) for _ in range(count)]


def batches(items: list[Point], batch_size: int) -> Iterable[list[Point]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


@dataclass(frozen=True)
class DatasetSummary:
    count: int
    mean_x: float
    mean_y: float
    min_x: float
    max_x: float
    min_y: float
    max_y: float


def summarize(points: list[Point]) -> DatasetSummary:
    if not points:
        raise ValueError("cannot summarize an empty dataset")
    xs = [x for x, _ in points]
    ys = [y for _, y in points]
    return DatasetSummary(
        count=len(points),
        mean_x=sum(xs) / len(xs),
        mean_y=sum(ys) / len(ys),
        min_x=min(xs),
        max_x=max(xs),
        min_y=min(ys),
        max_y=max(ys),
    )

