from __future__ import annotations

import random
from dataclasses import dataclass

from .data import Point


@dataclass(frozen=True)
class AugmentationPolicy:
    probability: float = 0.0
    jitter_std: float = 0.04
    x_scale_min: float = 0.96
    x_scale_max: float = 1.04
    y_shift: float = 0.0

    def validate(self) -> None:
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability must be between 0 and 1")
        if self.jitter_std < 0.0:
            raise ValueError("jitter_std cannot be negative")
        if self.x_scale_min <= 0.0 or self.x_scale_max <= 0.0:
            raise ValueError("scale bounds must be positive")
        if self.x_scale_min > self.x_scale_max:
            raise ValueError("x_scale_min cannot exceed x_scale_max")


def augment_points(points: list[Point], policy: AugmentationPolicy, *, seed: int | None = None) -> list[Point]:
    policy.validate()
    rng = random.Random(seed)
    augmented: list[Point] = []

    for x, y in points:
        if rng.random() > policy.probability:
            augmented.append((x, y))
            continue

        scale = rng.uniform(policy.x_scale_min, policy.x_scale_max)
        augmented.append(
            (
                x * scale + rng.gauss(0.0, policy.jitter_std),
                y + policy.y_shift + rng.gauss(0.0, policy.jitter_std),
            )
        )

    return augmented


def adapt_probability(
    current_probability: float,
    *,
    discriminator_real_accuracy: float,
    target_accuracy: float = 0.6,
    step_size: float = 0.02,
    max_probability: float = 0.8,
) -> float:
    if not 0.0 <= current_probability <= max_probability:
        raise ValueError("current_probability must be within the configured range")
    if not 0.0 <= discriminator_real_accuracy <= 1.0:
        raise ValueError("discriminator_real_accuracy must be between 0 and 1")
    if not 0.0 < target_accuracy < 1.0:
        raise ValueError("target_accuracy must be between 0 and 1")
    if step_size <= 0.0 or max_probability <= 0.0 or max_probability > 1.0:
        raise ValueError("invalid probability adaptation settings")

    deadband = step_size
    if discriminator_real_accuracy > target_accuracy + deadband:
        return min(max_probability, current_probability + step_size)
    if discriminator_real_accuracy < target_accuracy - deadband:
        return max(0.0, current_probability - step_size)
    return current_probability
