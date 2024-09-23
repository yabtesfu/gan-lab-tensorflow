from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


def one_hot(labels: Iterable[int], num_classes: int) -> list[list[float]]:
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    encoded: list[list[float]] = []
    for label in labels:
        if label < 0 or label >= num_classes:
            raise ValueError(f"label {label} is outside 0..{num_classes - 1}")
        row = [0.0] * num_classes
        row[label] = 1.0
        encoded.append(row)
    return encoded


def attach_labels(samples: list[tuple[float, ...]], labels: list[int], num_classes: int) -> list[tuple[float, ...]]:
    if len(samples) != len(labels):
        raise ValueError("samples and labels must have the same length")
    encoded = one_hot(labels, num_classes)
    return [tuple(sample) + tuple(label) for sample, label in zip(samples, encoded)]


@dataclass(frozen=True)
class ConditionalSpec:
    noise_dim: int
    num_classes: int

    @property
    def generator_input_dim(self) -> int:
        return self.noise_dim + self.num_classes

    def validate(self) -> None:
        if self.noise_dim <= 0:
            raise ValueError("noise_dim must be positive")
        if self.num_classes <= 1:
            raise ValueError("num_classes must be greater than one")


def build_conditional_generator(base_builder, spec: ConditionalSpec):
    spec.validate()
    model_config = base_builder.__self__ if hasattr(base_builder, "__self__") else None
    if model_config is not None:
        raise TypeError("pass a builder function, not a bound method")
    return base_builder(spec.generator_input_dim)

