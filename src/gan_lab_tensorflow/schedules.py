from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TTURSchedule:
    generator_lr: float = 1e-4
    discriminator_lr: float = 4e-4
    warmup_steps: int = 0
    decay: float = 0.0

    def validate(self) -> None:
        if self.generator_lr <= 0 or self.discriminator_lr <= 0:
            raise ValueError("learning rates must be positive")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps cannot be negative")
        if self.decay < 0:
            raise ValueError("decay cannot be negative")

    def value_at(self, step: int) -> tuple[float, float]:
        self.validate()
        if step < 0:
            raise ValueError("step cannot be negative")
        warmup = 1.0
        if self.warmup_steps:
            warmup = min(1.0, max(step, 1) / self.warmup_steps)
        decay_factor = 1.0 / (1.0 + self.decay * step)
        return (
            self.generator_lr * warmup * decay_factor,
            self.discriminator_lr * warmup * decay_factor,
        )


def discriminator_steps_for_phase(step: int, *, warmup_until: int = 500, base_steps: int = 1) -> int:
    if step < 0:
        raise ValueError("step cannot be negative")
    if warmup_until < 0 or base_steps <= 0:
        raise ValueError("invalid phase configuration")
    return base_steps + 1 if step < warmup_until else base_steps

