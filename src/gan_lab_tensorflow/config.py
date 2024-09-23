from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    noise_dim: int = 2
    data_dim: int = 2
    hidden_units: tuple[int, ...] = (64, 64)
    activation: str = "leaky_relu"
    image_shape: tuple[int, int, int] = (28, 28, 1)


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 128
    steps: int = 10_000
    learning_rate: float = 1e-4
    discriminator_steps: int = 1
    generator_steps: int = 1
    log_every: int = 100
    snapshot_every: int = 500
    loss: str = "vanilla"
    gradient_penalty_weight: float = 10.0
    seed: int = 42

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.loss not in {"vanilla", "wgan-gp"}:
            raise ValueError("loss must be either 'vanilla' or 'wgan-gp'")


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: str = "quadratic"
    output_dir: Path = field(default_factory=lambda: Path("outputs/quadratic"))
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def validate(self) -> None:
        if self.dataset not in {"quadratic", "sine", "mixture"}:
            raise ValueError("dataset must be one of: quadratic, sine, mixture")
        self.training.validate()

