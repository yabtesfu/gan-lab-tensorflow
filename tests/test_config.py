from pathlib import Path

import pytest

from gan_lab_tensorflow.config import ExperimentConfig, TrainingConfig


def test_training_config_validates_positive_steps():
    config = TrainingConfig(steps=0)
    with pytest.raises(ValueError):
        config.validate()


def test_experiment_config_accepts_quadratic_dataset():
    config = ExperimentConfig(dataset="quadratic", output_dir=Path("outputs/test"))
    config.validate()

