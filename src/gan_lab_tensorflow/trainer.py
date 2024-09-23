from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import ExperimentConfig
from .data import sample_curve, sample_mixture, sample_noise, sine_y
from .losses import (
    gradient_penalty,
    vanilla_discriminator_loss,
    vanilla_generator_loss,
    wasserstein_discriminator_loss,
    wasserstein_generator_loss,
)
from .models import build_mlp_discriminator, build_mlp_generator


@dataclass
class TrainingRecord:
    step: int
    generator_loss: float
    discriminator_loss: float


class GanTrainer:
    def __init__(self, config: ExperimentConfig):
        config.validate()
        self.config = config
        self.records: list[TrainingRecord] = []

    def _tf(self):
        try:
            import tensorflow as tf  # type: ignore
        except ImportError as exc:
            raise RuntimeError("TensorFlow is required to run training") from exc
        return tf

    def _real_batch(self, seed: int):
        training = self.config.training
        if self.config.dataset == "quadratic":
            points = sample_curve(training.batch_size, seed=seed)
        elif self.config.dataset == "sine":
            points = sample_curve(training.batch_size, fn=sine_y, noise=0.08, seed=seed)
        else:
            points = sample_mixture(training.batch_size, seed=seed)
        return points

    def train(self) -> list[TrainingRecord]:
        tf = self._tf()
        model_cfg = self.config.model
        training = self.config.training
        generator = build_mlp_generator(model_cfg)
        discriminator = build_mlp_discriminator(model_cfg)
        gen_opt = tf.keras.optimizers.Adam(training.learning_rate, beta_1=0.5)
        disc_opt = tf.keras.optimizers.Adam(training.learning_rate, beta_1=0.5)
        out_dir = Path(self.config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for step in range(1, training.steps + 1):
            real_points = self._real_batch(training.seed + step)
            real = tf.convert_to_tensor(real_points, dtype=tf.float32)
            noise = tf.convert_to_tensor(
                sample_noise(training.batch_size, model_cfg.noise_dim, seed=training.seed + step),
                dtype=tf.float32,
            )

            with tf.GradientTape() as disc_tape:
                fake = generator(noise, training=True)
                real_logits = discriminator(real, training=True)
                fake_logits = discriminator(fake, training=True)
                if training.loss == "wgan-gp":
                    disc_loss = wasserstein_discriminator_loss(real_logits, fake_logits)
                    disc_loss += training.gradient_penalty_weight * gradient_penalty(discriminator, real, fake)
                else:
                    disc_loss = vanilla_discriminator_loss(real_logits, fake_logits)
            disc_grads = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
            disc_opt.apply_gradients(zip(disc_grads, discriminator.trainable_variables))

            with tf.GradientTape() as gen_tape:
                fake = generator(noise, training=True)
                fake_logits = discriminator(fake, training=True)
                gen_loss = (
                    wasserstein_generator_loss(fake_logits)
                    if training.loss == "wgan-gp"
                    else vanilla_generator_loss(fake_logits)
                )
            gen_grads = gen_tape.gradient(gen_loss, generator.trainable_variables)
            gen_opt.apply_gradients(zip(gen_grads, generator.trainable_variables))

            if step % training.log_every == 0 or step == 1:
                self.records.append(TrainingRecord(step, float(gen_loss.numpy()), float(disc_loss.numpy())))

            if step % training.snapshot_every == 0:
                generator.save(out_dir / f"generator_step_{step}.keras")

        generator.save(out_dir / "generator_final.keras")
        return self.records

