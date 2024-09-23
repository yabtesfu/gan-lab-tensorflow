# GAN Lab TensorFlow

GAN Lab TensorFlow is a learning-focused project for building and analyzing Generative Adversarial Networks. It starts with the classic 2D quadratic distribution from the tutorial notes, then leaves room for image experiments with TensorFlow/Keras models, WGAN-GP losses, checkpoints, and visual diagnostics.

The goal is not only to train a generator, but to make the adversarial process visible: real samples, generated samples, discriminator feature space, loss curves, and stability metrics.

## Features

- Quadratic, sine, and Gaussian-mixture synthetic datasets
- TensorFlow/Keras MLP generator and discriminator builders
- DCGAN-style image generator/discriminator builders
- Conditional GAN data/model helpers
- Replay buffer for discriminator stabilization experiments
- TTUR and warmup learning-rate schedules
- Adaptive discriminator augmentation helpers
- Vanilla GAN and Wasserstein loss helpers
- Gradient-penalty utility for WGAN-GP experiments
- Training configuration dataclasses
- Alternating GAN training loop with checkpoint hooks
- Pure-Python metrics for moment distance, coverage, and RBF-MMD summaries
- Matplotlib visualization helpers
- CLI for synthetic-data training and sampling
- Unit tests for config, data generation, and metrics
- `push_project.sh` configured for the requested GitHub repo and date window

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e .[dev]
```

TensorFlow currently supports stable Python versions below Python 3.13. If you are using Python 3.14, create an environment with Python 3.11 or 3.12 for actual model training.

## Quick Start

```bash
gan-lab train --steps 2000 --dataset quadratic --out outputs/quadratic
gan-lab sample --checkpoint outputs/quadratic --count 256
```

Or run with Python directly:

```bash
python -m gan_lab_tensorflow.cli train --steps 2000
```

## Project Layout

```txt
src/gan_lab_tensorflow/
  augment.py      Adaptive discriminator augmentation helpers
  config.py       Experiment configuration
  data.py         Synthetic distributions and batching helpers
  models.py       TensorFlow generator/discriminator builders
  losses.py       GAN and WGAN-GP loss functions
  conditional.py  Conditional GAN helpers
  replay.py       Stabilization replay buffer
  schedules.py    TTUR/warmup schedules
  evaluation.py   Distribution-level evaluation metrics
  trainer.py      Alternating training loop
  metrics.py      Lightweight sample diagnostics
  visualize.py    Plotting helpers
  cli.py          Command-line entry point
tests/
  test_config.py
  test_data.py
  test_metrics.py
docs/
  experiment-plan.md
```

## Learning Path

1. Generate a visible 2D real distribution.
2. Train a small generator and discriminator.
3. Watch generated samples move toward the target distribution.
4. Track generator/discriminator losses.
5. Inspect discriminator feature-space separation.
6. Try stability improvements such as WGAN-GP.
7. Add conditional labels and compare class-conditioned samples.
8. Extend the same training loop toward MNIST or small image datasets.

## Notes

This is an educational lab, not a production image-generation system. The code is intentionally readable and modular so each GAN concept can be studied independently.
