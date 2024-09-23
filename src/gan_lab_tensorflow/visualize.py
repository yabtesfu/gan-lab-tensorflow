from __future__ import annotations

from pathlib import Path

from .data import Point


def plot_samples(real: list[Point], generated: list[Point], output: str | Path) -> None:
    import matplotlib.pyplot as plt

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    plt.scatter([x for x, _ in real], [y for _, y in real], s=8, alpha=0.45, label="real")
    plt.scatter([x for x, _ in generated], [y for _, y in generated], s=8, alpha=0.45, label="generated")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def plot_losses(records, output: str | Path) -> None:
    import matplotlib.pyplot as plt

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.plot([r.step for r in records], [r.generator_loss for r in records], label="generator")
    plt.plot([r.step for r in records], [r.discriminator_loss for r in records], label="discriminator")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output)
    plt.close()

