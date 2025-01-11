from __future__ import annotations

import argparse
from pathlib import Path

from .config import ExperimentConfig, TrainingConfig
from .data import sample_curve, sample_mixture, sine_y, summarize
from .trainer import GanTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GAN Lab TensorFlow")
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="train a synthetic-distribution GAN")
    train.add_argument("--dataset", choices=["quadratic", "sine", "mixture"], default="quadratic")
    train.add_argument("--steps", type=int, default=2000)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--loss", choices=["vanilla", "wgan-gp"], default="vanilla")
    train.add_argument("--out", type=Path, default=Path("outputs/quadratic"))

    sample = sub.add_parser("describe-data", help="summarize a synthetic dataset")
    sample.add_argument("--dataset", choices=["quadratic", "sine", "mixture"], default="quadratic")
    sample.add_argument("--count", type=int, default=512)

    serve = sub.add_parser("serve", help="launch the real-time GAN Observatory web app")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "describe-data":
        if args.dataset == "quadratic":
            points = sample_curve(args.count)
        elif args.dataset == "sine":
            points = sample_curve(args.count, fn=sine_y, noise=0.08)
        else:
            points = sample_mixture(args.count)
        print(summarize(points))
        return 0

    if args.command == "serve":
        from .live.server import run

        print(f"GAN Observatory on http://{args.host}:{args.port}")
        run(host=args.host, port=args.port)
        return 0

    training = TrainingConfig(steps=args.steps, batch_size=args.batch_size, loss=args.loss)
    config = ExperimentConfig(dataset=args.dataset, output_dir=args.out, training=training)
    records = GanTrainer(config).train()
    print(f"completed {args.steps} steps; logged {len(records)} records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

