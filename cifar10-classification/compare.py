"""Compare the baseline and residual models (Part 2).

Reads results/<run>/metrics.json and history.json for each run and writes
results/comparison.md and results/comparison.png.

Example:
    python compare.py
    python compare.py --runs residual residual_amp --output comparison_amp
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cifar10.plots import plot_comparison


def main() -> None:
    p = argparse.ArgumentParser(description="Compare trained CIFAR-10 models.")
    p.add_argument("--runs", nargs="+", default=["baseline", "residual"])
    p.add_argument("--output", default="comparison", help="file name (without extension) in results/")
    p.add_argument("--results-dir", default="results")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    metrics = {r: json.loads((results_dir / r / "metrics.json").read_text()) for r in args.runs}
    histories = {r: json.loads((results_dir / r / "history.json").read_text()) for r in args.runs}

    rows = [
        "| Run | Parameters | Test accuracy | Test loss | Best val accuracy | Best epoch | Train time |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, m in metrics.items():
        rows.append(
            f"| {name} | {m['parameters']:,} | **{m['test_accuracy']:.2%}** | {m['test_loss']:.4f} "
            f"| {m['best_val_accuracy']:.2%} | {m['best_epoch']}/{m['epochs_trained']} "
            f"| {m['train_time_seconds'] / 60:.1f} min |"
        )

    classes = list(next(iter(metrics.values()))["per_class_accuracy"])
    rows += ["", "| Class | " + " | ".join(metrics) + " |", "|---|" + "---|" * len(metrics)]
    for c in classes:
        rows.append(f"| {c} | " + " | ".join(f"{m['per_class_accuracy'][c]:.2%}" for m in metrics.values()) + " |")

    if len(metrics) == 2:
        (a, ma), (b, mb) = metrics.items()
        diff = 100 * (mb["test_accuracy"] - ma["test_accuracy"])
        rows += ["", f"**{b} vs {a}: {diff:+.2f} percentage points of test accuracy.**"]

    table = "\n".join(rows)
    (results_dir / f"{args.output}.md").write_text(table + "\n")
    plot_comparison(histories, results_dir / f"{args.output}.png")
    print(table)
    print(f"\nWritten {results_dir / args.output}.md and .png")


if __name__ == "__main__":
    main()
