"""Compare the baseline and residual models (Part 2).

Reads results/<model>/metrics.json and history.json for each model and writes
results/comparison.md and results/comparison.png.

Example:
    python compare.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cifar10.plots import plot_comparison


def main() -> None:
    p = argparse.ArgumentParser(description="Compare trained CIFAR-10 models.")
    p.add_argument("--models", nargs="+", default=["baseline", "residual"])
    p.add_argument("--results-dir", default="results")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    metrics = {m: json.loads((results_dir / m / "metrics.json").read_text()) for m in args.models}
    histories = {m: json.loads((results_dir / m / "history.json").read_text()) for m in args.models}

    rows = [
        "| Model | Parameters | Test accuracy | Test loss | Best val accuracy | Best epoch | Train time |",
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
    (results_dir / "comparison.md").write_text(table + "\n")
    plot_comparison(histories, results_dir / "comparison.png")
    print(table)
    print(f"\nWritten {results_dir / 'comparison.md'} and {results_dir / 'comparison.png'}")


if __name__ == "__main__":
    main()
