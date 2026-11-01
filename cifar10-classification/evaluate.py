"""Evaluate the best checkpoint of a trained model on the CIFAR-10 test set.

Writes to results/<model>/: metrics.json, curves.png, misclassified.png, confusion_matrix.png

Example:
    python evaluate.py --model baseline
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from torch import nn

from cifar10.data import CLASSES, get_dataloaders
from cifar10.engine import evaluate, predict
from cifar10.models import BLOCKS, build_model
from cifar10.plots import plot_confusion_matrix, plot_curves, plot_misclassified
from cifar10.utils import count_parameters, get_device, load_checkpoint


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained CIFAR-10 model on the test set.")
    p.add_argument("--model", choices=sorted(BLOCKS), default="baseline")
    p.add_argument("--checkpoint", default=None, help="defaults to checkpoints/<model>/best.pt")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    ckpt_path = Path(args.checkpoint or f"checkpoints/{args.model}/best.pt")
    ckpt = load_checkpoint(ckpt_path, device)

    model = build_model(ckpt["model_name"], **ckpt["model_kwargs"]).to(device)
    model.load_state_dict(ckpt["model_state"])

    config = ckpt["config"]
    loaders = get_dataloaders(args.data_dir, config["batch_size"], config["val_size"], args.num_workers, config["seed"])
    test_loss, test_acc = evaluate(model, loaders.test, nn.CrossEntropyLoss(), device)
    images, labels, preds, confidences = predict(model, loaders.test, device)

    num_classes = len(CLASSES)
    confusion = np.bincount(labels.numpy() * num_classes + preds.numpy(),
                            minlength=num_classes ** 2).reshape(num_classes, num_classes)
    per_class = confusion.diagonal() / confusion.sum(axis=1)

    results_dir = Path(args.results_dir) / ckpt["model_name"]
    history_path = results_dir / "history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else ckpt["history"]

    plot_curves(history, results_dir / "curves.png", title=f"{ckpt['model_name']} model")
    plot_misclassified(images, labels, preds, confidences, results_dir / "misclassified.png")
    plot_confusion_matrix(confusion, results_dir / "confusion_matrix.png")

    metrics = {
        "model": ckpt["model_name"],
        "parameters": count_parameters(model),
        "test_accuracy": test_acc,
        "test_loss": test_loss,
        "best_val_accuracy": ckpt["best_val_acc"],
        "best_epoch": ckpt["epoch"],
        "epochs_trained": len(history),
        "train_time_seconds": sum(h["seconds"] for h in history),
        "per_class_accuracy": {name: float(acc) for name, acc in zip(CLASSES, per_class)},
    }
    (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(f"Model: {metrics['model']} ({metrics['parameters']:,} parameters), best epoch {metrics['best_epoch']}")
    print(f"Test accuracy: {test_acc:.2%} | test loss: {test_loss:.4f}")
    for name, acc in metrics["per_class_accuracy"].items():
        print(f"  {name:<10} {acc:6.2%}")
    print(f"Plots and metrics written to {results_dir}/")


if __name__ == "__main__":
    main()
