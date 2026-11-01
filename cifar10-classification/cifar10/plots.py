"""Plots: training curves, misclassified samples, confusion matrix, model comparison."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")  # write files without needing a display
import matplotlib.pyplot as plt
import numpy as np
import torch

from .data import CLASSES, denormalize


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_curves(history: List[dict], path: Path, title: str = "") -> None:
    """Loss and accuracy per epoch for the training and validation sets."""
    epochs = [h["epoch"] for h in history]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))

    ax_loss.plot(epochs, [h["train_loss"] for h in history], label="train")
    ax_loss.plot(epochs, [h["val_loss"] for h in history], label="validation")
    ax_loss.set(xlabel="epoch", ylabel="cross-entropy loss", title="Loss")

    ax_acc.plot(epochs, [100 * h["train_acc"] for h in history], label="train")
    ax_acc.plot(epochs, [100 * h["val_acc"] for h in history], label="validation")
    ax_acc.set(xlabel="epoch", ylabel="accuracy (%)", title="Accuracy")

    for ax in (ax_loss, ax_acc):
        ax.grid(alpha=0.3)
        ax.legend()
    if title:
        fig.suptitle(title)
    _save(fig, path)


def plot_misclassified(
    images: torch.Tensor, labels: torch.Tensor, preds: torch.Tensor, confidences: torch.Tensor,
    path: Path, count: int = 25,
) -> None:
    """Grid of test images the model got wrong, with true label, prediction and confidence."""
    wrong = (preds != labels).nonzero(as_tuple=True)[0][:count]
    cols = 5
    rows = max(1, int(np.ceil(len(wrong) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.5))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, idx in zip(np.atleast_1d(axes).ravel(), wrong.tolist()):
        ax.imshow(denormalize(images[idx]).permute(1, 2, 0).numpy())
        ax.set_title(
            f"true: {CLASSES[labels[idx]]}\npred: {CLASSES[preds[idx]]} ({confidences[idx]:.0%})",
            fontsize=8, color="#b00020",
        )
    fig.suptitle("Sample misclassified test images")
    _save(fig, path)


def plot_confusion_matrix(matrix: np.ndarray, path: Path) -> None:
    """Row-normalized confusion matrix (each row sums to 100%)."""
    norm = matrix / matrix.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES)), CLASSES)
    ax.set(xlabel="predicted", ylabel="true", title="Confusion matrix (test set)")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{norm[i, j]:.0%}", ha="center", va="center", fontsize=7,
                    color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, path)


def plot_comparison(histories: Dict[str, Sequence[dict]], path: Path) -> None:
    """Validation loss and accuracy of several runs on the same axes."""
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))
    for name, history in histories.items():
        epochs = [h["epoch"] for h in history]
        ax_loss.plot(epochs, [h["val_loss"] for h in history], label=name)
        ax_acc.plot(epochs, [100 * h["val_acc"] for h in history], label=name)
    ax_loss.set(xlabel="epoch", ylabel="validation loss", title="Validation loss")
    ax_acc.set(xlabel="epoch", ylabel="validation accuracy (%)", title="Validation accuracy")
    for ax in (ax_loss, ax_acc):
        ax.grid(alpha=0.3)
        ax.legend()
    _save(fig, path)
