"""Training and evaluation loops."""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
) -> Tuple[float, float]:
    """Run one epoch of training. Returns (mean loss, accuracy).

    The scheduler, if given, is stepped after every batch (OneCycleLR is a per-step schedule).
    """
    model.train()
    # Accumulate on the device and read back once per epoch to avoid a sync on every batch.
    total_loss = torch.zeros((), device=device)
    correct = torch.zeros((), device=device, dtype=torch.long)
    seen = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.detach() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum()
        seen += labels.size(0)

    return total_loss.item() / seen, correct.item() / seen


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    """Evaluate without gradients. Returns (mean loss, accuracy)."""
    model.eval()
    total_loss = torch.zeros((), device=device)
    correct = torch.zeros((), device=device, dtype=torch.long)
    seen = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        total_loss += criterion(logits, labels) * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum()
        seen += labels.size(0)

    return total_loss.item() / seen, correct.item() / seen


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device):
    """Return (images, labels, predictions, confidences) for the whole loader, on the CPU."""
    model.eval()
    all_images, all_labels, all_preds, all_conf = [], [], [], []
    for images, labels in loader:
        probs = torch.softmax(model(images.to(device)), dim=1)
        conf, preds = probs.max(dim=1)
        all_images.append(images)
        all_labels.append(labels)
        all_preds.append(preds.cpu())
        all_conf.append(conf.cpu())
    return torch.cat(all_images), torch.cat(all_labels), torch.cat(all_preds), torch.cat(all_conf)
