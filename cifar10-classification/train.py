"""Train a CNN on CIFAR-10.

Examples:
    python train.py --model baseline
    python train.py --model residual --epochs 30
    python train.py --model baseline --resume      # continue from checkpoints/baseline/last.pt
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import OneCycleLR

from cifar10.data import get_dataloaders
from cifar10.engine import evaluate, train_one_epoch
from cifar10.models import BLOCKS, build_model
from cifar10.utils import count_parameters, get_device, load_checkpoint, save_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a CNN on CIFAR-10.")
    p.add_argument("--model", choices=sorted(BLOCKS), default="baseline")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.1, help="peak learning rate of the OneCycle schedule")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--dropout", type=float, default=0.1, help="dropout after each conv stage")
    p.add_argument("--head-dropout", type=float, default=0.3, help="dropout before the classifier")
    p.add_argument("--val-size", type=int, default=5000)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    p.add_argument("--resume", action="store_true", help="resume from <checkpoint-dir>/<model>/last.pt")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    loaders = get_dataloaders(args.data_dir, args.batch_size, args.val_size, args.num_workers, args.seed)
    model_kwargs = {"dropout": args.dropout, "head_dropout": args.head_dropout}
    model = build_model(args.model, **model_kwargs).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                    nesterov=True, weight_decay=args.weight_decay)
    # Warm up to the peak LR over the first 30% of steps, then anneal (cosine) towards zero.
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(loaders.train))

    ckpt_dir = Path(args.checkpoint_dir) / args.model
    results_dir = Path(args.results_dir) / args.model
    results_dir.mkdir(parents=True, exist_ok=True)

    start_epoch, history, best_val_acc = 1, [], 0.0
    if args.resume and (ckpt_dir / "last.pt").exists():
        ckpt = load_checkpoint(ckpt_dir / "last.pt", device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch, history, best_val_acc = ckpt["epoch"] + 1, ckpt["history"], ckpt["best_val_acc"]
        print(f"Resumed from epoch {ckpt['epoch']}")

    print(f"Model: {args.model} | parameters: {count_parameters(model):,} | device: {device}")
    print(f"Train / val / test batches: {len(loaders.train)} / {len(loaders.val)} / {len(loaders.test)}")

    for epoch in range(start_epoch, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, loaders.train, criterion, optimizer, device, scheduler)
        val_loss, val_acc = evaluate(model, loaders.val, criterion, device)
        seconds = time.time() - start

        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"], "seconds": seconds,
        })
        is_best = val_acc > best_val_acc
        best_val_acc = max(best_val_acc, val_acc)

        state = {
            "epoch": epoch, "model_name": args.model, "model_kwargs": model_kwargs,
            "model_state": model.state_dict(), "history": history,
            "best_val_acc": best_val_acc, "config": vars(args),
        }
        if is_best:
            save_checkpoint(ckpt_dir / "best.pt", **state)
        save_checkpoint(ckpt_dir / "last.pt", **state,
                        optimizer_state=optimizer.state_dict(), scheduler_state=scheduler.state_dict())
        (results_dir / "history.json").write_text(json.dumps(history, indent=2))

        print(f"epoch {epoch:3d}/{args.epochs} | train loss {train_loss:.4f} acc {train_acc:6.2%} "
              f"| val loss {val_loss:.4f} acc {val_acc:6.2%} | lr {history[-1]['lr']:.4f} "
              f"| {seconds:5.1f}s{'  * best' if is_best else ''}", flush=True)

    print(f"Best validation accuracy: {best_val_acc:.2%} (checkpoint: {ckpt_dir / 'best.pt'})")


if __name__ == "__main__":
    main()
