"""Train a CNN on CIFAR-10.

Examples:
    python train.py --model baseline
    python train.py --model residual --epochs 30
    python train.py --model residual --amp --run-name residual_amp   # mixed precision
    python train.py --model baseline --resume                         # continue from last.pt

    # Distributed data parallel (one process per GPU; falls back to CPU + gloo without CUDA)
    torchrun --nproc_per_node=2 train.py --model residual
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import SGD
from torch.optim.lr_scheduler import OneCycleLR

from cifar10.data import get_dataloaders
from cifar10.engine import evaluate, train_one_epoch
from cifar10.models import BLOCKS, build_model
from cifar10.utils import count_parameters, get_device, load_checkpoint, save_checkpoint, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a CNN on CIFAR-10.")
    p.add_argument("--model", choices=sorted(BLOCKS), default="baseline")
    p.add_argument("--run-name", default=None, help="name for checkpoint/results folders (default: model name)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128, help="per process when using DDP")
    p.add_argument("--lr", type=float, default=0.1, help="peak learning rate of the OneCycle schedule")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--dropout", type=float, default=0.1, help="dropout after each conv stage")
    p.add_argument("--head-dropout", type=float, default=0.3, help="dropout before the classifier")
    p.add_argument("--amp", action="store_true", help="mixed precision (float16 on CUDA/MPS, bfloat16 on CPU)")
    p.add_argument("--val-size", type=int, default=5000)
    p.add_argument("--train-subset", type=int, default=None, help="use only N training images (smoke tests)")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--device", default="auto", help="auto, cuda, mps or cpu")
    p.add_argument("--resume", action="store_true", help="resume from <checkpoint-dir>/<run-name>/last.pt")
    args = p.parse_args()
    args.run_name = args.run_name or args.model
    return args


def setup_distributed(args: argparse.Namespace):
    """Initialise DDP when launched by torchrun. Returns (distributed, rank, world_size, device)."""
    if "LOCAL_RANK" not in os.environ:
        return False, 0, 1, get_device(args.device)
    local_rank = int(os.environ["LOCAL_RANK"])
    if torch.cuda.is_available() and args.device in ("auto", "cuda"):
        torch.cuda.set_device(local_rank)
        device, backend = torch.device("cuda", local_rank), "nccl"
    else:
        # MPS does not support DDP, so without CUDA each process trains on the CPU.
        device, backend = torch.device("cpu"), "gloo"
    dist.init_process_group(backend)
    return True, dist.get_rank(), dist.get_world_size(), device


def main() -> None:
    args = parse_args()
    distributed, rank, world_size, device = setup_distributed(args)
    is_main = rank == 0
    set_seed(args.seed)  # same seed on every process -> identical initial weights

    loaders = get_dataloaders(args.data_dir, args.batch_size, args.val_size, args.num_workers, args.seed,
                              distributed=distributed, train_subset=args.train_subset)
    model_kwargs = {"dropout": args.dropout, "head_dropout": args.head_dropout}
    model = build_model(args.model, **model_kwargs).to(device)
    raw_model = model  # unwrapped model, used for evaluation and checkpoints
    if distributed:
        model = DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None)

    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                    nesterov=True, weight_decay=args.weight_decay)
    # Warm up to the peak LR over the first 30% of steps, then anneal (cosine) towards zero.
    scheduler = OneCycleLR(optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(loaders.train))

    amp_dtype = None
    if args.amp:
        amp_dtype = torch.bfloat16 if device.type == "cpu" else torch.float16
    # Loss scaling is only needed for float16 (bfloat16 has the same range as float32).
    scaler = torch.amp.GradScaler(device.type) if amp_dtype == torch.float16 else None

    ckpt_dir = Path(args.checkpoint_dir) / args.run_name
    results_dir = Path(args.results_dir) / args.run_name
    if is_main:
        results_dir.mkdir(parents=True, exist_ok=True)

    start_epoch, history, best_val_acc = 1, [], 0.0
    if args.resume and (ckpt_dir / "last.pt").exists():
        ckpt = load_checkpoint(ckpt_dir / "last.pt", device)
        raw_model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        if scaler is not None and ckpt.get("scaler_state"):
            scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch, history, best_val_acc = ckpt["epoch"] + 1, ckpt["history"], ckpt["best_val_acc"]
        if is_main:
            print(f"Resumed from epoch {ckpt['epoch']}")

    if is_main:
        print(f"Run: {args.run_name} | model: {args.model} | parameters: {count_parameters(raw_model):,} "
              f"| device: {device} | processes: {world_size} | mixed precision: {amp_dtype or 'off'}")
        print(f"Train / val / test batches per process: "
              f"{len(loaders.train)} / {len(loaders.val)} / {len(loaders.test)}")

    for epoch in range(start_epoch, args.epochs + 1):
        if distributed:
            loaders.train.sampler.set_epoch(epoch)  # different shuffle every epoch
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, loaders.train, criterion, optimizer, device,
                                                scheduler, amp_dtype, scaler)
        if not is_main:
            continue  # only rank 0 validates, logs and saves
        val_loss, val_acc = evaluate(raw_model, loaders.val, criterion, device)
        seconds = time.time() - start

        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"], "seconds": seconds,
        })
        is_best = val_acc > best_val_acc
        best_val_acc = max(best_val_acc, val_acc)

        state = {
            "epoch": epoch, "run_name": args.run_name, "model_name": args.model, "model_kwargs": model_kwargs,
            "model_state": raw_model.state_dict(), "history": history,
            "best_val_acc": best_val_acc, "config": vars(args),
        }
        if is_best:
            save_checkpoint(ckpt_dir / "best.pt", **state)
        save_checkpoint(ckpt_dir / "last.pt", **state,
                        optimizer_state=optimizer.state_dict(), scheduler_state=scheduler.state_dict(),
                        scaler_state=scaler.state_dict() if scaler is not None else None)
        (results_dir / "history.json").write_text(json.dumps(history, indent=2))

        print(f"epoch {epoch:3d}/{args.epochs} | train loss {train_loss:.4f} acc {train_acc:6.2%} "
              f"| val loss {val_loss:.4f} acc {val_acc:6.2%} | lr {history[-1]['lr']:.4f} "
              f"| {seconds:5.1f}s{'  * best' if is_best else ''}", flush=True)

    if is_main:
        print(f"Best validation accuracy: {best_val_acc:.2%} (checkpoint: {ckpt_dir / 'best.pt'})")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
