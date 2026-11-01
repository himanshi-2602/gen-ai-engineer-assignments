"""Export a trained checkpoint to ONNX, verify it against PyTorch and benchmark it.

Writes checkpoints/<run>/model.onnx and results/<run>/onnx_metrics.json.

Example:
    python export_onnx.py --run residual
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnx
import torch

from cifar10.data import get_dataloaders
from cifar10.inference import OnnxClassifier, export_onnx
from cifar10.models import build_model
from cifar10.utils import load_checkpoint


def benchmark(fn, batch: np.ndarray, repeats: int) -> float:
    """Median latency of fn(batch) in milliseconds."""
    fn(batch)  # warm-up
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn(batch)
        times.append((time.perf_counter() - start) * 1000)
    return float(np.median(times))


def main() -> None:
    p = argparse.ArgumentParser(description="Export a CIFAR-10 checkpoint to ONNX and verify it.")
    p.add_argument("--run", default="residual")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--results-dir", default="results")
    args = p.parse_args()

    cpu = torch.device("cpu")
    ckpt = load_checkpoint(Path(f"checkpoints/{args.run}/best.pt"), cpu)
    model = build_model(ckpt["model_name"], **ckpt["model_kwargs"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    onnx_path = export_onnx(model, Path(f"checkpoints/{args.run}/model.onnx"), args.opset)
    onnx.checker.check_model(onnx.load(str(onnx_path)))
    classifier = OnnxClassifier(onnx_path)
    print(f"Exported {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB), opset {args.opset}, checker passed")

    # Run the whole test set through both PyTorch and ONNX Runtime.
    config = ckpt["config"]
    test_loader = get_dataloaders(args.data_dir, 256, config["val_size"], num_workers=2, seed=config["seed"]).test
    max_diff, agree, correct_onnx, correct_torch, total = 0.0, 0, 0, 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            torch_logits = model(images).numpy()
            onnx_logits = classifier.logits(images.numpy())
            max_diff = max(max_diff, float(np.abs(torch_logits - onnx_logits).max()))
            agree += int((torch_logits.argmax(1) == onnx_logits.argmax(1)).sum())
            correct_torch += int((torch_logits.argmax(1) == labels.numpy()).sum())
            correct_onnx += int((onnx_logits.argmax(1) == labels.numpy()).sum())
            total += len(labels)

    # CPU latency: one image (online serving) and a batch of 256 (offline scoring).
    single, batch = np.random.randn(1, 3, 32, 32).astype(np.float32), np.random.randn(256, 3, 32, 32).astype(np.float32)
    with torch.no_grad():
        torch_fn = lambda x: model(torch.from_numpy(x))  # noqa: E731
        latency = {
            "torch_cpu_1_image_ms": benchmark(torch_fn, single, 200),
            "onnxruntime_cpu_1_image_ms": benchmark(classifier.logits, single, 200),
            "torch_cpu_batch256_ms": benchmark(torch_fn, batch, 20),
            "onnxruntime_cpu_batch256_ms": benchmark(classifier.logits, batch, 20),
        }

    metrics = {
        "run": args.run,
        "onnx_file": str(onnx_path),
        "opset": args.opset,
        "max_abs_logit_difference": max_diff,
        "prediction_agreement": agree / total,
        "pytorch_test_accuracy": correct_torch / total,
        "onnxruntime_test_accuracy": correct_onnx / total,
        **{k: round(v, 3) for k, v in latency.items()},
    }
    out = Path(args.results_dir) / args.run / "onnx_metrics.json"
    out.write_text(json.dumps(metrics, indent=2))
    for key, value in metrics.items():
        print(f"  {key:<30} {value}")
    print(f"Written {out}")


if __name__ == "__main__":
    main()
