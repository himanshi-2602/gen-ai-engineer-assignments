"""ONNX export and ONNX Runtime inference.

`OnnxClassifier` and `preprocess` only need numpy, Pillow and onnxruntime (no torch),
so the FastAPI server can run without PyTorch installed.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Union

import numpy as np
import onnxruntime as ort
from PIL import Image

from .constants import CIFAR10_MEAN, CIFAR10_STD, CLASSES

_MEAN = np.array(CIFAR10_MEAN, dtype=np.float32).reshape(3, 1, 1)
_STD = np.array(CIFAR10_STD, dtype=np.float32).reshape(3, 1, 1)


def export_onnx(model, path: Union[str, Path], opset: int = 17) -> Path:
    """Export a PyTorch model to ONNX with a dynamic batch dimension."""
    import torch  # only needed for export

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    torch.onnx.export(
        model,
        torch.randn(1, 3, 32, 32),
        str(path),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=opset,
        dynamo=False,
    )
    return path


def preprocess(image: Image.Image) -> np.ndarray:
    """PIL image of any size -> normalized float32 array of shape (1, 3, 32, 32)."""
    image = image.convert("RGB")
    if image.size != (32, 32):
        image = image.resize((32, 32), Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return ((array - _MEAN) / _STD)[None].astype(np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


class OnnxClassifier:
    """Runs a CIFAR-10 ONNX model with ONNX Runtime on the CPU."""

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.session = ort.InferenceSession(str(self.path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def logits(self, batch: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input_name: batch.astype(np.float32)})[0]

    def predict_image(self, image: Image.Image, top_k: int = 3) -> dict:
        probs = softmax(self.logits(preprocess(image)))[0]
        order = np.argsort(-probs)[:top_k]
        top: List[dict] = [{"label": CLASSES[i], "probability": round(float(probs[i]), 4)} for i in order]
        return {"label": top[0]["label"], "confidence": top[0]["probability"], "top_k": top}
