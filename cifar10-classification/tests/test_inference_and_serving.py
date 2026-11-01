import io

import numpy as np
import pytest
import torch
from PIL import Image

from cifar10.data import build_transforms
from cifar10.inference import OnnxClassifier, export_onnx, preprocess
from cifar10.models import build_model
from cifar10.utils import set_seed


@pytest.fixture(scope="module")
def onnx_model(tmp_path_factory):
    set_seed(0)
    model = build_model("residual").eval()
    path = export_onnx(model, tmp_path_factory.mktemp("onnx") / "model.onnx")
    return model, OnnxClassifier(path), path


def test_onnx_matches_pytorch_for_any_batch_size(onnx_model):
    model, classifier, _ = onnx_model
    for batch_size in (1, 7):
        images = torch.randn(batch_size, 3, 32, 32)
        with torch.no_grad():
            expected = model(images).numpy()
        np.testing.assert_allclose(classifier.logits(images.numpy()), expected, atol=1e-4)


def test_preprocess_matches_training_transforms():
    image = Image.fromarray(np.random.RandomState(0).randint(0, 256, (32, 32, 3), dtype=np.uint8))
    expected = build_transforms(augment=False)(image).numpy()
    np.testing.assert_allclose(preprocess(image)[0], expected, atol=1e-6)


def test_preprocess_resizes_any_image():
    assert preprocess(Image.new("RGBA", (300, 200))).shape == (1, 3, 32, 32)


def test_predict_image_returns_sorted_probabilities(onnx_model):
    _, classifier, _ = onnx_model
    result = classifier.predict_image(Image.new("RGB", (64, 64), (120, 180, 60)))
    probs = [p["probability"] for p in result["top_k"]]
    assert result["label"] == result["top_k"][0]["label"]
    assert probs == sorted(probs, reverse=True) and len(probs) == 3


def test_api(onnx_model, monkeypatch):
    from fastapi.testclient import TestClient

    import serve

    monkeypatch.setattr(serve, "MODEL_PATH", onnx_model[2])
    with TestClient(serve.app) as client:
        assert client.get("/health").json()["status"] == "ok"
        assert "CIFAR-10" in client.get("/").text

        buffer = io.BytesIO()
        Image.new("RGB", (32, 32), (200, 30, 30)).save(buffer, format="PNG")
        response = client.post("/predict", files={"file": ("red.png", buffer.getvalue(), "image/png")})
        assert response.status_code == 200
        assert response.json()["label"] in serve.CLASSES

        bad = client.post("/predict", files={"file": ("notes.txt", b"not an image", "text/plain")})
        assert bad.status_code == 400
