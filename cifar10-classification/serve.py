"""FastAPI service that classifies uploaded images with the exported ONNX model.

Run:
    uvicorn serve:app --port 8080
Then open http://localhost:8080 (upload page) or http://localhost:8080/docs (API docs).

    curl -F "file=@cat.png" http://localhost:8080/predict

The model file can be changed with the MODEL_PATH environment variable
(default: checkpoints/residual/model.onnx, created by export_onnx.py).
"""
from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image, UnidentifiedImageError

from cifar10.constants import CLASSES
from cifar10.inference import OnnxClassifier

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "checkpoints/residual/model.onnx"))
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        raise RuntimeError(f"{MODEL_PATH} not found. Run: python export_onnx.py --run residual")
    state["classifier"] = OnnxClassifier(MODEL_PATH)  # load once, reuse for every request
    yield
    state.clear()


app = FastAPI(title="CIFAR-10 classifier", version="1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": str(MODEL_PATH), "classes": list(CLASSES)}


@app.post("/predict")
async def predict(file: UploadFile = File(..., description="An image (PNG, JPEG, ...); resized to 32x32")) -> dict:
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is larger than 5 MB.")
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="The uploaded file is not a valid image.")
    return {"filename": file.filename, **state["classifier"].predict_image(image)}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CIFAR-10 Classifier</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 560px; margin: 40px auto; padding: 0 16px; color: #1c2430; }
  .card { border: 1px solid #d9dfe7; border-radius: 10px; padding: 20px; }
  img { width: 160px; height: 160px; object-fit: cover; image-rendering: pixelated; border-radius: 6px; margin-top: 12px; }
  .bar { background: #e8eef7; border-radius: 4px; margin: 6px 0; }
  .bar span { display: block; background: #0a84ff; color: #fff; padding: 2px 8px; border-radius: 4px; white-space: nowrap; }
  .muted { color: #5d6b7e; }
</style></head>
<body>
  <h1>CIFAR-10 Classifier</h1>
  <p class="muted">Upload a photo of an airplane, automobile, bird, cat, deer, dog, frog, horse, ship or truck.
     The image is resized to 32&times;32 and classified by the residual CNN (ONNX Runtime).</p>
  <div class="card">
    <input type="file" id="file" accept="image/*">
    <div><img id="preview" alt="" hidden></div>
    <div id="result"></div>
  </div>
<script>
  const file = document.getElementById("file"), preview = document.getElementById("preview"),
        result = document.getElementById("result");
  file.addEventListener("change", async () => {
    if (!file.files.length) return;
    preview.src = URL.createObjectURL(file.files[0]);
    preview.hidden = false;
    result.textContent = "Classifying...";
    const body = new FormData();
    body.append("file", file.files[0]);
    const response = await fetch("/predict", { method: "POST", body });
    const data = await response.json();
    if (!response.ok) { result.textContent = data.detail; return; }
    result.innerHTML = "<h2>" + data.label + "</h2>" + data.top_k.map(p =>
      '<div class="bar"><span style="width:' + Math.max(p.probability * 100, 12) + '%">' +
      p.label + " " + (p.probability * 100).toFixed(1) + "%</span></div>").join("");
  });
</script>
</body></html>"""
