import io
import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from tensorflow.keras.preprocessing import image as keras_image

from train import build_mobilenetv2_model

app = FastAPI(
    title="Elephant Classifier API",
    description="REST API for classifying African vs. Asian elephants using MobileNetV2",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CODE_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = CODE_DIR / "best_mobilenetv2.weights.h5"
CLASS_INDICES_PATH = CODE_DIR / "class_indices.json"

_model = None
_idx_to_class = {}


def get_model():
    global _model, _idx_to_class
    if _model is None:
        if not CLASS_INDICES_PATH.exists() or not WEIGHTS_PATH.exists():
            raise RuntimeError("Model artifacts missing.")
        with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
            class_indices = json.load(f)
        _idx_to_class = {v: k for k, v in class_indices.items()}
        _model = build_mobilenetv2_model(num_classes=len(_idx_to_class))
        _model.load_weights(str(WEIGHTS_PATH))
    return _model, _idx_to_class


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict[str, float]


@app.get("/health")
def health_check():
    return {"status": "ok", "model": "MobileNetV2"}


@app.post("/predict", response_model=PredictionResponse)
async def predict_elephant(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not an image.")

    contents = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")

    pil_img = pil_img.resize((224, 224))
    x = keras_image.img_to_array(pil_img) / 255.0
    x = np.expand_dims(x, axis=0)

    model, idx_to_class = get_model()
    preds = model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(preds))
    pred_class = idx_to_class[pred_idx]
    confidence = float(preds[pred_idx]) * 100

    return PredictionResponse(
        prediction=pred_class,
        confidence=round(confidence, 2),
        probabilities={idx_to_class[i]: round(float(p), 4) for i, p in enumerate(preds)},
    )

