import io
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from streamlit.testing.v1 import AppTest
from tensorflow.keras.preprocessing import image as keras_image

import sys

CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = CODE_DIR.parent.parent
TRAIN_DIR = PROJECT_ROOT / "train-20260105T160050Z-1-001" / "train"
TEST_DIR = PROJECT_ROOT / "test-20260105T160050Z-1-001" / "test"

if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from api import app as fastapi_app
from predict import predict
from train import build_mobilenetv2_model


def test_dataset_directories_exist_and_populated():
    for split_dir in (TRAIN_DIR, TEST_DIR):
        assert split_dir.exists(), f"Missing dataset directory: {split_dir}"
        for cls in ("African", "Asian"):
            cls_dir = split_dir / cls
            assert cls_dir.exists(), f"Missing class directory: {cls_dir}"
            files = [p for p in cls_dir.iterdir() if p.is_file()]
            assert len(files) > 0, f"No images found in {cls_dir}"


def test_class_indices_file():
    ci_path = CODE_DIR / "class_indices.json"
    assert ci_path.exists()
    with open(ci_path, encoding="utf-8") as f:
        ci = json.load(f)
    assert ci == {"African": 0, "Asian": 1}


def test_model_loads_weights_and_predicts():
    weights_path = CODE_DIR / "best_mobilenetv2.weights.h5"
    assert weights_path.exists()

    model = build_mobilenetv2_model(num_classes=2)
    model.load_weights(str(weights_path))

    african_img = next((TEST_DIR / "African").iterdir())
    asian_img = next((TEST_DIR / "Asian").iterdir())

    for img_path in (african_img, asian_img):
        img = keras_image.load_img(str(img_path), target_size=(224, 224))
        x = keras_image.img_to_array(img) / 255.0
        x = np.expand_dims(x, axis=0)
        preds = model.predict(x, verbose=0)
        assert preds.shape == (1, 2)
        assert np.isclose(np.sum(preds), 1.0, atol=1e-5)
        assert np.all(preds >= 0.0) and np.all(preds <= 1.0)


def test_cli_predict_module():
    african_img = next((TEST_DIR / "African").iterdir())
    res = predict(african_img, CODE_DIR)
    assert res["prediction"] in ("African", "Asian")
    assert 0.0 <= res["confidence"] <= 100.0
    assert "African" in res["probabilities"]
    assert "Asian" in res["probabilities"]


def test_fastapi_health_endpoint():
    client = TestClient(fastapi_app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model": "MobileNetV2"}


def test_fastapi_predict_endpoint():
    client = TestClient(fastapi_app)
    img_byte_arr = io.BytesIO()
    test_img = Image.new("RGB", (224, 224), color=(73, 109, 137))
    test_img.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)

    response = client.post(
        "/predict",
        files={"file": ("test.jpg", img_byte_arr, "image/jpeg")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in ("African", "Asian")
    assert "confidence" in data
    assert "probabilities" in data


def test_streamlit_app_renders_without_error(monkeypatch):
    monkeypatch.chdir(CODE_DIR)
    at = AppTest.from_file(str(CODE_DIR / "app.py"), default_timeout=30)
    at.run()
    assert not at.exception
    assert len(at.title) == 1
    assert at.title[0].value == "Elephant Classifier"
