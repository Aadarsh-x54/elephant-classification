import argparse
import json
from pathlib import Path

import numpy as np
from tensorflow.keras.preprocessing import image

from train import build_mobilenetv2_model


def predict(image_path: Path, artifacts_dir: Path):
    weights_path = artifacts_dir / "best_mobilenetv2.weights.h5"
    class_indices_path = artifacts_dir / "class_indices.json"

    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights: {weights_path}")
    if not class_indices_path.exists():
        raise FileNotFoundError(f"Missing class indices: {class_indices_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with open(class_indices_path, "r", encoding="utf-8") as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}

    model = build_mobilenetv2_model(num_classes=len(idx_to_class))
    model.load_weights(str(weights_path))

    img = image.load_img(str(image_path), target_size=(224, 224))
    x = image.img_to_array(img) / 255.0
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(preds))
    pred_class = idx_to_class[pred_idx]
    confidence = float(preds[pred_idx]) * 100

    print("=" * 45)
    print(f"Image File : {image_path.name}")
    print(f"Prediction : {pred_class}")
    print(f"Confidence : {confidence:.2f}%")
    print("Class Probabilities:")
    for idx, prob in enumerate(preds):
        print(f"  - {idx_to_class[idx]}: {prob * 100:.2f}%")
    print("=" * 45)

    return {
        "prediction": pred_class,
        "confidence": confidence,
        "probabilities": {idx_to_class[i]: float(p) for i, p in enumerate(preds)},
    }


def main():
    code_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Predict elephant species from an image")
    parser.add_argument("image_path", type=Path, help="Path to image file (.jpg, .png, etc.)")
    parser.add_argument("--artifacts-dir", type=Path, default=code_dir, help="Directory with weights & labels")
    args = parser.parse_args()

    predict(args.image_path, args.artifacts_dir)


if __name__ == "__main__":
    main()

