import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from train import build_mobilenetv2_model


def evaluate(test_dir: Path, artifacts_dir: Path, batch_size: int = 32) -> dict:
    """Evaluate the trained MobileNetV2 model on the test set and save metrics."""
    weights_path = artifacts_dir / "best_mobilenetv2.weights.h5"
    class_indices_path = artifacts_dir / "class_indices.json"

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    if not class_indices_path.exists():
        raise FileNotFoundError(f"Class indices file not found: {class_indices_path}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    with open(class_indices_path, encoding="utf-8") as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    model = build_mobilenetv2_model(num_classes=len(class_names))
    model.load_weights(str(weights_path))

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_set = test_datagen.flow_from_directory(
        str(test_dir),
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    preds = model.predict(test_set, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_set.classes

    acc = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    report_dict = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    report_text = classification_report(
        y_true, y_pred, target_names=class_names, zero_division=0
    )

    print(f"\nTest Accuracy: {acc:.4f} ({int(np.sum(y_true == y_pred))}/{len(y_true)})")
    print("\nClassification Report:")
    print(report_text)
    print("Confusion Matrix:")
    print(cm)

    # Save confusion matrix plot
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=f"Confusion Matrix (Acc: {acc:.2%})",
    )
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    cm_path = artifacts_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)

    results = {
        "accuracy": acc,
        "total_samples": int(len(y_true)),
        "correct_predictions": int(np.sum(y_true == y_pred)),
        "class_names": class_names,
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
    }
    report_path = artifacts_dir / "evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved evaluation report to {report_path}")
    print(f"Saved confusion matrix plot to {cm_path}")
    return results


def main():
    code_dir = Path(__file__).resolve().parent
    project_root = code_dir.parent.parent

    parser = argparse.ArgumentParser(description="Evaluate Elephant Classifier")
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=project_root / "test-20260105T160050Z-1-001" / "test",
    )
    parser.add_argument("--artifacts-dir", type=Path, default=code_dir)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    evaluate(
        test_dir=args.test_dir,
        artifacts_dir=args.artifacts_dir,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

