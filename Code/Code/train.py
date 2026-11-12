import argparse
import json
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def build_mobilenetv2_model(num_classes: int = 2, learning_rate: float = 1e-3) -> Model:
    """Build the MobileNetV2 transfer-learning model matching app.py."""
    base = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.4)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base.input, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train(
    train_dir: Path,
    test_dir: Path,
    output_dir: Path,
    epochs: int = 15,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
) -> dict:
    """Train MobileNetV2 on the elephant dataset and save best weights & class indices."""
    if not train_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
    )
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    training_set = train_datagen.flow_from_directory(
        str(train_dir),
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=42,
    )
    test_set = test_datagen.flow_from_directory(
        str(test_dir),
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    class_indices_path = output_dir / "class_indices.json"
    with open(class_indices_path, "w", encoding="utf-8") as f:
        json.dump(training_set.class_indices, f)
    print(f"Saved class_indices.json: {training_set.class_indices}")

    weights_path = output_dir / "best_mobilenetv2.weights.h5"
    model = build_mobilenetv2_model(
        num_classes=training_set.num_classes,
        learning_rate=learning_rate,
    )

    callbacks = [
        ModelCheckpoint(
            filepath=str(weights_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_accuracy",
            patience=6,
            restore_best_weights=True,
            verbose=1,
        ),
    ]

    history = model.fit(
        training_set,
        validation_data=test_set,
        epochs=epochs,
        callbacks=callbacks,
    )

    history_data = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    history_path = output_dir / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history_data, f, indent=2)

    # Load the best saved weights and run final evaluation
    model.load_weights(str(weights_path))
    loss, acc = model.evaluate(test_set, verbose=0)
    print(f"\nBest model evaluation -> loss: {loss:.4f}, accuracy: {acc:.4f}")
    return {"loss": float(loss), "accuracy": float(acc), "history": history_data}


def main():
    code_dir = Path(__file__).resolve().parent
    project_root = code_dir.parent.parent

    parser = argparse.ArgumentParser(description="Train Elephant Classifier (MobileNetV2)")
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=project_root / "train-20260105T160050Z-1-001" / "train",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=project_root / "test-20260105T160050Z-1-001" / "test",
    )
    parser.add_argument("--output-dir", type=Path, default=code_dir)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()

    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPUs available: {len(tf.config.list_physical_devices('GPU'))}")
    train(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()

