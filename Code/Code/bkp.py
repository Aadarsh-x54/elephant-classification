import streamlit as st
import numpy as np
import json
from tensorflow.keras.models import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image

st.set_page_config(page_title="Elephant Classifier", layout="centered")

st.title("🐘 Elephant Classifier")
st.write("Upload an image — the model will predict **African** or **Asian** elephant.")

# Load model + labels (cache for performance)
@st.cache_resource
def load_artifacts():
    num_classes = 2  # Set this based on your output, assuming 2 classes for elephants

    # Rebuild the model architecture
    base = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base.trainable = False  # Start with frozen base; can add fine-tuning later

    x = base.output
    x = GlobalAveragePooling2D()(x)

    # No hyperparameters needed as the best model was already selected
    x = Dense(256, activation='relu')(x)  # Assume dense_units = 256 as an example
    x = Dropout(0.3)(x)  # Assume dropout_rate = 0.3 as an example

    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base.input, outputs=outputs)

    # Load the best weights
    model.load_weights('best_mobilenetv2.weights.h5')

    # Load class indices
    with open("class_indices.json") as f:
        class_indices = json.load(f)

    # Invert mapping: {0:'african', 1:'asian'}
    idx_to_class = {v: k for k, v in class_indices.items()}

    return model, idx_to_class

model, idx_to_class = load_artifacts()

uploaded_file = st.file_uploader("Choose an elephant image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    st.image(uploaded_file, caption="Uploaded image", use_column_width=True)

    # Preprocess
    img = image.load_img(uploaded_file, target_size=(224, 224))
    x = image.img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)

    # Predict
    preds = model.predict(x)
    pred_idx = np.argmax(preds, axis=1)[0]
    pred_class = idx_to_class[pred_idx]
    confidence = float(np.max(preds)) * 100

    st.subheader(f"Prediction: **{pred_class}**")
    st.write(f"Confidence (accuracy for this image): **{confidence:.2f}%**")
    st.write("Note: Model accuracy may vary based on image quality and angle.")