# Elephant Classification (African vs. Asian)

A production-grade deep learning classification system that distinguishes between **African** and **Asian** elephants using transfer learning with **MobileNetV2**, accessible via an interactive **Streamlit** dashboard, high-performance **FastAPI** REST API, and a standalone **CLI** inference tool.

## Key Highlights
- **Model Architecture**: MobileNetV2 (ImageNet backbone) + GlobalAveragePooling2D + Dense(256, ReLU) + Dropout(0.4) + Dense(2, Softmax).
- **Test Performance**: **87.77% accuracy** across 188 test images (African F1: 0.89, Asian F1: 0.87).
- **Three Inference Interfaces**:
  1. **Streamlit UI** (`app.py`) — Interactive browser drag-and-drop image prediction.
  2. **FastAPI REST Service** (`api.py`) — Production REST API endpoints (`/predict`, `/health`) with OpenAPI Swagger documentation.
  3. **CLI Tool** (`predict.py`) — Direct command-line image classification.
- **Full Test Coverage**: 7 automated unit and integration tests passing (`pytest tests/test_project.py`).
- **Container Ready**: Production `Dockerfile` and `docker-compose.yml`.

---

## Project Structure

```text
Code/Code/
├── app.py                      # Streamlit web interface
├── api.py                      # FastAPI REST API (/predict, /health)
├── predict.py                  # Standalone CLI inference tool
├── train.py                    # Training pipeline (early stopping & checkpoints)
├── evaluate.py                 # Evaluation script (classification report & matrix)
├── tests/
│   └── test_project.py         # Comprehensive pytest suite (7 tests)
├── best_mobilenetv2.weights.h5 # Trained MobileNetV2 weights (87.77% test accuracy)
├── class_indices.json          # Label index mapping: {"African": 0, "Asian": 1}
├── requirements.txt            # Pinned Python package dependencies
├── Dockerfile                  # Container definition
├── docker-compose.yml          # Container orchestration
├── training_curves.png         # Accuracy & loss curves plot
├── confusion_matrix.png        # Test set confusion matrix visualization
├── evaluation_report.json      # Detailed precision/recall/F1 metrics per class
└── Deployment.txt              # Production deployment reference
```

---

## Dataset

- **Training set**: 840 images (420 African, 420 Asian)
- **Test set**: 188 images (97 African, 91 Asian)

---

## Quickstart

### 1. Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Launch the Streamlit Web App
```bash
streamlit run app.py
```
Open `http://localhost:8501` to upload elephant photos and view predictions.

### 3. Launch the FastAPI REST Server
```bash
uvicorn api:app --reload --port 8000
```
- Interactive Swagger docs: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`
- Predict endpoint: `POST http://localhost:8000/predict` (multipart file upload)

### 4. CLI Inference
```bash
python predict.py path/to/image.jpg
```

### 5. Run Automated Tests
```bash
pytest tests/test_project.py -v
```
All **7 tests** verify dataset integrity, weights, predictions, CLI, FastAPI, and Streamlit.

### 6. Containerized Deployment
```bash
docker compose up --build
```
