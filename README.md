# Product Classifier API

FastAPI-based production service for product classification, featuring ML model training, MLflow evaluation, multi-stage Docker builds, cross-platform health checks, and Kubernetes readiness.

---

## Project Structure

```text
MLE_Exercise_2026/
├── data/                    # Data Directory
│   ├── processed/
│   └── raw/
├── deploy/                  # Production Kubernetes manifests
│   └── manifests.yaml
├── docs/                    # Architectural and analysis documentation
│   ├── classification_service.md
│   ├── data_analysis_preprocessing.md
│   ├── deployment.md
│   └── model_training_analysis.md
├── model/                   # Exported model artifacts (/model in runtime)
│   ├── conda.yaml
│   ├── MLmodel
│   ├── model.skops
│   ├── python_env.yaml
│   └── requirements.txt
├── src/
│   └── product_classifier/  # Application source package
│       ├── api/             # FastAPI service & integration tests
│       │   ├── app.py
│       │   └── test_app.py
│       ├── data/            # Preprocessing & unit tests
│       │   ├── check_data.py
│       │   ├── preprocess.py
│       │   └── test_preprocess.py
│       ├── evaluation/      # MLflow evaluation & export scripts
│       │   ├── data_loader.py
│       │   ├── evaluate.py
│       │   └── export.py
│       └── models/          # Model architecture & training scripts
│           ├── classifier.py
│           ├── data_loader.py
│           ├── serve.py
│           └── train.py
├── .dockerignore            # Excludes local virtual environments and caches
├── .gitignore               # Git untracked file rules
├── Dockerfile               # Multi-stage Python build script with uv
├── docker-compose.yaml      # Local orchestration setup with healthchecks
├── pyproject.toml           # Project metadata & uv dependencies
└── uv.lock                  # Locked dependency graph
```

---

## Setup & Pipeline Execution

### 0. Copy Data 
You need to copy the provided data under data/raw/ before start.

### 1. Environment Setup
```bash
# Install dependencies using uv
uv sync
```

### 2. Preprocessing & Feature Engineering
```bash
# Execute data cleaning and preprocessing pipeline
python -m src.product_classifier.data.preprocess
```

### 3. Model Training, Evaluation & Export

Executing `train.py` outputs the generated MLflow **`run_id`** and logs key performance metrics:
* **`accuracy`**: Overall model classification accuracy.
* **`f1_macro`**: Macro-averaged F1 score across target classes.

```bash
# 1. Train model pipeline (prints run_id and logs accuracy & f1_macro metrics to MLflow)
python -m src.product_classifier.models.train

# 2. Evaluate specific MLflow run using the generated run_id
python -m src.product_classifier.evaluation.evaluate --run-id=<run_id>

# 3. Export trained model artifacts to the /model directory for deployment
python -m src.product_classifier.evaluation.export --run-id=<run_id>
```

---

## Testing Suite

### 1. Preprocessing Unit Tests (Offline)
> **Note**: These unit tests check data transformation logic locally and **do not** require a running API service.

```bash
# Run data preprocessing unit tests
uv run pytest -v src/product_classifier/data/test_preprocess.py
```

### 2. API Service Integration Tests
> ⚠️ **Prerequisite**: The API service **MUST** be running locally or inside Docker before executing these tests.

#### Step A: Start the Service First
* **Option 1 (Local Uvicorn)**:
  ```bash
  uvicorn src.product_classifier.api.app:app --host 0.0.0.0 --port 8000
  ```
* **Option 2 (Docker Compose)**:
  ```bash
  docker compose up -d
  ```

#### Step B: Run Service Tests
```bash
# Run API endpoint integration tests
uv run pytest -v src/product_classifier/api
```

---

## Docker Containerization

### Docker Compose (Recommended)
```bash
# Build image and run service in detached mode
docker compose up --build -d

# Check container health status
docker compose ps

# Stream runtime and model loading logs
docker compose logs -f product-classifier

# Stop and remove container resources
docker compose down
```

### Single Docker Command
```bash
# Build production image with version tag
docker build -t product-classifier:v1.0.0 -f Dockerfile .
```

---

## API Verification & Endpoints

### Linux / Bash / Command Prompt
```bash
# Liveness Probe
curl -i http://localhost:8000/live

# Readiness Probe
curl -i http://localhost:8000/ready

# Predict Endpoint
curl -i -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"products": ["Wireless Mouse", "Organic Almond Milk"]}'
```

### Windows PowerShell
```powershell
# Native curl.exe call
curl.exe -X POST http://localhost:8000/predict `
  -H "Content-Type: application/json" `
  -d "{\"products\": [\"Wireless Mouse\", \"Organic Almond Milk\"]}"

# PowerShell REST cmdlet
Invoke-RestMethod -Uri "http://localhost:8000/predict" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"products": ["Wireless Mouse", "Organic Almond Milk"]}'
```

---

## Kubernetes Production Deployment

```bash
# Apply combined Deployment and LoadBalancer Service
kubectl apply -f deploy/manifests.yaml

# Check rollout status
kubectl get pods -l app=product-classifier
kubectl get svc product-classifier-service
```

---

## Key Technical Features

* **Fast Multi-Stage Builds**: Uses `uv` inside `python:3.10-slim` builder stage for fast dependency resolution and low final image size.
* **Eager Model Initialization**: Models load during the FastAPI `lifespan` hook. `/ready` returns `503` until model setup is complete to prevent cold-start request drops.
* **Cross-Platform Health Checks**: Docker healthcheck uses Python's `urllib.request` against `/ready`, eliminating host OS/tool dependency issues (`curl`/`wget`).
* **Zero-Downtime Rolling Updates**: K8s strategy uses `maxUnavailable: 0` alongside dedicated `/live` and `/ready` probes.