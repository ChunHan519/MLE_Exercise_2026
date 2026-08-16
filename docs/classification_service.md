# Product Classification Service: Design Architecture

## Core Architecture
* **Framework**: FastAPI application served via Uvicorn, wrapping an MLflow scikit-learn classification pipeline.
* **Model Artifacts**: MLflow package serialized with `skops` (`MLmodel` metadata + `model.skops` binary).
* **Eager Initialization**: Loaded during app startup via FastAPI's `@asynccontextmanager` (`lifespan`) to eliminate cold-start latency for initial requests.

## Validation & Error Boundaries
* **Schema Enforcement**: Pydantic V2 model using `min_length=1` on `products: List[str]` to reject empty array payloads with `422 Unprocessable Entity`.
* **V2 Compatibility**: Uses `json_schema_extra` for OpenAPI example definitions to prevent Pydantic V2 deprecation warnings.
* **Exception Mapping**: Catches service-level `ValueError` exceptions in route handlers to explicitly return `400 Bad Request` instead of unhandled `500 Internal Server Error` crashes.

## Monitoring & Observability
* **Liveness Probe (`GET /live`)**: Lightweight check verifying the API process is alive and responding.
* **Readiness Probe (`GET /ready`)**: Validates model presence in memory and runs a dummy prediction before allowing routing traffic.
* **Inference Telemetry**: Captures per-batch sample count and execution timing (`latency_ms`).

---

## Endpoint Specification

| Endpoint | Method | Request Payload | Success Code | Error Behavior |
| :--- | :--- | :--- | :--- | :--- |
| `/live` | `GET` | *None* | `200 OK` | N/A |
| `/ready` | `GET` | *None* | `200 OK` | `503 Service Unavailable` (if uninitialized or probe fails) |
| `/predict` | `POST` | `{"products": ["Item Name"]}` | `200 OK` | `422` (empty list), `400` (bad input), `500` (runtime exception) |

---

## Monitoring Strategy

### Key Metrics to Track
* **System & Infrastructure**:
  * **Latency**: Track p95/p99 execution time via `latency_ms`.
  * **Error Rates**: Monitor ratio of HTTP `5xx` (service failure) vs `4xx` (invalid input).
  * **Resource Saturation**: Monitor container CPU, memory usage, and pod restart counts.
* **Daily Verification & Data Quality**:
  * **Output Sanity Audits**: Run daily automated checks on prediction outputs to ensure category distributions match expected historical baselines.
  * **Ground-Truth Sampling**: Spot-check a daily random sample of predicted outputs against ground-truth labels to monitor precision and recall.