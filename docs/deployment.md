# Containerization & Production Deployment

## Docker

- Use a **multi-stage Docker build** with `uv` for fast dependency installation.
- Store model artifacts in `/model`.
- Use `.dockerignore` to exclude `.venv/`, datasets, caches, and `mlruns/`.
- Keep the final image lightweight.

## Docker Compose

- Use **Docker Compose** as a quick and simple solution for local deployment.
- It allows the API and model service to be started easily.
- Include health checks using the `/ready` endpoint.
- Define CPU and memory limits for the service.

## Production Deployment

- For a production-standard deployment, use **Kubernetes** instead of Docker Compose.
- Run multiple replicas of the classifier API for availability.
- Use a Kubernetes `Service` to distribute traffic across replicas.
- Use **RollingUpdate** to support zero-downtime deployments.
- Use `/live` for liveness checks and `/ready` for readiness checks.

## Scaling & Operations

- Use **Horizontal Pod Autoscaler (HPA)** to automatically scale pods based on CPU utilization.
- A CPU target of around **70%** can be used as the initial scaling threshold.
- Define CPU and memory requests/limits to prevent resource issues.
- Keep application logs in `stdout/stderr` for centralized monitoring.
- This setup provides a simple path from **local Docker Compose → production Kubernetes**.