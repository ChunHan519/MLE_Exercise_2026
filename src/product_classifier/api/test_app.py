import pytest
from fastapi.testclient import TestClient
from product_classifier.api.app import app

@pytest.fixture
def client():
    """Fixture that starts the TestClient within a context manager 
    so FastAPI startup events (and model loading) execute properly.
    """
    with TestClient(app) as c:
        yield c


def test_liveness(client):
    """Test the liveness probe endpoint."""
    response = client.get("/live")
    assert response.status_code == 200
    assert response.json() == {"status": "I'm alive."}


def test_readiness(client):
    """Test the readiness probe endpoint and model pipeline execution."""
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["model_loaded"] is True


def test_predict_success(client):
    """Test product classification with a valid batch request."""
    payload = {
        "products": [
            "Colgate Total Toothpaste 150g",
            "Organic Whole Milk 1L",
            "Nescafe Gold Blend Coffee 200g"
        ]
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "success"
    assert data["count"] == 3
    assert len(data["predictions"]) == 3
    assert "predicted_category" in data["predictions"][0]
    assert isinstance(data["latency_ms"], float)


def test_predict_empty_list(client):
    """Test that sending an empty product list returns a 400/422 validation error."""
    payload = {"products": []}
    response = client.post("/predict", json=payload)
    # FastAPI validation error for empty list / schema
    assert response.status_code in [400, 422]