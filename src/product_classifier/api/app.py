import logging
import time
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException, status
import pandas as pd
from pydantic import BaseModel, Field

from product_classifier.models.serve import ProductClassifierService

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("production_api")

ml_service: ProductClassifierService = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Eagerly load the model during startup for fast container scaling."""
    global ml_service
    try:
        logger.info("Initializing ProductClassifierService during app startup...")
        ml_service = ProductClassifierService()
        logger.info("Service initialized successfully.")
    except Exception as e:
        logger.critical(f"Startup failed - could not load model: {e}")
        ml_service = None
    yield
    logger.info("Shutting down prediction service.")


app = FastAPI(
    title="Product Category Classification API",
    description="Production-ready service with separate liveness and readiness probes.",
    version="1.0.0",
    lifespan=lifespan
)


class PredictionRequest(BaseModel):
    products: List[str] = Field(
        ...,
        min_length=1,
        json_schema_extra={"example": ["Colgate Toothpaste 150g"]}
    )


class PredictionItem(BaseModel):
    input_product: str
    predicted_category: str


class PredictionResponse(BaseModel):
    status: str = "success"
    count: int
    predictions: List[PredictionItem]
    latency_ms: float


@app.get("/live", tags=["Monitoring"])
def liveness_check():
    """Liveness probe: verifies the container process is running."""
    return {"status": "I'm alive."}


@app.get("/ready", tags=["Monitoring"])
def readiness_check():
    """Readiness probe: verifies model is loaded and pipeline execution works via dummy prompt."""
    if ml_service is None or ml_service.pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded."
        )
    try:
        # Run a dummy prediction prompt to verify pipeline execution
        test_result = ml_service.predict("Health Check Product")
        if test_result.empty:
            raise ValueError("Dummy prediction returned empty results.")
    except Exception as e:
        logger.error(f"Readiness probe inference check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model loaded but prediction pipeline probe failed."
        )
    
    return {"status": "ready", "model_loaded": True}


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict_products(request: PredictionRequest):
    """Performs instant inference using the pre-loaded model."""
    if ml_service is None or ml_service.pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model service unavailable."
        )

    start_time = time.time()
    try:
        result_df = ml_service.predict(request.products)
        latency = (time.time() - start_time) * 1000

        predictions_list = [
            PredictionItem(
                input_product=row["input_product"],
                predicted_category=row["predicted_category"]
            )
            for _, row in result_df.iterrows()
        ]

        return PredictionResponse(
            status="success",
            count=len(predictions_list),
            predictions=predictions_list,
            latency_ms=round(latency, 2)
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )