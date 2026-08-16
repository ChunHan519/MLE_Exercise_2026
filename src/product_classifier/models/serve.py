import logging
from pathlib import Path
from typing import List, Union
import mlflow.sklearn
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ProductClassifierService")


class ProductClassifierService:

    def __init__(self, model_path: Union[str, Path] = None):
        if model_path is None:
            # Dynamically locate project root by searching upwards for the 'model' folder containing 'MLmodel'
            current_dir = Path(__file__).resolve().parent
            root_dir = None
            for parent in [current_dir] + list(current_dir.parents):
                if (parent / "model" / "MLmodel").exists():
                    root_dir = parent
                    break

            # Fallback to 2 levels up if dynamic search isn't triggered
            if root_dir is None:
                root_dir = Path(__file__).resolve().parents[2]

            model_path = root_dir / "model"

        try:
            logger.info(f"Eagerly loading model from path: {model_path}")
            # Format as file URI (file:///C:/...) required by MLflow on Windows
            model_uri = Path(model_path).resolve().as_uri()
            self.pipeline = mlflow.sklearn.load_model(model_uri)
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model artifact: {e}")
            raise RuntimeError(f"Model initialization failed: {e}")

    def predict(self, data: Union[str, List[str], pd.DataFrame, Path]) -> pd.DataFrame:
        try:
            if isinstance(data, str):
                if data.endswith(".csv") or Path(data).exists():
                    df_in = pd.read_csv(data)
                    X = df_in.iloc[:, 0]
                else:
                    X = pd.Series([data])
            elif isinstance(data, list):
                if not data:
                    raise ValueError("Input list cannot be empty.")
                X = pd.Series(data)
            elif isinstance(data, pd.DataFrame):
                if data.empty:
                    raise ValueError("Input DataFrame cannot be empty.")
                X = data.iloc[:, 0]
            elif isinstance(data, Path):
                df_in = pd.read_csv(data)
                X = df_in.iloc[:, 0]
            else:
                raise TypeError("Unsupported input type.")

            predictions = self.pipeline.predict(X)
            logger.info(f"Inference complete | samples={len(X)}")

            return pd.DataFrame({
                "input_product": X.values,
                "predicted_category": predictions
            })

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise RuntimeError(f"Prediction failed: {e}")