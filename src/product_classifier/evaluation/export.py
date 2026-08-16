import argparse
from pathlib import Path
import mlflow
import mlflow.sklearn
import logging

logging.getLogger("mlflow").setLevel(logging.WARNING)


def export_model(run_id: str):
    root_dir = Path(__file__).resolve().parents[3]
    
    mlruns_dir = root_dir / "mlruns"
    db_file_path = mlruns_dir / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_file_path.as_posix()}")

    model_uri = f"runs:/{run_id}/model"
    print(f"Loading model from MLflow run: {model_uri}")
    
    try:
        pipeline = mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        raise RuntimeError(f"Failed to load model from run ID {run_id}: {e}")

    export_dir = root_dir / "model"
    export_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting model to project root: {export_dir}")
    mlflow.sklearn.save_model(sk_model=pipeline, path=str(export_dir))
    
    print("\nModel export complete! The `model/` folder in your project root is ready for Dockerization.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export MLflow model to project root for Docker.")
    parser.add_argument("--run-id", type=str, required=True, help="The MLflow run ID of the model to export.")
    args = parser.parse_args()
    
    export_model(run_id=args.run_id)