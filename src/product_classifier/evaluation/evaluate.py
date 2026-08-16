import argparse
from pathlib import Path
import tempfile
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import logging
import warnings
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from product_classifier.env import EXPECTED_CATEGORIES
from .data_loader import DataLoader

logging.getLogger("mlflow").setLevel(logging.WARNING)
logging.getLogger("mlflow.utils.environment").setLevel(logging.ERROR)
logging.getLogger("mlflow.utils.uv_utils").setLevel(logging.ERROR)

def run_final_evaluation(run_id: str):
    root_dir = Path(__file__).resolve().parents[3]
    
    mlruns_dir = root_dir / "mlruns"
    db_file_path = mlruns_dir / "mlflow.db"
    
    mlflow.set_tracking_uri(f"sqlite:///{db_file_path.as_posix()}")
    
    experiment_name = "category_classification_evaluation"
    try:
        mlflow.create_experiment(experiment_name, artifact_location=mlruns_dir.as_uri())
    except Exception:
        pass
    mlflow.set_experiment(experiment_name)

    processed_dir = root_dir / "data" / "processed"
    eval_loader = DataLoader(processed_data_dir=processed_dir)
    X_val, y_val, X_prod = eval_loader.load_evaluation_data()

    val_mask = y_val.isin(EXPECTED_CATEGORIES)
    X_val, y_val = X_val[val_mask], y_val[val_mask]

    model_uri = f"runs:/{run_id}/model"
    print(f"Loading local MLflow model artifact from: {model_uri}")
    
    pipeline = mlflow.sklearn.load_model(model_uri=model_uri)

    with mlflow.start_run(run_name=f"Eval_Run_{run_id[:8]}"):
        mlflow.set_tag("source_training_run_id", run_id)

        val_preds = pipeline.predict(X_val)
        
        if len(val_preds) > 0 and isinstance(val_preds[0], (int, np.integer)):
            sorted_cats = sorted(EXPECTED_CATEGORIES)
            val_preds = [sorted_cats[int(p)] for p in val_preds]

        acc = accuracy_score(y_val, val_preds)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_val, val_preds, average="macro", zero_division=0
        )

        mlflow.log_metrics({
            "final_val_accuracy": acc,
            "final_val_precision": prec,
            "final_val_recall": rec,
            "final_val_f1": f1,
        })

        prod_preds = pipeline.predict(X_prod)
        if len(prod_preds) > 0 and isinstance(prod_preds[0], (int, np.integer)):
            sorted_cats = sorted(EXPECTED_CATEGORIES)
            prod_preds = [sorted_cats[int(p)] for p in prod_preds]

        with tempfile.TemporaryDirectory() as tmp_dir:
            val_df = pd.DataFrame({
                "product_name": X_val.values,
                "actual_category": y_val.values,
                "predicted_category": val_preds,
            })
            val_path = Path(tmp_dir) / "ground_truth_comparison.csv"
            val_df.to_csv(val_path, index=False)
            mlflow.log_artifact(str(val_path), artifact_path="evaluation_reports")

            prod_df = pd.DataFrame({
                "product_name": X_prod.values,
                "predicted_category": prod_preds,
            })
            prod_path = Path(tmp_dir) / "production_query_predictions.csv"
            prod_df.to_csv(prod_path, index=False)
            mlflow.log_artifact(str(prod_path), artifact_path="production_outputs")

        print(
            f"\nFinal Evaluation Complete.\n"
            f"Accuracy: {acc:.4f} | Macro F1: {f1:.4f}\n"
            f"Reports logged in local MLflow run artifacts."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run final evaluation using a specific MLflow run ID.")
    parser.add_argument("--run-id", type=str, required=True, help="The MLflow run ID of the trained model.")
    args = parser.parse_args()
    
    run_final_evaluation(run_id=args.run_id)