from pathlib import Path
import time
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from product_classifier.env import EXPECTED_CATEGORIES
from .classifier import (
    BaseClassifier,
    LinearSVMClassifier,
    LogisticRegressionClassifier,
    XGBoostClassifier,
)
from .data_loader import DataLoader


def train_and_log():
    root_dir = Path(__file__).resolve().parents[3]
    
    mlruns_dir = root_dir / "mlflow_runs"
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    
    db_file_path = mlruns_dir / "mlflow.db"
    
    mlflow.set_tracking_uri(f"sqlite:///{db_file_path.as_posix()}")
    mlflow.set_experiment("category_classification_training")

    mlflow.sklearn.autolog(log_models=True, silent=True)

    processed_dir = root_dir / "data" / "processed"
    loader = DataLoader(processed_data_dir=processed_dir)
    X_train, X_val, y_train, y_val = loader.load_train_splits()

    # Filter data to only include the 5 fixed categories
    train_mask = y_train.isin(EXPECTED_CATEGORIES)
    X_train, y_train = X_train[train_mask], y_train[train_mask]

    val_mask = y_val.isin(EXPECTED_CATEGORIES)
    X_val, y_val = X_val[val_mask], y_val[val_mask]

    # Use fixed categories for a stable label mapping
    label_mapping = {
        cat: idx for idx, cat in enumerate(sorted(EXPECTED_CATEGORIES))
    }

    classifiers: list[BaseClassifier] = [
        LogisticRegressionClassifier(),
        LinearSVMClassifier(),
        XGBoostClassifier(label_mapping=label_mapping),
    ]

    for model in classifiers:
        with mlflow.start_run(run_name=f"Train_{model.name}") as run:
            print(f"Training and logging {model.name} to local MLflow run: {run.info.run_id}")
            start_time = time.time()

            model.fit(X_train, y_train)
            training_time = round(time.time() - start_time, 2)

            preds = model.predict(X_val)
            acc = accuracy_score(y_val, preds)
            prec, rec, f1, _ = precision_recall_fscore_support(
                y_val, preds, average="macro", zero_division=0
            )

            mlflow.log_metrics({
                "internal_val_accuracy": acc,
                "internal_val_precision": prec,
                "internal_val_recall": rec,
                "internal_val_f1": f1,
                "training_time_sec": training_time,
            })

            if model.pipeline:
                input_ex = X_train.to_frame().head(3) if isinstance(X_train, pd.Series) else X_train.head(3)
                mlflow.sklearn.log_model(
                    sk_model=model.pipeline,
                    artifact_path="model",
                    input_example=input_ex,
                    skops_trusted_types=[
                        "xgboost.core.Booster",
                        "xgboost.sklearn.XGBClassifier",
                    ],
                )

            print(f"Finished {model.name} | Val Acc: {acc:.4f} | Val F1: {f1:.4f}\n")


if __name__ == "__main__":
    train_and_log()