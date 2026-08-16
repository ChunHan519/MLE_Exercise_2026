import argparse
from pathlib import Path
import mlflow
import mlflow.sklearn
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation on unlabeled query data using an MLflow run_id."
    )
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="MLflow run_id of the trained model",
    )
    args = parser.parse_args()

    root_dir = Path(__file__).resolve().parents[3]
    mlruns_dir = root_dir / "mlruns"
    db_file_path = mlruns_dir / "mlflow.db"

    mlflow.set_tracking_uri(f"sqlite:///{db_file_path.as_posix()}")

    input_path = root_dir / "data" / "processed" / "Query_and_Validation_data.csv"
    output_path = root_dir / "data" / "processed" / f"load_test_result_{args.run_id}.csv"

    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found at: {input_path}")

    df = pd.read_csv(input_path)
    unlabeled_mask = df["category"].isna() | (
        df["category"].astype(str).str.strip() == ""
    )
    unlabeled_df = df[unlabeled_mask].copy()

    if unlabeled_df.empty:
        print("No unlabeled records found.")
        return

    print(
        f"Processing {len(unlabeled_df)} unlabeled records using MLflow run_id: {args.run_id}..."
    )

    model_uri = f"runs:/{args.run_id}/model"
    print(f"Loading local MLflow model artifact from: {model_uri}")
    model = mlflow.sklearn.load_model(model_uri=model_uri)

    predictions = model.predict(unlabeled_df["product_name"])

    result_df = pd.DataFrame({
        "product_name": unlabeled_df["product_name"],
        "category": predictions,
    })

    result_df.to_csv(output_path, index=False)

    print(f"Done! Results saved to: {output_path}")


if __name__ == "__main__":
    main()