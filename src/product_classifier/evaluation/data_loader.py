from pathlib import Path
import pandas as pd


class DataLoader:
    """Loads Query_and_Validation_data.csv and separates ground truth vs query data."""

    def __init__(self, processed_data_dir: Path):
        self.processed_data_dir = processed_data_dir

    def load_evaluation_data(self) -> tuple[pd.Series, pd.Series, pd.Series]:
        query_val_path = (
            self.processed_data_dir / "Query_and_Validation_data.csv"
        )
        df = pd.read_csv(query_val_path)

        # Marked rows -> Ground truth final evaluation
        val_mask = df["category"].notna() & (df["category"].str.strip() != "")
        val_df = df[val_mask]
        X_val, y_val = val_df["product_name"], val_df["category"]

        # Unmarked rows -> Production simulation query rows
        prod_df = df[~val_mask]
        X_prod = prod_df["product_name"]

        return X_val, y_val, X_prod