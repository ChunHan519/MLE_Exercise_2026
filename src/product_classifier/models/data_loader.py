from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


class DataLoader:
    """Loads Training_data.csv and splits it for internal model training."""

    def __init__(self, processed_data_dir: Path):
        self.processed_data_dir = processed_data_dir

    def load_train_splits(
        self, test_size: float = 0.2, random_state: int = 42
    ) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        train_path = self.processed_data_dir / "Training_data.csv"
        df = pd.read_csv(train_path).dropna(subset=["product_name", "category"])

        X = df["product_name"]
        y = df["category"]

        return train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )