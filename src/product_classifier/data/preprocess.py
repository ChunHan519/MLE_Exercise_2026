import csv
from pathlib import Path
import pandas as pd
from product_classifier.env import EXPECTED_CATEGORIES

EXPECTED_COLUMNS = ["product_name", "category"]


def preprocess_file(file_path: Path) -> pd.DataFrame:
    with file_path.open("r", encoding="utf-8") as file:
        rows = file.readlines()

    rows = rows[1:]
    cleaned_rows = [row.replace(";;;;", "").rstrip("\n") for row in rows]

    processed_data = []

    for raw_line in cleaned_rows:
        if not raw_line.strip():
            continue

        parsed = list(csv.reader([raw_line]))[0]
        
        ends_with_comma = raw_line.strip().endswith(",")

        if ends_with_comma:
            if parsed and parsed[-1] == "":
                product_name = ",".join(parsed[:-1])
            else:
                product_name = ",".join(parsed)
            category = ""
        else:
            if len(parsed) > 1:
                category = parsed[-1].strip()
                product_name = ",".join(parsed[:-1])
            else:
                product_name = parsed[0] if parsed else ""
                category = ""

        processed_data.append({
            "product_name": product_name,
            "category": category
        })

    df = pd.DataFrame(processed_data, columns=EXPECTED_COLUMNS)

    df["product_name"] = (
        df["product_name"]
        .str.replace('"', "", regex=False)
        .str.replace("\\", "", regex=False)
        .str.strip()
    )

    if "category" in df.columns:
        df["category"] = (
            df["category"]
            .str.replace('"', "", regex=False)
            .str.replace("\\", "", regex=False)
            .str.strip()
        )

    df = df.drop_duplicates().reset_index(drop=True)

    return df


def write_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)


if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parents[3]
    raw_path = root_dir / "data" / "raw"
    processed_path = root_dir / "data" / "processed"

    processed_path.mkdir(parents=True, exist_ok=True)

    training_file_name = "Training_data.csv"
    validation_file_name = "Query_and_Validation_data.csv"

    print(f"Process {training_file_name}")
    train_input = raw_path / training_file_name
    train_output = processed_path / training_file_name
    if train_input.exists():
        train_df = preprocess_file(train_input)
        write_csv(train_df, str(train_output))
        print(f"Done process {training_file_name}")

    print(f"Process {validation_file_name}")
    val_input = raw_path / validation_file_name
    val_output = processed_path / validation_file_name
    if val_input.exists():
        val_df = preprocess_file(val_input)
        write_csv(val_df, str(val_output))
        print(f"Done process {validation_file_name}")