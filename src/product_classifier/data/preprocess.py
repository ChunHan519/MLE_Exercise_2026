import csv
from pathlib import Path
import pandas as pd
from product_classifier.env import EXPECTED_CATEGORIES

EXPECTED_COLUMNS = ["product_name", "category"]


def load_data(input_data: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(input_data, (str, Path)):
        file_path = Path(input_data)
        with file_path.open("r", encoding="utf-8") as file:
            raw_lines = file.readlines()
        rows = raw_lines[1:]
        cleaned_lines = [row.replace(";;;;", "").rstrip("\n") for row in rows]
        parsed_rows = list(csv.reader(cleaned_lines))
        return pd.DataFrame(parsed_rows)
    return input_data.copy()


def fix_product_description_comma(df: pd.DataFrame) -> pd.DataFrame:
    processed_data = []
    for _, row in df.iterrows():
        values = [str(val).strip() for val in row if pd.notna(val) and str(val).strip() != ""]
        if not values:
            continue

        category = ""
        if values[-1] in EXPECTED_CATEGORIES:
            category = values[-1]
            product_values = values[:-1]
        else:
            product_values = values

        product_name = ",".join(product_values)
        processed_data.append({
            "product_name": product_name,
            "category": category
        })
    return pd.DataFrame(processed_data)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = EXPECTED_COLUMNS
    return df


def clean_product_description(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    def clean_name(name: str) -> str:
        if not isinstance(name, str):
            return name
        name = name.replace("\\", "").strip()
        if name.startswith('"') and name.endswith('"'):
            inner = name[1:-1]
            if inner and inner[-1].isdigit():
                name = name[1:]
            else:
                name = inner
        return name

    df["product_name"] = df["product_name"].apply(clean_name)
    return df


def clean_category(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "category" in df.columns:
        df["category"] = (
            df["category"]
            .str.replace('"', "", regex=False)
            .str.replace("\\", "", regex=False)
            .str.rstrip(";")
            .str.strip()
        )
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True)


def preprocess_data(input_data: str | Path | pd.DataFrame) -> pd.DataFrame:
    df = load_data(input_data)
    df = fix_product_description_comma(df)
    df = standardize_columns(df)
    df = clean_product_description(df)
    df = clean_category(df)
    df = remove_duplicates(df)
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
        train_df = preprocess_data(train_input)
        write_csv(train_df, str(train_output))
        print(f"Done process {training_file_name}")

    print(f"Process {validation_file_name}")
    val_input = raw_path / validation_file_name
    val_output = processed_path / validation_file_name
    if val_input.exists():
        val_df = preprocess_data(val_input)
        write_csv(val_df, str(val_output))
        print(f"Done process {validation_file_name}")