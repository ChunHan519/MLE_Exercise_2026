import csv
import pandas as pd
from pathlib import Path


EXPECTED_COLUMNS = ["product_name", "category"]


def preload_csv(path: str) -> pd.DataFrame:
    with Path(path).open("r", encoding="utf-8") as file:
        rows = file.readlines()

    # Remove header
    rows = rows[1:]

    # Remove trailing ";;;;"
    rows = [
        row.replace(";;;;", "").rstrip("\n")
        for row in rows
    ]

    return pd.DataFrame(csv.reader(rows))


def fix_product_description_comma(df: pd.DataFrame) -> pd.DataFrame:
    cleaned_rows = []

    for _, row in df.iterrows():
        # Check and strip all values in the row
        values = []
        for value in row:
            if pd.notna(value):
                value = str(value).strip()
                values.append(value)

        if not values:
            continue

        # Check if the category is missing
        category_missing = values[-1] == ""

        # Remove empty values
        values = [value for value in values if value != ""]

        # Join product name
        if category_missing:
            product_name = ",".join(values)
            category = ""

        else:
            product_name = ",".join(values[:-1])
            category = values[-1]

        cleaned_rows.append({
            "product_name": product_name,
            "category": category
        })

    return pd.DataFrame(cleaned_rows)


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = EXPECTED_COLUMNS

    return df


def clean_product_description(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["product_name"] = (
        df["product_name"]
        .str.replace('"', "", regex=False)
        .str.replace("\\", "", regex=False)
        .str.strip()
    )

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = fix_product_description_comma(df)
    df = standardize_columns(df)
    df = clean_product_description(df)
    df = remove_duplicates(df)

    return df


def write_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)


if __name__ == "__main__":
    base_path = r"C:\Users\tiowt\Desktop\Work\Interview\NIQ_2026\TiowChunHan_NIQ_Interview"
    raw_path = r"data\raw"
    processed_path = r"data\processed"

    training_file_name = "Training_data.csv"
    validation_file_name = "Query_and_Validation_data.csv"

    print(f"Process {training_file_name}")
    training_df = preload_csv(rf"{base_path}\{raw_path}\{training_file_name}")
    processed_training_df = preprocess_data(training_df)
    write_csv(processed_training_df, rf"{base_path}\{processed_path}\{training_file_name}")
    print(f"Done process {training_file_name}")

    print(f"Process {validation_file_name}")
    validation_df = preload_csv(rf"{base_path}\{raw_path}\{validation_file_name}")
    processed_validation_df = preprocess_data(validation_df)
    write_csv(processed_validation_df, rf"{base_path}\{processed_path}\{validation_file_name}")
    print(f"Done process {validation_file_name}")

     