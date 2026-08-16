import csv
import pandas as pd
from pathlib import Path


EXPECTED_COLUMNS = ["product_name", "category"]


def preload_csv(path: str) -> pd.DataFrame:
    with Path(path).open("r", encoding="utf-8") as file:
        rows = file.readlines()

    # Remove header
    rows = rows[1:]

    # Remove ;;;; but keep trailing comma
    rows = [
        row.replace(";;;;", "").rstrip("\n")
        for row in rows
    ]

    return pd.DataFrame(csv.reader(rows))


def fix_product_description_comma(df: pd.DataFrame) -> pd.DataFrame:
    product_names = []
    categories = []

    for _, row in df.iterrows():
        values = [
            str(value).strip()
            for value in row
            if pd.notna(value)
        ]

        # Detect trailing comma.
        # csv.reader represents it as an empty final value.
        no_category = values and values[-1] == ""

        # Remove empty values
        values = [
            value
            for value in values
            if value.strip()
        ]

        if not values:
            continue

        if no_category:
            # Entire row is product description
            product_name = ",".join(values)
            category = ""

        else:
            # Last value is category
            product_name = ",".join(values[:-1])
            category = values[-1]

        product_names.append(product_name)
        categories.append(category)

    return pd.DataFrame({
        "product_name": product_names,
        "category": categories,
    })


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = EXPECTED_COLUMNS

    return df


def strip_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for column in df.columns:
        df[column] = df[column].str.strip()

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
    df = strip_columns(df)
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

    print(f"Process {training_file_name}")
    validation_df = preload_csv(rf"{base_path}\{raw_path}\{validation_file_name}")
    processed_validation_df = preprocess_data(validation_df)
    write_csv(processed_validation_df, rf"{base_path}\{processed_path}\{validation_file_name}")
    print(f"Done process {training_file_name}")

    