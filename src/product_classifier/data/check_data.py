import argparse
import pandas as pd


def check_data(csv_path):
    df = pd.read_csv(csv_path)
    print(df)
    return df


def check_shape(df):
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True)
    parser.add_argument(
        "--function",
        choices=["check_data", "check_shape"],
        required=True
    )

    args = parser.parse_args()

    if args.function == "check_data":
        check_data(args.csv_path)

    elif args.function == "check_shape":
        df = pd.read_csv(args.csv_path)
        check_shape(df)