from pathlib import Path
import pandas as pd
import argparse


from src.preprocess import convert_dataframe_to_bool, create_binary_label

data_path = Path("data")
raw_path = data_path / "raw"
interim_path = data_path / "interim"
output_file_name = "train.parquet"

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="pipeline for running cloud classification model"
    )

    parser.add_argument(
        "step",
        default="preprocess",
        help="Which step to run",
        choices=["preprocess"],
    )

    args = parser.parse_args()

    if args.step == "preprocess":
        try:
            interim_path.mkdir(parents=True, exist_ok=False)
        except:
            print("Folder data/interim already exists!")

        df = pd.read_csv(raw_path / "train.csv")
        df_tmp = create_binary_label(df)
        df_processed = convert_dataframe_to_bool(df_tmp)
        df_processed.to_parquet(interim_path / output_file_name)
        print(f"The file is written to {interim_path / output_file_name}")
