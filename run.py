from pathlib import Path
import pandas as pd
import argparse

from sklearn.model_selection import train_test_split

from src.preprocess import convert_dataframe_to_bool, create_binary_label
from src.fasttext_utility import df_to_txt

data_path = Path("data")
raw_path = data_path / "raw"
interim_path = data_path / "interim"

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="pipeline for running cloud classification model"
    )

    parser.add_argument(
        "step",
        default="preprocess-train",
        help="Which step to run",
        choices=[
            "preprocess-train",
            "preprocess-test",
            "fasttext-prepare",
        ],
    )

    args = parser.parse_args()

    if args.step == "preprocess-train":
        output_file_name = "train.parquet"
        try:
            interim_path.mkdir(parents=True, exist_ok=False)
        except:
            print("Folder data/interim already exists!")

        df = pd.read_csv(raw_path / "train.csv")
        df_tmp = create_binary_label(df)
        df_processed = convert_dataframe_to_bool(df_tmp)
        df_processed.to_parquet(interim_path / output_file_name)
        print(f"The file is written to {interim_path / output_file_name}")

    elif args.step == "preprocess-test":
        output_file_name = "test.parquet"
        df_test = pd.read_csv(raw_path / "test_private_expanded.csv")
        df_test = create_binary_label(df_test, target_col="toxicity")
        df_test = convert_dataframe_to_bool(df_test)
        df_test = df_test.head(10000).reset_index(drop=True)
        df_test.to_parquet(interim_path / "test.parquet")
        print(f"The file is written to {interim_path / output_file_name}")

    elif args.step == "fasttext-prepare":
        df = pd.read_parquet(interim_path / "train.parquet")
        df["comment_text"] = df["comment_text"].str.lower()
        df_train, df_valid = train_test_split(df, test_size=0.2, random_state=32)
        for a_df, name in zip([df_train, df_valid], ["train", "valid"]):
            output_file_name = interim_path / f"fasttext_{name}.txt"
            df_to_txt(
                a_df,
                input_col="comment_text",
                label_col="label",
                output_file_path=output_file_name,
            )
            print(f"The file is written to {interim_path / output_file_name}")
