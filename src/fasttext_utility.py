import pandas as pd


def df_to_txt(
    df: pd.DataFrame, input_col: str, label_col: str, output_file_path: str
) -> None:
    with open(output_file_path, "w") as f:
        for label, feature in zip(df[label_col], df[input_col]):
            one_line = (
                "__label__"
                + str(label).replace("\n", "").replace("\r", "")
                + " "
                + str(feature).replace("\n", "").replace("\r", "")
                + "\n"
            )
            f.write(one_line)
