from typing import List

import numpy as np
import pandas as pd

identity_columns = [
    "male",
    "female",
    "homosexual_gay_or_lesbian",
    "christian",
    "jewish",
    "muslim",
    "black",
    "white",
    "psychiatric_or_mental_illness",
]


def convert_dataframe_to_bool(
    df: pd.DataFrame, identity_columns: List[str] = identity_columns
) -> pd.DataFrame:
    """Convert identity columns to boolen columns"""
    bool_df = df.copy()
    for col in identity_columns:
        bool_df[col] = np.where(df[col] >= 0.5, True, False)
    return bool_df


def create_binary_label(
    df: pd.DataFrame, target_col: str = "target", label_col: str = "label"
) -> pd.DataFrame:
    """Create binary labels from continuous information"""
    df_new = df.copy()
    df_new[label_col] = (df_new[target_col] >= 0.5).astype(int)
    return df_new
