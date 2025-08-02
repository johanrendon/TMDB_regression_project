from os import name
from typing import List

import numpy as np
import pandas as pd

original_df: pd.DataFrame = pd.read_csv("./data/raw/TMDB_movie_dataset_v11.csv")


def rvote_average(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Conserva solo los valores de los votos que son mayores a 0
    """

    df = (
        dataframe[(dataframe["vote_average"] > 0) & (dataframe["vote_count"] > 0)]
        .reset_index(drop=True)
        .copy()
    )
    return df


def drop_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop: List[str] = [
        "backdrop_path",
        "keywords",
        "tagline",
        "imdb_id",
        "original_title",
        "poster_path",
        "spoken_languages",
        "homepage",
        "status",
    ]
    df_clean = dataframe.drop(columns=columns_to_drop, axis=1)

    return df_clean


def rbudget_revenue(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Conserva solo los valores de el budget y revenue que son mayores a 0
    """

    df = (
        dataframe[(dataframe["budget"] > 0) & (dataframe["revenue"] > 0)]
        .reset_index(drop=True)
        .copy()
    )
    return df


if __name__ == "__main__":
    cleaned_df = rbudget_revenue(drop_columns(rvote_average(original_df)))
    cleaned_df.to_csv("./data/interm/cleaned_df.csv", index=False)
