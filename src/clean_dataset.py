from typing import List

import pandas as pd

original_df: pd.DataFrame = pd.read_csv("./data/raw/TMDB_movie_dataset_v11.csv")


def rvote_average(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Filter movies with vote_average and vote_count greater than zero.

    Args:
        dataframe (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: Filtered DataFrame containing only rows with positive
        vote_average and vote_count.
    """
    df = (
        dataframe[(dataframe["vote_average"] > 0) & (dataframe["vote_count"] > 0)]
        .reset_index(drop=True)
        .copy()
    )
    return df


def drop_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Remove unnecessary columns from the DataFrame.

    Args:
        dataframe (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: DataFrame without the specified columns.
    """
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
    return dataframe.drop(columns=columns_to_drop, axis=1)


def rbudget_revenue(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Filter movies with budget and revenue greater than zero.

    Args:
        dataframe (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: Filtered DataFrame containing only rows with positive
        budget and revenue.
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
