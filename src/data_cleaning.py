import logging
from pathlib import Path
from typing import List, Union

import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def save_data(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Save a DataFrame to a CSV file in a predefined directory."""
    output_path = Path("data/interm/") / f"{name}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"Saving DataFrame to: {output_path}")
    df.to_csv(output_path, index=False)
    return df


class CleanHandler:
    """A container class for data cleaning methods on DataFrames."""

    @staticmethod
    def _save_if_requested(df: pd.DataFrame, save: bool, name: str) -> pd.DataFrame:
        """Helper method to save DataFrame if requested."""
        if save:
            if not name:
                raise ValueError(
                    "The 'name' parameter cannot be empty if 'save' is True."
                )
            return save_data(name, df)
        return df

    @staticmethod
    def clean_missing_values(
        df: pd.DataFrame,
        method: str = "mean",
        fill_value: Union[str, int, float] = None,
        axis: int = 0,
        thresh: int = None,
        save: bool = False,
        name: str = "",
    ) -> pd.DataFrame:
        logging.info(f"Starting missing value cleaning with method: '{method}'")
        df_cleaned = df.copy()

        if method in ["mean", "median"]:
            numeric_columns = df_cleaned.select_dtypes(include="number").columns
            if numeric_columns.empty:
                logging.warning(
                    "No numeric columns found to impute using mean or median."
                )
            else:
                fill_values = (
                    df_cleaned[numeric_columns].mean()
                    if method == "mean"
                    else df_cleaned[numeric_columns].median()
                )
                df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(
                    fill_values
                )

        elif method == "mode":
            for column in df_cleaned.columns:
                df_cleaned[column] = df_cleaned[column].fillna(
                    df[column].mode().iloc[0]
                )

        elif method == "constant":
            if fill_value is None:
                raise ValueError(
                    "A 'fill_value' must be provided for the 'constant' method."
                )
            df_cleaned = df_cleaned.fillna(fill_value)

        elif method == "drop":
            original_rows = len(df_cleaned)
            df_cleaned = df_cleaned.dropna(axis=axis, thresh=thresh)
            logging.info(
                f"Removed {original_rows - len(df_cleaned)} rows/columns with null values."
            )

        else:
            logging.warning(
                f"Unknown method '{method}'. Missing values were not handled."
            )
            return df_cleaned

        logging.info("Missing value cleaning process completed.")
        return CleanHandler._save_if_requested(df_cleaned, save, name)

    @staticmethod
    def select_features(
        df: pd.DataFrame, features: List[str], save: bool = False, name: str = ""
    ) -> pd.DataFrame:
        logging.info(f"Selecting {len(features)} features: {features}")
        df_selected = df[features].copy()
        return CleanHandler._save_if_requested(df_selected, save, name)

    @staticmethod
    def remove_invalid_rows(
        df: pd.DataFrame, save: bool = False, name: str = ""
    ) -> pd.DataFrame:
        logging.info("Removing rows with invalid numeric values.")
        df_selected = df[
            (df["vote_average"] > 0)
            & (df["vote_count"] > 0)
            & (df["revenue"] > 0)
            & (df["budget"] > 0)
        ]
        return CleanHandler._save_if_requested(df_selected, save, name)
