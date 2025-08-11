import logging
from pathlib import Path
from typing import List, Union

import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def save_data(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Save a DataFrame to a CSV file in a predefined directory.

    The function builds a path inside the 'data/interm/' directory,
    ensures the directory exists, and then saves the provided DataFrame.

    Args:
        name (str): The base name for the output file (without the .csv extension).
        df (pd.DataFrame): The DataFrame to save.

    Returns:
        pd.DataFrame: The same DataFrame passed as input, allowing method chaining if needed.
    """
    output_path = Path("data/interm/") / f"{name}.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"Saving DataFrame to: {output_path}")
    df.to_csv(output_path, index=False)
    return df


class CleanHandler:
    """A container class for data cleaning methods on DataFrames.

    This class groups static functionalities related to dataset preparation and cleaning.
    """

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
        """Clean missing values in a DataFrame using various methods.

        This function takes a DataFrame and applies a strategy to handle NaN values.
        It can fill (impute) them using mean, median, mode, or a constant value,
        or remove the rows/columns containing them.

        Args:
            df (pd.DataFrame): The input DataFrame containing data to clean.
            method (str, optional): The method to use. Valid values are:
                'mean', 'median', 'mode', 'constant', 'drop'. Defaults to "mean".
            fill_value (Union[str, int, float], optional): The value to use when
                the method is 'constant'. Defaults to None.
            axis (int, optional): Axis for the 'drop' method. 0 for rows, 1 for columns.
                Defaults to 0.
            thresh (int, optional): For the 'drop' method, the minimum number of
                non-null values required to avoid removal. Defaults to None.
            save (bool, optional): If True, saves the cleaned DataFrame to a CSV file.
                Defaults to False.
            name (str, optional): The file name (without extension) if `save` is True.
                Required if `save` is True. Defaults to "".

        Returns:
            pd.DataFrame: A new DataFrame with missing values handled.

        Raises:
            ValueError: If `save` is True but `name` is not provided.
            ValueError: If `method` is 'constant' but no `fill_value` is provided.
        """
        logging.info(f"Starting missing value cleaning with method: '{method}'")

        df_cleaned = df.copy()

        if method in ["mean", "median"]:
            numeric_columns = df_cleaned.select_dtypes(include="number").columns
            if numeric_columns.empty:
                logging.warning(
                    "No numeric columns found to impute using mean or median."
                )
            else:
                if method == "mean":
                    fill_values = df_cleaned[numeric_columns].mean()
                else:
                    fill_values = df_cleaned[numeric_columns].median()
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
            final_rows = len(df_cleaned)
            logging.info(
                f"Removed {original_rows - final_rows} rows/columns with null values."
            )

        else:
            logging.warning(
                f"Unknown method '{method}'. Missing values were not handled."
            )
            return df_cleaned

        logging.info("Missing value cleaning process completed.")

        if save:
            if not name:
                raise ValueError(
                    "The 'name' parameter cannot be empty if 'save' is True."
                )
            return save_data(name, df_cleaned)

        return df_cleaned

    @staticmethod
    def select_features(
        df: pd.DataFrame, features: List[str], save: bool = False, name: str = ""
    ) -> pd.DataFrame:
        """Select a subset of features (columns) from a DataFrame.

        This function creates a copy of the DataFrame containing only the columns
        specified in the features list. Optionally, it can save the resulting DataFrame
        to a CSV file.

        Args:
            df (pd.DataFrame): The original DataFrame from which features will be selected.
            features (List[str]): A list of column names to select.
            save (bool, optional): If True, saves the DataFrame with selected features.
                Defaults to False.
            name (str, optional): The file name (without extension) to save if `save` is True.
                Required if `save` is True. Defaults to "".

        Returns:
            pd.DataFrame: A new DataFrame containing only the selected columns.

        Raises:
            ValueError: If `save` is True but `name` is not provided.
            KeyError: If any feature in the `features` list is not found in the DataFrame columns.
        """
        logging.info(f"Selecting {len(features)} features: {features}")

        df_selected: pd.DataFrame = df[features].copy()

        if save:
            if not name:
                raise ValueError(
                    "The 'name' parameter cannot be empty if 'save' is True."
                )
            return save_data(name, df_selected)

        return df_selected
