import logging
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class MissingValueStrategy(ABC):
    @abstractmethod
    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Abstract method to handle missing values.

        Args:
            df (pd.DataFrame): DataFrame containing missing values.

        Returns:
            pd.DataFrame: DataFrame with missing values handled.
        """
        pass


class DropMissingValues(MissingValueStrategy):
    def __init__(self, axis: int = 0, thresh: int = None) -> None:
        """Drop missing values using a specific strategy.

        Args:
            axis (int, optional): Axis to drop values from.
                `0` for rows, `1` for columns. Defaults to 0.
            thresh (int, optional): Minimum number of non-null values required
                to keep the row/column. Defaults to None.
        """
        self._axis = axis
        self._thresh = thresh

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove missing values based on the specified axis and threshold.

        Args:
            df (pd.DataFrame): DataFrame containing missing values.

        Returns:
            pd.DataFrame: DataFrame with missing values removed.
        """
        logging.info(
            f"Dropping missing values on axis {self._axis} with threshold {self._thresh}"
        )
        df_cleaned = df.dropna(axis=self._axis, thresh=self._thresh)
        logging.info("Missing values dropped.")
        return df_cleaned


class FillMissingValues(MissingValueStrategy):
    def __init__(self, method="mean", fill_value=None) -> None:
        """Fill missing values using the specified method or constant value.

        Args:
            method (str, optional): Method to fill missing values.
                Options: `"mean"`, `"median"`, `"mode"`, `"constant"`. Defaults to `"mean"`.
            fill_value (Any, optional): Value used when `method` is `"constant"`. Defaults to None.
        """
        self.fill_value = fill_value
        self.method = method

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values using the specified method or constant value.

        Args:
            df (pd.DataFrame): DataFrame containing missing values.

        Returns:
            pd.DataFrame: DataFrame with missing values filled.
        """
        logging.info(f"Filling missing values using method: {self.method}")

        df_cleaned = df.copy()
        if self.method == "mean":
            numeric_columns = df_cleaned.select_dtypes(include="number").columns
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(
                df[numeric_columns].mean()
            )
        elif self.method == "median":
            numeric_columns = df_cleaned.select_dtypes(include="number").columns
            df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(
                df[numeric_columns].median()
            )
        elif self.method == "mode":
            for column in df_cleaned.columns:
                df_cleaned[column].fillna(df[column].mode().iloc[0], inplace=True)
        elif self.method == "constant":
            df_cleaned = df_cleaned.fillna(self.fill_value)
        else:
            logging.warning(
                f"Unknown method '{self.method}'. No missing values handled."
            )

        logging.info("Missing values filled.")
        return df_cleaned


class MissingValueHandler:
    def __init__(self, strategy: MissingValueStrategy):
        """Initialize the MissingValueHandler with a specific strategy.

        Args:
            strategy (MissingValueStrategy): Strategy used to handle missing values.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: MissingValueStrategy):
        """Set a new strategy for handling missing values.

        Args:
            strategy (MissingValueStrategy): New strategy to handle missing values.
        """
        logging.info("Switching missing value handling strategy.")
        self._strategy = strategy

    def handle_missing_values(
        self, df: pd.DataFrame, save: bool = False, name: str = ""
    ) -> pd.DataFrame:
        """Execute the missing value handling using the current strategy.

        Args:
            df (pd.DataFrame): DataFrame containing missing values.
            save (bool, optional): Whether to save the cleaned DataFrame as a CSV file. Defaults to False.
            name (str, optional): Name of the output file (without extension).
                Required if `save` is True. Defaults to "".

        Returns:
            pd.DataFrame: DataFrame with missing values handled.

        Raises:
            ValueError: If `save` is True and `name` is empty.
        """
        logging.info("Executing missing value handling strategy.")
        df_cleaned = self._strategy.handle(df)

        if save:
            if not name:
                raise ValueError(
                    "The 'name' parameter cannot be empty if 'save' is True."
                )

            output_path = Path(f"../data/interm/{name}.csv")
            output_path.parent.mkdir(parents=True, exist_ok=True)

            logging.info(f"Saving DataFrame without missing values to: {output_path}")
            df_cleaned.to_csv(output_path, index=False)

        return df_cleaned

