from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd


class DataInspectionStrategy(ABC):
    @abstractmethod
    def inspect(self, df: pd.DataFrame) -> None:
        """Perform a specific type of inspection.

        Args:
            df (pd.DataFrame): DataFrame to inspect.

        Returns:
            None: Prints the inspection results directly.
        """
        pass


class DataTypesInspection(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame) -> None:
        """Inspect the data types and null values in the DataFrame.

        Args:
            df (pd.DataFrame): DataFrame to inspect.

        Returns:
            None: Prints data types and null values.
        """
        print("Data types and null values:")
        print(df.info())


class SummaryStatistics(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame) -> None:
        """Print numerical and categorical statistics.

        Args:
            df (pd.DataFrame): DataFrame to inspect.

        Returns:
            None: Prints descriptive statistics for numerical and
            categorical features.
        """
        print("Statistics (Numerical Features):")
        print(df.describe())
        print("Statistics (Categorical Features):")
        print(df.describe(include="O"))


class DataInspector:
    def __init__(self, strategy: DataInspectionStrategy) -> None:
        """Initialize the DataInspector with a given strategy.

        Args:
            strategy (DataInspectionStrategy): The strategy to use for
            data inspection.
        """
        self._strategy: DataInspectionStrategy = strategy

    def set_strategy(self, strategy: DataInspectionStrategy) -> None:
        """Set the strategy to use for data inspection.

        Args:
            strategy (DataInspectionStrategy): The new strategy to use.

        Returns:
            None
        """
        self._strategy = strategy

    def execute_inspection(self, df: pd.DataFrame) -> None:
        """Execute the inspection using the current strategy.

        Args:
            df (pd.DataFrame): DataFrame to inspect.

        Returns:
            None
        """
        self._strategy.inspect(df)
