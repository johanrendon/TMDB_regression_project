import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

import pandas as pd

# Basic logging configuration to display process messages.
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class FeatureSelectionStrategy(ABC):
    """Abstract base class (ABC) defining the interface for feature selection strategies.

    Acts as a contract that all concrete feature selection strategies must implement.
    This ensures that the `FeatureSelectionHandler` can work with any strategy
    following this interface.
    """

    @abstractmethod
    def select(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """Abstract method to select features from a DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame from which features will be selected.
            features (List[str]): The list of features to consider or use in the strategy.

        Returns:
            pd.DataFrame: A new DataFrame containing only the selected features.
        """
        pass


class SelectFeatures(FeatureSelectionStrategy):
    """Concrete strategy that selects columns based on a list of names.

    This is a simple implementation of `FeatureSelectionStrategy` that directly
    selects the columns specified in the `features` list.
    """

    def select(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """Select a subset of columns from the DataFrame.

        Creates a copy of the input DataFrame containing only the columns whose
        names are specified in the `features` list.

        Args:
            df (pd.DataFrame): The input DataFrame.
            features (List[str]): The list of column names to select.

        Returns:
            pd.DataFrame: A copy of the DataFrame with the selected columns.
        """
        logging.info(f"Selecting {len(features)} features: {features}")
        df_selected = df[features].copy()
        return df_selected


class FeatureSelectionHandler:
    """Manages and executes a feature selection strategy.

    This class follows the Strategy Design Pattern. It maintains a reference
    to a strategy object and delegates the selection task to it. The strategy
    can be dynamically changed at runtime.

    Attributes:
        _strategy (FeatureSelectionStrategy): The current feature selection strategy.
    """

    def __init__(self, strategy: FeatureSelectionStrategy) -> None:
        """Initialize the FeatureSelectionHandler with an initial strategy.

        Args:
            strategy (FeatureSelectionStrategy): The feature selection strategy to use.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: FeatureSelectionStrategy) -> None:
        """Change the feature selection strategy at runtime.

        Args:
            strategy (FeatureSelectionStrategy): The new strategy to set.
        """
        logging.info(f"Changing selection strategy to: {type(strategy).__name__}")
        self._strategy = strategy

    def execute_selection(
        self, df: pd.DataFrame, features: List[str], save: bool = False, name: str = ""
    ) -> Optional[pd.DataFrame]:
        """Execute the feature selection process on a DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame.
            features (List[str]): The list of features to consider.
            save (bool, optional): If True, saves the result to a CSV and returns None.
                                   If False, returns the DataFrame with selected features.
                                   Defaults to False.
            name (str, optional): The file name if `save` is True.

        Returns:
            Optional[pd.DataFrame]: The DataFrame with selected features if `save` is False,
                                    otherwise None.

        Raises:
            ValueError: If `save` is True but no `name` is provided.
        """
        logging.info("Executing selection with strategy.")

        features_selected: pd.DataFrame = self._strategy.select(df, features)

        if save:
            if not name:
                raise ValueError(
                    "The 'name' parameter cannot be empty if 'save' is True."
                )

            output_path = Path(f"../data/interm/{name}.csv")
            output_path.parent.mkdir(parents=True, exist_ok=True)

            logging.info(f"Saving selected features to: {output_path}")
            features_selected.to_csv(output_path, index=False)

            return features_selected
        else:
            return features_selected
