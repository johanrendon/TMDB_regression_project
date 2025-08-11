from abc import ABC, abstractmethod
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class UnivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(
        self, df: pd.DataFrame, feature: str, box: bool = False, log: bool = False
    ) -> None:
        """Perform univariate analysis on a specific feature.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.
            feature (str): The name of the feature to analyze.
            box (bool, optional): Whether to include a boxplot. Defaults to False.
            log (bool, optional): Whether to use a logarithmic scale. Defaults to False.

        Returns:
            None
        """
        pass


class NumericalUnivariate(UnivariateAnalysisStrategy):
    def analyze(
        self, df: pd.DataFrame, feature: str, box: bool = True, log: bool = False
    ) -> None:
        """Plot the distribution of a numerical feature using histogram and optional boxplot.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.
            feature (str): The numerical feature to analyze.
            box (bool, optional): Whether to include a boxplot. Defaults to True.
            log (bool, optional): Whether to use a logarithmic scale. Defaults to False.

        Returns:
            None
        """
        if box:
            fig, axes = plt.subplots(1, 2, figsize=(20, 10))
            sns.histplot(data=df[feature], bins=30, kde=True, ax=axes[0], log_scale=log)
            sns.boxplot(data=df[feature], log_scale=log)
            print(df[feature].describe())
            plt.suptitle(f"Distribution of {feature}")
        else:
            plt.figure(figsize=(10, 6))
            sns.histplot(data=df[feature], kde=True, bins=30, log_scale=log)
            plt.title(f"Distribution of {feature}")

        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.show()


class CategoricalUnivariate(UnivariateAnalysisStrategy):
    def analyze(
        self, df: pd.DataFrame, feature: str, box: bool = False, log: bool = False
    ) -> None:
        """Plot the distribution of a categorical feature using a bar plot.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.
            feature (str): The categorical feature to analyze.
            box (bool, optional): Ignored for categorical features. Defaults to False.
            log (bool, optional): Ignored for categorical features. Defaults to False.

        Returns:
            None
        """
        plt.figure(figsize=(10, 6))
        sns.countplot(x=feature, data=df, palette="muted")
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Count")
        plt.show()


class UnivariateAnalyzer:
    def __init__(self, strategy: UnivariateAnalysisStrategy) -> None:
        """Initialize the UnivariateAnalyzer with a given strategy.

        Args:
            strategy (UnivariateAnalysisStrategy): The strategy to use for univariate analysis.
        """
        self._strategy: UnivariateAnalysisStrategy = strategy

    def set_strategy(self, strategy: UnivariateAnalysisStrategy) -> None:
        """Set a new strategy for univariate analysis.

        Args:
            strategy (UnivariateAnalysisStrategy): The new strategy to use.

        Returns:
            None
        """
        self._strategy = strategy

    def execute_analysis(
        self, df: pd.DataFrame, feature: str, box: bool = False, log: bool = False
    ) -> None:
        """Execute the univariate analysis using the current strategy.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.
            feature (str): The feature to analyze.
            box (bool, optional): Whether to include a boxplot (if applicable). Defaults to False.
            log (bool, optional): Whether to use a logarithmic scale (if applicable). Defaults to False.

        Returns:
            None
        """
        self._strategy.analyze(df, feature, box=box, log=log)
