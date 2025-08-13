import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.io.xml import file_exists

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def save_image(
    filename: str, base_dir: str = "../artifacts/images", subdir: Optional[str] = None
) -> None:
    os.makedirs(base_dir, exist_ok=True)

    if subdir:
        path = os.path.join(base_dir, subdir)
    else:
        path = base_dir

    filepath = os.path.join(path, filename)
    plt.savefig(filepath)
    logging.info(f"Image {filename} saved in {path}")


class UnivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(
        self,
        df: pd.DataFrame,
        feature: str,
        box: bool = False,
        log: bool = False,
        save: bool = False,
        filename: str | None = None,
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
        self,
        df: pd.DataFrame,
        feature: str,
        box: bool = True,
        log: bool = False,
        save: bool = False,
        filename: Optional[str] = None,
    ) -> None:
        """Plot the distribution of a numerical feature using histogram and optional boxplot.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.
            feature (str): The numerical feature to analyze.
            box (bool, optional): Whether to include a boxplot. Defaults to True.
            log (bool, optional): Whether to use a logarithmic scale. Defaults to False.
            save (bool, optional): Save image. Defaults to False.
            filename (str, optional): Name for the image when save is True. Defaults to None.

        Returns:
            None
        """
        if box:
            fig, axes = plt.subplots(1, 2, figsize=(20, 10))
            sns.histplot(data=df[feature], bins=30, kde=True, ax=axes[0], log_scale=log)
            sns.boxplot(x=df[feature], ax=axes[1])
            axes[0].set_title(f"Histogram - {feature}")
            axes[1].set_title(f"Boxplot - {feature}")
            fig.suptitle(f"Distribution of {feature}")

        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(data=df[feature], kde=True, bins=30, log_scale=log, ax=ax)
            ax.set_title(f"Distribution of {feature}")

        if save:
            if not filename:
                raise ValueError(
                    "The 'filename' parameter cannot be empty if 'save' is True."
                )
            save_image(filename)

        plt.show()
        # plt.close(fig)

        print(df[feature].describe())


class CategoricalUnivariate(UnivariateAnalysisStrategy):
    def analyze(
        self,
        df: pd.DataFrame,
        feature: str,
        box: bool = False,
        log: bool = False,
        save: bool = False,
        filename: str | None = None,
    ) -> None:
        """Plot the distribution of a categorical feature using a bar plot.

        Args:
            df (pd.DataFrame): The DataFrame containing the data.
            feature (str): The numerical feature to analyze.
            box (bool, optional): Whether to include a boxplot. Defaults to True.
            save (bool, optional): Save image. Defaults to False.
            filename (str, optional): Name for the image when save is True. Defaults to None.

        Returns:
            None
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x=feature, data=df, palette="muted", ax=ax)
        ax.set_title(f"Distribution of {feature}")
        ax.set_xlabel(feature)
        ax.set_ylabel("Count")
        if save:
            if not filename:
                raise ValueError(
                    "The 'filename' parameter cannot be empty if 'save' is True."
                )
            save_image(filename)
        plt.show()
        # plt.close(fig)

        print(df[feature].describe())


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
        self,
        df: pd.DataFrame,
        feature: str,
        box: bool = False,
        log: bool = False,
        save: bool = False,
        filename: str | None = None,
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
        self._strategy.analyze(
            df, feature, box=box, log=log, save=save, filename=filename
        )
