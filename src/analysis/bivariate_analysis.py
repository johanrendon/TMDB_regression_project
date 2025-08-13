import logging
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


class BivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(
        self, df: pd.DataFrame, feature1: str, feature2: str, corr: bool = False
    ) -> None:
        pass


class NumericalVsNumerical(BivariateAnalysisStrategy):
    def analyze(
        self, df: pd.DataFrame, feature1: str, feature2: str, corr: bool = False
    ) -> None:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=df, x=feature1, y=feature2)
        plt.title(f"{feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.show()
        if corr:
            print(np.corrcoef(df[feature1], df[feature2])[0, 1])


class NumericalVsCategoricalAnalysis(BivariateAnalysisStrategy):
    def analyze(
        self, df: pd.DataFrame, feature1: str, feature2: str, corr: bool = False
    ) -> None:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=feature1, y=feature2, data=df)
        plt.title(f"{feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        plt.xticks(rotation=45)
        plt.show()


class BivariateAnalyzer:
    def __init__(self, strategy: BivariateAnalysisStrategy) -> None:
        self._strategy = strategy

    def set_strategy(self, strategy: BivariateAnalysisStrategy) ->
            self._strategy = strategy

    def execute_analysis(
        self, df: pd.DataFrame, feature1: str, feature2: str, corr: bool = False
    ) -> None:
        self._strategy.analyze(df, feature1, feature2, corr=corr)
