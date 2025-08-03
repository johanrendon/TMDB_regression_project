from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class UnivariateAnalysisStrategy(ABC):
    @abstractmethod
    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Realiza el análisi univariado de una feature en específico.

        Parametros:
        df (pd.DataFrame): El dataframe que contiene los datos.
        feature (str): El nombre de la feature a analizar.
        """
        pass


class NumericalUnivariate(UnivariateAnalysisStrategy):
    def analyze(
        self, df: pd.DataFrame, feature: str, box: bool = True, log: bool = False
    ):
        """
        Grafica la distribución de una variable numérica usando un histograma y KDE.


        Parametros:
        df (pd.DataFrame): El dataframe que contiene los datos.
        feature (str): El nombre de la feature a analizar.

        Returns:
        None

        """

        if box:
            fig, axes = plt.subplots(1, 2, figsize=(20, 10))

            sns.histplot(data=df[feature], bins=30, kde=True, ax=axes[0], log_scale=log)

            sns.boxplot(data=df[feature], log_scale=log)

            print(df[feature].describe())

            plt.title(f"Distribución de {feature}")
            plt.xlabel((feature))
            plt.ylabel("Frecuencia")
            plt.show()

        else:
            plt.figure(figsize=(10, 6))
            sns.histplot(data=df[feature], kde=True, bins=30, log_scale=log)
            plt.title(f"Distribución de {feature}")
            plt.xlabel((feature))
            plt.ylabel("Frecuencia")
            plt.show()


class CategoricalUnivariate(UnivariateAnalysisStrategy):
    def analyze(self, df: pd.DataFrame, feature: str):
        """
        Grafica la distribución de una variable categórica usando un barplot.


        Parametros:
        df (pd.DataFrame): El dataframe que contiene los datos.
        feature (str): El nombre de la feature a analizar.

        Returns:
        None

        """

        plt.figure(figsize=(10, 6))
        sns.countplot(x=feature, data=df, palette="muted")
        plt.title(f"Distribución de {feature}")
        plt.xlabel((feature))
        plt.ylabel("Cantidad")
        plt.show()


class UnivariateAnalyzer:
    def __init__(self, strategy: UnivariateAnalysisStrategy) -> None:
        self._strategy = strategy

    def set_strategy(self, strategy: UnivariateAnalysisStrategy):
        self._strategy = strategy

    def execute_analysis(
        self, df: pd.DataFrame, feature: str, box: bool = True, log: bool = False
    ):
        self._strategy.analyze(df, feature, box=box, log=log)
