from abc import ABC, abstractmethod

import pandas as pd


class DataInspectionStrategy(ABC):
    @abstractmethod
    def inspect(self, df: pd.DataFrame):
        """
        Realiza una determinada inspección.

        Parametros:
        df (pd.DataFrame): Dataframe a inspeccionar.

        Returns:
        None: Este método imprime los resultados de la inspección directamente.

        """
        pass


class DataTypesInspection(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame):
        """
        Inspecciona los tipos de datos y los valores nulos en el dataframe

        Parametros:
        df (pd.DataFrame): DataFrame a inspeccionar.

        Returns:
        None: Imprime los tipos de datos y los valores nulos.
        """

        print("Tipos de datos y valores nulos:")
        print(df.info())


class SummaryStatistics(DataInspectionStrategy):
    def inspect(self, df: pd.DataFrame) -> None:
        """
        Imprime las estadísticas numéricas y categóricas.

        Parametros:
        df (pd.DataFrame): DataFrame a inspeccionar.

        Returns:
        None: Imprime las estadísticas en la consola.
        """

        print("Estadísitcas (Numerical Features):")
        print(df.describe())
        print("Estadísitcas (Categorical Features):")
        print(df.describe(include="O"))


class DataInspector:
    def __init__(self, strategy: DataInspectionStrategy) -> None:
        self._strategy = strategy

    def set_strategy(self, strategy: DataInspectionStrategy):
        """
        Establece la estrategia a utilizar.

        Parametros:
        strategy (DataInspectionStrategy): La nueva estrategia a usar para la inspección de los datos.

        Returns:
        None
        """
        self._strategy = strategy

    def execute_inspection(self, df: pd.DataFrame):
        """
        Ejecuta la inspección usando la estrategia actual.

        Parametros:
        df (pd.DataFrame): Dataframe a inspeccionar.

        Returns:
        None
        """

        self._strategy.inspect(df)
