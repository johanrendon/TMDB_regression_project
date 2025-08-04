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
        """
        Método abstracto para manejar los valos nulos.

        Parámetros:
        df (pd.DataFrame): Dataframe que contiene los valores nulos.

        Returns:
        pd.DataFrame: Dataframe con los datos nulos tratados.
        """

        pass


class DropMissingValues(MissingValueStrategy):
    def __init__(self, axis: int = 0, thresh: int = None) -> None:
        """
        Dropea los valores nulos usando determinada estrategia.

        Parámetros:
        axis (int): 0 para eliminar las filas, 1 para eliminar las columnas.
        thresh (int): Es el threshold para los valores no nulos. Rows/columns con menos valores no nulos que el tresh son eliminados.
        """

        self._axis = axis
        self._thresh = thresh

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info(
            f"Eliminando valores nulos en el eje {self._axis} con el threshhold {self._thresh}"
        )
        df_cleaned = df.dropna(axis=self._axis, thresh=self._thresh)
        logging.info("Valores nulos eliminados")
        return df_cleaned


class FillMissingValues(MissingValueStrategy):
    def __init__(self, method="mean", fill_value=None) -> None:
        self.fill_value = fill_value
        self.method = method

    def handle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fills missing values using the specified method or constant value.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing missing values.

        Returns:
        pd.DataFrame: The DataFrame with missing values filled.
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
        """
        Initializes the MissingValueHandler with a specific missing value handling strategy.

        Parameters:
        strategy (MissingValueHandlingStrategy): The strategy to be used for handling missing values.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: MissingValueStrategy):
        """
        Sets a new strategy for the MissingValueHandler.

        Parameters:
        strategy (MissingValueHandlingStrategy): The new strategy to be used for handling missing values.
        """
        logging.info("Switching missing value handling strategy.")
        self._strategy = strategy

    def handle_missing_values(
        self, df: pd.DataFrame, save: bool = False, name: str = ""
    ) -> pd.DataFrame:
        """
        Executes the missing value handling using the current strategy.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing missing values.

        Returns:
        pd.DataFrame: The DataFrame with missing values handled.
        """
        logging.info("Executing missing value handling strategy.")

        df_cleaned = self._strategy.handle(df)
        if save:
            if not name:
                raise ValueError(
                    "El parámetro 'name' no puede estar vacío si 'save' es True."
                )

            output_path = Path(f"../data/interm/{name}.csv")
            output_path.parent.mkdir(parents=True, exist_ok=True)

            logging.info(f"Guardando df sin missing values: {output_path}")
            df_cleaned.to_csv(output_path, index=False)

            return df_cleaned
        else:
            return df_cleaned
