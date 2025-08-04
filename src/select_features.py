import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

import pandas as pd

# Configuración básica de logging para ver los mensajes del proceso.
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class FeatureSelectionStrategy(ABC):
    """
    Clase base abstracta (ABC) que define la interfaz para las estrategias de selección.

    Actúa como un contrato que todas las estrategias concretas de selección de
    características deben implementar. Esto asegura que el `FeatureSelectionHandler`
    pueda trabajar con cualquier estrategia que siga esta interfaz.
    """

    @abstractmethod
    def select(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """
        Método abstracto para seleccionar características de un DataFrame.

        Args:
            df (pd.DataFrame): El DataFrame de entrada del cual se seleccionarán las
                               características.
            features (List[str]): La lista de características a considerar o utilizar
                                  por la estrategia.

        Returns:
            pd.DataFrame: Un nuevo DataFrame que contiene únicamente las
                          características seleccionadas.
        """
        pass


class SelectFeatures(FeatureSelectionStrategy):
    """
    Una estrategia concreta que selecciona columnas basándose en una lista de nombres.

    Esta es una implementación simple del `FeatureSelectionStrategy` que toma
    directamente las columnas especificadas en la lista `features`.
    """

    def select(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """
        Selecciona un subconjunto de columnas del DataFrame.

        Crea una copia del DataFrame de entrada conteniendo únicamente las columnas
        cuyos nombres se especifican en la lista `features`.

        Args:
            df (pd.DataFrame): El DataFrame de entrada.
            features (List[str]): La lista de nombres de columnas a seleccionar.

        Returns:
            pd.DataFrame: Una copia del DataFrame con las columnas seleccionadas.
        """
        logging.info(f"Seleccionando {len(features)} características: {features}")
        df_selected = df[features].copy()
        return df_selected


class FeatureSelectionHandler:
    """
    Gestiona y ejecuta una estrategia de selección de características.

    Esta clase sigue el Patrón de Diseño Strategy. Mantiene una referencia a un
    objeto de estrategia y delega la tarea de selección a dicho objeto. La estrategia
    puede ser intercambiada dinámicamente en tiempo de ejecución.

    Attributes:
        _strategy (FeatureSelectionStrategy): La estrategia de selección actual.
    """

    def __init__(self, strategy: FeatureSelectionStrategy) -> None:
        """
        Inicializa el FeatureSelectionHandler con una estrategia inicial.

        Args:
            strategy (FeatureSelectionStrategy): La estrategia de selección de
                                                 características a utilizar.
        """
        self._strategy = strategy

    def set_strategy(self, strategy: FeatureSelectionStrategy) -> None:
        """
        Permite cambiar la estrategia de selección en tiempo de ejecución.

        Args:
            strategy (FeatureSelectionStrategy): La nueva estrategia a establecer.
        """
        logging.info(
            f"Cambiando la estrategia de selección a: {type(strategy).__name__}"
        )
        self._strategy = strategy

    def execute_selection(
        self, df: pd.DataFrame, features: List[str], save: bool = False, name: str = ""
    ) -> Optional[pd.DataFrame]:
        """
        Ejecuta el proceso de selección de características sobre un DataFrame.

        Args:
            df (pd.DataFrame): El DataFrame de entrada.
            features (List[str]): La lista de características a considerar.
            save (bool): Si es True, guarda el resultado en un CSV y no devuelve nada.
                         Si es False, devuelve el DataFrame con las características seleccionadas.
            name (str): El nombre del archivo si save es True.

        Returns:
            Optional[pd.DataFrame]: Un DataFrame con las características seleccionadas si save=False,
                                    de lo contrario None.

        Raises:
            ValueError: Si `save` es True pero no se proporciona un `name` para el archivo.
        """
        logging.info("Executing selection with strategy.")

        features_selected: pd.DataFrame = self._strategy.select(df, features)

        if save:
            if not name:
                raise ValueError(
                    "El parámetro 'name' no puede estar vacío si 'save' es True."
                )

            output_path = Path(f"../data/interm/{name}.csv")
            # Asegurarse de que el directorio padre existe
            output_path.parent.mkdir(parents=True, exist_ok=True)

            logging.info(f"Guardando características seleccionadas en: {output_path}")
            features_selected.to_csv(output_path, index=False)

            return features_selected
        else:
            return features_selected
