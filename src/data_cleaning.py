import logging
from pathlib import Path
from typing import List, Union

import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def save_data(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """Guarda un DataFrame en un archivo CSV en un directorio predefinido.

    La función construye una ruta dentro del directorio 'data/interm/', se asegura
    de que este directorio exista, y luego guarda el DataFrame proporcionado.

    Args:
        name (str): El nombre base para el archivo de salida (sin la extensión .csv).
        df (pd.DataFrame): El DataFrame que se va a guardar.

    Returns:
        pd.DataFrame: El mismo DataFrame que se pasó como entrada, permitiendo
            el encadenamiento de métodos si fuera necesario.
    """
    output_path = Path("data/interm/") / f"{name}.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"Guardando DataFrame en: {output_path}")
    df.to_csv(output_path, index=False)
    return df


class CleanHandler:
    """
    Una clase contenedora para métodos de limpieza de datos en DataFrames.

    Esta clase agrupa funcionalidades estáticas relacionadas con la preparación
    y limpieza de conjuntos de datos.
    """

    @staticmethod
    def clean_missing_values(
        df: pd.DataFrame,
        method: str = "mean",
        fill_value: Union[str, int, float] = None,
        axis: int = 0,
        thresh: int = None,
        save: bool = False,
        name: str = "",
    ) -> pd.DataFrame:
        """Limpia los valores faltantes de un DataFrame utilizando varios métodos.

        Esta función toma un DataFrame y aplica una estrategia para manejar los
        valores NaN. Puede rellenarlos (imputar) con la media, mediana, moda o
        un valor constante, o puede eliminar las filas/columnas que los contengan.

        Args:
            df (pd.DataFrame): El DataFrame de entrada que contiene los datos a limpiar.
            method (str, optional): El método a utilizar. Los valores válidos son:
                'mean', 'median', 'mode', 'constant', 'drop'. Por defecto es "mean".
            fill_value (Union[str, int, float], optional): El valor a utilizar cuando
                el método es 'constant'. Por defecto es None.
            axis (int, optional): El eje para el método 'drop'. 0 para filas, 1 para
                columnas. Por defecto es 0.
            thresh (int, optional): Para el método 'drop', es el número mínimo de
                valores no nulos requeridos para no ser eliminado. Por defecto es None.
            save (bool, optional): Si es True, guarda el DataFrame limpio en un
                archivo CSV. Por defecto es False.
            name (str, optional): El nombre del archivo (sin extensión) si save es True.
                Es obligatorio si `save` es True. Por defecto es "".

        Returns:
            pd.DataFrame: Un nuevo DataFrame con los valores faltantes tratados.

        Raises:
            ValueError: Si `save` es True pero no se proporciona un `name`.
            ValueError: Si `method` es 'constant' pero no se proporciona un `fill_value`.
        """
        logging.info(
            f"Iniciando limpieza de valores faltantes con el método: '{method}'"
        )

        df_cleaned = df.copy()

        if method in ["mean", "median"]:
            numeric_columns = df_cleaned.select_dtypes(include="number").columns
            if numeric_columns.empty:
                logging.warning(
                    "No se encontraron columnas numéricas para imputar con media o mediana."
                )
            else:
                if method == "mean":
                    fill_values = df_cleaned[numeric_columns].mean()
                else:
                    fill_values = df_cleaned[numeric_columns].median()
                df_cleaned[numeric_columns] = df_cleaned[numeric_columns].fillna(
                    fill_values
                )

        elif method == "mode":
            for column in df_cleaned.columns:
                df_cleaned[column] = df_cleaned[column].fillna(
                    df[column].mode().iloc[0]
                )

        elif method == "constant":
            if fill_value is None:
                raise ValueError(
                    "Se debe proveer un 'fill_value' para el método 'constant'."
                )
            df_cleaned = df_cleaned.fillna(fill_value)

        elif method == "drop":
            original_rows = len(df_cleaned)
            df_cleaned = df_cleaned.dropna(axis=axis, thresh=thresh)
            final_rows = len(df_cleaned)
            logging.info(
                f"Se eliminaron {original_rows - final_rows} filas/columnas con valores nulos."
            )

        else:
            logging.warning(
                f"Método desconocido '{method}'. No se manejaron los valores faltantes."
            )
            return df_cleaned

        logging.info("Proceso de limpieza de valores faltantes completado.")

        if save:
            if not name:
                raise ValueError(
                    "El parámetro 'name' no puede estar vacío si 'save' es True."
                )
            return save_data(name, df_cleaned)

        return df_cleaned

    @staticmethod
    def select_features(
        df: pd.DataFrame, features: List[str], save: bool = False, name: str = ""
    ) -> pd.DataFrame:
        """Selecciona un subconjunto de características (columnas) de un DataFrame.

        Esta función crea una copia del DataFrame que contiene únicamente las
        columnas especificadas en la lista de características. Opcionalmente,
        puede guardar el DataFrame resultante en un archivo CSV.

        Args:
            df (pd.DataFrame): El DataFrame original del cual se seleccionarán
                las características.
            features (List[str]): Una lista de nombres de las columnas a seleccionar.
            save (bool, optional): Si es True, guarda el DataFrame con las
                características seleccionadas. Por defecto es False.
            name (str, optional): El nombre del archivo (sin extensión) a guardar
                si `save` es True. Es obligatorio en ese caso. Por defecto es "".

        Returns:
            pd.DataFrame: Un nuevo DataFrame que contiene solo las columnas seleccionadas.

        Raises:
            ValueError: Si `save` es True pero no se proporciona un `name`.
            KeyError: Si alguna de las características en la lista `features`
                no se encuentra en las columnas del DataFrame.
        """
        logging.info(f"Seleccionando {len(features)} características: {features}")

        df_selected = df[features].copy()

        if save:
            if not name:
                raise ValueError(
                    "El parámetro 'name' no puede estar vacío si 'save' es True."
                )
            return save_data(name, df_selected)

        return df_selected
