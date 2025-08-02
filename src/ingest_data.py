import os
import zipfile
from abc import ABC, abstractmethod

import pandas as pd


class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path: str) -> pd.DataFrame:
        """Método abstracto para la ingesta de los datos de un archivo dado"""
        pass


class ZipDataIngestor(DataIngestor):
    def ingest(self, file_path: str) -> pd.DataFrame:
        if not file_path.endswith(".zip"):
            raise ValueError("El archivo no es un .zip")

        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall("./data/raw")

        extracted_files = os.listdir("./data/raw")
        csv_files = [f for f in extracted_files if f.endswith(".csv")]

        if len(csv_files) == 0:
            raise FileNotFoundError("No hay archivos csv en los archivos extraidos")

        csv_file_path = os.path.join("extracted_data", csv_files[0])
        df = pd.read_csv(csv_file_path)

        return df


class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(file_extension: str) -> DataIngestor:
        """Devuelve el apropiado Data ingestor basado en la extension del archivo.

        De momento solo está implementado el .zip
        """

        if file_extension == ".zip":
            return ZipDataIngestor()
        else:
            raise ValueError(f"No hay un ingestor para los archivos {file_extension}")
