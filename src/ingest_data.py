import os
import zipfile
from abc import ABC, abstractmethod

import pandas as pd


class DataIngestor(ABC):
    """Abstract base class for data ingestion."""

    @abstractmethod
    def ingest(self, file_path: str) -> pd.DataFrame:
        """Abstract method to ingest data from a given file.

        Args:
            file_path (str): Path to the file to ingest.

        Returns:
            pd.DataFrame: DataFrame containing the ingested data.
        """
        pass


class ZipDataIngestor(DataIngestor):
    """Data ingestor for .zip files containing CSV files."""

    def ingest(self, file_path: str) -> pd.DataFrame:
        """Extracts and loads the first CSV file found in a .zip archive.

        Args:
            file_path (str): Path to the .zip file.

        Returns:
            pd.DataFrame: DataFrame with the contents of the extracted CSV file.

        Raises:
            ValueError: If the file is not a .zip.
            FileNotFoundError: If no CSV files are found in the extracted contents.
        """
        if not file_path.endswith(".zip"):
            raise ValueError("The file is not a .zip archive.")

        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall("../data/raw")

        extracted_files = os.listdir("../data/raw")
        csv_files = [f for f in extracted_files if f.endswith(".csv")]

        if len(csv_files) == 0:
            raise FileNotFoundError("No CSV files found in the extracted contents.")

        csv_file_path = os.path.join("../data/raw/", csv_files[0])
        df = pd.read_csv(csv_file_path)

        return df


class DataIngestorFactory:
    """Factory class to obtain the appropriate DataIngestor implementation."""

    @staticmethod
    def get_data_ingestor(file_extension: str) -> DataIngestor:
        """Returns the appropriate DataIngestor based on the file extension.

        Currently, only .zip is implemented.

        Args:
            file_extension (str): The file extension, including the dot (e.g., ".zip").

        Returns:
            DataIngestor: The DataIngestor implementation for the given extension.

        Raises:
            ValueError: If there is no ingestor available for the given file extension.
        """
        if file_extension == ".zip":
            return ZipDataIngestor()
        else:
            raise ValueError(f"No ingestor available for file type: {file_extension}")
