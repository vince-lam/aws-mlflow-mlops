import os
import urllib.request as request
import zipfile

from src.aws_mlflow_mlops import logger
from src.aws_mlflow_mlops.utils.common import get_size
from src.aws_mlflow_mlops.entity.config_entity import DataIngestionConfig
from pathlib import Path


class DataIngestion:
    """
    The DataIngestion class is responsible for managing the data ingestion process.
    It downloads and extracts data files based on the provided configuration.

    Attributes:
        config (DataIngestionConfig): An object containing the configuration for the data ingestion process.
    """

    def __init__(self, config: DataIngestionConfig):
        """
        The constructor for DataIngestion class.

        Parameters:
            config (DataIngestionConfig): An object containing the configuration for the data ingestion process.
        """
        self.config = config

    def download_file(self):
        """
        The method to download a data file from a URL.
        If the file already exists, it logs the file size.
        """
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url=self.config.source_URL, filename=self.config.local_data_file
            )
            logger.info(f"{filename} download! with following info: \n{headers}")
        else:
            logger.info(
                f"File already exists of size: {get_size(Path(self.config.local_data_file))}"
            )

    def extract_zip_file(self):
        """
        The method to extract a zip file to a specified directory.
        If the directory does not exist, it creates the directory.
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, "r") as zip_ref:
            zip_ref.extractall(unzip_path)
        logger.info(f"File extracted to: {unzip_path}")
