from src.aws_mlflow_mlops.constants import *
from src.aws_mlflow_mlops.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
)
from src.aws_mlflow_mlops.utils.common import create_directories, read_yaml


class ConfigurationManager:
    """
    The ConfigurationManager class is responsible for managing the configuration of the data ingestion process.
    It reads configuration, parameters, and schema from YAML files, and creates necessary directories.

    Attributes:
        config (dict): Configuration read from a YAML file.
        params (dict): Parameters read from a YAML file.
        schema (dict): Schema read from a YAML file.
    """

    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        params_filepath=PARAMS_FILE_PATH,
        schema_filepath=SCHEMA_FILE_PATH,
    ):
        """
        The constructor for ConfigurationManager class.

        Parameters:
            config_filepath (str): The path to the configuration YAML file.
            params_filepath (str): The path to the parameters YAML file.
            schema_filepath (str): The path to the schema YAML file.
        """
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        The method to get the configuration for the data ingestion process.

        Returns:
            DataIngestionConfig: An object containing the configuration for the data ingestion process.
        """
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir,
        )

        return data_ingestion_config

    def get_data_validation_config(self) -> DataValidationConfig:
        """
        The method to get the configuration for the data validation process.

        This method reads the data validation configuration and schema from the class attributes,
        creates necessary directories, and returns a DataValidationConfig object.

        Returns:
            DataValidationConfig: An object containing the configuration for the data validation process.
        """
        config = self.config.data_validation
        schema = self.schema.COLUMNS

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE=config.STATUS_FILE,
            unzip_data_dir=config.unzip_data_dir,
            all_schema=schema,
        )

        return data_validation_config

    def get_data_transformation_config(self) -> DataTransformationConfig:
        """
        The method to get the data transformation configuration.

        This method retrieves the data transformation configuration from the main configuration object,
        creates a DataTransformationConfig instance with the root directory and data path from the configuration,
        and returns this instance.

        Returns:
            DataTransformationConfig: An instance of DataTransformationConfig with the root directory and data path from the configuration.
        """
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
        )

        return data_transformation_config
