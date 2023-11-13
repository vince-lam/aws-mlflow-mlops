from src.aws_mlflow_mlops.constants import *
from src.aws_mlflow_mlops.utils.common import read_yaml, create_directories
from src.aws_mlflow_mlops.entity.config_entity import DataIngestionConfig


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
