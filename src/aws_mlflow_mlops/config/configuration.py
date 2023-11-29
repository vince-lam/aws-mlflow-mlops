from pathlib import Path
from typing import Dict

from src.aws_mlflow_mlops.constants import (
    CONFIG_FILE_PATH,
    PARAMS_FILE_PATH,
    SCHEMA_FILE_PATH,
)
from src.aws_mlflow_mlops.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    DataValidationConfig,
    ModelEvaluationConfig,
    ModelTrainerConfig,
)
from src.aws_mlflow_mlops.utils.common import (
    create_directories,
    get_mlflow_tracking_uri,
    read_yaml,
)


class ConfigurationManager:
    """
    The ConfigurationManager class is responsible for managing the configuration of the data
    ingestion process. It reads configuration, parameters, and schema from YAML files, and creates
    necessary directories.

    Attributes:
        config (dict): Configuration read from a YAML file.
        params (dict): Parameters read from a YAML file.
        schema (dict): Schema read from a YAML file.
    """

    def __init__(
        self,
        config_filepath: Path = CONFIG_FILE_PATH,
        params_filepath: Path = PARAMS_FILE_PATH,
        schema_filepath: Path = SCHEMA_FILE_PATH,
    ):
        """
        The constructor for ConfigurationManager class.

        Parameters:
            config_filepath (str): The path to the configuration YAML file.
            params_filepath (str): The path to the parameters YAML file.
            schema_filepath (str): The path to the schema YAML file.
        """
        self.config: Dict = read_yaml(config_filepath)
        self.params: Dict = read_yaml(params_filepath)
        self.schema: Dict = read_yaml(schema_filepath)

        create_directories([self.config["artifacts_root"]])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        The method to get the configuration for the data ingestion process.

        Returns:
            DataIngestionConfig: An object containing the configuration for data ingestion process.
        """
        config = self.config["data_ingestion"]

        create_directories([config["root_dir"]])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config["root_dir"],
            source_URL=config["source_URL"],
            local_data_file=config["local_data_file"],
            unzip_dir=config["unzip_dir"],
        )

        return data_ingestion_config

    def get_data_validation_config(self) -> DataValidationConfig:
        """
        The method to get the configuration for the data validation process.

        This method reads the data validation configuration and schema from the class attributes,
        creates necessary directories, and returns a DataValidationConfig object.

        Returns:
            DataValidationConfig: An object containing configuration for data validation process.
        """
        config = self.config["data_validation"]
        schema = self.schema["COLUMNS"]

        create_directories([config["root_dir"]])

        data_validation_config = DataValidationConfig(
            root_dir=config["root_dir"],
            STATUS_FILE=config["STATUS_FILE"],
            unzip_data_dir=config["unzip_data_dir"],
            all_schema=schema,
        )

        return data_validation_config

    def get_data_transformation_config(self) -> DataTransformationConfig:
        """
        The method to get the data transformation configuration.

        This method retrieves the data transformation configuration from the main configuration
        object, creates a DataTransformationConfig instance with the root directory and data path
        from the configuration, and returns this instance.

        Returns:
            DataTransformationConfig: An instance of DataTransformationConfig with the root
            directory and data path from the configuration.
        """
        config = self.config["data_transformation"]

        create_directories([config["root_dir"]])

        data_transformation_config = DataTransformationConfig(
            root_dir=config["root_dir"],
            data_path=config["data_path"],
        )

        return data_transformation_config

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        """
        The method to get the model trainer configuration.

        This method retrieves the model trainer configuration from the main configuration object,
        the ElasticNet parameters, and the target column schema.
        It creates the directories for the root directory in the configuration if they do not exist,
        creates a ModelTrainerConfig instance with the root directory, train data path, test data
        path, model name, alpha, l1_ratio, and target column name from the configuration,
        parameters, and schema,and returns this instance.

        Returns:
            ModelTrainerConfig: An instance of ModelTrainerConfig with the root directory, train
            data path, test data path, model name, alpha, l1_ratio, and target column name from the
            configuration, parameters, and schema.
        """
        config = self.config["model_trainer"]
        params = self.params["ElasticNet"]
        schema = self.schema["TARGET_COLUMN"]

        create_directories([config["root_dir"]])

        model_trainer_config = ModelTrainerConfig(
            root_dir=config["root_dir"],
            train_data_path=config["train_data_path"],
            test_data_path=config["test_data_path"],
            model_name=config["model_name"],
            alpha=params["alpha"],
            l1_ratio=params["l1_ratio"],
            target_column=schema["name"],
        )

        return model_trainer_config

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        """
        Retrieve the configuration for model evaluation.

        This method fetches the model evaluation configuration, ElasticNet parameters, and the
        target column schema from the instance's attributes. It creates necessary directories
        specified in the config and constructs a ModelEvaluationConfig object with the retrieved
        and processed information.

        Returns:
        ModelEvaluationConfig: An object containing the root directory, test data path, model path,
        ElasticNet parameters, metric file name, target column name, and MLflow URI for evaluation.
        """
        config = self.config["model_evaluation"]
        params = self.params["ElasticNet"]
        schema = self.schema["TARGET_COLUMN"]
        mlflow_tracking_uri = get_mlflow_tracking_uri()

        create_directories([config["root_dir"]])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config["root_dir"],
            test_data_path=config["test_data_path"],
            model_path=config["model_path"],
            all_params=params,
            metric_file_name=config["metric_file_name"],
            target_column=schema["name"],
            mlflow_uri=mlflow_tracking_uri,
        )

        return model_evaluation_config
