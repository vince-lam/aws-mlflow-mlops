from src.aws_mlflow_mlops import logger
from src.aws_mlflow_mlops.components.data_ingestion import DataIngestion
from src.aws_mlflow_mlops.config.configuration import ConfigurationManager

STAGE_NAME = "Data Ingestion Stage"


class DataIngestionTrainingPipeline:
    """
    The DataIngestionTrainingPipeline class is responsible for managing the data ingestion process in the training pipeline.

    This class uses the ConfigurationManager to get the data ingestion configuration, creates a DataIngestion instance with this configuration,
    and then downloads and extracts the data file.
    """

    def __init__(self):
        """
        The constructor for DataIngestionTrainingPipeline class.
        """
        pass

    def main(self) -> None:
        """
        The main method to run the data ingestion process in the training pipeline.

        This method gets the data ingestion configuration from the ConfigurationManager, creates a DataIngestion instance with this configuration,
        downloads the data file, and extracts the data file.

        Returns:
            None
        """
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()


if __name__ == "__main__":
    # This script runs the data ingestion stage of the training pipeline.
    # It creates an instance of the DataIngestionTrainingPipeline class and calls its main method.
    # If any exceptions are raised during the execution of the pipeline, they are logged and re-raised.
    try:
        logger.info(f"Running {STAGE_NAME}...")
        data_ingestion_pipeline = DataIngestionTrainingPipeline()
        data_ingestion_pipeline.main()
        logger.info(f"Completed {STAGE_NAME}\n\n")
    except Exception as e:
        logger.exception(e)
        raise e
