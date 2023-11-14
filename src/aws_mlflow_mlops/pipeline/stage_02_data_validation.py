from src.aws_mlflow_mlops import logger
from src.aws_mlflow_mlops.components.data_validation import DataValidation
from src.aws_mlflow_mlops.config.configuration import ConfigurationManager

STAGE_NAME = "Data Validation Stage"


class DataValidationTrainingPipeline:
    """
    The DataValidationTrainingPipeline class is responsible for managing the data validation process in the training pipeline.

    This class uses the ConfigurationManager to get the data validation configuration, creates a DataValidation instance with this configuration,
    and then validates all columns in the data.
    """

    def __init__(self):
        """
        The constructor for DataValidationTrainingPipeline class.
        """
        pass

    def main(self) -> None:
        """
        The main method to run the data validation process in the training pipeline.

        This method gets the data validation configuration from the ConfigurationManager, creates a DataValidation instance with this configuration,
        validates all columns in the data, and logs a message indicating that the data validation is completed.

        Returns:
            None
        """
        config_manager = ConfigurationManager()

        data_validation_config = config_manager.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        data_validation.validate_all_columns()
        logger.info("Data validation completed")


if __name__ == "__main__":
    # This script runs the data validation stage of the training pipeline.
    # It creates an instance of the DataValidationTrainingPipeline class and calls its main method.
    # If any exceptions are raised during the execution of the pipeline, they are logged and re-raised.
    try:
        logger.info(f"Running {STAGE_NAME}...")
        data_validation_pipeline = DataValidationTrainingPipeline()
        data_validation_pipeline.main()
        logger.info(f"Completed {STAGE_NAME}\n\n")
    except Exception as e:
        logger.exception(e)
        raise e
