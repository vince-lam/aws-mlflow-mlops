from pathlib import Path

from src.aws_mlflow_mlops import logger
from src.aws_mlflow_mlops.components.data_transformation import DataTransformation
from src.aws_mlflow_mlops.config.configuration import ConfigurationManager

STAGE_NAME = "Data Transformation Stage"


class DataTransformationTrainingPipeline:
    """
    The DataTransformationTrainingPipeline class is responsible for managing the data transformation process in the training pipeline.

    This class checks the status of the data validation stage, and if the data validation was successful,
    it gets the data transformation configuration, creates a DataTransformation instance with this configuration,
    and splits the data into training and testing sets.
    If the data validation was not successful, it raises an exception.
    """

    def __init__(self):
        """
        The constructor for DataTransformationTrainingPipeline class.
        """
        pass

    def main(self) -> None:
        """
        The main method to run the data transformation process in the training pipeline.

        This method checks the status of the data validation stage from a status file.
        If the data validation was successful, it logs a message, gets the data transformation configuration,
        creates a DataTransformation instance with this configuration, splits the data into training and testing sets,
        and logs a message indicating that the data transformation is completed.
        If the data validation was not successful, it raises an exception.
        If an error occurs during the data transformation process, it logs the error and raises the exception.

        Returns:
            None
        """
        try:
            with open(Path("artifacts/data_validation/status.txt"), "r") as f:
                status = f.read().split(" ")[-1]
            if status == "True":
                logger.info(f"Running {STAGE_NAME}...")
                config = ConfigurationManager()
                data_transformation_config = config.get_data_transformation_config()
                data_transformation = DataTransformation(
                    config=data_transformation_config
                )
                data_transformation.train_test_splitting()
                logger.info(f"Completed {STAGE_NAME}\n\n")
            else:
                raise Exception("Data Validation Failed, Schema Not Valid")
        except Exception as e:
            logger.exception(e)
            raise e
