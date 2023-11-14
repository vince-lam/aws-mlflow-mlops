import os

import pandas as pd

from src.aws_mlflow_mlops import logger
from src.aws_mlflow_mlops.entity.config_entity import DataValidationConfig


class DataValidation:
    """
    The DataValidation class is responsible for validating the data based on a provided schema.

    Attributes:
        config (DataValidationConfig): An object containing the configuration for the data validation process.
    """

    def __init__(self, config: DataValidationConfig):
        """
        The constructor for DataValidation class.

        Parameters:
            config (DataValidationConfig): An object containing the configuration for the data validation process.
        """
        self.config = config

    def validate_all_columns(self) -> bool:
        """
        The method to validate all columns in the data against the schema.

        This method reads the data from a CSV file, gets the column names, and checks if each column is in the schema.
        If a column is not in the schema, it logs a message and writes the validation status to a file.
        If a column is in the schema, it writes the validation status to a file.
        If an error occurs during the validation process, it logs the error and raises the exception.

        Returns:
            bool: True if all columns are in the schema, False otherwise.
        """
        try:
            validation_status = False

            data = pd.read_csv(self.config.unzip_data_dir)
            all_columns = list(data.columns)
            all_schema = self.config.all_schema.keys()

            for column in all_columns:
                if column not in all_schema:
                    validation_status = False
                    logger.info(f"Column {column} is not in the schema")
                    with open(self.config.STATUS_FILE, "w") as f:
                        f.write(f"Validation status: {validation_status}")
                else:
                    validation_status = True
                    with open(self.config.STATUS_FILE, "w") as f:
                        f.write(f"Validation status: {validation_status}")
            return validation_status
        except Exception as e:
            logger.error(f"Error in validate_all_columns: {e}")
            raise e
