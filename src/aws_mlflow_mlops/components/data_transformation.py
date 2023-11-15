import os

import pandas as pd
from sklearn.model_selection import train_test_split

from src.aws_mlflow_mlops import logger
from src.aws_mlflow_mlops.entity.config_entity import DataTransformationConfig


class DataTransformation:
    """
    The DataTransformation class is responsible for transforming the data for the training pipeline.

    This class uses the DataTransformationConfig to get the configuration for data transformation,
    and provides a method to split the data into training and testing sets.
    """

    def __init__(self, config: DataTransformationConfig):
        """
        The constructor for DataTransformation class.

        Args:
            config (DataTransformationConfig): The configuration for data transformation.
        """
        self.config = config

    def train_test_splitting(self) -> None:
        """
        The method to split the data into training and testing sets.

        This method reads the data from a CSV file, splits the data into training and testing sets,
        writes the training and testing sets to CSV files, logs the shapes of the training and testing sets,
        and prints the shapes of the training and testing sets.

        Returns:
            None
        """
        data = pd.read_csv(self.config.data_path)

        train, test = train_test_split(data, test_size=0.2, random_state=42)

        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        logger.info("Split data into train and test sets.")
        logger.info(f"Train shape: {train.shape}\n Test shape:{test.shape}")

        print(f"Train shape: {train.shape}\n Test shape:{test.shape}")
