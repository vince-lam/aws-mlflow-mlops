# Components
import os
from pathlib import Path
from urllib.parse import urlparse

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from aws_mlflow_mlops.config.configuration import ModelEvaluationConfig
from aws_mlflow_mlops.utils.common import save_json


class ModelEvaluation:
    """
    A class used to evaluate a trained model and log the results into MLflow.

    ...

    Attributes
    ----------
    config : ModelEvaluationConfig
        The configuration settings for model evaluation.

    Methods
    -------
    evaluation_metrics(actual, pred):
        Calculates and returns the RMSE, MAE, and R2 score of the model predictions.

    log_into_mlflow():
        Loads the test data and the trained model, makes predictions on the test data, calculates
        evaluation metrics, and logs the results into MLflow.
    """

    def __init__(self, config: ModelEvaluationConfig):
        """
        Constructs all the necessary attributes for the ModelEvaluation object.

        Parameters
        ----------
            config : ModelEvaluationConfig
                The configuration settings for model evaluation.
        """
        self.config = config

    def evaluation_metrics(self, actual, pred):
        """
        Calculates and returns the RMSE, MAE, and R2 score of the model predictions.

        Parameters
        ----------
            actual : array-like
                The actual target values.

            pred : array-like
                The predicted target values.

        Returns
        -------
            rmse : float
                The Root Mean Square Error of the model predictions.

            mae : float
                The Mean Absolute Error of the model predictions.

            r2 : float
                The R2 score of the model predictions.
        """
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def log_into_mlflow(self):
        """
        Loads the test data and the trained model, makes predictions on the test data, calculates
        evaluation metrics, and logs the results into MLflow.

        This method also saves the evaluation metrics locally as a JSON file.
        """
        load_dotenv()

        # Load test data and model
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]

        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_registry_uri()).scheme

        with mlflow.start_run():
            predicted_qualities = model.predict(test_x)
            (rmse, mae, r2) = self.evaluation_metrics(test_y, predicted_qualities)

            # Save metrics as local
            scores = {
                "rmse": rmse,
                "r2": r2,
                "mae": mae,
            }
            print(scores)
            save_json(path=Path(self.config.metric_file_name), data=scores)

            print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (model.alpha, model.l1_ratio))
            print("  RMSE: %s" % rmse)
            print("  MAE: %s" % mae)
            print("  R2: %s" % r2)

            mlflow.log_params(self.config.all_params)

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mae", mae)

            # Model registry does not work with file store
            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(model, "model", registered_model_name="ElasticnetModel")
            else:
                mlflow.sklearn.log_model(model, "model")
