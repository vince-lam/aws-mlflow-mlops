from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd


class PredictionPipeline:
    """
    A class used to load a trained model and make predictions.

    Attributes
    ----------
    model : object
        The trained model loaded from a joblib file.

    Methods
    -------
    predict(data: List[float]) -> float:
        Makes a prediction using the trained model and the provided data.
    """

    def __init__(self):
        """
        Constructs all the necessary attributes for the PredictionPipeline object.

        Attributes
        ----------
        model : object
            The trained model loaded from a joblib file.
        """
        self.model = joblib.load(Path("artifacts/model_trainer/model.joblib"))

    def predict(self, data: List[float]) -> float:
        """
        Makes a prediction using the trained model and the provided data.

        Parameters
        ----------
        data : List[float]
            The input data to make a prediction on.

        Returns
        -------
        prediction : float
            The prediction made by the model.
        """
        prediction = self.model.predict(data)

        return self.model.predict(data)
