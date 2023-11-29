import os

import numpy as np
import pandas as pd
from flask import Flask, redirect, render_template, request

from src.aws_mlflow_mlops.pipeline.prediction import PredictionPipeline

# from pathlib import Path

app = Flask(__name__)  # Initialize the flask App


@app.route("/", methods=["GET"])  # Route to the home page
def home():
    return render_template("index.html")


@app.route("/train", methods=["GET"])  # Route to train pipeline
def training():
    os.system("python3 main.py")
    return "Training successful"


@app.route("/predict", methods=["GET", "POST"])  # Route to show predictions in web UI
def index():
    if request.method == "POST":
        try:
            # Read user inputs
            fixed_acidity = float(request.form["fixed_acidity"])
            volatile_acidity = float(request.form["volatile_acidity"])
            citric_acid = float(request.form["citric_acid"])
            residual_sugar = float(request.form["residual_sugar"])
            chlorides = float(request.form["chlorides"])
            free_sulfur_dioxide = float(request.form["free_sulfur_dioxide"])
            total_sulfur_dioxide = float(request.form["total_sulfur_dioxide"])
            density = float(request.form["density"])
            pH = float(request.form["pH"])
            sulphates = float(request.form["sulphates"])
            alcohol = float(request.form["alcohol"])

            data = [
                fixed_acidity,
                volatile_acidity,
                citric_acid,
                residual_sugar,
                chlorides,
                free_sulfur_dioxide,
                total_sulfur_dioxide,
                density,
                pH,
                sulphates,
                alcohol,
            ]
            data = np.array(data).reshape(1, 11)

            # Load model
            # model_path = Path("artifacts/model_trainer/model.joblib")
            prediction_pipeline = PredictionPipeline()
            prediction = prediction_pipeline.predict(data)

            return render_template("results.html", prediction=str(prediction))
        except Exception as e:
            print("The Exception message is: ", e)
            return "An error occurred. Please try again."
    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
