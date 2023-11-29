from src.aws_mlflow_mlops import logger
from src.aws_mlflow_mlops.components.model_evaluation import ModelEvaluation
from src.aws_mlflow_mlops.config.configuration import ConfigurationManager

STAGE_NAME = "Model Evaluation Stage"


class ModelEvaluationTrainingPipeline:
    """
    A class used to run the model evaluation pipeline.

    ...

    Methods
    -------
    main():
        Runs the model evaluation pipeline.
    """

    def __init__(self):
        pass

    def main(self):
        """
        Runs the model evaluation pipeline.

        This method creates a ConfigurationManager object, gets the model evaluation configuration,
        creates a ModelEvaluation object with the configuration, and logs the model evaluation
        into MLflow.
        """
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation_config = ModelEvaluation(config=model_evaluation_config)
        model_evaluation_config.log_into_mlflow()


if __name__ == "__main__":
    """
    The main entry point for the script.

    This block is executed when the script is run directly. It creates a
    ModelEvaluationTrainingPipeline object, runs the model evaluation pipeline, and logs the start
    and completion of the pipeline. If an exception occurs, it logs the error and re-raises the
    exception.
    """
    try:
        logger.info(f"Starting {STAGE_NAME}")
        pipeline = ModelEvaluationTrainingPipeline()
        pipeline.main()
        logger.info(f"Completed {STAGE_NAME}")
    except Exception as e:
        logger.error(f"Failed to complete {STAGE_NAME} with error: {e}")
        raise e
