from src.aws_mlflow_mlops import logger
from src.aws_mlflow_mlops.components.model_evaluation import ModelEvaluation
from src.aws_mlflow_mlops.config.configuration import ConfigurationManager

STAGE_NAME = "Model Evaluation Stage"


class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation_config = ModelEvaluation(config=model_evaluation_config)
        model_evaluation_config.log_into_mlflow()


if __name__ == "__main__":
    try:
        logger.info(f"Starting {STAGE_NAME}")
        pipeline = ModelEvaluationTrainingPipeline()
        pipeline.main()
        logger.info(f"Completed {STAGE_NAME}")
    except Exception as e:
        logger.error(f"Failed to complete {STAGE_NAME} with error: {e}")
        raise e
