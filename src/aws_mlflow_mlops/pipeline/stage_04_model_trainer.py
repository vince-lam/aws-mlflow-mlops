from src.aws_mlflow_mlops import logger
from src.aws_mlflow_mlops.components.model_trainer import ModelTrainer
from src.aws_mlflow_mlops.config.configuration import ConfigurationManager

STAGE_NAME = "Model Training Stage"


class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer_config = ModelTrainer(config=model_trainer_config)
        model_trainer_config.train()


if __name__ == "__main__":
    try:
        logger.info(f"Starting {STAGE_NAME}")
        pipeline = ModelTrainerTrainingPipeline()
        pipeline.main()
        logger.info(f"Completed {STAGE_NAME}")
    except Exception as e:
        logger.error(f"Failed to complete {STAGE_NAME} with error: {e}")
        raise e
