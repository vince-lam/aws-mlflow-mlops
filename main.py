from src.aws_mlflow_mlops import logger
from src.aws_mlflow_mlops.pipeline.stage_01_data_ingestion import (
    DataIngestionTrainingPipeline,
)
from src.aws_mlflow_mlops.pipeline.stage_02_data_validation import (
    DataValidationTrainingPipeline,
)
from src.aws_mlflow_mlops.pipeline.stage_03_data_transformation import (
    DataTransformationTrainingPipeline,
)
from src.aws_mlflow_mlops.pipeline.stage_04_model_trainer import (
    ModelTrainerTrainingPipeline,
)

STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f"Running {STAGE_NAME}...")
    data_ingestion_pipeline = DataIngestionTrainingPipeline()
    data_ingestion_pipeline.main()
    logger.info(f"Completed {STAGE_NAME}\n\n")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Validation Stage"
try:
    logger.info(f"Running {STAGE_NAME}...")
    data_validation_pipeline = DataValidationTrainingPipeline()
    data_validation_pipeline.main()
    logger.info(f"Completed {STAGE_NAME}\n\n")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Transformation Stage"
try:
    logger.info(f"Running {STAGE_NAME}...")
    data_transformation_pipeline = DataTransformationTrainingPipeline()
    data_transformation_pipeline.main()
    logger.info(f"Completed {STAGE_NAME}\n\n")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Model Training Stage"
try:
    logger.info(f"Starting {STAGE_NAME}")
    pipeline = ModelTrainerTrainingPipeline()
    pipeline.main()
    logger.info(f"Completed {STAGE_NAME}")
except Exception as e:
    logger.error(f"Failed to complete {STAGE_NAME} with error: {e}")
    raise e
