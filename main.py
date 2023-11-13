from src.aws_mlflow_mlops import logger
from src.aws_mlflow_mlops.pipeline.stage_01_data_ingestion import (
    DataIngestionTrainingPipeline,
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
