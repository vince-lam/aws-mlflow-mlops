from src.aws_mlflow_mlops.config.configuration import ConfigurationManager
from src.aws_mlflow_mlops.components.data_ingestion import DataIngestion
from src.aws_mlflow_mlops import logger


STAGE_NAME = "Data Ingestion Stage"


class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()


if __name__ == "__main__":
    try:
        logger.info(f"Running {STAGE_NAME}...")
        data_ingestion_pipeline = DataIngestionTrainingPipeline()
        data_ingestion_pipeline.main()
        logger.info(f"Completed {STAGE_NAME}\n\n")
    except Exception as e:
        logger.exception(e)
        raise e
