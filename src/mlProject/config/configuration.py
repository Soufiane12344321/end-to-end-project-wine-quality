from src.mlProject.constants import *
from src.mlProject.utils.common import read_yaml, create_directories
from src.mlProject.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig
)

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            raw_data_path=Path(config.raw_data_path),
            mongodb_connection_string=self.config.mongodb.connection_string,
            database_name=self.config.mongodb.database_name,
            collection_name=self.config.mongodb.collection_name
        )

        return data_ingestion_config

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.COLUMNS

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=Path(config.root_dir),
            raw_data_path=Path(config.raw_data_path),
            validated_data_path=Path(config.validated_data_path),
            status_file=Path(config.status_file),  # Changed to match config.yaml
            required_columns=schema,
            target_column=self.schema.TARGET_COLUMN
        )

        return data_validation_config

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            transformed_data_path=Path(config.transformed_data_path),
            preprocessor_path=Path(config.preprocessor_path)
        )

        return data_transformation_config

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.ElasticNet
        
        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            transformed_data_path=Path(config.transformed_data_path),
            train_data_path=Path(config.train_data_path),
            test_data_path=Path(config.test_data_path),
            model_path=Path(config.model_path),
            target_column=config.target_column,
            test_size=params.test_size,
            random_state=params.random_state,
            alpha=params.alpha,
            l1_ratio=params.l1_ratio
        )

        return model_trainer_config

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=Path(config.root_dir),
            test_data_path=Path(config.test_data_path),
            model_path=Path(config.model_path),
            metric_file_path=Path(config.metric_file_path),
            target_column=config.target_column
        )

        return model_evaluation_config