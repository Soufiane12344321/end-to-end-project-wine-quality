from src.mlProject.config.configuration import ConfigurationManager
from src.mlProject.components.data_ingestion import DataIngestion
from src.mlProject.components.data_validation import DataValidation
from src.mlProject.components.data_transformation import DataTransformation
from src.mlProject.components.model_training import ModelTrainer
from src.mlProject.components.model_evaluation import ModelEvaluation
from src.mlProject import logging
from src.mlProject.exception import CustomException
import sys

class TrainingPipeline:
    def __init__(self):
        self.config_manager = ConfigurationManager()

    def start_data_ingestion(self):
        """Execute data ingestion stage"""
        try:
            logging.info("Starting data ingestion stage")
            data_ingestion_config = self.config_manager.get_data_ingestion_config()
            
            data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
            data_path = data_ingestion.download_file()
            logging.info(f"Data ingestion completed. Data saved at: {data_path}")
            return data_path
        except Exception as e:
            raise CustomException(e, sys)

    def start_data_validation(self):
        """Execute data validation stage"""
        try:
            logging.info("Starting data validation stage")
            data_validation_config = self.config_manager.get_data_validation_config()
            data_validation = DataValidation(config=data_validation_config)
            validation_status = data_validation.validate_all_columns()
            logging.info(f"Data validation completed with status: {validation_status}")
            return validation_status
        except Exception as e:
            raise CustomException(e, sys)

    def start_data_transformation(self):
        """Execute data transformation stage"""
        try:
            logging.info("Starting data transformation stage")
            data_transformation_config = self.config_manager.get_data_transformation_config()
            data_transformation = DataTransformation(config=data_transformation_config)
            transformed_data, preprocessor = data_transformation.initiate_data_transformation()
            logging.info("Data transformation completed")
            return transformed_data, preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        
    def start_model_training(self):
        """Execute model training stage"""
        try:
            logging.info("Starting model training stage")
            model_trainer_config = self.config_manager.get_model_trainer_config()
            model_trainer = ModelTrainer(config=model_trainer_config)
            train_metrics, test_metrics, model = model_trainer.train()
            logging.info(f"Model training completed. Test R2 Score: {test_metrics['r2_score']:.3f}")
            return train_metrics, test_metrics, model
        except Exception as e:
            raise CustomException(e, sys)
        
    def start_model_evaluation(self):
        """Execute model evaluation stage"""
        try:
            logging.info("Starting model evaluation stage")
            model_evaluation_config = self.config_manager.get_model_evaluation_config()
            model_evaluation = ModelEvaluation(config=model_evaluation_config)
            metrics = model_evaluation.evaluate()
            logging.info(f"Model evaluation completed. R2 Score: {metrics['r2_score']:.3f}")
            return metrics
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    try:
        logging.info("Starting training pipeline")
        pipeline = TrainingPipeline()
        
        # Execute pipeline stages
        pipeline.start_data_ingestion()
        pipeline.start_data_validation()
        pipeline.start_data_transformation()
        train_metrics, test_metrics, model = pipeline.start_model_training()
        pipeline.start_model_evaluation()
        
        logging.info(f"Training pipeline completed successfully")
        logging.info(f"Final Model Metrics:")
        logging.info(f"Train R2 Score: {train_metrics['r2_score']:.3f}")
        logging.info(f"Test R2 Score: {test_metrics['r2_score']:.3f}")
        
        logging.info("Training pipeline completed successfully")
        
    except Exception as e:
        logging.error(f"Training pipeline failed: {e}")
        raise e