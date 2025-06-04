from src.mlProject import logging
from src.mlProject.pipeline.training_pipeline import TrainingPipeline
from src.mlProject.exception import CustomException
import sys

def main():
    try:
        logging.info("Training Pipeline Started")
        
        # Initialize and run training pipeline
        pipeline = TrainingPipeline()
       
        # Uncomment below lines as you implement each stage
        pipeline.start_data_ingestion()
        pipeline.start_data_validation()
        pipeline.start_data_transformation()
        pipeline.start_model_training()
        pipeline.start_model_evaluation()
        
        logging.info("Training Pipeline Completed Successfully")
        
    except Exception as e:
        logging.error("Training Pipeline Failed")
        raise CustomException(e, sys)

if __name__ == "__main__":
    main()