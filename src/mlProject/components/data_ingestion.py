import os
import sys
from pymongo import MongoClient
import pandas as pd
from src.mlProject.entity.config_entity import DataIngestionConfig
from src.mlProject import logging
from src.mlProject.exception import CustomException

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        """
        Initialize with DataIngestionConfig
        """
        self.config = data_ingestion_config

    def download_file(self) -> str:
        """
        Download data from MongoDB and save as CSV
        """
        try:
            logging.info("Trying to download data from MongoDB")
            
            # Create directory
            os.makedirs(os.path.dirname(self.config.raw_data_path), exist_ok=True)
            
            # Connect to MongoDB
            client = MongoClient(self.config.mongodb_connection_string)
            db = client[self.config.database_name]
            collection = db[self.config.collection_name]
            
            # Get data
            cursor = collection.find({})
            df = pd.DataFrame(list(cursor))
            
            # Remove MongoDB's _id column if present
            if '_id' in df.columns:
                df = df.drop('_id', axis=1)
            
            # Save to CSV
            df.to_csv(self.config.raw_data_path, index=False)
            
            logging.info(f"Data saved to {self.config.raw_data_path}")
            return self.config.raw_data_path
            
        except Exception as e:
            raise CustomException(e, sys)
        finally:
            if 'client' in locals():
                client.close()