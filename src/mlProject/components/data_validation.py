import os
import pandas as pd
from src.mlProject import logging
from src.mlProject.entity.config_entity import DataValidationConfig
from src.mlProject.exception import CustomException
import sys

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_columns(self) -> bool:
        try:
            validation_status = False
            
            # Create directories
            os.makedirs(os.path.dirname(self.config.status_file), exist_ok=True)
            
            # Read data
            data = pd.read_csv(self.config.raw_data_path)
            all_cols = list(data.columns)

            # Validate columns
            all_schema = self.config.required_columns
            
            if sorted(all_schema) == sorted(all_cols):
                validation_status = True
                with open(self.config.status_file, 'w') as f:
                    f.write(f"Validation status: {validation_status}")
                    
                # Save validated data
                os.makedirs(os.path.dirname(self.config.validated_data_path), exist_ok=True)
                data.to_csv(self.config.validated_data_path, index=False)
                logging.info("Validated data saved successfully")
            else:
                with open(self.config.status_file, 'w') as f:
                    f.write(f"Validation status: {validation_status}")
                logging.error("Schema validation failed")

            return validation_status

        except Exception as e:
            raise CustomException(e, sys)