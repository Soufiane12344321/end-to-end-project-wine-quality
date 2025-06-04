import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from src.mlProject import logging
from src.mlProject.entity.config_entity import DataTransformationConfig
from src.mlProject.exception import CustomException
import joblib

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        """
        Initialize with DataTransformationConfig
        """
        self.config = config

    def get_data_transformer_object(self):
        """
        Create the preprocessing pipeline
        """
        try:
            numerical_features = [
                'fixed acidity', 'volatile acidity', 'citric acid', 
                'residual sugar', 'chlorides', 'free sulfur dioxide',
                'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'
            ]

            # Create numeric pipeline
            numeric_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            # Combine pipelines
            preprocessor = ColumnTransformer([
                ('numeric_pipeline', numeric_pipeline, numerical_features)
            ])

            return preprocessor

        except Exception as e:
            logging.error("Error in get_data_transformer_object")
            raise CustomException(e, sys)

    def initiate_data_transformation(self):
        """
        Perform data transformation
        """
        try:
            # Read the data
            df = pd.read_csv(self.config.data_path)
            logging.info("Read data completed")

            # Prepare feature and target data
            numerical_features = [
                'fixed acidity', 'volatile acidity', 'citric acid', 
                'residual sugar', 'chlorides', 'free sulfur dioxide',
                'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'
            ]
            target_feature = 'quality'

            X = df[numerical_features]
            y = df[target_feature]

            # Create preprocessor
            preprocessor = self.get_data_transformer_object()
            
            # Fit and transform the data
            X_transformed = preprocessor.fit_transform(X)

            # Convert to DataFrame
            transformed_df = pd.DataFrame(
                X_transformed,
                columns=numerical_features
            )
            transformed_df[target_feature] = y

            # Create transformation output directory
            os.makedirs(os.path.dirname(self.config.transformed_data_path), exist_ok=True)

            # Save transformed data
            transformed_df.to_csv(self.config.transformed_data_path, index=False)
            logging.info("Saved transformed data")

            # Save preprocessor
            joblib.dump(
                preprocessor, 
                self.config.preprocessor_path
            )
            logging.info("Saved preprocessor object")

            return (
                transformed_df,
                preprocessor
            )

        except Exception as e:
            logging.error("Error in initiate_data_transformation")
            raise CustomException(e, sys)