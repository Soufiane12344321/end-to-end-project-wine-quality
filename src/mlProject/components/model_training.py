import os
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from src.mlProject.entity.config_entity import ModelTrainerConfig
from src.mlProject import logging
from src.mlProject.exception import CustomException
import joblib

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        try:
            # Load transformed data
            data = pd.read_csv(self.config.transformed_data_path)
            logging.info("Loaded transformed data")

            # Split features and target
            X = data.drop(self.config.target_column, axis=1)
            y = data[self.config.target_column]

            # Split data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.config.test_size,
                random_state=self.config.random_state
            )

            # Initialize and train model
            model = ElasticNet(
                alpha=self.config.alpha,
                l1_ratio=self.config.l1_ratio,
                random_state=self.config.random_state
            )
            model.fit(X_train, y_train)

            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate metrics
            train_metrics = {
                'r2_score': r2_score(y_train, y_train_pred),
                'mae': mean_absolute_error(y_train, y_train_pred),
                'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred))
            }

            test_metrics = {
                'r2_score': r2_score(y_test, y_test_pred),
                'mae': mean_absolute_error(y_test, y_test_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred))
            }

            # Log metrics
            logging.info("Model Training Results:")
            logging.info(f"Train Metrics: {train_metrics}")
            logging.info(f"Test Metrics: {test_metrics}")

            # Save train and test data
            train_data = pd.concat([X_train, y_train], axis=1)
            test_data = pd.concat([X_test, y_test], axis=1)

            os.makedirs(os.path.dirname(self.config.train_data_path), exist_ok=True)
            train_data.to_csv(self.config.train_data_path, index=False)
            test_data.to_csv(self.config.test_data_path, index=False)

            # Save model
            os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)
            joblib.dump(model, self.config.model_path)

            # Return metrics for evaluation
            return train_metrics, test_metrics, model

        except Exception as e:
            logging.error("Error in model training")
            raise CustomException(e, sys)