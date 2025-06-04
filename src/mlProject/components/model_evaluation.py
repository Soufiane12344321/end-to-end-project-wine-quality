import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import json
from src.mlProject.entity.config_entity import ModelEvaluationConfig
from src.mlProject import logging
from src.mlProject.exception import CustomException
from pathlib import Path

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def evaluate_metrics(self, actual, pred):
        try:
            r2 = r2_score(actual, pred)
            mae = mean_absolute_error(actual, pred)
            rmse = np.sqrt(mean_squared_error(actual, pred))
            return {'r2_score': float(r2), 'mae': float(mae), 'rmse': float(rmse)}
        except Exception as e:
            raise CustomException(e, sys)

    def evaluate(self):
        try:
            # Load test data and model
            test_data = pd.read_csv(self.config.test_data_path)
            model = joblib.load(self.config.model_path)

            # Separate features and target
            X_test = test_data.drop(self.config.target_column, axis=1)
            y_test = test_data[self.config.target_column]

            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate metrics
            metrics = self.evaluate_metrics(y_test, y_pred)

            # Save metrics
            os.makedirs(os.path.dirname(self.config.metric_file_path), exist_ok=True)
            with open(self.config.metric_file_path, 'w') as f:
                json.dump(metrics, f, indent=4)

            logging.info(f"Model Evaluation Results:")
            logging.info(f"R2 Score: {metrics['r2_score']:.3f}")
            logging.info(f"MAE: {metrics['mae']:.3f}")
            logging.info(f"RMSE: {metrics['rmse']:.3f}")

            return metrics

        except Exception as e:
            logging.error("Error in model evaluation")
            raise CustomException(e, sys)