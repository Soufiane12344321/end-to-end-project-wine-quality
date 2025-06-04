from dataclasses import dataclass
from pathlib import Path

@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion"""
    root_dir: Path
    raw_data_path: Path
    mongodb_connection_string: str
    database_name: str
    collection_name: str

@dataclass
class DataValidationConfig:
    """Configuration for data validation"""
    root_dir: Path
    raw_data_path: Path
    validated_data_path: Path
    status_file: Path  # Changed from status_file_path to status_file
    required_columns: list
    target_column: str

@dataclass
class DataTransformationConfig:
    """Configuration for data transformation"""
    root_dir: Path
    data_path: Path
    transformed_data_path: Path
    preprocessor_path: Path
    

@dataclass
class ModelTrainerConfig:
    """Configuration for model training"""
    root_dir: Path
    transformed_data_path: Path
    train_data_path: Path
    test_data_path: Path
    model_path: Path
    target_column: str
    test_size: float
    random_state: int
    alpha: float
    l1_ratio: float
   

@dataclass
class ModelEvaluationConfig:
    """Configuration for model evaluation"""
    root_dir: Path
    test_data_path: Path
    model_path: Path
    metric_file_path: Path
    target_column: str