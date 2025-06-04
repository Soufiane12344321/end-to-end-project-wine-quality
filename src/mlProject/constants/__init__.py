from pathlib import Path

# Root directory of the project
ROOT_DIR = Path("artifacts")

# Configuration file path
CONFIG_FILE_PATH = Path("config/config.yaml")

# Parameters file path
PARAMS_FILE_PATH = Path("params.yaml")

# Schema file path
SCHEMA_FILE_PATH = Path("schema.yaml")

# MongoDB related constants
MONGODB_URL_KEY = "mongodb_url"
DATABASE_NAME = "winedata"
COLLECTION_NAME = "datasets_wine"

# Data related constants
TARGET_COLUMN = "quality"
TRAIN_TEST_SPLIT_RATIO = 0.2
RANDOM_STATE = 42

# Required columns for validation
REQUIRED_COLUMNS = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "quality"
]