import os
from box.exceptions import BoxValueError
import yaml
from src.mlProject import logging
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
from pymongo import MongoClient
from typing import Dict, List
import pandas as pd
from src.mlProject.exception import CustomException
from dotenv import load_dotenv
import sys



@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads a yaml file and returns a ConfigBox object
    Args:
        path_to_yaml (Path): path to yaml file
    Raises:
        ValueError: if yaml file is empty
        Exception: if any other error occurs
    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        if not os.path.exists(path_to_yaml):
            raise FileNotFoundError(f"YAML file not found at: {path_to_yaml}")
            
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            
            if content is None:
                raise ValueError(f"YAML file is empty at: {path_to_yaml}")
                
            logging.info(f"YAML file loaded successfully from: {path_to_yaml}")
            return ConfigBox(content)
            
    except Exception as e:
        logging.error(f"Error in reading YAML file: {e}")
        raise e
    


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logging.info(f"created directory at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logging.info(f"json file saved at: {path}")




@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json files data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)

    logging.info(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path):
    """save binary file

    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    joblib.dump(value=data, filename=path)
    logging.info(f"binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """load binary data

    Args:
        path (Path): path to binary file

    Returns:
        Any: object stored in the file
    """
    data = joblib.load(path)
    logging.info(f"binary file loaded from: {path}")
    return data



@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"

def read_yaml_file(filepath: str) -> Dict:
    try:
        with open(filepath) as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise CustomException(e)

def connect_to_mongodb(connection_string: str) -> MongoClient:
    try:
        client = MongoClient(connection_string)
        return client
    
    except Exception as e:
        raise CustomException(e)

def extract_data_from_mongodb(
    connection_string: str,
    database_name: str,
    collection_name: str
) -> pd.DataFrame:
    try:
        # Connect to MongoDB
        client = connect_to_mongodb(connection_string)
        
        # Get database and collection
        db = client[database_name]
        collection = db[collection_name]
        
        # Extract data
        data = list(collection.find({}))
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Drop MongoDB's default _id column
        if '_id' in df.columns:
            df = df.drop('_id', axis=1)
            
        return df
        
    except Exception as e:
        raise CustomException(e)
    finally:
        client.close()

def load_mongodb_env():
    """Load MongoDB environment variables"""
    env_path = Path('.env')
    load_dotenv(dotenv_path=env_path)
    
    return {
        "MONGODB_URL": os.getenv("MONGODB_URL"),
        "DATABASE_NAME": os.getenv("DATABASE_NAME"),
        "COLLECTION_NAME": os.getenv("COLLECTION_NAME")
    }

@ensure_annotations
def save_object(file_path: str, obj: object) -> None:
    """
    Save a Python object as a pickle file using joblib
    
    Args:
        file_path (str): path to save the object
        obj (object): Python object to be saved
    """
    try:
        # Create directory if it doesn't exist
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        # Save the object
        joblib.dump(obj, file_path)
        logging.info(f"Object saved successfully to: {file_path}")
        
    except Exception as e:
        logging.error(f"Error saving object: {e}")
        raise CustomException(e, sys)