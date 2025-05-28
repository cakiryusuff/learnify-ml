import pandas as pd
from learnify_ml.src.custom_exception import CustomException
from learnify_ml.src.logger import get_logger

logger = get_logger(__name__)

def load_data(file_path):
    try:
        """
        Load data from a CSV file.
        
        Parameters:
        file_path (str): The path to the CSV file.
        
        Returns:
        pd.DataFrame: The loaded data as a pandas DataFrame.
        """
        logger.info(f"Loading data from {file_path}")
        return pd.read_csv(file_path)
    except Exception as e:
        raise CustomException(e, "Error loading data from file: {}".format(file_path))

def save_data(data, file_path):
    try:
        """
        Save data to a CSV file.
        
        Parameters:
        data (pd.DataFrame): The data to save.
        file_path (str): The path to the CSV file where the data will be saved.
        """
        logger.info(f"Saving data to {file_path}")
        data.to_csv(file_path, index=False)
    except Exception as e:
        raise CustomException(e, "Error saving data to file: {}".format(file_path))