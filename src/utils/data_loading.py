import pandas as pd
import os


def load_dataset(path: str = "data/Filter_Toxic_Comments_dataset.csv") -> pd.DataFrame:
    """
    Loads the dataset and returns a pandas DataFrame.
    Checks that the file exists and is readable.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"\033[91mThe file {path} does not exist. Make sure you put it in the 'data/' folder.\033[0m")
    
    try:
        df = pd.read_csv(path)
        print(f"\033[92mDataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns.\033[0m")
        return df
    except Exception as e:
        raise RuntimeError(f"\033[91mError loading dataset: {e}.\033[0m")