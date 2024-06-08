"""
Common functions used within the pipeline.

Author: Raymond Liu
Date: May 2024
"""

import json
import pandas as pd


def load_json(data_path):
    """
    Load a json file and return its contents as a dictionary
    """
    # Open and read the JSON file
    with open(data_path, 'r') as file:
        data = json.load(file)
    
    return data

def load_csv(data_path):
    """
    Load a csv file into a list of rows, with an index column corresponding to the index
    """
    result = pd.read_csv(data_path, index_col="index")
    return result
