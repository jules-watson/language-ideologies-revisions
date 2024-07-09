"""
Common functions used within the pipeline.

Author: Raymond Liu
Date: May 2024
"""

import json
import pandas as pd


gender_list_dict = {
    2: ['masculine', 'feminine'],
    3: ['neutral', 'masculine', 'feminine'],
}

def load_json(data_path):
    """
    Load a json file and return its contents as a dictionary
    """
    # Open and read the JSON file
    with open(data_path, 'r') as file:
        data = json.load(file)
    
    return data

def load_csv(data_path, with_index_col=True):
    """
    Load a csv file into a list of rows, potentially with an index column corresponding to the index
    """
    if with_index_col:
        result = pd.read_csv(data_path, index_col="index")
    else:
        result = pd.read_csv(data_path)

    return result
