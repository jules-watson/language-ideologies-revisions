"""
Common functions used within the pipeline.
"""

import json

def load_config(data_path):
    """
    Load the query api configuration for this experiment
    """
    # Open and read the JSON file
    with open(data_path, 'r') as file:
        data = json.load(file)
    
    return data
