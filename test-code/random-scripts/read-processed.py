"""
Try to read the processed CSV file into pandas
"""

import pandas as pd

# Confirms that pandas can read the CSV
result = pd.read_csv("test_experiment/gpt-3.5-turbo/processed.csv")

# Confirm this prints the full response with the new line
print(result["responses"][0])
