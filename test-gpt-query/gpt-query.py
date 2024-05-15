# Initially taken from https://github.com/juliawatson/language-ideologies-2024/blob/main/fall_2023_main/part_2_query_gpt.py

# Requires openai==0.28

import pandas as pd
# import tqdm
import csv
import numpy as np
import datetime
import os
import openai
import time

import constants


def load_stimuli(data_path):
    """
    Load the stimuli from csv into a list of rows, each corresponding to a prompt
    """
    result = pd.read_csv(data_path, index_col="index")
    result["form_set"] = [eval(item) for item in result["form_set"]]
    return result


def set_up_api():
    """
    Sets up connection to the API, need to put OPEN_API_KEY as a variable in the terminal before running
    """
    # export OPENAI_API_KEY="INSERT_KEY_HERE"
    # Set up access to the api
    openai.organization = "org-tl7YMxuVpm9XVcU5fHxH2QDh"
    openai.api_key = os.getenv("OPENAI_API_KEY")    

                
def query_gpt(num_generations=2):
    """
    For each prompt, queries gpt-3.5 to generate num_generations text responses.  
    """
    set_up_api()

    raw_path = "raw.csv"

    # Initialize varialbes to track API rate limitations
    # For help understanding token limit: 
    #       https://platform.openai.com/tokenizer shows how gpt-3.5 tokenizer breaks words down
    queries_avail = 5
    tokens_avail = 250000
    max_tokens_per_request = 200*num_generations
    start_time = time.time()
    mins = 0  

    with open(raw_path, "w") as f:
        # Repeatedly attempts to query the api until success- 
        # (the code should not stop prematurely and result in a loss of progress and tokens, 
        # although this try-except does not cover all errors that may occur)
        # If code stops prematurely:
        #       look at the last index of the raw file + 1, or the tqdm progress bar
        #       uncomment lines 66-67 and replace STARTING_INDEX with the last index
        #       before running, change line 51, "w" to "a"
        # if i < STARTING_INDEX: 
        #     # < 2854
        #     # missing i = 1987
        #     continue
        prompt = "Revise and justify the following sentence: Bill is a congressperson."

        completed = False
        while not completed: 
            try:
                output = openai.ChatCompletion.create(model ="gpt-3.5-turbo", 
                                                messages = [{"role":"user", "content":prompt}], n=num_generations)
                completed = True
            except openai.error.RateLimitError as e: 
                print(e)
                time.sleep(10)
                completed = False
            except Exception as e: 
                print(e)
                completed = False

    
if __name__ == "__main__":
    query_gpt()