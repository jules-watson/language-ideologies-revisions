# Initially taken from https://github.com/juliawatson/language-ideologies-2024/blob/main/fall_2023_main/part_2_query_gpt.py

from openai import OpenAI, RateLimitError
from tqdm import trange

import pandas as pd
import csv
import os
import time

from constants import (
    REQUESTS_PER_MINUTE_LIMIT, 
    TOKENS_PER_MINUTE_LIMIT, 
    MODEL_NAME
)

def load_stimuli(data_path):
    """
    Load the stimuli from csv into a list of rows, each corresponding to a prompt
    """
    result = pd.read_csv(data_path, index_col="index")
    result["form_set"] = [eval(item) for item in result["form_set"]]
    return result


def query_gpt(raw_path, loaded_stimuli, num_generations=2):
    """
    For each stimuli sentence: queries GPT-3's API for num_generation text responses.
    Unbatched: for potential batching, see
    https://platform.openai.com/docs/guides/rate-limits/error-mitigation
    """
    # Initialize varialbes to track API rate limitations
    # For help understanding token limit: 
    #       https://platform.openai.com/tokenizer shows how gpt-3.5 tokenizer breaks words down
    queries_avail = REQUESTS_PER_MINUTE_LIMIT
    tokens_avail = TOKENS_PER_MINUTE_LIMIT
    max_tokens_per_request = 500*num_generations

    client = OpenAI()

    start_time = time.time()
    mins = 0 

    with open(raw_path, "w") as f:
        fieldnames = [
            "prompt", "finish_reason", "usage", "responses", "id", "object", "created", "model"
        ]
        csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
        csv_writer.writeheader()

        for i in trange(len(loaded_stimuli)):
            prompt = loaded_stimuli.loc[i, "prompt_text"]

            # Repeatedly attempts to query the api until success- 
            # (the code should not stop prematurely and result in a loss of progress and tokens, 
            # although this try-except does not cover all errors that may occur)
            completed = False

            while not completed: 
                try:
                    output = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "user", "content": prompt},
                        ],
                        max_tokens=None,
                        n=2,
                    )
                    completed = True
                except RateLimitError as e: 
                    print(f"Index:{i}\n", e, "\n")
                    time.sleep(10)
                    completed = False
                except Exception as e: 
                    print(f"Index:{i}\n", e, "\n")
                    completed = False
            
            # Format raw GPT output into a row in output CSV
            responses = [output.choices[i].message.content for i in range(len(output.choices))]
            csv_writer.writerow({
                "prompt": prompt,
                "finish_reason": output.choices[0].finish_reason,
                "usage": dict({
                    "prompt_tokens": output.usage.prompt_tokens, 
                    "completion_tokens": output.usage.completion_tokens,                             
                    "total_tokens": output.usage.total_tokens
                }),
                "responses": responses,
                "id": output.id,
                "object": output.object,
                "created": output.created,
                "model": output.model
            })
        
        # Sleep to ensure that request per minute or token per minute limits are not breached
        # Source: https://platform.openai.com/docs/guides/rate-limits/overview
        end_time = time.time()
        queries_avail -= num_generations
        tokens_avail -= output.usage.total_tokens
        if queries_avail <= 1 or tokens_avail <= max_tokens_per_request and (end_time - start_time) < 60:
            remaining = 60 - (end_time - start_time)
            remaining = remaining if remaining > 0 else 0
            time.sleep(remaining)
            queries_avail = REQUESTS_PER_MINUTE_LIMIT
            tokens_avail = TOKENS_PER_MINUTE_LIMIT
            mins += 1
            start_time = time.time()
        elif (end_time - start_time) >= 60:
            start_time = time.time()


def main(config):
    """
    For each stimuli sentence: query the model and save the output of the model.
    """
    print(f"Collecting data from model: {MODEL_NAME}")
    print(f"config: {config}")

    dirname = "/".join(config.split("/")[:-1])
    input_path = f"{dirname}/stimuli.csv"
    raw_path = f"{dirname}/{MODEL_NAME}/raw.csv"

    if not os.path.exists(f"{dirname}/{MODEL_NAME}"):
        os.mkdir(f"{dirname}/{MODEL_NAME}")

    input_sentences = load_stimuli(input_path)
    query_gpt(raw_path, input_sentences, num_generations=5)

if __name__ == "__main__":
    main("test_experiment/config.json")
