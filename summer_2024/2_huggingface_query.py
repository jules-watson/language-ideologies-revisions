"""
Query huggingface models for responses to prompts, for a specific configuration and set of stimuli.

Adapted from 2_gpt_query.py.
Used this resource: https://huggingface.co/docs/transformers/main/en/conversations

Author: Julia Watson
Date: September 2024
"""

from tqdm import trange

import csv
import datetime
import os
import time
import torch
from transformers import pipeline


from constants import (
    MODEL_NAME,
    EXPERIMENT_PATH
)
from common import load_json, load_csv


# TODO - download this model and test it
MODEL_NAME_TO_MODEL_PATH = {
    "llama-3.1-8B-Instruct": "/scratch/ssd004/scratch/jwatson/meta-llama/Meta-Llama-3.1-8B-Instruct",
}


def query_huggingface(processed_path, loaded_stimuli, config):
    """
    For each stimuli sentence: queries GPT-3's API for num_generation text responses.
    Unbatched: for potential batching, see
    https://platform.openai.com/docs/guides/batch
    """

    query_api_args = config["query_api_args"]

    # initialize model
    pipe = pipeline(
        "text-generation",
        MODEL_NAME_TO_MODEL_PATH[MODEL_NAME],
        torch_dtype=torch.bfloat16,
        device=0  # uses first available GPU
    )

    with open(processed_path, "w") as f:
        fieldnames = ["index"] + config["ind_var_cols"] + config["keep_cols"] + [
            "prompt", "finish_reason", "usage", "response", "id", "object", "created", "model"
        ]        
        csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
        csv_writer.writeheader()

        start_time = time.time()

        for i in trange(len(loaded_stimuli)):
            prompt = loaded_stimuli.loc[i, "prompt_text"]
            chat = {
                "role": "user",
                "content": prompt
            }

            # query huggingface model
            responses = pipe(
                chat, 
                max_new_tokens=query_api_args["max_tokens_per_response"],
                num_return_sequences=query_api_args["num_responses"])
            
            for response in responses:
                assert len(response["generated_text"]) == 2
                assert response["generated_text"][0]["role"] == "user"
                assert response["generated_text"][1]["role"] == "assistant"
                
                row_dict = {
                    "index": i,
                    "prompt": prompt,
                    "usage": {},
                    "response": response["generated_text"][1]["content"],
                    "id": "",
                    "object": "",
                    "created": time.time(),
                    "model": MODEL_NAME
                }
                for v in config["ind_var_cols"] + config["keep_cols"]: # add independent variables to output
                    row_dict[v] = loaded_stimuli.loc[i, v]
                csv_writer.writerow(row_dict)

        end_time = time.time()        
        return end_time - start_time


def main(config):
    """
    For each stimuli sentence: query the model and save the output of the model.
    """
    print(f"Collecting data from model: {MODEL_NAME}")
    print(f"config: {config}")
    date = str(datetime.datetime.now(datetime.timezone.utc))
    print(date)

    dirname = "/".join(config.split("/")[:-1])
    input_path = f"{dirname}/stimuli.csv"
    config_path = f"{dirname}/config.json"
    processed_path = f"{dirname}/{MODEL_NAME}/processed.csv"

    if not os.path.exists(f"{dirname}/{MODEL_NAME}"):
        os.mkdir(f"{dirname}/{MODEL_NAME}")

    input_sentences = load_csv(input_path)
    config = load_json(config_path)

    with open(f"{dirname}/{MODEL_NAME}/running_metadata.txt", "a") as metadata:
        metadata.write(f"{date}\nTotal prompts: {str(len(input_sentences))}")
        metadata.write("Total seconds: " + str(query_huggingface(processed_path, input_sentences, config)))


if __name__ == "__main__":
    main(f"{EXPERIMENT_PATH}/config.json")