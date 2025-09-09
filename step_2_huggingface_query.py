"""
Query huggingface models for responses to prompts, for a specific configuration and set of stimuli.

Adapted from 2_gpt_query.py.
Used this resource: https://huggingface.co/docs/transformers/main/en/conversations

Author: Jules Watson
Date: September 2024

To download models, use commands like:
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct --local-dir meta-llama/Meta-Llama-3.1-8B-Instruct
"""

from tqdm import trange

import csv
import datetime
import os
import pandas as pd
import time
import torch


from constants import (
    MODEL_NAME,
    EXPERIMENT_PATH,
    USE_SHARDS,
    NON_INSTRUCTION_FINETUNED_MODELS
)
from common import load_json, load_csv


# This downloads huggingface models 
# relevnat for gemini models (not llama)
os.environ['HF_HOME'] = "/scratch/ssd004/scratch/jwatson/hf_models/"

from transformers import pipeline


# What can we do to speed this up?
# [tried it - no] does passing eos_token_id=terminators help ()https://medium.com/@manuelescobar-dev/implementing-and-running-llama-3-with-hugging-faces-transformers-library-40e9754d8c80
# [tried it - no] does setting max_new_tokens to a power of two help? (512 instead of 500)
# [] does changing the GPU configuration help - running with a different GPU or more GPUs (instead of gpu:rtx6000:1) 
# [] does parallelizing calls to the pipeline help, 
#    -> possibly by using a dataset (I was getting this warning: You seem to be using the pipelines sequentially on GPU. In order to maximize efficiency please use a dataset)
#    -> possibly by setting batch + passing multiple samples at once
MODEL_NAME_TO_MODEL_PATH = {
    "llama-3.1-8B-Instruct": "/scratch/ssd004/scratch/jwatson/meta-llama/Meta-Llama-3.1-8B-Instruct",
    "gemma-2-9b-it": "google/gemma-2-9b-it",
    "Mistral-Nemo-Instruct-2407": "mistralai/Mistral-Nemo-Instruct-2407",

    "llama-3.1-8B": "/scratch/ssd004/scratch/jwatson/meta-llama/Meta-Llama-3.1-8B",
    "gemma-2-9b": "google/gemma-2-9b",
    "Mistral-Nemo-Base-2407": "mistralai/Mistral-Nemo-Base-2407",
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
            if MODEL_NAME in NON_INSTRUCTION_FINETUNED_MODELS:
                chat = prompt
            else:
                chat = [{
                    "role": "user",
                    "content": prompt
                }]

            # query huggingface model
            # Note: Takes one minute for 3 responses of max 500 tokens, but 10 seconds for 
            # 1 response of max 500 tokens, so we should query separately, rather than 
            # using the num_return_sequences paramter.
            for _ in range(query_api_args["num_responses"]):
                response = pipe(
                    chat, 
                    eos_token_id=pipe.tokenizer.eos_token_id,
                    pad_token_id=pipe.tokenizer.eos_token_id,
                    max_new_tokens=query_api_args["max_tokens_per_response"])[0]

                if MODEL_NAME in NON_INSTRUCTION_FINETUNED_MODELS:
                    response_text = response["generated_text"].lstrip(chat)
                else:
                    assert len(response["generated_text"]) == 2
                    assert response["generated_text"][0]["role"] == "user"
                    assert response["generated_text"][1]["role"] == "assistant"
                    response_text = response["generated_text"][1]["content"]

                row_dict = {
                    "index": i,
                    "prompt": prompt,
                    "usage": {},
                    "response": response_text,
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


def store_shards(shard_format_str, full_sentences, n_shards):
    shard_size = len(full_sentences) // n_shards

    # store first n-1 shards
    for shard_i in range(n_shards - 1):
        start = shard_i * shard_size
        end = (shard_i + 1) * shard_size
        curr_data = full_sentences[start:end]

        curr_path = shard_format_str.format(shard_i + 1)
        curr_data.to_csv(curr_path, index=False)
    
    # store the nth shard
    start = (n_shards - 1) * shard_size
    curr_data = full_sentences[start:]
    curr_path = shard_format_str.format(n_shards)
    curr_data.to_csv(curr_path, index=False)


def all_shards_complete(processed_path_format_str, n_shards):
    for i in range(1, n_shards + 1):
        curr_path = processed_path_format_str.format(i)
        if not os.path.exists(curr_path):
            return False
    return True


def merge_shards(processed_path_format_str, n_shards):
    result = pd.DataFrame()
    for i in range(1, n_shards + 1):
        curr_path = processed_path_format_str.format(i)
        curr_data = pd.read_csv(curr_path)
        result = pd.concat([result, curr_data])
    return result.reset_index(drop=True)


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

    if not os.path.exists(f"{dirname}/{MODEL_NAME}"):
        os.mkdir(f"{dirname}/{MODEL_NAME}")

    config = load_json(config_path)

    if "n_shards" in config and USE_SHARDS:

        # Split into shards if needed for this model (skip if already done)
        n_shards = config["n_shards"]
        shard_format_str = dirname + "/stimuli-shard-{}-of" + f"-{n_shards}.csv"
        shard_1_path = shard_format_str.format(1)
        if not os.path.exists(shard_1_path):
            full_sentences = load_csv(input_path)
            store_shards(shard_format_str, full_sentences, n_shards)
    
        # Determine which shard to run for now
        processed_path_format_str = f"{dirname}/{MODEL_NAME}/" + "processed-{}-of" + f"-{n_shards}.csv"
        processed_path = None
        for shard_i in range(1, n_shards + 1):
            if os.path.exists(processed_path_format_str.format(shard_i)):
                continue
            processed_path = processed_path_format_str.format(shard_i)
            break
        if processed_path is None:
            raise ValueError(f"All shards are complete for: {processed_path_format_str}")
        
        # Load corresponding sentence data for this shard
        input_sentences = pd.read_csv(shard_format_str.format(shard_i))

    else:
        processed_path = f"{dirname}/{MODEL_NAME}/processed.csv"
        input_sentences = pd.read_csv(input_path, index_col=False)
    

    query_time = query_huggingface(processed_path, input_sentences, config)

    with open(f"{dirname}/{MODEL_NAME}/running_metadata.txt", "a") as metadata:
        metadata.write(f"{date}\nTotal prompts: {str(len(input_sentences))}")
        metadata.write("\nTotal seconds: " + str(query_time))

    # Merge shards, if they're all done.
    if "n_shards" in config and USE_SHARDS:
        n_shards = config["n_shards"]
        processed_path_format_str = f"{dirname}/{MODEL_NAME}/" + "processed-{}-of" + f"-{n_shards}.csv"
        if all_shards_complete(processed_path_format_str, n_shards):
            result = merge_shards(processed_path_format_str, n_shards)
            merged_output_path = f"{dirname}/{MODEL_NAME}/processed.csv"
            result.to_csv(merged_output_path)



if __name__ == "__main__":
    main(f"{EXPERIMENT_PATH}/config.json")
