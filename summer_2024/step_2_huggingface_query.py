"""
Query huggingface models for responses to prompts, for a specific configuration and set of stimuli.

Adapted from 2_gpt_query.py.
Used this resource: https://huggingface.co/docs/transformers/main/en/conversations

Author: Julia Watson
Date: September 2024

To download models, use commands like:
huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct --local-dir meta-llama/Meta-Llama-3.1-8B-Instruct
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


# What can we do to speed this up?
# [tried it - no] does passing eos_token_id=terminators help ()https://medium.com/@manuelescobar-dev/implementing-and-running-llama-3-with-hugging-faces-transformers-library-40e9754d8c80
# [tried it - no] does setting max_new_tokens to a power of two help? (512 instead of 500)
# [] does changing the GPU configuration help - running with a different GPU or more GPUs (instead of gpu:rtx6000:1) 
# [] does parallelizing calls to the pipeline help, 
#    -> possibly by using a dataset (I was getting this warning: You seem to be using the pipelines sequentially on GPU. In order to maximize efficiency please use a dataset)
#    -> possibly by setting batch + passing multiple samples at once
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
    query_time = query_huggingface(processed_path, input_sentences, config)

    with open(f"{dirname}/{MODEL_NAME}/running_metadata.txt", "a") as metadata:
        metadata.write(f"{date}\nTotal prompts: {str(len(input_sentences))}")
        metadata.write("\nTotal seconds: " + str(query_time))


if __name__ == "__main__":
    main(f"{EXPERIMENT_PATH}/config.json")
