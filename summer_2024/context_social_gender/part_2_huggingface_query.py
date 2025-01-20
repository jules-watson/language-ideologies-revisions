"""
TODO: this needs to be adapted to fit our pipeline and prompts.

You will want to use the single-token parts of this script
(since we're looking at the probabilities of "man", "woman", and "person",
which are each one token).
"""


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
import time
import os
import tqdm
import csv
import datetime
import numpy as np

import constants
from common import load_json


# To download models:
# cd /scratch/ssd004/scratch/jwatson
# huggingface-cli download meta-llama/Meta-Llama-3-8B --local-dir meta-llama/Meta-Llama-3-8B
# huggingface-cli download meta-llama/Meta-Llama-3.1-8B --local-dir meta-llama/Meta-Llama-3.1-8B
# huggingface-cli download meta-llama/Llama-2-7b-hf --local-dir meta-llama/Llama-2-7b-hf


MODEL_NAME_TO_MODEL_PATH = {
    "llama-3-8B": "/scratch/ssd004/scratch/jwatson/meta-llama/Meta-Llama-3-8B",
    "llama-3.1-8B": "/scratch/ssd004/scratch/jwatson/meta-llama/Meta-Llama-3.1-8B",
    "llama-2-7B":  "/scratch/ssd004/scratch/jwatson/meta-llama/Llama-2-7b-hf"
}

MODEL_NAME_TO_WHITESPACE_CHARACTER = {
    "llama-3-8B": "Ġ",
    "llama-3.1-8B": "Ġ",
    "llama-2-7B": "▁"
}
WHITESPACE_CHARACTER = MODEL_NAME_TO_WHITESPACE_CHARACTER[constants.MODEL_NAME]


def set_up_model(imported=False, device_name="cuda"):
    """
    Sets up variables to use model
    """
    global model
    global tokenizer
    global device
    device = device_name

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_TO_MODEL_PATH[constants.MODEL_NAME])
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME_TO_MODEL_PATH[constants.MODEL_NAME],
        torch_dtype=torch.float16,
        device_map=device
    )
    model.eval()
        
    if imported:
        return model, tokenizer, device
    return None, None, None


def compute_probability(sentence_logprobs, string_tokens, continuation):

    def convert_tokens_to_string(token_list):
        return "".join(token_list).replace(WHITESPACE_CHARACTER, " ").strip()

    # Figure out how many tokens make up the continuation
    i = len(string_tokens) - 1
    continuation_str = convert_tokens_to_string(string_tokens[i])
    while continuation_str != continuation and i > 0:
        i -=1
        continuation_str = convert_tokens_to_string(string_tokens[i:])
    assert continuation_str == continuation, f"{continuation_str} != {continuation}; string_tokens={string_tokens}"
    token_logprobs = sentence_logprobs[i:]
    
    return np.sum(token_logprobs)


def get_sentence_token_logprobs(logprobs, sentence_tokens):
    """For each token in sentence_tokens, extract that
    token's probability in logprobs.
    
    input shape: [n_tokens, vocab_size]
    output shape: [n_tokens]
    """
    result = []
    # Note that:
    # - logprobs contains probability distribution over tokens *after*
    #   the token at that position in sentence_tokens.input_ids.
    # - sentence_tokens starts with <|start_of_text|>.
    # This means the probabability of the first word in the sentence (after
    # <|start_of_text|>) will be at logprobs[0, sentence_tokens.input_ids[0][1]]
    # This is why we iterate over sentence_tokens.input_ids[0], starting at
    # index 1.
    for token_i, vocab_i in enumerate(sentence_tokens["input_ids"][0][1:]):
        result.append(logprobs[token_i, vocab_i].item())
    return np.array(result)


def run_model(sentence, continuation):
    """
    Query the model for the log probability of a given variant. 
    """

    # Tokenize the sentence
    sentence_tokens = tokenizer(sentence, return_tensors="pt")
    sentence_tokens.to("cuda")

    # Feed the sentence into the model
    with torch.no_grad():
        # shape: [n_tokens, vocab_size]
        output_logits = model(input_ids=sentence_tokens.input_ids).logits[0]

    # normalize to get probabilities -- make this a testable function
    logprobs = torch.log_softmax(output_logits, axis=1)
    # assert torch.allclose(torch.sum(torch.exp(logprobs), axis=1), torch.tensor(1.).to("cuda"))

    if isinstance(continuation, dict):
        logprob_dict = {}
        for adj, adj_id in continuation.items():
            # Access the log probability of the adjective as the last token continuation (last position means i = -1), j = id
            adj_logprob = logprobs[-1, adj_id].item()
            clean_adj = adj.lstrip(WHITESPACE_CHARACTER) # removes leading whitespace
            logprob_dict[clean_adj] = adj_logprob
        return logprob_dict
    else:
        # use normalized probabilities to compute the probability of the continuation -- make this a testable function
        sentence_logprobs = get_sentence_token_logprobs(logprobs, sentence_tokens)
        string_tokens = tokenizer.convert_ids_to_tokens(sentence_tokens["input_ids"][0])[1:]
        return compute_probability(sentence_logprobs, string_tokens, continuation)  


def query_llama(output_path, loaded_sentences_df, continuation_col, model_name):
    """
    For each row in loaded_sentences_df, calculate the log probability of the continuation 
    and save into output_path. 
    """
    if isinstance(continuation_col, dict): # continuation_col is dict if single-token
        # Set fieldnames with one column per adjective
        fieldnames = list(loaded_sentences_df.columns) + ["model"] + list(continuation_col.keys())
    else: 
        fieldnames = list(loaded_sentences_df.columns) + ["model", "logprob"]

    with open(output_path, "w") as f:
        csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
        csv_writer.writeheader()
        start_time = time.time()
        
        for i,row in tqdm.tqdm(loaded_sentences_df.iterrows()):
            output_row = {}
            for fname in fieldnames:
                if fname in row:
                    output_row[fname] = row[fname]
            
            output_row["model"] = model_name

            if isinstance(continuation_col, dict): 
                model_output = run_model(row["prompt"], continuation_col)
                output_row.update(model_output)
            else:
                output_row["logprob"] = run_model(row["prompt"], row[continuation_col])

            csv_writer.writerow(output_row)
        end_time = time.time()
        
        return end_time - start_time
            
        
def main(config_path):  
    """
    For each stimuli sentence, queries the model for probabilities of pronoun variants
    and saves the result in output_path file. This uses raw_path file as an intermediate.
    """
    print(f"Collecting data from model: {constants.MODEL_NAME}")
    print(f"config_path: {config_path}")
    date = str(datetime.datetime.now(datetime.timezone.utc))
    print(date)

    config = load_json(config_path)
    dirname = "/".join(config_path.split("/")[:-1])
    input_path = f"{dirname}/stimuli.csv"
    output_path = f"{dirname}/{constants.MODEL_NAME}/logprobs.csv"
    experiment_type = config["experiment_type"]

    if experiment_type=="multi-token": # multi-token case
        continuation_col = config["continuation_col"]
    elif experiment_type=="single-token": # single-token case
        item_path = config.get("item_path")
        item_df = pd.read_csv(item_path) #only considering case for adj for now 
        single_token_adjectives = {}
        # Add token IDs for single-token adjectives
        for item in item_df["adjectives"]:
            # item_token = MODEL_NAME_TO_WHITESPACE_CHARACTER[constants.MODEL_NAME] + adj
            item_token = WHITESPACE_CHARACTER + item
            vocab = tokenizer.get_vocab()
            if item_token in vocab:
                single_token_adjectives[item] = vocab[item_token]
        continuation_col = single_token_adjectives

    if not os.path.exists(f"{dirname}/{constants.MODEL_NAME}"):
        os.mkdir(f"{dirname}/{constants.MODEL_NAME}")

    input_sentences_df = pd.read_csv(input_path)
    runtime_str = query_llama(
        output_path, 
        input_sentences_df, 
        continuation_col=continuation_col,
        model_name=constants.MODEL_NAME)

    with open(f"{dirname}/{constants.MODEL_NAME}/running_metadata.txt", "a") as metadata:
        metadata.write(f"{date}\nTotal prompts: {str(len(input_sentences_df))}\nDevice: {device}\n")
        metadata.write("Total seconds: " + str(runtime_str))
      


if __name__ == "__main__":    
    assert("llama" in constants.MODEL_NAME.lower())
    set_up_model()
    # main("analyses/pilot_adjective/config.json")
    # main("analyses/pilot_adjectives_expanded_single_token/config.json")
    # main("analyses/pilot_context_gender_associations/config.json")
    main("analyses/pilot_overt_stereotypes/config.json")