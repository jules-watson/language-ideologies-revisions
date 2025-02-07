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
import sys

# Get the parent directory path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add parent directory to sys.path
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

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


def run_model(sentence, continuation, is_single_token):
    """
    Query the model for the log probability of a given noun. 
    """

    # Tokenize the sentence
    sentence_tokens = tokenizer(sentence, return_tensors="pt")
    sentence_tokens.to("cuda")

    # Feed the sentence into the model
    with torch.no_grad():
        # shape: [n_tokens, vocab_size]
        output_logits = model(input_ids=sentence_tokens.input_ids).logits[0]

    # Normalize to get probabilities
    logprobs = torch.log_softmax(output_logits, axis=1)

    if is_single_token:
        logprob_dict = {}
        
        for noun, noun_id in continuation.items():
            # Access the log probability of the noun as the next token in the sequence
            noun_logprob = logprobs[-1, noun_id].item()
            
            # Remove leading whitespace from the noun
            noun_stripped = noun.lstrip(WHITESPACE_CHARACTER)
            
            # Store the log probability in the dictionary with the stripped noun as the key
            logprob_dict[noun_stripped] = noun_logprob
        
        return logprob_dict
    else:
        # Use normalized probabilities to compute the probability of the continuation
        sentence_logprobs = get_sentence_token_logprobs(logprobs, sentence_tokens)
        string_tokens = tokenizer.convert_ids_to_tokens(sentence_tokens["input_ids"][0])[1:]
        return compute_probability(sentence_logprobs, string_tokens, continuation)

def get_fieldnames(loaded_sentences_df, continuation, is_single_token):
    """
    Determine the fieldnames for the CSV output based on if the continuation is a sinlge or multiple tokens.
    """
    base_fieldnames = list(loaded_sentences_df.columns) + ["model"]
    
    if is_single_token:  
        continuation_fieldnames = list(continuation.keys())
        return base_fieldnames + continuation_fieldnames
    else:  
        return base_fieldnames + ["logprob"]

def create_output_row(row, fieldnames, model_name, continuation, is_single_token):
    """
    Create an output row for the CSV file with log probabilities of the next token(s).
    """
    output_row = {fname: row[fname] for fname in fieldnames if fname in row}
    output_row["model"] = model_name
    
    if is_single_token:  
        model_output = run_model(row["prompt_text"], continuation, is_single_token)
        output_row.update(model_output)
    else:  
        output_row["logprob"] = run_model(row["prompt_text"], row[continuation], is_single_token)
    
    return output_row
            

def query_llama(output_path, loaded_sentences_df, continuation, model_name):
    """
    For each row in loaded_sentences_df, calculate the log probability of the 
    next token(s) and save it to output_path. 
    """
    # continuation will only be a dict in the single-token case
    is_single_token = isinstance(continuation, dict)
    
    # Determine fieldnames for the CSV output depending on if we have a single or multiple next tokens
    fieldnames = get_fieldnames(loaded_sentences_df, continuation, is_single_token)
    
    # Open the output file and write the header
    with open(output_path, "w") as f:
        csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
        csv_writer.writeheader()
        
        start_time = time.time()
        
        # Process each row in the DataFrame
        for _, row in tqdm.tqdm(loaded_sentences_df.iterrows(), total=loaded_sentences_df.shape[0]):
            output_row = create_output_row(row, fieldnames, model_name, continuation, is_single_token)
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

    # Determine the continuation column based on single or multi-token case
    if experiment_type=="multi-token":  # multi-token case
        continuation = config["continuation"]

    elif experiment_type == "single-token":  # single-token case
        nouns = config.get("single_next_tokens", [])
        single_token_nouns = {}
        # Add token IDs for single-token nouns
        for noun in nouns:
            token = WHITESPACE_CHARACTER + noun
            vocab = tokenizer.get_vocab()
            if token in vocab:
                single_token_nouns[noun] = vocab[token]
        continuation = single_token_nouns

    if not os.path.exists(f"{dirname}/{constants.MODEL_NAME}"):
        os.mkdir(f"{dirname}/{constants.MODEL_NAME}")

    input_sentences_df = pd.read_csv(input_path)
    runtime_str = query_llama(
        output_path, 
        input_sentences_df, 
        continuation=continuation,
        model_name=constants.MODEL_NAME)

    with open(f"{dirname}/{constants.MODEL_NAME}/running_metadata.txt", "a") as metadata:
        metadata.write(f"{date}\nTotal prompts: {str(len(input_sentences_df))}\nDevice: {device}\n")
        metadata.write("Total seconds: " + str(runtime_str))
      

if __name__ == "__main__":    
    assert("llama" in constants.MODEL_NAME.lower())
    set_up_model()
    main(f"{constants.EXPERIMENT_PATH}/config.json")