"""
Create JSON outputs of the model for various stimuli and set prompts,
to be used to test the efficacy of splitting.
"""

from openai import OpenAI
from tqdm import tqdm

import json

from config import *

def query_gpt(dir_path):
    data = {} # output dict

    client = OpenAI()

    for m in models:
        print(f"Querying {m}")
        
        with tqdm(total=len(prompts) * len(stimuli)) as pbar: # progress bar
            data[m] = {}

            for p in prompts:
                data[m][p] = {}

                for s in stimuli:
                    prompt = f"{prompts[p]} {stimuli[s]}"

                    output = client.chat.completions.create(
                        model=m,
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=MAX_TOKENS,
                        n=n_responses
                    )

                    responses = [output.choices[i].message.content for i in range(len(output.choices))]
                    data[m][p][s] = responses

                    pbar.update(1)
    
        with open(f"{dir_path}/{m}.json", "w") as json_file:
            json.dump(data[m], json_file, indent=4)


if __name__ == "__main__":
    query_gpt("random_scripts/test_split/outputs")