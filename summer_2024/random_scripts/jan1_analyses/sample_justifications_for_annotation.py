"""Sample justifications for annotation.

We are considering training a model to classify justifications
(e.g., "gender-related", "text-quality-related positive valence", 
"text-quality-related positive valence").

However, we want to first assess whether an annotation scheme
like this is suitable for our data. This script samples 100
justifications where a revision was made (considering responses from a
llama model and a gpt model; selected evenly from across 
the 4 prompt wordings, including "revise", "revise-if-needed", "improve",
"improve-if-needed")
"""

import pandas as pd


def main(config_paths, output_path, n=100):
    
    # Load justifications into a single spreadsheet
    df = pd.DataFrame()
    for config_path in config_paths:
        justifications_processed_path = config_path.replace("config.json", "sbert_input_for_justifications.csv")
        curr_df = pd.read_csv(justifications_processed_path)
        curr_df["prompt_wording"] = config_path.split("/")[-2]
        df = pd.concat([df, curr_df])

    # Sample n cases
    sample_df = df.sample(n)
    sample_df.to_csv(output_path)


if __name__ == "__main__":
    config_paths=[
        "analyses/piloting_jan1/improve/config.json",
        "analyses/piloting_jan1/improve_if_needed/config.json",
        "analyses/piloting_jan1/revise/config.json",
        "analyses/piloting_jan1/revise_if_needed/config.json"
    ]
    output_path = "random_scripts/jan1_analyses/sample_justifications_100.csv"
    main(config_paths, output_path)