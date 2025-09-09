import pandas as pd


def main(stimuli_path, output_path):
    df = pd.read_csv(stimuli_path, index_col=0)
    df = df[df["task_wording"] == "unknown gender"]
    df = df.drop(['task_wording', 'prompt_text'], axis="columns")
    df.to_csv(output_path, index=False)


if __name__ == "__main__":
    stimuli_path = "../../analyses/piloting_pronouns_genders/stimuli.csv"
    output_path = "../../data/piloting_pronouns_genders_sample.csv"
    # NOTE THAT there are 3 duplicate sentences in this sample :/
    main(stimuli_path, output_path)