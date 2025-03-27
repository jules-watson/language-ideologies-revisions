import os
import pandas as pd
import sys

summer_2024_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(summer_2024_dir)

from constants import EXPERIMENT_PATH, MODEL_NAMES


def sample_train_test(config_path, n_train=200, n_test=50):
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
    for model_name in MODEL_NAMES:
        curr_path = config_path.replace("config.json", f"{model_name}/processed.csv")
        df = pd.read_csv(curr_path, index_col=0)
        sample_df = df.sample(n_train + n_test).reset_index(drop=True)

        curr_train_df = sample_df.loc[:n_train-1]
        curr_test_df = sample_df.loc[n_train:]

        train_df = pd.concat([train_df, curr_train_df])
        test_df = pd.concat([test_df, curr_test_df])

    train_df.to_csv("train.csv", index=False)
    test_df.to_csv("test.csv", index=False)


if __name__ == "__main__":
    sample_train_test(f"../../{EXPERIMENT_PATH}/config.json")