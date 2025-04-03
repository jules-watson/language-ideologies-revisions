import pandas as pd

from constants import MODEL_NAMES


def get_column(model_results_df, p_val_threshold):
    POSITIVE_EFFECT_COLOR = "bdffea"  # blue-green
    NEGATIVE_EFFECT_COLOR = "fac0dc"  # pink
    INTERCEPT_EFFECT_COLOR = "dbd9d9" # gray
    NO_PREDICTION = [
        "(Intercept)", "context_fem", "context_neut", "prompt_fem", "prompt_masc"
    ]
    result = []
    # for estimate, p_value in zip(model_results_df["Estimate"], model_results_df["Pr(>|z|)"]):
    for predictor, row in model_results_df.iterrows():
        estimate = row["Estimate"]
        p_value = row["Pr(>|z|)"]
        if predictor in NO_PREDICTION and p_value < p_val_threshold:  # significant effect for Intercept or control predictor -- make it gray bc no prediction
            # result.append(f"\\boldnumcolor{{{estimate:.2f}}}{{{INTERCEPT_EFFECT_COLOR}}}")
            result.append(f"\cellcolor[HTML]{{{INTERCEPT_EFFECT_COLOR}}} {estimate:.2f}")
        elif p_value < p_val_threshold:
            curr_cell_color = NEGATIVE_EFFECT_COLOR if estimate < 0 else POSITIVE_EFFECT_COLOR
            # result.append(f"\\boldnumcolor{{{estimate:.2f}}}{{{curr_cell_color}}}")
            result.append(f"\cellcolor[HTML]{{{curr_cell_color}}} {estimate:.2f}")
        else:
            result.append(f"{estimate:.2f}")
    return result


def combine_results(config_path, models, experiment_name):
    p_val_threshold = 0.05 / len(models)  # bonferroni correction

    result_df = None
    for model in models:
        model_results_path = config_path.replace("config.json", f"{model}/regression_results_{experiment_name}.csv")
        model_results_df = pd.read_csv(model_results_path, index_col=0)

        if result_df is None:
            result_df = pd.DataFrame(index = model_results_df.index)
        else:
            assert (result_df.index == model_results_df.index).all()

        result_df[model] = get_column(model_results_df, p_val_threshold)


    output_path = config_path.replace("config.json", f"overall_regression_results_{experiment_name}.csv")
    result_df.to_csv(output_path)

    output_path_latex = output_path.replace(".csv", ".tex_table")
    result_df.to_latex(output_path_latex)



if __name__ == "__main__":
    combine_results("analyses/full_revise_if_needed/config.json", MODEL_NAMES, "exp1")
    combine_results("analyses/full_revise_if_needed/config.json", MODEL_NAMES, "exp2")
