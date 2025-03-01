"""
Plot graphs representing the revision and justification statistics.

TODO: make this use the stacked_grouped_bar_graph function from visualizations.py

Author: Raymond Liu
Date: Aug 2024
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from common import load_json
import seaborn as sns
import os

from matplotlib.colors import ListedColormap
from constants import EXPERIMENT_PATH, MODEL_NAMES

# Colors for revision bar graph
base_colors = [
    (0/255, 121/255, 255/255, 1),    # blue
    (0/255, 223/255, 162/255, 1),    # green
    (255/255, 165/255, 0/255, 1),  # orange
    (255/255, 0/255, 96/255, 1)      # red
]

model_names_shortened = {
    'gpt-3.5-turbo': 'GPT-3.5',
    'gpt-4-turbo': 'GPT-4',
    'gpt-4o': 'GPT-4o',
    'llama-3.1-8B-Instruct': 'Llama-3.1-8B'
}


def plot_revision_rates(data_frames, output_path, prompt_wording, axe=None, legend=True, **kwargs):
    """Create a clustered stacked bar plot.

    data_frames is a list of pandas dataframes. The dataframes should have
        identical columns and index

    Adapted from https://github.com/juliawatson/language-ideologies-2024/blob/ae9ddbeb2cb4c78dc8cbcd8f72f7de670b6675ab/fall_2023_main/exploratory/visualize_by_gender.py#L45
    """
    n_df = len(data_frames)
    n_col = len(data_frames[0].columns)-2  # = total number of columns - number of columns before removed_rate
    n_ind = len(data_frames[0].index)

    save_figure = False
    if axe is None:
        if n_ind > 2:
            figsize=[8, 5]
        else:
            figsize=[4.8, 6]

        plt.figure(figsize=figsize)
        axe = plt.subplot(111)
        save_figure = True

    alphas = [0.7, 0.4, 1]
    for i, df in enumerate(data_frames):  # for each dataframe
        alpha = alphas[i]
        colormap = ListedColormap([
            (r, g, b, alpha) for r, g, b, _ in base_colors
        ])
        axe = df.plot(kind="bar",
                      linewidth=0,
                      stacked=True,
                      ax=axe,
                      legend=True,
                      grid=False,
                      colormap=colormap,
                      **kwargs)  # make bar plots

    h, l = axe.get_legend_handles_labels()  # get the handles we want to modify
    for i in range(0, n_df * n_col, n_col):  # len(h) = n_col * n_df
        for j, pa in enumerate(h[i:i + n_col]):
            for rect in pa.patches:  # for each index
                rect.set_x(rect.get_x() + 1 / float(n_df + 1) * i / float(n_col))
                rect.set_width(1 / float(n_df + 1))

    xtick_offset = 1 / float(n_df + 1) / 2
    xticks = (np.arange(0, 2 * n_ind, 2) + xtick_offset) / 2.

    model_ticks = []
    for item in xticks:
        if n_df == 2:
            model_ticks += [item - xtick_offset, item + xtick_offset]
        else:
            model_ticks += [item - xtick_offset, item + xtick_offset, item + 3*xtick_offset]
    xticks = sorted(list(xticks) + model_ticks)

    index_items = list(df.index)
    xtick_labels = []
    for item in index_items:
        labels = [model_names_shortened[model_name] for model_name in MODEL_NAMES]
        labels.insert(1, '\n')
        xtick_labels += (labels)

    axe.set_xticks(xticks)
    axe.set_xticklabels(xtick_labels, rotation=0, fontsize=8)
    axe.tick_params(axis=u'both', which=u'both',length=0)

    # Create a new axis below the primary x-axis for gender labels
    axe2 = axe.secondary_xaxis('bottom')
    axe2.xaxis.set_ticks_position('none') 
    axe2.spines['bottom'].set_visible(False)  # Hides the horizontal line (spine)
    axe2.set_xticks((np.arange(0, 2 * n_ind, 2) + xtick_offset) / 2.)
    axe2.set_xticklabels(['neutral', 'masculine', 'feminine'])
    axe2.spines['bottom'].set_position(('outward', 8))  # Adjust the distance of the secondary axis below

    axe.set_xlabel("Role noun gender in original sentence", labelpad=10)
    axe.set_ylabel("Proportion")

    axe.set_title(f"Revision rates for {EXPERIMENT_PATH} - {prompt_wording}", pad=60)
    axe.set_xlim(xticks[0] - xtick_offset * 4, xticks[-1] + xtick_offset * 4)
    axe.set_ylim(0, 1)

    if legend:
        legend_labels = ["Revised to alternative wording", "Revised to neutral variant", "Revised to masculine variant", "Revised to feminine variant"]
        handles, _ = axe.get_legend_handles_labels()
        handles = handles[:4]
        l1 = axe.legend(handles, legend_labels, loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.2))
        plt.subplots_adjust(top=0.85)

    if save_figure:
        plt.savefig(output_path, #bbox_extra_artists=(l1,),
                    bbox_inches='tight',
                    dpi=700)


def plot_justification_words(justification_freqs_df, output_path, desired_words=None):
    """
    Plot the frequency of certain words in the justification.
    """
    if desired_words is None: # plot all the words
        desired_words = justification_freqs_df['word'].tolist()
    
    # Filter the dataframe for only the desired words
    df_filtered = justification_freqs_df[justification_freqs_df['word'].isin(desired_words)]

    # Prepare data for plotting
    words = df_filtered['word'].tolist()
    settings = MODEL_NAMES
    names = [f'frequency_{model_name}' for model_name in MODEL_NAMES]
    frequencies = df_filtered[names].values

    # Set the number of groups and bars
    n_groups = len(words)
    n_bars = len(settings)

    # Set up the bar width and positions
    bar_width = 0.2
    index = np.arange(n_groups)

    # Create the plot
    fig, ax = plt.subplots()

    # Generate a unique color for each word
    colors = plt.cm.get_cmap('tab10', n_groups)

    # Plot each word's frequencies as a group of bars
    alphas = [0.7, 0.4, 1]
    for i in range(n_groups):
        color = colors(i)
        for j in range(n_bars):
            ax.bar(index[i] + j * bar_width, frequencies[i][j], bar_width,
                label=f'{words[i]}',
                color=color, alpha=alphas[j])
            ax.text(index[i] + j * bar_width, frequencies[i][j] + 0 * max(frequencies[i]), settings[j],
                ha='center', va='top', rotation=90, fontsize=10)

    # Add labels, title, and legend
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(words)
    #ax.legend(loc='best')

    # Show the plot
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # adjust the layout to make room for the title and legend
    plt.savefig(output_path, #bbox_extra_artists=(l1,),
                bbox_inches='tight',
                dpi=700)


def plot_gender_association_scores(
    data_frames, 
    output_dir, 
    model_names, 
    gender_associations_df
):
    """
    Read a CSV (already loaded here as `gender_associations_df`) that contains
    sentence_format, feminine_score, and genderedness_score. Merge those score
    columns into each dataframe in `data_frames` by matching on 'sentence_format'.
    Then, generate a grouped bar plot for each dataframe:

      - 3 bar groups on the x-axis: neutral, masculine, feminine.
      - Each group has 2 bars: "revised" and "not revised" (distinct colors).
      - The bar height indicates the proportion of sentences in each category.
      - Overlay data points with y-values as the feminine_score or genderedness_score.
      - Add box plots to show the distribution of feminine and genderedness scores.
      - Save summary statistics of the feminine and genderedness scores in a CSV file.
      
    Parameters
    ----------
    data_frames : list of pd.DataFrame
        Each item is a dataframe representing a model's outputs.
    output_dir : str
        Directory path to save the plots.
    model_names : list
        Names of the models (parallel to data_frames).
    gender_associations_df : pd.DataFrame
        A dataframe that holds sentence_format and the associated
        feminine_score and genderedness_score columns of each sentence.
    """

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        print(f"The output directory {output_dir} for the genderedness bar graph doesn't exist.")
        return

    # Define distinct color palettes: one for bars, one for data points
    bar_palette = [base_colors[0], base_colors[2]]  # Blue for "not revised", Orange for "revised"
    point_palette = [base_colors[1], base_colors[3]]  # Red for "not revised", Green for "revised"

    # Remove duplicate sentence_format / feminine and genderedness scores in gender_associations_df 
    gender_associations_df = gender_associations_df.drop_duplicates(subset=['sentence_format'])

    for model_name, df in zip(model_names, data_frames):
        # Remove duplicates in df based on sentence - not sure if needed?
        df = df.drop_duplicates(subset=['sentence']) 

        # Attach feminine_score and genderedness_score to every sentence
        # Matching across dataframes will be based on the 'sentence_format' column
        df_merged = df.merge(
            gender_associations_df[["sentence_format", "feminine_score", "genderedness_score"]],
            on="sentence_format",
            how="left"
        )

        # Classify sentence based on whether the role noun variant was revised
        def classify_revision_status(row):
            if ((not row['variant_removed']) and (pd.isna(row['variant_added']))):
                return "not revised"
            return "revised"

        df_merged["revision_status"] = df_merged.apply(classify_revision_status, axis=1)

        # Group and compute proportions
        group_counts = (
            df_merged.groupby(["role_noun_gender", "revision_status"])
            .size()
            .reset_index(name="count")
        )
        # Calculate total count for each role_noun_gender group
        total_counts = df_merged["role_noun_gender"].value_counts().reset_index()
        total_counts.columns = ["role_noun_gender", "total_count"]

        # Merge total counts into group_counts
        group_counts = group_counts.merge(total_counts, on="role_noun_gender")

        # Calculate proportions based on total counts for each group
        group_counts["proportion"] = group_counts["count"] / group_counts["total_count"]

        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

        # y-axis starts at 0, ends at 1
        for ax in axes:
            ax.set_ylim(0, 1)

        # ----------------
        # Left subplot: feminine_score distribution
        # ----------------
        sns.barplot(
            data=group_counts,
            x="role_noun_gender",
            y="proportion",
            hue="revision_status",
            dodge=True,
            errorbar=None,         # no confidence interval
            palette=bar_palette,  
            ax=axes[0],
        )
        # Overlay data points for feminine_score
        sns.stripplot(
            data=df_merged,
            x="role_noun_gender",
            y="feminine_score",
            hue="revision_status",
            dodge=True,
            palette=point_palette, 
            alpha=0.6,
            edgecolor="black",      # outline for better visibility
            linewidth=0.5,
            ax=axes[0],
        )
        # Add box plot for feminine_score
        sns.boxplot(
            data=df_merged,
            x="role_noun_gender",
            y="feminine_score",
            hue="revision_status",
            dodge=True,
            palette=point_palette,
            ax=axes[0],
            showcaps=True,
            boxprops={'facecolor':'None'},
            flierprops={'marker':'x', 'color':'black', 'markersize':5, 'linestyle':'none'},  # Customize outliers
        )
        axes[0].set_title(f"{model_names_shortened[model_name]} - Feminine Score")
        axes[0].set_ylabel("Proportion (bars) / Feminine Score (points)")

        # ----------------
        # Right subplot: genderedness_score distribution
        # ----------------
        sns.barplot(
            data=group_counts,
            x="role_noun_gender",
            y="proportion",
            hue="revision_status",
            dodge=True,
            errorbar=None,
            palette=bar_palette,
            ax=axes[1],
        )
        sns.stripplot(
            data=df_merged,
            x="role_noun_gender",
            y="genderedness_score",
            hue="revision_status",
            dodge=True,
            palette=point_palette,
            alpha=0.6,
            edgecolor="black",
            linewidth=0.5,
            ax=axes[1],
        )
        # Add box plot for genderedness_score
        sns.boxplot(
            data=df_merged,
            x="role_noun_gender",
            y="genderedness_score",
            hue="revision_status",
            dodge=True,
            palette=point_palette,
            ax=axes[1],
            showcaps=True,
            boxprops={'facecolor':'None'},
            flierprops={'marker':'x', 'color':'black', 'markersize':5, 'linestyle':'none'},  # Customize outliers
        )
        axes[1].set_title(f"{model_names_shortened[model_name]} - Genderedness Score")
        axes[1].set_ylabel("Proportion (bars) / Genderedness Score (points)")

        for ax in axes:
            handles, labels = ax.get_legend_handles_labels()
            # We only need two labels: "not revised" and "revised"
            if "revised" in labels and "not revised" in labels:
                # Show first 2 handles/labels
                ax.legend(handles[:2], labels[:2], loc="upper right")
            else:
                ax.legend([], [], frameon=False)

        # ---- Set custom x-ticks and multi-level axis labels ----
        # Each x-axis has 3 groups (feminine, masculine, neutral), each with 2 bars
        # We want a row of gender labels, a row of "Revised" vs "Not Revised" subtitles
        # We'll remove the default x-axis ticks, then manually place them.

        for ax in axes:
            # Remove default x labels
            ax.set_xticks([0, 1, 2])  # Set the positions of the ticks
            ax.set_xticklabels([""] * 3)  # 3 is the number of variants (feminine, masculine, neutral)

            # Place gender labels at the center of each group
            # By default, Seaborn groups the bars at integer positions [0,1,2],
            # near-labeled as valid_genders in alphabetical or data order.
            # We'll place "feminine", "masculine", "neutral" near x=0,1,2. 
            # Then we place "Revised" / "Not Revised" at sub-locations for each group.
            ax.set_xlabel("")

            # Create a secondary axis for sub-labels
            secax = ax.secondary_xaxis('bottom')
            secax.spines["bottom"].set_visible(False)
            secax.set_xlabel("")
            # Hardcode the positions: 
            # bar 1 for group 0 is at ~ -0.2, bar 2 is at ~ +0.2, group 1 is at 0.8/1.2, group 2 is at 1.8/2.2, etc.
            # We'll approximate them below. 
            # "neutral" center ~ 0.0, "masculine" center ~ 1.0, "feminine" center ~ 2.0
            # We'll place "Revised" near -0.2, +0.8, +1.8, and "Not Revised" near +0.2, +1.2, +2.2
            secax.set_xticks([-0.2, 0.2, 0.8, 1.2, 1.8, 2.2])
            secax.set_xticklabels(["Not Revised","Revised","Not Revised","Revised","Not Revised","Revised"], rotation=0)

            # Another axis for the main labels (or reuse the same axis with more major ticks):
            # We'll simply use a second secondary_xaxis or manual annotation:
            ax2 = ax.secondary_xaxis('bottom')
            ax2.spines["bottom"].set_visible(False)
            ax2.set_xlabel("")
            # Put main group labels for x=0,1,2
            ax2.set_xticks([0,1,2])
            ax2.set_xticklabels(["Feminine","Masculine","Neutral"])
            ax2.tick_params(axis='x', pad=30)  # Move them down to avoid overlap

        plt.tight_layout()
        save_path = os.path.join(output_dir, f"{model_names_shortened[model_name]}_gender_score_grouped_bars.png")
        plt.savefig(save_path, dpi=300)
        plt.close(fig)

        # Calculate summary statistics for feminine_score and genderedness_score
        summary_stats = df_merged.groupby(["role_noun_gender", "revision_status"]).agg(
            feminine_mean=("feminine_score", "mean"),
            feminine_std=("feminine_score", "std"),
            feminine_min=("feminine_score", "min"),
            feminine_q1=("feminine_score", lambda x: x.quantile(0.25)),
            feminine_median=("feminine_score", "median"),
            feminine_q3=("feminine_score", lambda x: x.quantile(0.75)),
            feminine_max=("feminine_score", "max"),
            genderedness_mean=("genderedness_score", "mean"),
            genderedness_std=("genderedness_score", "std"),
            genderedness_min=("genderedness_score", "min"),
            genderedness_q1=("genderedness_score", lambda x: x.quantile(0.25)),
            genderedness_median=("genderedness_score", "median"),
            genderedness_q3=("genderedness_score", lambda x: x.quantile(0.75)),
            genderedness_max=("genderedness_score", "max")
        ).reset_index()

        # Save summary statistics to CSV
        summary_stats_path = os.path.join(output_dir, f"{model_names_shortened[model_name]}_gender_score_summary_stats.csv")
        summary_stats.to_csv(summary_stats_path, index=False)

        print(summary_stats)


def main(config):
    dirname = "/".join(config.split("/")[:-1])  # (same as EXPERIMENT_PATH) analyses/piloting_jan1/revise_if_needed
    config_path = f"{dirname}/config.json"  # path to the config file
    config = load_json(config_path)  # load the config file

    df_paths = [f'{dirname}/{model_name}/revision_stats.csv' for model_name in MODEL_NAMES]  # for every model, retrieve the path to the revision_stats csv 
    dfs = [pd.read_csv(df_path) for df_path in df_paths]  # load the revision_stats csvs for every model as dataframes
    
    task_wording_dict = config['task_wording']  # get the task wording dict 
    for prompt_wording in task_wording_dict.keys():  # every task wording name
    
        filtered_revision_stats_dfs = [df[df['prompt_wording'] == prompt_wording] for df in dfs] # filter the dataframes to keep only the rows for the current prompt wording

        # Escape slashes in prompt_wording
        escaped_prompt_wording = prompt_wording.replace('/', ' ')        
        
        # revision rates plot for the current prompt
        output_path = f'{dirname}/revision_bar_graph_{escaped_prompt_wording}.png'
        # plot_revision_rates(data_frames=filtered_revision_stats_dfs,
        #                     output_path=output_path,
        #                     prompt_wording=prompt_wording)

        if prompt_wording == 'unknown gender' and 'gender_association_scores' in config: 
            revision_df_paths = [f'{dirname}/{model_name}/revision.csv' for model_name in MODEL_NAMES]  # for every model, retrieve the path to the revision csv 
            revision_dfs = [pd.read_csv(revision_df_path) for revision_df_path in revision_df_paths]  # load the revision csvs for every model as dataframes
            filtered_revision_dfs = [df[df['task_wording'] == 'unknown gender'] for df in revision_dfs]

            gender_associations_df = pd.read_csv(config['gender_association_scores'])

            plot_gender_association_scores(
                data_frames=filtered_revision_dfs,
                output_dir=dirname,
                model_names=MODEL_NAMES, 
                gender_associations_df=gender_associations_df
            )

    # justification word counts plot
    # justification_freqs_df = pd.read_csv(f"{dirname}/justification_word_frequencies.csv")
    # output_path = f'{dirname}/justification_plot.png'
    # plot_justification_words(justification_freqs_df,
    #                          output_path = output_path,
    #                          desired_words=['professional', 'tone', 'clarity', 'inclusive', 'engaging'])

        


if __name__=="__main__":
    main(f'{EXPERIMENT_PATH}/config.json')
